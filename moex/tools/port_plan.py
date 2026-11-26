#!/usr/bin/env python3
"""moex/tools/port_plan.py - the per-model port plan for Strata-X: every component mapped to Strata's code
(REUSE / ADAPT / NEW) and every blocking dependency checked live on this box.

Strata's capabilities (v0.1.40.1, from its source: include/strata/core/layout.hpp ModelGeometry, src/core/*,
src/kernels/*; 88K lines, model-specific: PLE 83 files, QSA 42, GDN 27, HC 18, router top-10 17, MTP 19).

    python moex/tools/port_plan.py [--models moex/models]
"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import argparse, glob, json, os, re, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GG = _os.path.join(_MODELS, 'GGUFs'); EX = _os.path.join(_MODELS, 'EXL3')
# GPU expert kernels (native_mmvq_supported + dequant_bf16 geometry); the CPU path takes any ggml vec_dot type.
# Q2_K: ported in Strata-X (moex, 2026-10-07: native_q2_k_mmvq + dequant_bf16 type 10; GPU parity test pending).
STRATA_EXPERT_TYPES = {"Q4_0", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "Q2_0",
                       "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_M"}

# component -> (status, what Strata has, what the port needs), per target architecture
COMPONENTS = {
    "deepseek_v4": [
        ("expert arena / GPU expert cache / CPU pool / layer split / profile placement", "ADAPT",
         "generic machinery; kernels assume 2560x640 experts -> 4096x2048 (8x larger GEMVs), 256 experts, top-6"),
        ("expert quant kernels", "ADAPT", "Q3_K and IQ3_XXS run on Strata's GPU kernels (Q3_K was never blocked); MXFP4 (2.3% of UD-IQ3_S) runs on the CPU via ggml and needs a GPU MMVQ port like Q2_K's"),
        ("router", "ADAPT", "softmax top-10 -> sqrtsoftplus + noaux_tc bias, top-6, routed_scaling 1.5, 1 shared expert"),
        ("attention", "NEW", "compressed sparse attention: latent heads 512-dim, 1 KV head, compress x4 / x128 by layer, "
                             "indexer 64x128 top-512, SWA-128 window; reuse QSA's indexer/top-k + KV streaming ideas"),
        ("residual mixing", "ADAPT", "HC x4 low-rank -> Sinkhorn mHC x4 (shared with GLM)"),
        ("linear attention", "DROP", "none (Strata's 27 GDN files unused)"),
        ("PLE n-gram table", "DROP", "none (83 PLE files unused; frees the 28.8 GB page-cache need)"),
        ("MTP / drafting", "BLOCKED", "3 nextn layers not in the local GGUF; local DSpark drafter (dspark-DeepSeek-V4-Flash-Vision-Exp.gguf) "
                                      "needs a DSpark verify path, or fetch the MTP tensors like tools/mtp_fetch.py"),
        ("vision", "ADAPT", "mmproj-DeepSeek-V4-Flash-Vision-Exp-BF16.gguf is local; mtmd knows deepseek4 (pinned and mainline)"),
    ],
    "glm5_next_text": [
        ("expert arena / GPU expert cache / CPU pool / layer split / profile placement", "ADAPT",
         "generic machinery; 4096x2048 experts, 288 experts, top-8, 3 dense layers first"),
        ("expert quant kernels", "ADAPT", "test file glm53-flash-q2k-fast-111.gguf: Q2_K experts - GPU kernels ported in Strata-X (parity test pending), CPU via ggml"),
        ("router", "ADAPT", "sigmoid + noaux_tc, top-8, routed_scaling 2.5, 1 shared expert"),
        ("attention (full)", "ADAPT", "DSA = MLA (q_lora 1536, kv_lora 512, 64x256) + indexer 32x128, kpool 4, top-2048 - "
                                      "QSA's indexer is the same shape of idea (compress 4, budget 2048); MLA replaces GQA"),
        ("linear attention", "ADAPT", "KDA (64x128, channel-wise gates) <- Strata's Gated DeltaNet kernels (3:1 pattern identical)"),
        ("residual mixing", "ADAPT", "HC x4 low-rank -> Sinkhorn mHC x4"),
        ("PLE n-gram table", "DROP", "none"),
        ("MTP / drafting", "ADAPT", "1 nextn layer, like Qwen's single MTP layer (llama.cpp MTP: PR #27917 open)"),
        ("vision", "BLOCKED", "glm5-next is in llama.cpp mainline (merged 2026-09-30/10-01), not in Strata's pinned llama.cpp; "
                              "image path needs the bump + a GLM mmproj"),
    ],
    "mimo_v2": [
        ("expert arena / GPU expert cache / CPU pool / layer split / profile placement", "ADAPT",
         "generic machinery; 4096x2048 experts, 256 experts, top-8, no shared expert"),
        ("expert quant kernels", "BLOCKED", "no MiMo GGUF on disk (EXL3 only); needs a GGUF with Strata-supported expert types"),
        ("router", "ADAPT", "sigmoid + noaux_tc, top-8, no shared expert"),
        ("attention", "NEW", "plain GQA: 39 SWA-128 layers (64/8 heads, 192/128 dims, sink bias) + 9 full layers (64/4); "
                             "simplest of the three, but nothing of QSA/GDN carries over"),
        ("residual mixing", "DROP", "no hyper-connections"),
        ("linear attention", "DROP", "none"),
        ("PLE n-gram table", "DROP", "none"),
        ("MTP / drafting", "ADAPT", "3 nextn layers (in the EXL3 pack); Strata drafts with one"),
        ("vision", "ADAPT", "mimo2 in pinned + mainline llama.cpp; mtmd knows mimo; extras dir has the tower as safetensors, "
                            "a GGUF mmproj must be made"),
    ],
}

def gguf_first(pattern):
    hits = sorted(glob.glob(os.path.join(GG, "**", pattern), recursive=True))
    return hits[0] if hits else None

def expert_types(first):
    if not first:
        return None
    out = subprocess.run(["python3", "-I", os.path.join(ROOT, "tools", "gguf_inventory.py"), first], capture_output=True, text=True)
    m = re.search(r"routed expert tensors by part/type: (\{.*\})", out.stdout)
    return sorted({k.split(":")[1] for k in eval(m.group(1))}) if m else None   # noqa: S307 (own tool's output)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default=os.path.join(ROOT, "models")); a = ap.parse_args()
    free_gb = shutil.disk_usage("/").free / 1e9
    files = {
        "deepseek_v4": {"gguf": gguf_first("DeepSeek-V4-Flash*-00001-of-*.gguf"), "mmproj": gguf_first("mmproj-DeepSeek-V4*.gguf"),
                        "drafter": gguf_first("dspark-DeepSeek-V4*.gguf"), "capture": None},
        "glm5_next_text": {"gguf": gguf_first("GLM-5.3-Flash*-00001-of-*.gguf") or gguf_first("glm53-flash*.gguf"), "mmproj": gguf_first("mmproj-GLM*.gguf"),
                           "drafter": None, "capture": os.path.join(EX, "GLM-5.3-Flash-exl3-3.05bpw/moe_profiles/code.npz")},
        "mimo_v2": {"gguf": gguf_first("MiMo*-00001-of-*.gguf") or gguf_first("MiMo*.gguf"), "mmproj": gguf_first("mmproj-MiMo*.gguf"),
                    "drafter": None, "capture": None},
    }
    print(f"disk free on /: {free_gb:.0f} GB (a ~2.6-3.06 bpw target GGUF needs 90-115 GB)\n")
    for mt, comps in COMPONENTS.items():
        d = json.load(open(os.path.join(a.models, f"{mt}.json")))
        f = files[mt]
        et = expert_types(f["gguf"])
        cap = f["capture"] if f["capture"] and os.path.exists(f["capture"]) else None
        print(f"=== {d['architecture']} [{mt}]")
        for name, status, note in comps:
            print(f"  {status:8s} {name}: {note}")
        blockers = []
        if not f["gguf"]:
            blockers.append(f"no GGUF on disk -> download/convert one ({'room' if free_gb > 110 else 'NOT enough room: free disk first'})")
        elif et and not set(et) <= STRATA_EXPERT_TYPES:
            blockers.append(f"GGUF experts are {et}: not Strata-supported -> requant to IQ3_XXS/IQ2_M (needs ~100 GB free, have {free_gb:.0f}) "
                            "or add a K-quant expert kernel to the fork")
        if not f["mmproj"]:
            blockers.append("no GGUF mmproj on disk (images)")
        if not cap:
            blockers.append("no per-prompt routing capture -> moex/experiments/capture_routing.sh (GPUs needed: LAST)")
        print(f"  local: gguf={os.path.basename(f['gguf']) if f['gguf'] else '-'} experts={et or '-'} mmproj={'yes' if f['mmproj'] else 'no'} "
              f"drafter={'DSpark' if f['drafter'] else '-'} capture={'yes' if cap else 'no'}")
        print("  BLOCKED BY: " + ("; ".join(blockers) if blockers else "nothing local") + "\n")

if __name__ == "__main__":
    main()
