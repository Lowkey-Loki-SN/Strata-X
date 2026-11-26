#!/usr/bin/env python3
"""moex/tools/gguf_inventory.py - what a (sharded) GGUF is made of, for a Strata-style port.  Reads headers only.

Per category (routed experts, shared expert, router, attention, linear attention, indexer, hyper-connections, MTP,
embeddings/head, other): bytes and quant types.  Routed-expert quant types are checked against the formats Strata's
expert kernels already run (src/kernels/cpu + GPU expert cache, v0.1.40).

    python moex/tools/gguf_inventory.py <first-shard.gguf>
"""
import os, re, struct, sys
from collections import Counter, defaultdict

GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
        12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
        20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
        29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4"}
# Strata v0.1.40.1 expert paths, read from its source (2026-10-07):
#   GPU decode (src/kernels/cuda/native_mmvq.cu native_mmvq_supported): the set below;
#   GPU prompt  (src/kernels/cuda/dequant_bf16.cu geometry + the i-quant kernels): the same set;
#   CPU         (src/kernels/cpu/native_expert.cpp): ANY type ggml-cpu has a vec_dot for (tuned AVX2/AVX-512 kernels
#               for the common ones, ggml's own vec_dot otherwise) - so Q2_K / MXFP4 run on the CPU, only not on the GPU.
STRATA_GPU_EXPERT = {"Q4_0", "Q5_0", "Q5_1", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS", "Q2_0",
                     "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_M"}
# Strata-X (moex): Q2_K on the GPU (native_mmvq + dequant_bf16 type 10, 2026-10-07), then IQ1_S, MXFP4 and NVFP4 - the
# fork's native_mmvq_supported today (serve/quant_check.py GPU; moex/docs/QUANTS.md has the prompt path's and the dense rules)
STRATA_X_GPU_EXPERT = STRATA_GPU_EXPERT | {"Q2_K", "IQ1_S", "MXFP4", "NVFP4"}
STRATA_EXPERT = STRATA_X_GPU_EXPERT          # kept for callers: "runs everywhere Strata-X runs experts"
SC = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

def rd(f, fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def rstr(f): return f.read(rd(f, "<Q")).decode("utf-8", "replace")
def skip_val(f, t):
    if t in SC: f.seek(struct.calcsize(SC[t]), 1); return None
    if t == 8: return rstr(f)
    if t == 9:
        et, n = rd(f, "<I"), rd(f, "<Q")
        if et in SC: f.seek(struct.calcsize(SC[et]) * n, 1); return None
        for _ in range(n): skip_val(f, et)
        return None
    raise ValueError(t)

def read_shard(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver = rd(f, "<I"); nt = rd(f, "<Q"); nkv = rd(f, "<Q")
        kv = {}; align = 32
        for _ in range(nkv):
            k = rstr(f); t = rd(f, "<I")
            if t in SC:
                v = rd(f, SC[t])
            else:
                v = skip_val(f, t)
            kv[k] = v
            if k == "general.alignment": align = v
        tensors = []
        for _ in range(nt):
            name = rstr(f); nd = rd(f, "<I"); dims = [rd(f, "<Q") for _ in range(nd)]; ty = rd(f, "<I"); off = rd(f, "<Q")
            tensors.append([name, dims, GGML.get(ty, f"type{ty}"), off])
        data = (f.tell() + align - 1) // align * align
    size = os.path.getsize(path)
    tensors.sort(key=lambda t: t[3])
    for i, t in enumerate(tensors):
        nxt = tensors[i + 1][3] if i + 1 < len(tensors) else size - data
        t.append(nxt - t[3])
    return kv, tensors

def category(name, n_layers):
    m = re.match(r"blk\.(\d+)\.", name)
    if m and n_layers is not None and int(m.group(1)) >= n_layers:
        return "mtp"
    if "nextn" in name or name.startswith("mtp"): return "mtp"
    if "_exps" in name: return "routed_experts"
    if "_shexp" in name: return "shared_expert"
    if "ffn_gate_inp" in name or "exp_probs" in name: return "router"
    if "indexer" in name: return "indexer"
    if re.search(r"(^|\.)hc_|_hc_|hc\.", name): return "hyper_connections"
    if "ssm" in name or "kda" in name or "linear_attn" in name: return "linear_attention"
    if "attn" in name: return "attention"
    if "ffn_" in name: return "dense_ffn"
    if name.startswith(("token_embd", "output", "per_layer")): return "embed_head"
    return "other"

def main():
    first = sys.argv[1]
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first)
    shards = [first] if not m else [re.sub(r"-\d{5}-of-", "-%05d-of-" % i, first) for i in range(1, int(m.group(2)) + 1)]
    kv, tens = {}, []
    for s in shards:
        k, t = read_shard(s); kv.update({a: b for a, b in k.items() if b is not None}); tens += t
    arch = kv.get("general.architecture", "?")
    nl = kv.get(f"{arch}.block_count")
    bytes_by = Counter(); types_by = defaultdict(Counter); exp_types = Counter(); exp_bytes = Counter()
    for name, dims, ty, off, nb in tens:
        c = category(name, nl)
        bytes_by[c] += nb; types_by[c][ty] += 1
        if c == "routed_experts":
            part = "gate_up" if "gate_up" in name else re.sub(r".*ffn_(\w+?)_exps.*", r"\1", name)
            exp_types[(part, ty)] += 1; exp_bytes[ty] += nb
    total = sum(bytes_by.values())
    print(f"{os.path.basename(first)}: arch {arch}, {len(shards)} shard(s), {len(tens)} tensors, {total/1e9:.1f} GB, block_count {nl}")
    for c, b in bytes_by.most_common():
        print(f"  {c:18s} {b/1e9:7.2f} GB  {dict(types_by[c].most_common(6))}")
    print("  routed expert tensors by part/type:", {f"{p}:{t}": n for (p, t), n in sorted(exp_types.items())})
    bad = sorted(t for t in exp_bytes if t not in STRATA_GPU_EXPERT)
    print(f"  Strata GPU expert kernels cover: {sorted(t for t in exp_bytes if t in STRATA_GPU_EXPERT) or '-'}; "
          f"GPU kernel missing: {bad or 'none'}" + (f" ({sum(exp_bytes[t] for t in bad)/1e9:.1f} GB of experts; "
          "the CPU path runs them through ggml's vec_dot)" if bad else ""))

if __name__ == "__main__":
    main()
