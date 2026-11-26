#!/usr/bin/env python3
"""moex/tools/describe.py - a normalized MoE geometry from a Hugging Face config.json.

Strata hard-codes Qwen3.8-Flash-Next's geometry (include/strata/core/layout.hpp, ModelGeometry).  A port needs the
same numbers for the target model, plus what its layers are made of, so the gap to Strata's kernels is explicit.
Understands: qwen4_exp_text (Strata's model), glm5_next_text, deepseek_v4, mimo_v2.

    python moex/tools/describe.py <model_dir_or_config.json> [...] [--out moex/models]
"""
import argparse, json, os, sys
from collections import Counter

def load(path):
    p = os.path.join(path, "config.json") if os.path.isdir(path) else path
    c = json.load(open(p))
    t = c.get("text_config") if isinstance(c.get("text_config"), dict) else {}
    return c, {**c, **t}

def describe(path):
    raw, c = load(path)
    mt = c.get("model_type", "?")
    L = c["num_hidden_layers"]; H = c["hidden_size"]
    d = {"model_type": mt, "architecture": (c.get("architectures") or ["?"])[0], "hidden": H, "layers": L}
    # --- which layers are MoE
    if "mlp_layer_types" in c:
        moe = [i for i, x in enumerate(c["mlp_layer_types"]) if x == "sparse"]
    elif "moe_layer_freq" in c and isinstance(c["moe_layer_freq"], list):
        moe = [i for i, x in enumerate(c["moe_layer_freq"]) if x]
    else:
        k = c.get("first_k_dense_replace") or 0
        moe = list(range(k, L))
    E = c.get("n_routed_experts") or c.get("num_experts")
    K = c.get("num_experts_per_tok")
    inter = c["moe_intermediate_size"]
    nsh = c.get("n_shared_experts") or (1 if c.get("shared_expert_intermediate_size") else 0)
    sh_inter = c.get("shared_expert_intermediate_size") or (inter * nsh if nsh else 0)
    d["moe"] = {"moe_layers": len(moe), "dense_layers": L - len(moe), "routed_experts": E, "top_k": K,
                "expert_intermediate": inter, "shared_experts": nsh, "shared_intermediate": sh_inter,
                "scoring": c.get("scoring_func", "softmax"), "topk_method": c.get("topk_method", "greedy"),
                "routed_scaling": c.get("routed_scaling_factor")}
    ep = 3 * H * inter                                  # gate + up + down
    d["derived"] = {"expert_params": ep, "experts_total": E * len(moe), "routed_params_total": ep * E * len(moe),
                    "routed_active_params_per_token": ep * K * len(moe),
                    "shared_active_params_per_token": 3 * H * sh_inter * len(moe)}
    # --- attention families per layer
    fam = Counter(); attn = {}
    lt = c.get("layer_types")
    if mt == "qwen4_exp_text":
        fam.update(lt)
        attn = {"linear": "GatedDeltaNet", "linear_heads_k/v": [c["linear_num_key_heads"], c["linear_num_value_heads"]],
                "full": "QSA (GQA + indexer)", "heads/kv": [c["num_attention_heads"], c["num_key_value_heads"]],
                "head_dim": c["head_dim"], "indexer": {"heads": c["indexer_n_heads"], "dim": c["indexer_head_dim"],
                "compress": c["indexer_compress_ratio"], "budget": c["indexer_budget"]}}
    elif mt == "glm5_next_text":
        fam.update(lt)
        la = c.get("linear_attn_config", {})
        attn = {"linear": "KDA (Kimi delta attention)", "linear_heads": la.get("num_heads"), "linear_head_dim": la.get("head_dim"),
                "full": "DSA (MLA + indexer)", "heads": c["num_attention_heads"], "q_lora": c["q_lora_rank"],
                "kv_lora": c["kv_lora_rank"], "qk_head_dim": c.get("qk_head_dim"), "v_head_dim": c.get("v_head_dim"),
                "indexer": {"heads": c["index_n_heads"], "dim": c["index_head_dim"], "kpool": c.get("index_kpool"),
                            "topk": c["index_topk"]}}
    elif mt == "deepseek_v4":
        cr = c.get("compress_ratios", [])
        fam.update({f"compressed x{r}" if r else "dense/SWA": 1 for r in []})
        for r in cr[:L]:
            fam["sparse (compress %d)" % r if r else "window only"] += 1
        attn = {"full": "compressed sparse attention (MLA-style latent, indexer)", "heads": c["num_attention_heads"],
                "kv_heads": c["num_key_value_heads"], "head_dim": c["head_dim"], "q_lora": c.get("q_lora_rank"),
                "rope_dim": c.get("qk_rope_head_dim"), "sliding_window": c.get("sliding_window"),
                "indexer": {"heads": c["index_n_heads"], "dim": c["index_head_dim"], "topk": c["index_topk"]},
                "compress_ratios": sorted(set(cr))}
    elif mt == "mimo_v2":
        # MiMo-V2: hybrid sliding-window (128) and global attention; global layers from the layer pattern
        pat = c.get("hybrid_layer_pattern") or c.get("attention_type_pattern")
        if isinstance(pat, list) and len(pat) == L:
            fam.update("swa" if x else "full" for x in pat)
        attn = {"swa": {"window": c.get("sliding_window"), "heads/kv": [c.get("swa_num_attention_heads"), c.get("swa_num_key_value_heads")],
                        "qk/v dim": [c.get("swa_head_dim"), c.get("swa_v_head_dim")], "sink_bias": c.get("add_swa_attention_sink_bias")},
                "full": {"heads/kv": [c["num_attention_heads"], c["num_key_value_heads"]], "qk/v dim": [c["head_dim"], c.get("v_head_dim")]}}
    d["attention"] = {"per_layer": dict(fam), **attn}
    hc = c.get("hc_count") or c.get("hc_mult")
    d["hyper_connections"] = {"streams": hc, "kind": "low-rank (Qwen)" if c.get("hc_lowrank") else ("Sinkhorn (mHC)" if c.get("hc_sinkhorn_iters") else None)} if hc else None
    d["mtp_layers"] = c.get("mtp_num_hidden_layers") or c.get("num_nextn_predict_layers") or 0
    d["ngram_ple"] = bool(c.get("ngram_vocab_size_base"))
    d["vision"] = bool(raw.get("vision_config") or c.get("vision_dim"))
    d["vocab"] = c.get("vocab_size"); d["max_context"] = c.get("max_position_embeddings")
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+"); ap.add_argument("--out")
    a = ap.parse_args()
    for p in a.paths:
        d = describe(p)
        name = os.path.basename(os.path.normpath(p if os.path.isdir(p) else os.path.dirname(p)))
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            json.dump(d, open(os.path.join(a.out, f"{d['model_type']}.json"), "w"), indent=1)
        m, x = d["moe"], d["derived"]
        print(f"{name} [{d['model_type']}]: {d['layers']} layers ({m['moe_layers']} MoE), hidden {d['hidden']}, "
              f"{m['routed_experts']} experts x {m['expert_intermediate']} top-{m['top_k']} + {m['shared_experts']} shared, "
              f"{m['scoring']}; expert {x['expert_params']/1e6:.1f}M, routed active {x['routed_active_params_per_token']/1e9:.2f}B/token, "
              f"all routed {x['routed_params_total']/1e9:.0f}B; attention {d['attention']['per_layer']}; "
              f"HC {d['hyper_connections']}; MTP {d['mtp_layers']}; PLE {d['ngram_ple']}; vision {d['vision']}")

if __name__ == "__main__":
    main()
