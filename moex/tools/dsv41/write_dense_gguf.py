"""moex/tools/dsv41/write_dense_gguf.py - the V4.1 trunk's dense weights as a `deepseek41` GGUF for strata-x.

    python write_dense_gguf.py --out FILE.gguf [--dense DIR] [--index index.json] [--hf DIR] [--pack DIR]

Reads the pack's dense safetensors (the release's bytes) and writes everything strata-x loads on the GPUs, named as
its DeepSeek-V4 path names them (blk.N.attn_q_a, ...) plus V4.1's new tensors. The routed experts stay in the
pack's record files and the engram tables in its table files; this file only describes where they are
(deepseek41.expert_record.*, deepseek41.engram.*), relative to a pack directory given to the engine.

Types follow the deployed DSV4-Flash Q3_K file's dense layout (the user's recipe, DSV41_PLAN.md §6):
  Q8_0  from FP8 x E8M0 (32x32 blocks) via f32: attention projections, wo_a (dequantized to BF16 first, as the
        release's convert.py does), wo_b, the shared expert, engram wkv; and the head (from BF16)
  BF16  the embedding and the router, as released (the engine makes an exact F32 copy of the router)
  F32   norms, hyper-connection weights, sinks, biases (exact) - and, unlike DSV4's file, the tensors that feed V4.1's
        top-k selections exactly: the compressor wkv/wgate (the reference pools in fp32), the indexer wq_b (FP8 x
        scale, exact in f32), wk and weights_proj (DSV41_PORT.md, decision 2)
Q8_0 blocks are ggml's quantize_row_q8_0_ref (gguf-py's Q8_0: amax/127, roundf).

Engram tensors (layers 1 and 14: wkv, q_weight, k_weight) come from shards 47/48's dense files; while those are
missing the file is written without them and deepseek41.engram.present = false.

The DSpark drafter (the release's mtp.0-2: three ratio-0 blocks, 128 experts top-3 each) is written as dspark.blk.J.*
with the trunk's names and types, plus dspark.main_proj / main_norm (block 0) and dspark.output_norm, markov_embed,
markov_head (BF16 [vocab, rank]) and confidence (F32) from block 2; its experts are the pack's experts/M0-2.bin records
(deepseek41.dspark.*). --no-dspark leaves it out.
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import engram_tables  # noqa: E402

D = pathlib.Path(_os.path.join(_MODELS, 'dsv41-1017'))
RECORD_MXFP4 = 18_800_640
GGML_MXFP4 = 39


def dequant_fp8(w, s) -> np.ndarray:
    """FP8 E4M3 [out, in] x E8M0 [ceil(out/32), ceil(in/32)] -> f32, exact."""
    import torch
    out, inn = w.shape
    sf = s.float().repeat_interleave(32, 0)[:out].repeat_interleave(32, 1)[:, :inn]
    return (w.float() * sf).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--dense", type=pathlib.Path, default=D / "dense")
    ap.add_argument("--index", type=pathlib.Path, default=D / "index.json")
    ap.add_argument("--hf", type=pathlib.Path, default=D / "hf-small", help="config.json + tokenizer")
    ap.add_argument("--reference", type=pathlib.Path, default=D / "reference" / "inference")
    ap.add_argument("--layers", type=int, default=40, help="trunk layers to write (fewer: a small test file)")
    ap.add_argument("--no-dspark", action="store_true", help="leave the DSpark drafter out")
    a = ap.parse_args()

    import gguf
    import torch
    from safetensors import safe_open

    t0 = time.time()
    cfg = json.loads((a.reference / "config.json").read_text())
    hf = json.loads((a.hf / "config.json").read_text())["text_config"]
    weight_map = json.loads(a.index.read_text())["weight_map"]
    opened = {}

    def tensor(name):
        shard = weight_map[name]
        if shard not in opened:
            p = a.dense / shard
            if not p.exists():
                raise FileNotFoundError(p)
            opened[shard] = safe_open(str(p), framework="pt", device="cpu")
        return opened[shard].get_tensor(name)

    def have(name):
        return name in weight_map and (a.dense / weight_map[name]).exists()

    W = gguf.GGUFWriter(str(a.out), "deepseek41")
    Q8 = gguf.GGMLQuantizationType.Q8_0
    stats = {"Q8_0": 0, "F32": 0, "BF16": 0}

    def put_q8(gname, f32):
        q = gguf.quants.quantize(np.ascontiguousarray(f32, dtype=np.float32), Q8)
        W.add_tensor(gname, q, raw_dtype=Q8)
        stats["Q8_0"] += q.nbytes

    def put_f32(gname, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        W.add_tensor(gname, arr)
        stats["F32"] += arr.nbytes

    def put_bf16(gname, t):
        arr = t.contiguous().view(torch.int16).numpy().view(np.uint16)
        W.add_tensor(gname, arr, raw_dtype=gguf.GGMLQuantizationType.BF16)
        stats["BF16"] += arr.nbytes

    def fp8_q8(gname, rel):
        put_q8(gname, dequant_fp8(tensor(rel + ".weight"), tensor(rel + ".scale")))

    def to_f32(t):
        return t.float().numpy()

    n_layer = a.layers
    ratios = list(cfg["compress_ratios"])[:40]
    kv_src, idx_src = list(cfg["kv_source_layers"]), list(cfg["index_source_layers"])
    engram_ids = list(cfg["engram_layer_ids"])
    engram_present = all(have(f"layers.{i}.engram.wkv.weight") for i in engram_ids)

    # ---- metadata (the deepseek4 keys strata-x already reads, plus V4.1's) ----
    p = "deepseek41."
    W.add_name("DeepSeek-V4.1-Flash-1017 (dense, Q8_0)")
    for k, v in (("embedding_length", cfg["dim"]), ("block_count", n_layer), ("context_length", 1048576),
                 ("attention.head_count", cfg["n_heads"]), ("attention.key_length", cfg["head_dim"]),
                 ("attention.value_length", cfg["head_dim"]), ("rope.dimension_count", cfg["rope_head_dim"]),
                 ("attention.q_lora_rank", cfg["q_lora_rank"]), ("attention.sliding_window", cfg["window_size"]),
                 ("expert_count", cfg["n_routed_experts"]), ("expert_used_count", cfg["n_activated_experts"]),
                 ("expert_feed_forward_length", cfg["moe_inter_dim"]), ("expert_shared_count", cfg["n_shared_experts"]),
                 ("attention.indexer.head_count", cfg["index_n_heads"]), ("attention.indexer.key_length", cfg["index_head_dim"]),
                 ("attention.indexer.top_k", cfg["index_topk"]), ("attention.output_group_count", cfg["o_groups"]),
                 ("attention.output_lora_rank", cfg["o_lora_rank"]), ("hyper_connection.count", cfg["hc_mult"]),
                 ("hyper_connection.sinkhorn_iterations", cfg["hc_sinkhorn_iters"]), ("hash_layer_count", 0),
                 ("rope.scaling.original_context_length", cfg["original_seq_len"]), ("expert_gating_func", 4),
                 ("attention.candidate_source_layer", cfg["candidate_source_layer"]),
                 ("attention.candidate_top_blocks", cfg["candidate_topk_blocks"]),
                 ("attention.candidate_block_size", cfg["candidate_block_size"])):
        W.add_uint32(p + k, int(v))
    for k, v in (("attention.layer_norm_rms_epsilon", cfg["norm_eps"]), ("hyper_connection.epsilon", cfg["hc_eps"]),
                 ("attention.compress_rope_freq_base", cfg["compress_rope_theta"]), ("rope.freq_base", cfg["rope_theta"]),
                 ("rope.scaling.factor", cfg["rope_factor"]), ("rope.scaling.yarn_beta_fast", cfg["beta_fast"]),
                 ("rope.scaling.yarn_beta_slow", cfg["beta_slow"]), ("expert_weights_scale", cfg["route_scale"]),
                 ("swiglu_clamp_exp", cfg["swiglu_limit"]), ("swiglu_clamp_shexp", cfg["swiglu_limit"])):
        W.add_float32(p + k, float(v))
    W.add_string(p + "rope.scaling.type", "yarn")
    W.add_bool(p + "expert_weights_norm", bool(hf.get("norm_topk_prob", True)))
    W.add_array(p + "attention.compress_ratios", [int(r) for r in ratios[:n_layer]])
    W.add_array(p + "attention.kv_source_layers", kv_src)
    W.add_array(p + "attention.index_source_layers", idx_src)
    W.add_uint32("tokenizer.ggml.bos_token_id", int(hf.get("bos_token_id", 0)))
    W.add_uint32("tokenizer.ggml.eos_token_id", int(hf.get("eos_token_id", 1)))
    # the routed experts: one record file per layer in the pack (fp4_repack.expert_record; convert_pack.py)
    W.add_array(p + "expert_record.type", [GGML_MXFP4] * n_layer)
    W.add_array(p + "expert_record.stride", [RECORD_MXFP4] * n_layer)
    W.add_array(p + "expert_record.file", [f"experts/L{i:02d}.bin" for i in range(n_layer)])
    # engram: the fixed hash inputs (engram_tables.py, checked against the release's engram.py) and the table files
    def add_i64(key, vals):   # gguf-py types python ints as INT32; the multipliers need 64 bits
        W.add_key_value(key, [int(v) for v in vals], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT64)

    et = engram_tables.build(a.hf)
    W.add_bool(p + "engram.present", engram_present)
    W.add_array(p + "engram.layer_ids", [int(x) for x in et["layer_ids"]])
    add_i64(p + "engram.num_embeddings", et["num_embeddings"])
    W.add_uint32(p + "engram.max_ngram_size", int(et["max_ngram"]))
    W.add_uint32(p + "engram.head_count", int(et["n_heads"]))
    W.add_uint32(p + "engram.head_dim", int(et["head_dim"]))
    W.add_uint32(p + "engram.compressed_vocab_size", int(et["n_compressed"]))
    W.add_int32(p + "engram.pad_id", int(et["pad_id"]))
    add_i64(p + "engram.multipliers", [x for row in et["multipliers"] for x in row])   # [layer][max_ngram]
    add_i64(p + "engram.primes", [x for l in et["primes"] for g in l for x in g])        # [layer][ngram-1][head]
    add_i64(p + "engram.offsets", [x for row in et["offsets"] for x in row])             # [layer][(ngram-1)*head]
    # the token map is a tensor, not a metadata array: strata-x's GGUF reader keeps only the first 64 items of an array
    token_map = np.asarray(et["token_map"], dtype=np.int32)
    W.add_array(p + "engram.weight_file", [f"engram/L{i:02d}.weight.bin" for i in et["layer_ids"]])
    W.add_array(p + "engram.scale_file", [f"engram/L{i:02d}.scale.bin" for i in et["layer_ids"]])
    n_dsp = 0 if a.no_dspark else int(cfg.get("n_mtp_layers", 0))
    if n_dsp:
        q = p + "dspark."
        W.add_uint32(q + "block_count", n_dsp)
        W.add_uint32(q + "block_size", int(cfg["dspark_block_size"]))
        W.add_uint32(q + "noise_token_id", int(cfg["dspark_noise_token_id"]))
        W.add_array(q + "target_layers", [int(x) for x in cfg["dspark_target_layer_ids"]])
        W.add_uint32(q + "markov_rank", int(cfg["dspark_markov_rank"]))
        W.add_uint32(q + "expert_count", int(cfg["dspark_n_routed_experts"]))
        W.add_uint32(q + "expert_used_count", int(cfg["dspark_n_activated_experts"]))
        W.add_array(q + "expert_record.type", [GGML_MXFP4] * n_dsp)
        W.add_array(q + "expert_record.stride", [RECORD_MXFP4] * n_dsp)
        W.add_array(q + "expert_record.file", [f"experts/M{j}.bin" for j in range(n_dsp)])

    # ---- tensors ----
    W.add_tensor("engram.token_map", token_map, raw_dtype=gguf.GGMLQuantizationType.I32)
    put_bf16("token_embd.weight", tensor("embed.weight"))
    put_f32("output_norm.weight", to_f32(tensor("norm.weight")))
    put_q8("output.weight", to_f32(tensor("head.weight")))
    def put_block(pre, g):
        """The tensors every block has (attention without compression, mHC, router, shared expert)."""
        put_f32(g + "attn_norm.weight", to_f32(tensor(pre + "attn_norm.weight")))
        put_f32(g + "ffn_norm.weight", to_f32(tensor(pre + "ffn_norm.weight")))
        for part in ("attn", "ffn"):
            put_f32(g + f"hc_{part}_fn.weight", to_f32(tensor(pre + f"hc_{part}_fn")))
            put_f32(g + f"hc_{part}_base.weight", to_f32(tensor(pre + f"hc_{part}_base")))
            put_f32(g + f"hc_{part}_scale.weight", to_f32(tensor(pre + f"hc_{part}_scale")))
        fp8_q8(g + "attn_q_a.weight", pre + "attn.wq_a")
        put_f32(g + "attn_q_a_norm.weight", to_f32(tensor(pre + "attn.q_norm.weight")))
        fp8_q8(g + "attn_q_b.weight", pre + "attn.wq_b")
        fp8_q8(g + "attn_kv.weight", pre + "attn.wkv")
        put_f32(g + "attn_kv_a_norm.weight", to_f32(tensor(pre + "attn.kv_norm.weight")))
        put_f32(g + "attn_sinks.weight", to_f32(tensor(pre + "attn.attn_sink")))
        woa = dequant_fp8(tensor(pre + "attn.wo_a.weight"), tensor(pre + "attn.wo_a.scale"))
        put_q8(g + "attn_output_a.weight", torch.from_numpy(woa).to(torch.bfloat16).float().numpy())
        fp8_q8(g + "attn_output_b.weight", pre + "attn.wo_b")
        put_bf16(g + "ffn_gate_inp.weight", tensor(pre + "ffn.gate.weight"))
        put_f32(g + "exp_probs_b.bias", to_f32(tensor(pre + "ffn.gate.bias")))
        if have(pre + "ffn.gate.bias_vl"):
            put_f32(g + "exp_probs_b_vl.bias", to_f32(tensor(pre + "ffn.gate.bias_vl")))
        fp8_q8(g + "ffn_gate_shexp.weight", pre + "ffn.shared_experts.w1")
        fp8_q8(g + "ffn_up_shexp.weight", pre + "ffn.shared_experts.w3")
        fp8_q8(g + "ffn_down_shexp.weight", pre + "ffn.shared_experts.w2")

    for il in range(n_layer):
        pre, g = f"layers.{il}.", f"blk.{il}."
        r = ratios[il]
        put_block(pre, g)
        if il in kv_src:   # the layer's compressor: ratio 2 pools with a gate, ratio 1 only projects
            put_f32(g + "attn_compressor_kv.weight", to_f32(tensor(pre + "attn.compressor.wkv.weight")))
            put_f32(g + "attn_compressor_norm.weight", to_f32(tensor(pre + "attn.compressor.norm.weight")))
            if r > 1:
                put_f32(g + "attn_compressor_gate.weight", to_f32(tensor(pre + "attn.compressor.wgate.weight")))
            # the index keys (only a layer that compresses its own KV owns them)
            put_f32(g + "indexer.attn_k.weight", to_f32(tensor(pre + "attn.indexer.wk.weight")))
            put_f32(g + "indexer.k_norm.weight", to_f32(tensor(pre + "attn.indexer.k_norm.weight")))
        if il in idx_src:
            put_f32(g + "indexer.attn_q_b.weight",
                    dequant_fp8(tensor(pre + "attn.indexer.wq_b.weight"), tensor(pre + "attn.indexer.wq_b.scale")))
            put_f32(g + "indexer.proj.weight", to_f32(tensor(pre + "attn.indexer.weights_proj.weight")))
        if il in engram_ids and engram_present:
            fp8_q8(g + "engram.wkv.weight", pre + "engram.wkv")
            put_f32(g + "engram.q_weight", to_f32(tensor(pre + "engram.q_weight")))
            put_f32(g + "engram.k_weight", to_f32(tensor(pre + "engram.k_weight")))
        print(f"  layer {il:2d} done ({time.time() - t0:.0f} s)", flush=True)
    for j in range(n_dsp):   # the DSpark drafter's blocks (ratio 0: no compressor, indexer or engram)
        pre, g = f"mtp.{j}.", f"dspark.blk.{j}."
        put_block(pre, g)
        if j == 0:
            fp8_q8("dspark.main_proj.weight", pre + "main_proj")
            put_f32("dspark.main_norm.weight", to_f32(tensor(pre + "main_norm.weight")))
        if j == n_dsp - 1:
            put_f32("dspark.output_norm.weight", to_f32(tensor(pre + "norm.weight")))
            put_bf16("dspark.markov_embed.weight", tensor(pre + "markov_head.embed.weight").to(torch.bfloat16))
            put_bf16("dspark.markov_head.weight", tensor(pre + "markov_head.head.weight").to(torch.bfloat16))
            put_f32("dspark.confidence.weight", to_f32(tensor(pre + "confidence_head.proj.weight")))
        print(f"  DSpark block {j} done ({time.time() - t0:.0f} s)", flush=True)

    W.write_header_to_file()
    W.write_kv_data_to_file()
    W.write_tensors_to_file(progress=False)
    W.close()
    total = sum(stats.values())
    print(f"{a.out}: {total / 1e9:.2f} GB (Q8_0 {stats['Q8_0'] / 1e9:.2f}, F32 {stats['F32'] / 1e9:.2f}, "
          f"BF16 {stats['BF16'] / 1e9:.2f}), engram tensors {'in' if engram_present else 'NOT yet'}, DSpark blocks {n_dsp}; "
          f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
