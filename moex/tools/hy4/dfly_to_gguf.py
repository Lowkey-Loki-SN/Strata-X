#!/usr/bin/env python3
"""moex/tools/hy4/dfly_to_gguf.py - Hy4-flash's DFly2 drafter (tencent/Hy4-flash, model.dfly_layers.*) as a GGUF for
strata-x's --draft-dfly.

    ~/AI/Strata/.venv/bin/python -I moex/tools/hy4/dfly_to_gguf.py \\
        ~/AI/Models/hy4-flash/dfly2/model-00148-of-00148.safetensors CONFIG.json OUT.gguf [--type q8_0|bf16]

The shard holds the 87 drafter tensors (BF16); CONFIG.json is the release's config.json (its dfly_* keys and text_config).
What changes, nothing else:
- every 2-D matrix as Q8_0 (default; ggml's quantize_row_q8_0_ref: d = amax / 127 in F32, codes roundf(x / d), d as FP16)
  or BF16 as stored; norms, the fusion gate and the sinks as F32 (exact: BF16 values);
- kv_b_proj [H x (192 + 256), 512] split per head as llama.cpp's DeepSeek converter splits it: attn_k_b [192, 512, H] (each
  head's nope rows transposed: q_nope -> latent) and attn_v_b [512, 256, H] (latent -> v);
- the confidence head is left out (vLLM ignores it: dfly.py _ignored_prefixes).
Tensor names: dfly.fusion.{enorm,hnorm,gate,eh_proj}, dfly.corr.{hidden_norm,embed_norm,gate,up,down}, dfly.output_norm,
dfly.blk.N.{attn_norm, attn_q_a, attn_q_a_norm, attn_q_b, attn_kv_a_mqa, attn_kv_a_norm, attn_k_b, attn_v_b, attn_sinks,
attn_gate, attn_output, attn_post_norm, ffn_norm, ffn_gate, ffn_up, ffn_down}.weight - ggml's order: dims[0] the input.
"""
import argparse
import json
import struct
import sys

import numpy as np

STR, ARR, U32, F32, I32, BOOL = 8, 9, 4, 6, 5, 7
T_F32, T_Q8_0, T_BF16 = 0, 8, 30


def read_safetensors(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    base = 8 + n

    def load(name):
        v = hdr[name]
        if v["dtype"] != "BF16":
            raise SystemExit(f"{name}: {v['dtype']}, BF16 expected")
        a, b = v["data_offsets"]
        f.seek(base + a)
        raw = np.frombuffer(f.read(b - a), dtype=np.uint16).reshape(v["shape"])
        return raw

    return hdr, load


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def q8_0(x):
    """ggml's quantize_row_q8_0_ref over the last axis (a multiple of 32): 34-byte blocks {fp16 d, 32 int8}."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, n = x.reshape(-1, x.shape[-1]).shape
    if n % 32:
        raise SystemExit(f"rows of {n} values are not whole Q8_0 blocks")
    b = x.reshape(rows, n // 32, 32)
    amax = np.max(np.abs(b), axis=-1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    v = (b * idv[..., None]).astype(np.float32)
    q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int8)   # roundf: half away from zero
    out = np.empty((rows, n // 32, 34), dtype=np.uint8)
    out[..., 0:2] = d.astype(np.float16).view(np.uint8).reshape(rows, n // 32, 2)
    out[..., 2:] = q.view(np.uint8)
    return out.tobytes()


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")

    def w(self, fmt, v):
        self.f.write(struct.pack(fmt, v))

    def s(self, x):
        b = x.encode("utf-8")
        self.w("<Q", len(b))
        self.f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--type", default="q8_0", choices=("q8_0", "bf16"))
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    tc = cfg.get("text_config", cfg)
    hdr, load = read_safetensors(a.shard)
    P = "model.dfly_layers."
    n_layer = int(cfg["dfly_num_hidden_layers"])
    H = int(tc["num_attention_heads"]) if isinstance(tc.get("num_attention_heads"), int) else None
    kv_b0 = hdr[P + "0.self_attn.kv_b_proj.weight"]["shape"]          # [H x (nope + v), kv_lora]
    sinks0 = hdr[P + "0.self_attn.learnable_sink_param"]["shape"]
    H = int(sinks0[0])
    nope, rope, vdim = int(tc["qk_nope_head_dim"]), int(tc["qk_rope_head_dim"]), int(tc["v_head_dim"])
    kv_lora, q_lora, n = int(tc["kv_lora_rank"]), int(tc["q_lora_rank"]), int(tc["hidden_size"])
    if kv_b0 != [H * (nope + vdim), kv_lora]:
        raise SystemExit(f"kv_b_proj {kv_b0}: not [{H} x ({nope} + {vdim}), {kv_lora}]")
    n_ff = int(hdr[P + "0.mlp.gate_proj.weight"]["shape"][0])
    tensors = []   # (name, ggml type, dims (ggml order), bytes)

    def mat(name, w_u16, dims=None):
        """A [out, in] matrix (or a stack) as the chosen type; dims: ggml order (in first)."""
        dims = dims or [w_u16.shape[-1], w_u16.shape[-2]] + list(reversed(w_u16.shape[:-2]))
        if a.type == "bf16":
            tensors.append((name, T_BF16, dims, np.ascontiguousarray(w_u16).tobytes()))
        else:
            tensors.append((name, T_Q8_0, dims, q8_0(bf16_to_f32(w_u16))))

    def vec(name, w_u16):
        tensors.append((name, T_F32, [int(np.prod(w_u16.shape))], bf16_to_f32(w_u16).astype(np.float32).tobytes()))

    vec("dfly.fusion.enorm.weight", load(P + "mtp_fusion.enorm.weight"))
    vec("dfly.fusion.hnorm.weight", load(P + "mtp_fusion.hnorm.weight"))
    vec("dfly.fusion.gate.weight", load(P + "mtp_fusion.gate"))
    mat("dfly.fusion.eh_proj.weight", load(P + "mtp_fusion.eh_proj.weight"))
    vec("dfly.corr.hidden_norm.weight", load(P + "hidden_correction.hidden_norm.weight"))
    vec("dfly.corr.embed_norm.weight", load(P + "hidden_correction.embed_norm.weight"))
    mat("dfly.corr.gate.weight", load(P + "hidden_correction.gate_proj.weight"))
    mat("dfly.corr.up.weight", load(P + "hidden_correction.up_proj.weight"))
    mat("dfly.corr.down.weight", load(P + "hidden_correction.down_proj.weight"))
    vec("dfly.output_norm.weight", load(P + "final_norm.weight"))
    for i in range(n_layer):
        L = f"{P}{i}."
        o = f"dfly.blk.{i}."
        vec(o + "attn_norm.weight", load(L + "input_layernorm.weight"))
        mat(o + "attn_q_a.weight", load(L + "self_attn.q_a_proj.weight"))
        vec(o + "attn_q_a_norm.weight", load(L + "self_attn.q_a_layernorm.weight"))
        mat(o + "attn_q_b.weight", load(L + "self_attn.q_b_proj.weight"))
        mat(o + "attn_kv_a_mqa.weight", load(L + "self_attn.kv_a_proj_with_mqa.weight"))
        vec(o + "attn_kv_a_norm.weight", load(L + "self_attn.kv_a_layernorm.weight"))
        kvb = load(L + "self_attn.kv_b_proj.weight").reshape(H, nope + vdim, kv_lora)
        k_b = np.ascontiguousarray(np.transpose(kvb[:, :nope, :], (0, 2, 1)))   # [H, kv_lora, nope]: q_nope -> latent
        v_b = np.ascontiguousarray(kvb[:, nope:, :])                            # [H, v, kv_lora]: latent -> v
        mat(o + "attn_k_b.weight", k_b, [nope, kv_lora, H])
        mat(o + "attn_v_b.weight", v_b, [kv_lora, vdim, H])
        vec(o + "attn_sinks.weight", load(L + "self_attn.learnable_sink_param"))
        mat(o + "attn_gate.weight", load(L + "self_attn.linear_gate.weight"))
        mat(o + "attn_output.weight", load(L + "self_attn.o_proj.weight"))
        vec(o + "attn_post_norm.weight", load(L + "self_attn.post_proj_norm.weight"))
        vec(o + "ffn_norm.weight", load(L + "post_attention_layernorm.weight"))
        mat(o + "ffn_gate.weight", load(L + "mlp.gate_proj.weight"))
        mat(o + "ffn_up.weight", load(L + "mlp.up_proj.weight"))
        mat(o + "ffn_down.weight", load(L + "mlp.down_proj.weight"))
    kv_layers = [int(x) for x in cfg["dfly_target_kv_layers"]]
    idx_layers = cfg["dfly_target_index_layer"]
    idx_layers = [int(idx_layers)] * n_layer if isinstance(idx_layers, int) else [int(x) for x in idx_layers]
    meta = [
        ("general.architecture", STR, "hy4_dfly"),
        ("general.name", STR, "Hy4-flash DFly2 (moex dfly_to_gguf)"),
        ("hy4_dfly.block_count", U32, n_layer),
        ("hy4_dfly.embedding_length", U32, n),
        ("hy4_dfly.feed_forward_length", U32, n_ff),
        ("hy4_dfly.attention.head_count", U32, H),
        ("hy4_dfly.attention.q_lora_rank", U32, q_lora),
        ("hy4_dfly.attention.kv_lora_rank", U32, kv_lora),
        ("hy4_dfly.attention.key_length_mla", U32, nope + rope),
        ("hy4_dfly.attention.value_length_mla", U32, vdim),
        ("hy4_dfly.rope.dimension_count", U32, rope),
        ("hy4_dfly.attention.layer_norm_rms_epsilon", F32, float(tc["rms_norm_eps"])),
        ("hy4_dfly.mask_token_id", U32, int(cfg["dfly_mask_token_id"])),
        ("hy4_dfly.target_index_topk", U32, int(cfg["dfly_target_index_topk"])),
        ("hy4_dfly.context_local_window", U32, int(cfg.get("dfly_context_local_window") or 0)),
        ("hy4_dfly.target_kv_layers", ARR, (U32, kv_layers)),
        ("hy4_dfly.target_index_layers", ARR, (U32, idx_layers)),
    ]
    w = Writer(a.out)
    w.f.write(b"GGUF")
    w.w("<I", 3)
    w.w("<Q", len(tensors))
    w.w("<Q", len(meta))
    for k, t, v in meta:
        w.s(k)
        w.w("<I", t)
        if t == STR:
            w.s(v)
        elif t == U32:
            w.w("<I", v)
        elif t == F32:
            w.w("<f", v)
        elif t == ARR:
            et, items = v
            w.w("<I", et)
            w.w("<Q", len(items))
            for it in items:
                w.w("<I", it)
    align, off = 32, 0
    for name, t, dims, data in tensors:
        w.s(name)
        w.w("<I", len(dims))
        for d in dims:
            w.w("<Q", int(d))
        w.w("<I", t)
        w.w("<Q", off)
        off += (len(data) + align - 1) // align * align
    pos = w.f.tell()
    w.f.write(b"\0" * ((pos + align - 1) // align * align - pos))
    total = 0
    for name, t, dims, data in tensors:
        w.f.write(data)
        w.f.write(b"\0" * ((len(data) + align - 1) // align * align - len(data)))
        total += len(data)
    w.f.close()
    print(f"{a.out}: {len(tensors)} tensors, {total / 1e9:.3f} GB ({a.type}), {n_layer} layers of {H} heads, FFN {n_ff}, "
          f"target KV layers {kv_layers}, index layers {idx_layers}, mask {cfg['dfly_mask_token_id']}")


if __name__ == "__main__":
    sys.exit(main())
