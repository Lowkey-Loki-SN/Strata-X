"""moex/tools/dsv41/ref/ref_oracle.py - an op-level oracle for strata-x's DeepSeek-V4.1 path (DSV41_PORT.md stage S2).

    python ref_oracle.py --text FILE|--ids FILE.npy --steps N [--layers 0-39] [--gemm-act fp8|fp32] --no-engram --out DIR

Runs the release's model.py one token at a time (start_pos = the step, seqlen 1: its decode path; step 0 is a prefill
of one token) and writes every value strata-x probes in decode windows, in llama_dump's format (moex/ref/recorder:
DIR/index.tsv + DIR/dump.bin, record "<probe>-<layer>" of step s).  `strata-x --oracle DIR -t <the ids>` then runs the
same tokens one at a time, compares each probed op with this record and continues from the record's value - so every
op's error is its own.

The layers run one at a time (a block's weights, its routed experts read only when routed); the state that crosses
layers per step - the streams, the carried pre-mix, the index sources' top-k lists and layer 20's candidates - is kept
per step.  The ops mirror model.py's Block / Attention / Compressor / Indexer / MoE line by line, adding the records;
where the engine computes a value the reference skips, the record is the engine's convention: a ratio-2 KV source's
latent, index key and compressed row are zeros at an even position (no block completes), as the engine's buffers hold.
The routed experts' outputs are recorded unweighted (the engine's rows; model.py scales the SwiGLU hidden first), and
the MoE sum as model.py computes it.  The GPUs are never touched.
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import gc
import json
import os
import pathlib
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ref_model import Reference  # noqa: E402

D = pathlib.Path(_os.path.join(_MODELS, 'dsv41-1017'))


class Dump:
    """llama_dump's format: index.tsv lines "step name type ne0 ne1 ne2 ne3 offset", payloads back to back in dump.bin."""

    def __init__(self, out: pathlib.Path):
        out.mkdir(parents=True, exist_ok=True)
        self.bin = open(out / "dump.bin", "wb")
        self.idx = open(out / "index.tsv", "w")
        self.off = 0
        self.step = 0

    def put(self, name: str, il: int, t, ne=None, i32: bool = False):
        a = np.ascontiguousarray((t.float() if not i32 else t.int()).numpy() if torch.is_tensor(t) else t,
                                 dtype=np.int32 if i32 else np.float32).reshape(-1)
        ne = list(ne) if ne is not None else [a.size]
        ne = ne + [1] * (4 - len(ne))
        assert int(np.prod(ne)) == a.size, (name, ne, a.size)
        n = f"{name}-{il}" if il >= 0 else name
        self.bin.write(a.tobytes())
        self.idx.write(f"{self.step}\t{n}\t{'i32' if i32 else 'f32'}\t{ne[0]}\t{ne[1]}\t{ne[2]}\t{ne[3]}\t{self.off}\n")
        self.off += a.nbytes

    def close(self):
        self.bin.close()
        self.idx.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=pathlib.Path)
    src.add_argument("--ids", type=pathlib.Path)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--gemm-act", choices=["fp8", "fp32"], default="fp8")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--pack", type=pathlib.Path, default=D / "native")
    ap.add_argument("--dense", type=pathlib.Path, default=D / "dense")
    ap.add_argument("--reference", type=pathlib.Path, default=D / "reference" / "inference")
    ap.add_argument("--tokenizer", type=pathlib.Path, default=D / "hf-small")
    ap.add_argument("--index", type=pathlib.Path, default=D / "index.json")
    a = ap.parse_args()
    os.environ["DSV41_REF_GEMM_ACT"] = a.gemm_act
    lo, _, hi = a.layers.partition("-")
    layers = list(range(int(lo), int(hi or lo) + 1))
    if layers[0] != 0:
        raise SystemExit("ref_oracle: the layers start at 0")

    if a.ids:
        ids = np.load(a.ids).astype(np.int64).reshape(-1)[: a.steps]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(a.tokenizer))
        ids = np.asarray(tok.encode(a.text.read_text(), add_special_tokens=False)[: a.steps], dtype=np.int64)
    S = len(ids)
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "ids.npy", ids)
    (a.out / "ids.txt").write_text(",".join(str(int(x)) for x in ids) + "\n")

    ref = Reference(a.pack, a.dense, a.reference, a.tokenizer, a.index, max_seq_len=max(256, S + 8),
                    no_engram=a.no_engram, threads=a.threads)
    M, args = ref.M, ref.args
    shared = M.shared_attn
    dump = Dump(a.out)
    n, hc = args.dim, args.hc_mult

    with torch.inference_mode():
        # the engram's row numbers of every position (model.py Transformer.forward: one call over the sequence)
        hashes = None if a.no_engram else ref.engram_hash(torch.from_numpy(ids).view(1, -1), 0, None)
        emb = ref.embed()(torch.from_numpy(ids).view(1, -1))           # [1, S, n] bf16
        h = emb.unsqueeze(2).repeat(1, 1, hc, 1)                       # [1, S, 4, n]
        pre = M.make_identity_pre_mix(h, hc)                           # [1, S, 4] fp32
        for s in range(S):
            dump.step = s
            dump.put("inp_embd", -1, emb[0, s], [n])
            dump.put("hc_init", -1, h[0, s], [n, hc])
        sel_of, cand_of = {}, {}   # (source layer, step) -> model.py's top-k list (with the window offset) / candidates
        t_all = time.time()
        for il in layers:
            t0 = time.time()
            blk = ref.block(il, experts=False)
            att, moe = blk.attn, blk.ffn
            t_load = time.time() - t0
            for s in range(S):
                dump.step = s
                x = h[:, s:s + 1].clone()
                pm = pre[:, s:s + 1].clone()
                if blk.engram is not None:   # before the layer's mixes, as Transformer.forward runs it
                    x = engram_step(blk.engram, il, x, hashes[:, s:s + 1, blk.engram.layer_hash_index, :], dump)
                x, pm = block_step(ref, M, blk, att, moe, il, x, s, pm, dump, shared, sel_of, cand_of)
                h[:, s:s + 1] = x
                pre[:, s:s + 1] = pm
            print(f"  layer {il:2d}: load {t_load:5.1f} s, {S} steps {time.time() - t0 - t_load:6.1f} s", flush=True)
            del blk, att, moe
            gc.collect()
        if layers[-1] == args.n_layers - 1:
            norm, head = ref.head()
            for s in range(S):
                dump.step = s
                hf = M.Block.hc_pre(None, h[:, s:s + 1], pre[:, s:s + 1])
                dump.put("hc_head", -1, hf[0, 0], [n])
                hn = norm(hf)
                dump.put("result_norm", -1, hn[0, 0], [n])
                lg = head(hn, full_logits=True)[0, 0]
                dump.put("result_output", -1, lg, [lg.numel()])
                dump.put("logits", -1, lg, [lg.numel()])
    dump.close()
    (a.out / "run.json").write_text(json.dumps(dict(steps=S, layers=[layers[0], layers[-1]], gemm_act=a.gemm_act,
                                                    no_engram=a.no_engram, seconds=round(time.time() - t_all, 1)), indent=1))
    print(f"oracle: {S} steps x layers {layers[0]}-{layers[-1]} in {time.time() - t_all:.0f} s -> {a.out}", flush=True)
    return 0


def engram_step(eng, il, x, hash_ids, dump):
    """model.py Engram.forward for one token, with the engine's probes (eg_emb: the 24 rows dequantized; eg_kv: wkv's
    output; eg_out: the streams after the write)."""
    emb = eng.embed(hash_ids)                                   # [1, 1, cols, dim]
    dump.put("eg_emb", il, emb[0, 0].flatten(), [emb[0, 0].numel()])
    kv = eng.wkv(emb.flatten(-2))
    dump.put("eg_kv", il, kv[0, 0], [kv.size(-1)])
    key, value = kv.split([eng.hc_mult * eng.dim, eng.dim], dim=-1)
    key = key.float().unflatten(-1, (eng.hc_mult, eng.dim))
    weight = eng.q_weight.float() * eng.k_weight.float()
    h, eps = x.float(), eng.eps
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * weight * key).sum(-1) * rstd * eng.dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(eng.clamp_value).sqrt(), dot))
    out = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
    dump.put("eg_out", il, out[0, 0], [eng.dim, eng.hc_mult])
    return out


def hc_mixes(blk, x, fn, scale, base):
    """Block.hc_mixes, also returning the raw mixes (the engine's hc_mixes probe)."""
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + blk.norm_eps)
    mixes = F.linear(xf, fn) * rsqrt
    pre, post, comb = blk_split(blk, mixes, scale, base)
    return pre, post, comb, mixes


def blk_split(blk, mixes, scale, base):
    import kernel  # ref_kernels, installed as model.py's kernel module
    return kernel.hc_split_sinkhorn(mixes, scale, base, blk.hc_mult, blk.hc_sinkhorn_iters, blk.hc_eps)


def block_step(ref, M, blk, att, moe, il, x, s, pre_mix, dump, shared, sel_of, cand_of):
    """model.py Block.forward for one token at position s, with the engine's probes."""
    n = blk.attn_norm.dim
    residual = x
    a_pre, a_post, a_comb, mixes = hc_mixes(blk, x, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
    dump.put("hc_mixes", il, mixes[0, 0], [24])
    dump.put("hc_pre", il, a_pre[0, 0], [4])
    dump.put("hc_post", il, a_post[0, 0], [4])
    dump.put("hc_comb", il, a_comb[0, 0].flatten(), [4, 4])
    x = blk.hc_pre(x, pre_mix)
    dump.put("hc_attn_pre", il, x[0, 0], [n])
    x = blk.attn_norm(x)
    dump.put("attn_norm", il, x[0, 0], [n])
    x = attention_step(M, att, il, x, s, dump, shared, sel_of, cand_of)
    x = blk.hc_post(x, residual, a_post, a_comb)
    dump.put("hc_attn_post", il, x[0, 0], [n, 4])
    residual = x
    f_pre, f_post, f_comb, mixes = hc_mixes(blk, x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
    dump.put("hc_mixes", il, mixes[0, 0], [24])
    dump.put("hc_pre", il, f_pre[0, 0], [4])
    dump.put("hc_post", il, f_post[0, 0], [4])
    dump.put("hc_comb", il, f_comb[0, 0].flatten(), [4, 4])
    x = blk.hc_pre(x, a_pre)
    dump.put("hc_ffn_pre", il, x[0, 0], [n])
    x = blk.ffn_norm(x)
    dump.put("ffn_norm", il, x[0, 0], [n])
    x = moe_step(ref, M, moe, il, x, dump)
    x = blk.hc_post(x, residual, f_post, f_comb)
    dump.put("l_last", il, x[0, 0], [n, 4])
    return x, f_pre


def attention_step(M, a, il, x, start_pos, dump, shared, sel_of, cand_of):
    """model.py Attention.forward (and _window_kv, _compress_kv, _compress_topk_idxs, Compressor, Indexer) for seqlen 1."""
    bsz, seqlen, _ = x.size()
    freqs_cis = a.freqs_cis[start_pos:start_pos + seqlen]
    rd = a.rope_head_dim
    qr = a.wq_a(x)
    dump.put("qr", il, qr[0, 0], [qr.size(-1)])
    qr = a.q_norm(qr)
    dump.put("qr_norm", il, qr[0, 0], [qr.size(-1)])
    q = a.wq_b(qr).unflatten(-1, (a.n_local_heads, a.head_dim))
    M.apply_rotary_emb(q[..., -rd:], freqs_cis)
    dump.put("q", il, q[0, 0], [a.head_dim, a.n_local_heads])
    # _window_kv
    win = a.window_size
    kv = a.kv_norm(a.wkv(x))
    M.apply_rotary_emb(kv[..., -rd:], freqs_cis)
    M.act_quant(kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    dump.put("kv", il, kv[0, 0], [a.head_dim])
    if start_pos == 0:
        a.window_kv_cache[:bsz, :seqlen] = kv
        window_kv = kv
    else:
        a.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
        window_kv = a.window_kv_cache[:bsz]
    topk_idxs = M.get_window_topk_idxs(win, bsz, seqlen, start_pos)
    kvcat = window_kv
    if a.compress_ratio:
        ckv, cidx = compress_kv_step(M, a, il, x, qr, start_pos, window_kv.size(1), dump, shared, sel_of, cand_of)
        kvcat = torch.cat([window_kv, ckv], dim=1)
        topk_idxs = torch.cat([topk_idxs, cidx], dim=-1)
    o = M.sparse_attn(q, kvcat, a.attn_sink, topk_idxs, a.softmax_scale)
    dump.put("attn", il, o[0, 0], [a.head_dim * a.n_local_heads])
    M.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    o = o.view(bsz, seqlen, a.n_local_groups, -1)
    M.act_quant(o, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    dump.put("attn_derope", il, o[0, 0].flatten(), [a.head_dim, a.n_local_heads])
    wo_a = a.wo_a.weight.view(a.n_local_groups, a.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    dump.put("attn_wo_a", il, o[0, 0].flatten(), [a.o_lora_rank, 1, a.n_local_groups])
    x = a.wo_b(o.flatten(2))
    dump.put("attn_out", il, x[0, 0], [x.size(-1)])
    return x


def compress_kv_step(M, a, il, x, qr, start_pos, offset, dump, shared, sel_of, cand_of):
    bsz, seqlen, _ = x.size()
    ratio, rd = a.compress_ratio, a.rope_head_dim
    compress_len = (start_pos + seqlen) // ratio
    latent = None
    if a.is_kv_source:
        c = a.compressor
        if ratio == 1:
            dump.put("cmp_kv", il, c.wkv(x)[0, 0], [a.head_dim])
        else:
            xf = x.float()
            dump.put("cmp_kv", il, c.wkv(xf)[0, 0], [a.head_dim])
            dump.put("cmp_gate", il, c.wgate(xf)[0, 0], [a.head_dim])
        latent = c(x, start_pos)
        dump.put("cmp_latent", il, latent[0, 0] if latent is not None else torch.zeros(a.head_dim), [a.head_dim])
        shared.compress_kv = a.compress_kv_cache
    idxs = topk_idxs_step(M, a, il, x, qr, latent, start_pos, offset, compress_len, dump, shared, sel_of, cand_of)
    if latent is not None:
        freqs = (a.freqs_cis[: seqlen - seqlen % ratio: ratio] if start_pos == 0
                 else a.freqs_cis[start_pos + 1 - ratio].unsqueeze(0))
        M.apply_rotary_emb(latent[..., -rd:], freqs)
        M.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
        dump.put("cmp_row", il, latent[0, 0], [a.head_dim])
        a.compress_kv_cache[:bsz, start_pos // ratio: start_pos // ratio + latent.size(1)] = latent
    elif a.is_kv_source:
        dump.put("cmp_row", il, torch.zeros(a.head_dim), [a.head_dim])
    return shared.compress_kv[:bsz, :compress_len], idxs


def topk_idxs_step(M, a, il, x, qr, latent, start_pos, offset, compress_len, dump, shared, sel_of, cand_of):
    bsz, seqlen, _ = x.size()
    if not a.is_index_source:   # the list of this step's index source (model.py's shared slot, kept per step here)
        src = max(l for (l, s) in sel_of if s == start_pos and l < il)
        shared.topk_idxs = sel_of[(src, start_pos)]
        return shared.topk_idxs
    ix = a.indexer
    if ix.freqs_cis is None:
        ix.freqs_cis = a.freqs_cis
    ratio, rd, end_pos = ix.compress_ratio, ix.rope_head_dim, start_pos + seqlen
    if ix.owns_k:
        if latent is not None:
            freqs = (ix.freqs_cis[: seqlen - seqlen % ratio: ratio] if start_pos == 0
                     else ix.freqs_cis[start_pos + 1 - ratio].unsqueeze(0))
            k = ix.k_norm(ix.wk(latent))
            M.apply_rotary_emb(k[..., -rd:], freqs)
            M.fp4_act_quant(k, M.fp4_block_size, True)
            dump.put("idx_k", il, k[0, 0], [ix.index_head_dim])
            ix.k_cache[:bsz, start_pos // ratio: start_pos // ratio + k.size(1)] = k
        else:
            dump.put("idx_k", il, torch.zeros(ix.index_head_dim), [ix.index_head_dim])
        shared.index_k = ix.k_cache
    if compress_len == 0:   # model.py skips the indexer while no row is visible
        idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32)
        shared.topk_idxs = idxs
        sel_of[(il, start_pos)] = idxs
        return idxs
    q = ix.wq_b(qr).unflatten(-1, (ix.n_local_heads, ix.index_head_dim))
    M.apply_rotary_emb(q[..., -rd:], ix.freqs_cis[start_pos:end_pos])
    M.fp4_act_quant(q, M.fp4_block_size, True)
    dump.put("idx_q", il, q[0, 0], [ix.index_head_dim, ix.n_local_heads])
    index_k = shared.index_k[:bsz, : end_pos // ratio]
    weights = ix.weights_proj(x) * (ix.softmax_scale * ix.n_heads ** -0.5)
    dump.put("idx_w", il, weights[0, 0], [ix.n_local_heads])
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if start_pos == 0:
        compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
        index_score.masked_fill_(torch.arange(seqlen // ratio) >= compress_lens, -torch.inf)
    else:
        compress_lens = end_pos // ratio
    if ix.is_candidate_source:
        shared.candidates = M.select_candidate_blocks(index_score, compress_lens, ix.candidate_topk_blocks,
                                                      ix.candidate_block_size)
        cand_of[start_pos] = shared.candidates
    elif ix.uses_candidates:
        index_score = index_score.masked_fill(~cand_of[start_pos], -torch.inf)
    topk = min(ix.index_topk, end_pos // ratio)
    raw = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    rows = torch.full((ix.index_topk,), -1, dtype=torch.int32)
    rows[:topk] = raw[0, 0].int()
    dump.put("idx_sel", il, rows, [ix.index_topk], i32=True)
    idxs = torch.where(raw < compress_lens, raw + offset, -1).int()
    shared.topk_idxs = idxs
    sel_of[(il, start_pos)] = idxs
    return idxs


def moe_step(ref, M, moe, il, x, dump):
    """model.py MoE.forward for one token: the router's logits, ids and weights, each routed expert's (unweighted) rows,
    the routed sum as model.py forms it (the weight on the SwiGLU hidden, experts in id order), the shared expert."""
    shape = x.size()
    xf = x.view(-1, moe.dim)
    g = moe.gate
    dump.put("ffn_moe_logits", il, M.linear(xf.float(), g.weight.float())[0], [g.weight.size(0)])
    weights, indices = g(xf, None)
    for e in indices[0].tolist():
        if not getattr(moe.experts[e], "loaded", False):
            ref.load_expert(moe, il, e)
    dump.put("ffn_moe_topk", il, indices[0], [indices.size(1)], i32=True)
    dump.put("ffn_moe_weights_scaled", il, weights[0], [1, weights.size(1)])
    rows = torch.stack([moe.experts[e](xf, None)[0].float() for e in indices[0].tolist()])
    dump.put("ffn_moe_down", il, rows, [moe.dim, indices.size(1)])
    y = torch.zeros_like(xf, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=moe.n_routed_experts).tolist()
    for i in range(moe.n_routed_experts):
        if counts[i] == 0:
            continue
        idx, top = torch.where(indices == i)
        y[idx] += moe.experts[i](xf[idx], weights[idx, top, None])
    dump.put("ffn_moe_out", il, y[0], [moe.dim])
    sh = moe.shared_experts(xf)
    dump.put("ffn_shexp", il, sh[0], [moe.dim])
    y += sh
    y = y.type_as(xf).view(shape)
    dump.put("ffn_out", il, y[0, 0], [moe.dim])
    return y


if __name__ == "__main__":
    raise SystemExit(main())
