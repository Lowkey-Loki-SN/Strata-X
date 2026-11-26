"""moex/tools/ernie55/ref/dspark_accept.py - DSpark's acceptance on ERNIE-5.5's own text, offline (ERNIE55_PLAN.md: before a
port). The release's drafter (model.mtp.*: three window-only HCA stages with 512-expert MoEs, the Markov head) drafts a
block of 7 from the trunk's stream-mean features after layers 40-42; this measures how many of those drafts the trunk keeps.

1. The trunk (model.py, the FP8 release exactly, fp32) over each conversation (prompt + a reply the engine generated
   greedily), layer-major: each layer's attention half a conversation at a time, its FFN half over every row at once (each
   routed expert dequantized once a layer); the captured features (model.py: R.mean over the streams after layers 40, 41,
   42) and the trunk's own greedy next token at every position.
2. DSpark as vLLM-ERNIE runs it (nvidia/dspark.py; vLLM's DSpark speculator; vllm/models/deepseek_v4/nvidia/dspark.py):
   main_x = feature_norm(feature_projection([f40 | f41 | f42])); each stage's context K = V rows = kv_layernorm(
   linear_kv_proj(main_x)) with RoPE at their positions (plain, theta 8e6: the stages' rope); the block for an anchor at
   position p = [token p, then dspark_noise x 6] at positions p .. p + 6, embedded with the target's table (sample from
   anchor), through the three stages - the attention NON-causal inside the block (every row sees the 7 block rows) and
   over the context rows in the 128-position window (query p + k: positions p + k - 127 .. p - 1); then the stages' hc
   head, model.mtp.norm, the target's head (base logits), and drafts 1..7 left to right: argmax(MultiMax(base_i +
   markov_w2 . markov_w1[:, previous token])), the previous token the anchor, then each draft (greedy, as T 0 serving).
3. Per anchor the accepted prefix: drafts against the text's next tokens (the engine's greedy reply) and against the
   trunk's own greedy predictions; per draft depth the conditional acceptance; the expected tokens a verify window of w
   rows (the first w - 1 drafts kept while they match, + 1).

    VENV/bin/python -I moex/tools/ernie55/ref/dspark_accept.py --model DIR \\    (VENV: one with torch, as model.py's)
        --seq NAME:PROMPT_IDS:REPLY_IDS ... [--max-anchors N] [--threads N] [--out FILE.json]

REPLY_IDS: the engine's greedy continuation (strata-x's stdout: comma-separated ids), cut after its first <|ernie:im_end|>.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M  # noqa: E402

F32 = torch.float32
IM_END = 8   # <|ernie:im_end|>


def read_ids(path: str) -> list[int]:
    return [int(x) for x in pathlib.Path(path).read_text().replace("\n", ",").split(",") if x.strip()]


def trunk_pass(mdl: M.Model, seqs: list[list[int]]):
    """features {layer: [T, 4096]} and the trunk's greedy next token [T] for each sequence"""
    c = mdl.cfg
    R = [mdl.embed(ids) for ids in seqs]
    caches = [mdl.new_cache() for _ in seqs]
    pos = [torch.arange(len(ids)) for ids in seqs]
    feats = [dict() for _ in seqs]
    sizes = [len(ids) for ids in seqs]
    for il in range(c.n_layer):
        t0 = time.time()
        for s in range(len(seqs)):
            R[s] = mdl.attn_half(il, R[s], pos[s], caches[s].layers[il])
            caches[s].layers[il] = None   # (the whole sequence ran: the layer's rows are not needed again)
        allr = mdl.ffn_half(il, torch.cat(R, 0))
        R = list(torch.split(allr, sizes, 0))
        if il in c.dspark_targets:
            for s in range(len(seqs)):
                feats[s][il] = R[s].mean(1).clone()
        mdl.clear()
        print(f"  trunk layer {il} {time.time() - t0:.1f} s", flush=True)
    nxt = []
    for s in range(len(seqs)):
        h = mdl.final_norm(R[s])
        lg = mdl.logits(h)
        nxt.append(lg.argmax(-1))
        del lg
    return feats, nxt


def stage_attn(mdl: M.Model, st: int, x: torch.Tensor, blocks: list[tuple[int, int]], ctx_kv: list[torch.Tensor],
               inv: torch.Tensor) -> torch.Tensor:
    """stage st's attention for every block's 7 rows (x [B * 7, 4096]); blocks[b] = (sequence, anchor position);
    ctx_kv[s] [T_s, 512] the stage's context rows (RoPE'd)"""
    c = mdl.cfg
    W = mdl.W
    a = f"model.mtp.{st}.self_attn."
    K = c.dspark_block
    BK = x.shape[0]
    pos = torch.cat([torch.arange(p, p + K) for _, p in blocks])
    qr = M.rms(x @ W(a + "linear_q_down_proj.weight").T, W(a + "q_layernorm.weight"), c.rms_eps)
    kv = M.rms(x @ W(a + "linear_kv_proj.weight").T, W(a + "kv_layernorm.weight"), c.rms_eps)
    g = qr @ W(a + "gate_proj.weight").T
    q = (qr @ W(a + "linear_q_up_proj.weight").T).view(BK, c.n_head, c.head_dim)
    q = M.rms(q, None, c.rms_eps)
    nope = c.head_dim - c.rope_dim
    q = torch.cat([q[..., :nope], M.rope_gptj(q[..., nope:], pos, inv)], -1)
    kv = torch.cat([kv[..., :nope], M.rope_gptj(kv[..., nope:], pos, inv)], -1)
    sink = W(a + "core_attention.attn_sink")
    scale = c.head_dim ** -0.5
    o = torch.empty(BK, c.n_head, c.head_dim)
    for b, (s, p) in enumerate(blocks):
        lo = max(0, p - c.window + 1)
        rows = torch.cat([ctx_kv[s][lo:p], kv[b * K:(b + 1) * K]], 0)            # [n_ctx + 7, 512], K = V
        n_ctx = p - lo
        qb = q[b * K:(b + 1) * K]                                                 # [7, H, D]
        sc = torch.einsum("khd,nd->khn", qb, rows) * scale
        if n_ctx > 0:   # query p + k sees context positions p + k - 127 .. p - 1 (the window), every block row
            kk = torch.arange(K)[:, None]
            cpos = torch.arange(lo, p)[None, :]
            drop = cpos < (p + kk - c.window + 1)                                  # [7, n_ctx]
            sc[:, :, :n_ctx] = sc[:, :, :n_ctx].masked_fill(drop[:, None, :], float("-inf"))
        m = torch.maximum(sc.amax(-1), sink[None, :])
        e = torch.exp(sc - m[..., None])
        o[b * K:(b + 1) * K] = torch.einsum("khn,nd->khd", e, rows) / (e.sum(-1) + torch.exp(sink[None, :] - m))[..., None]
    o = mdl.vha(o, a)
    o = torch.cat([o[..., :nope], M.rope_gptj(o[..., nope:], pos, inv, inverse=True)], -1)
    wa = W(a + "linear_o_group_proj").view(c.o_groups, c.o_lora, -1)
    z = torch.einsum("tgr,gdr->tgd", o.reshape(BK, c.o_groups, -1), wa).reshape(BK, -1)
    u = z * torch.sigmoid(g)
    return u @ W(a + "o_proj.weight").T


def dspark_pass(mdl: M.Model, seqs: list[list[int]], feats, blocks: list[tuple[int, int]]) -> torch.Tensor:
    """drafts [B, 7] for every block"""
    c = mdl.cfg
    W = mdl.W
    K = c.dspark_block
    inv = M.inv_freq_plain(c.rope_dim, c.rope_theta)
    main_x = []
    for s in range(len(seqs)):
        f = torch.cat([feats[s][il] for il in c.dspark_targets], -1)
        main_x.append(M.rms(f @ W("model.mtp.feature_projection.weight").T, W("model.mtp.feature_norm.weight"), c.rms_eps))
    toks = []
    for s, p in blocks:
        toks += [seqs[s][p]] + [c.dspark_noise] * (K - 1)
    R = mdl.embed(torch.tensor(toks))                                                  # [B * 7, 4, 4096]
    for st in range(3):
        t0 = time.time()
        pre = f"model.mtp.{st}."
        a = pre + "self_attn."
        ctx_kv = []
        for s in range(len(seqs)):
            k = M.rms(main_x[s] @ W(a + "linear_kv_proj.weight").T, W(a + "kv_layernorm.weight"), c.rms_eps)
            nope = c.head_dim - c.rope_dim
            ps = torch.arange(k.shape[0])
            ctx_kv.append(torch.cat([k[..., :nope], M.rope_gptj(k[..., nope:], ps, inv)], -1))
        ha = M.hc_params(mdl.w, pre + "self_attention_hyper_connection")
        xa, post, comb = M.mhc_pre(R, ha, c)
        xa = M.rms(xa, W(pre + "input_layernorm.weight"), c.rms_eps)
        R = M.mhc_post(stage_attn(mdl, st, xa, blocks, ctx_kv, inv), R, post, comb)
        hf = M.hc_params(mdl.w, pre + "mlp_hyper_connection")
        xf, post, comb = M.mhc_pre(R, hf, c)
        xf = M.rms(xf, W(pre + "post_attention_layernorm.weight"), c.rms_eps)
        R = M.mhc_post(mdl.ffn(c.n_layer + st, xf, pre), R, post, comb)
        mdl.clear()
        print(f"  DSpark stage {st} {time.time() - t0:.1f} s", flush=True)
    h = M.hc_head(R, mdl.w.f32("model.mtp.mhc_contract.hc_head_fn").T.contiguous(), mdl.w.f32("model.mtp.mhc_contract.hc_head_base"),
                  mdl.w.f32("model.mtp.mhc_contract.hc_head_scale"), c)
    h = M.rms(h, mdl.w.f32("model.mtp.norm.weight"), c.rms_eps)                         # [B * 7, 4096]
    lm = mdl.big("lm_head.weight")
    w1 = mdl.w.f32("model.mtp.markov_head.markov_w1.weight")                            # [rank, V]
    w2 = mdl.w.f32("model.mtp.markov_head.markov_w2.weight")                            # [V, rank]
    mr, mt = mdl.w.f32("model.lm_head.multimax_ranges"), mdl.w.f32("model.lm_head.multimax_ts")
    B = len(blocks)
    drafts = torch.empty(B, K, dtype=torch.long)
    CH = 64
    for b0 in range(0, B, CH):
        b1 = min(B, b0 + CH)
        hb = h[b0 * K:b1 * K]
        base = torch.cat([hb @ lm[i:i + 16384].to(F32).T for i in range(0, lm.shape[0], 16384)], -1).view(b1 - b0, K, -1)
        prev = torch.tensor([seqs[s][p] for s, p in blocks[b0:b1]])
        for i in range(K):
            bias = w1[:, prev].T @ w2.T                                                   # [n, V]
            d = M.multimax(base[:, i] + bias, mr, mt).argmax(-1)
            drafts[b0:b1, i] = d
            prev = d
    return drafts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", action="append", required=True, help="NAME:PROMPT_IDS:REPLY_IDS")
    ap.add_argument("--max-anchors", type=int, default=0, help="at most N anchors a sequence (0: every reply position)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    mdl = M.Model(a.model)
    c = mdl.cfg
    names, seqs, starts = [], [], []
    for spec in a.seq:
        name, pf, rf = spec.split(":")
        prompt, reply = read_ids(pf), read_ids(rf)
        if IM_END in reply:
            reply = reply[:reply.index(IM_END) + 1]
        names.append(name)
        seqs.append(prompt + reply)
        starts.append(len(prompt))
        print(f"{name}: prompt {len(prompt)}, reply {len(reply)} tokens", flush=True)
    t0 = time.time()
    feats, nxt = trunk_pass(mdl, seqs)
    print(f"trunk {time.time() - t0:.0f} s", flush=True)
    K = c.dspark_block
    blocks = []
    for s, ids in enumerate(seqs):
        ps = list(range(starts[s], len(ids) - 1))
        if a.max_anchors and len(ps) > a.max_anchors:
            step = len(ps) / a.max_anchors
            ps = [ps[int(i * step)] for i in range(a.max_anchors)]
        blocks += [(s, p) for p in ps]
    t0 = time.time()
    drafts = dspark_pass(mdl, seqs, feats, blocks)
    print(f"DSpark {time.time() - t0:.0f} s, {len(blocks)} blocks", flush=True)

    def accepted(ref):   # the kept prefix of each block's drafts; draft i is checked only where position p + 1 + i exists
        lens, depth_ok, depth_n = [], [0] * K, [0] * K
        for b, (s, p) in enumerate(blocks):
            n = 0
            for i in range(K):
                q = p + 1 + i
                if q >= len(seqs[s]):
                    break
                depth_n[i] += 1
                if int(drafts[b, i]) != ref(s, q):
                    break
                depth_ok[i] += 1
                n += 1
            lens.append(n)
        return lens, depth_ok, depth_n

    res = {}
    for label, ref in (("text", lambda s, q: seqs[s][q]), ("trunk", lambda s, q: int(nxt[s][q - 1]))):
        lens, ok, nn = accepted(ref)
        tok_w = {w: sum(min(L, w - 1) for L in lens) / len(lens) + 1 for w in range(2, K + 2)}
        per = {}
        for s, name in enumerate(names):
            ls = [L for L, (bs, _) in zip(lens, blocks) if bs == s]
            per[name] = {"anchors": len(ls), "mean_accepted": sum(ls) / max(1, len(ls)),
                         "tokens_w3": sum(min(L, 2) for L in ls) / max(1, len(ls)) + 1,
                         "tokens_w4": sum(min(L, 3) for L in ls) / max(1, len(ls)) + 1}
        hist = [lens.count(k) for k in range(K + 1)]
        cond = [ok[i] / max(1, (ok[i - 1] if i else nn[0])) for i in range(K)]
        res[label] = {"anchors": len(lens), "mean_accepted_of_7": sum(lens) / len(lens), "histogram_0_to_7": hist,
                      "conditional_accept_by_depth": cond, "tokens_per_window": tok_w, "per_sequence": per}
        print(f"\nagainst the {label}: {len(lens)} anchors, mean accepted {res[label]['mean_accepted_of_7']:.3f} of 7")
        print("  kept-prefix histogram 0..7:", hist)
        print("  P(draft i kept | drafts before kept):", " ".join(f"{x:.3f}" for x in cond))
        print("  tokens a verify window of w rows:", " ".join(f"w{w} {v:.2f}" for w, v in tok_w.items()))
        for name, d in per.items():
            print(f"  {name:8s} anchors {d['anchors']:4d}  mean accepted {d['mean_accepted']:.2f}  w3 {d['tokens_w3']:.2f}  w4 {d['tokens_w4']:.2f}")
    agree = sum(int(nxt[s][q - 1]) == seqs[s][q] for s in range(len(seqs)) for q in range(starts[s], len(seqs[s])))
    total = sum(len(seqs[s]) - starts[s] for s in range(len(seqs)))
    print(f"\nthe trunk's greedy token = the engine's reply token at {agree} of {total} reply positions ({100 * agree / total:.1f}%)")
    res["trunk_vs_text_agreement"] = agree / total
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
