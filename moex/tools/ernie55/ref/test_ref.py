"""moex/tools/ernie55/ref/test_ref.py - the reference model's modules against the serving references' own torch functions
(copied verbatim from vLLM-ERNIE ba322a5 and vLLM 5df63a7, cited), and its caches against themselves: a prompt read in one
call and the same prompt fed one token at a time must give the same outputs.

    ~/AI/vLLM/.venv/bin/python -I moex/tools/ernie55/ref/test_ref.py [--model DIR] [--index FILE] [--real]

Without --real only synthetic inputs (no model files); with --real, layers 0, 1 and 7 of the release (an HCA layer with the
dense FFN, an HCA layer with the MoE, an MLA + DSA layer) on a short prompt.
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M  # noqa: E402

torch.manual_seed(0)
FAILS = []


def check(ok: bool, what: str) -> None:
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


# ------------------------------------------------------------------------------------------------ the references, verbatim
def _mhc_pre_ref(residual, fn, hc_scale, hc_base, rms_eps, pre_eps, sk_eps, alpha, n):
    """vLLM-ERNIE tests/models/ernie5_5/test_mhc.py:18-33 (the layer input returned in fp32 here, not cast to bf16)"""
    num_tokens, hc_mult, hidden_size = residual.shape
    x = residual.view(num_tokens, -1).float()
    mixes = (x @ fn.t()) / (x.square().mean(dim=-1, keepdim=True).sqrt() + rms_eps)
    pre_mix = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
    post = mixes[:, hc_mult: 2 * hc_mult] * hc_scale[1]
    post_mix = torch.sigmoid(post + hc_base[hc_mult: 2 * hc_mult]) * alpha
    comb = mixes[:, 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]
    comb = torch.softmax(comb.view(-1, hc_mult, hc_mult), dim=-1) + sk_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + sk_eps)
    for _ in range(n - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + sk_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + sk_eps)
    layer_input = ((pre_mix + pre_eps).unsqueeze(-1) * residual.float()).sum(dim=1)
    return post_mix.unsqueeze(-1), comb, layer_input


def _situ_native(x, beta, linear_beta):
    """vLLM model_executor/layers/activation.py SituAndMul.forward_native (162-202), fp32"""
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def _multimax_eager(logits, r, t):
    """vLLM-ERNIE tests/models/ernie5_5/test_multimax.py _eager"""
    return (logits + t[0] * (r[0] - logits).relu() + t[1] * (logits - r[1]).relu() + t[2] * (r[2] - logits).relu().square()
            + t[3] * (logits - r[3]).relu().square())


def _routing_torch(gating_output, topk, renormalize, bias, scale):
    """vLLM-ERNIE nvidia/moe.py _ernie5_5_routing_torch (170-188), without the padding mask"""
    ids = torch.topk(gating_output + bias, topk, dim=-1, sorted=False).indices
    weights = gating_output.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    weights = weights * scale[ids]
    return weights.float(), ids.int()


# ------------------------------------------------------------------------------------------------ synthetic checks
def synthetic(cfg: M.Cfg) -> None:
    T, S, n = 7, 4, 4096
    R = torch.randn(T, S, n)
    hp = M.HcParams(fn=torch.randn(24, S * n) * 1e-2, base=torch.randn(24) * 0.1, scale=torch.tensor([0.9, 1.1, 0.8]))
    xi, post, comb = M.mhc_pre(R, hp, cfg)
    p2, c2, l2 = _mhc_pre_ref(R, hp.fn, hp.scale, hp.base, cfg.rms_eps, cfg.hc_eps, cfg.hc_eps, 2.0, cfg.sinkhorn)
    check(torch.allclose(xi, l2, rtol=1e-6, atol=1e-6) and torch.allclose(post, p2.squeeze(-1), rtol=1e-6, atol=1e-7) and
          torch.allclose(comb, c2, rtol=1e-6, atol=1e-7), "mHC pre = vLLM-ERNIE's torch reference (test_mhc.py)")
    y = torch.randn(T, n)
    Rn = M.mhc_post(y, R, post, comb)
    ref = post.unsqueeze(-1) * y.unsqueeze(1) + torch.einsum("tjk,tjh->tkh", comb, R)
    exp = torch.stack([post[:, k:k + 1] * y + sum(comb[:, j, k:k + 1] * R[:, j] for j in range(S)) for k in range(S)], 1)
    check(torch.allclose(Rn, exp, rtol=1e-5, atol=1e-5) and torch.equal(Rn, ref), "mHC post: R'[k] = post[k] y + sum_j comb[j][k] R[j]")
    g, u = torch.randn(64, 2048) * 6, torch.randn(64, 2048) * 40
    check(torch.allclose(M.situ_glu(g, u, 4.0, 25.0), _situ_native(torch.cat([g, u], -1), 4.0, 25.0), rtol=1e-6, atol=1e-6),
          "SiTU-GLU = vLLM's SituAndMul.forward_native")
    r = torch.tensor([15.625, 15.9375, 8.75, 0.3671875])
    t = torch.tensor([-0.1337890625, 0.279296875, -0.09765625, -0.0301513671875])
    lg = torch.randn(3, 4096) * 12
    check(torch.equal(M.multimax(lg, r, t), _multimax_eager(lg, r, t)), "MultiMax = vLLM-ERNIE's eager form, bit for bit")
    gate = torch.randn(9, 1024) * 4
    sc = torch.sigmoid(gate[:, :512]) + torch.sigmoid(gate[:, 512:])
    bias, scale = torch.randn(512) * 0.05, torch.rand(512) * 3
    w_ref, i_ref = _routing_torch(sc, 10, True, bias, scale)
    ids = torch.topk(sc + bias, 10, -1).indices
    wk = sc.gather(1, ids)
    wk = wk / wk.sum(-1, keepdim=True).clamp_min(1e-20) * scale[ids]
    same = all(set(ids[k].tolist()) == set(i_ref[k].tolist()) for k in range(9))
    o1, o2 = ids.sort(-1), i_ref.long().sort(-1)
    wsort = wk.gather(1, o1.indices)
    wref = w_ref.gather(1, o2.indices)
    check(same and torch.allclose(wsort, wref, rtol=1e-6, atol=0), "MoE routing = vLLM-ERNIE's _ernie5_5_routing_torch")
    # FP8 (UE8M0) rounding: every value an E4M3 code times a power-of-two scale, within half a step
    x = torch.randn(5, 64, 128) * torch.logspace(-3, 3, 64)[None, :, None]
    q = M.fp8_fake(x)
    s = torch.exp2(torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448)))
    codes = (q / s).to(torch.float8_e4m3fn).to(torch.float32)
    check(torch.equal(codes, q / s) and bool(((q / s).abs() <= 448).all()), "FP8 fake-quant: E4M3 codes x a power-of-two scale")
    check(bool(((q - x).abs() <= 0.0625 * x.abs() + 1e-30 + s * 2 ** -9).all()), "FP8 fake-quant: within E4M3's half step")
    h = M.hadamard(torch.eye(128))
    check(torch.allclose(h @ h.T, torch.eye(128), atol=1e-6), "Hadamard 128: orthonormal")
    a, b = torch.randn(10, 128), torch.randn(10, 128)
    check(torch.allclose((M.hadamard(a) * M.hadamard(b)).sum(-1), (a * b).sum(-1), rtol=1e-5, atol=1e-5), "Hadamard keeps q . k")
    # YaRN: vLLM-ERNIE's correction range for dim 64, base 1.6e5, 65,536 (vllm_spec.md §1.4: 15 and 25)
    def corr(rot):
        return (64 * math.log(65536 / (rot * 2 * math.pi))) / (2 * math.log(160000.0))
    check(math.floor(corr(32)) == 15 and math.ceil(corr(1)) == 25, "YaRN correction range 15 .. 25")
    inv = M.inv_freq_yarn(64, 160000.0, 16.0, 65536, 32.0, 1.0)
    base = 160000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    check(torch.allclose(inv[:15], 1 / base[:15]) and torch.allclose(inv[25:], 1 / (16 * base[25:])), "YaRN: kept below 15, / 16 from 25")
    # RoPE: a rotation (norms kept), and the inverse undoes it
    x = torch.randn(6, 3, 64)
    pos = torch.arange(6) * 1000
    y = M.rope_gptj(x, pos, inv)
    check(torch.allclose(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-5) and torch.allclose(M.rope_gptj(y, pos, inv, inverse=True), x, atol=1e-4),
          "GPT-J RoPE: norm kept, inverse undoes it")


# ------------------------------------------------------------------------------------------------ the release's layers
def real(model_dir: str, index: str | None) -> None:
    m = M.Model(model_dir, index)
    cfg = m.cfg
    check(cfg.n_layer == 43 and [i for i in range(43) if cfg.is_dsa(i)] == [7, 14, 21, 28, 35, 42], "the layer kinds: DSA at 7, 14, 21, 28, 35, 42")
    w = m.w.f32("model.layers.7.self_attn.kv_a_proj_with_mqa.weight")
    raw = m.w.raw("model.layers.7.self_attn.kv_a_proj_with_mqa.weight")
    sc = m.w.raw("model.layers.7.self_attn.kv_a_proj_with_mqa.weight_scale_inv")
    check(tuple(sc.shape) == (5, 32) and w[575, 4095].item() == raw[575, 4095].float().item() * sc[4, 31].item(),
          "FP8 dequant: the edge block (576 rows: 5 blocks of 128)")
    pw = torch.log2(sc.float())
    check(torch.equal(pw, pw.round()), "the block scales are powers of two")
    # one call against one token at a time, through layers 0 (HCA + dense), 1 (HCA + MoE) and 7 (DSA + MoE)
    torch.manual_seed(1)
    T = 12
    x = torch.randn(T, cfg.hidden) * 0.05
    for il, kind in ((0, "hca"), (1, "hca"), (7, "dsa")):
        pre = f"model.layers.{il}."
        attn = m.attn_dsa if kind == "dsa" else m.attn_hca
        lc1 = M.LayerCache()
        y1 = attn(il, x, torch.arange(T), lc1, pre)
        lc2 = M.LayerCache()
        y2 = torch.cat([attn(il, x[t:t + 1], torch.tensor([t]), lc2, pre) for t in range(T)], 0)
        f1 = m.ffn(il, y1, pre)
        f2 = torch.cat([m.ffn(il, y1[t:t + 1], pre) for t in range(T)], 0)
        m.clear()
        da = ((y1 - y2).abs().max() / y1.abs().max()).item()
        df = ((f1 - f2).abs().max() / f1.abs().max()).item()
        ok = math.isfinite(da) and math.isfinite(df) and da < 1e-5 and df < 1e-5
        check(ok, f"layer {il} ({kind}): one call = one token at a time (attention {da:.1e}, FFN {df:.1e} relative)")


def long(model_dir: str, index: str | None) -> None:
    """The attention caches past their short-prompt regime: 2,400 positions through layer 1 (HCA: 18 compressed blocks)
    and layer 7 (DSA: the indexer selects 2,048 of the candidates from position 2,176 on), in one call against chunks of
    uneven sizes that straddle the blocks (and single tokens at the end)"""
    m = M.Model(model_dir, index)
    cfg = m.cfg
    torch.manual_seed(2)
    T = 2400
    x = torch.randn(T, cfg.hidden) * 0.05
    sizes = [97, 128, 1, 255, 300, 513, 129, 700]
    sizes += [1] * (T - sum(sizes))
    for il, kind, fp8 in ((1, "hca", True), (7, "dsa", False), (7, "dsa", True)):
        M.IDX_FP8 = fp8
        pre = f"model.layers.{il}."
        attn = m.attn_dsa if kind == "dsa" else m.attn_hca
        sel1, sel2 = [], []
        m.hooks = {"dsa_sel": lambda n, l, t: sel1.append(t)}
        lc1 = M.LayerCache()
        y1 = attn(il, x, torch.arange(T), lc1, pre)
        m.hooks = {"dsa_sel": lambda n, l, t: sel2.append(t)}
        lc2 = M.LayerCache()
        out, at = [], 0
        for n in sizes:
            out.append(attn(il, x[at:at + n], torch.arange(at, at + n), lc2, pre))
            at += n
        y2 = torch.cat(out, 0)
        m.hooks = {}
        m.clear()
        d = ((y1 - y2).abs().max() / y1.abs().max()).item()
        extra = ""
        if kind == "hca":
            extra = f", {len(lc1.comp)} / {len(lc2.comp)} compressed rows"
            ok = len(lc1.comp) == len(lc2.comp) == T // 128
            check(ok and math.isfinite(d) and d < 1e-5, f"layer {il} ({kind}), {T} positions: one call = chunks ({d:.1e} relative{extra})")
        else:
            same = len(sel1) == len(sel2) and all(torch.equal(a, b) for a, b in zip(sel1, sel2))
            diff = [len(set(a.tolist()) ^ set(b.tolist())) // 2 for a, b in zip(sel1, sel2)]
            extra = (f", {len(sel1)} selections of {cfg.idx_topk}: {sum(x > 0 for x in diff)} differ, by at most {max(diff, default=0)} "
                     f"of the {cfg.idx_topk}")
            n_ok = len(sel1) == T - (cfg.idx_topk + cfg.window)
            if not fp8:   # exact scores: the same selections and outputs
                check(n_ok and same and math.isfinite(d) and d < 1e-5,
                      f"layer {il} (dsa, exact index scores), {T} positions: one call = chunks ({d:.1e} relative{extra})")
            else:   # E4M3-rounded scores: a key at a rounding edge can move a candidate across the 2,048th place
                check(n_ok and max(diff, default=0) <= 8,
                      f"layer {il} (dsa, the references' E4M3 index scores), {T} positions: one call vs chunks "
                      f"({d:.1e} relative{extra})")
        M.IDX_FP8 = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_os.path.join(_MODELS, 'HF/ERNIE-5.5-Flash-Preview-FP8'))
    ap.add_argument("--index", default=None, help="model.safetensors.index.json, when the model directory has none yet")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--long", action="store_true", help="also the 2,400-position cache check (layers 1 and 7)")
    a = ap.parse_args()
    cfg = M.Cfg.load(pathlib.Path(a.model) / "config.json")
    synthetic(cfg)
    if a.real:
        real(a.model, a.index)
    if a.long:
        long(a.model, a.index)
    print(f"{len(FAILS)} FAILED" if FAILS else "ALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
