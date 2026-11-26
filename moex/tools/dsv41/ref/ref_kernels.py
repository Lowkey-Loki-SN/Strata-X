"""moex/tools/dsv41/ref/ref_kernels.py - pure-torch stand-ins for DeepSeek-V4.1's inference/kernel.py (tilelang).

DeepSeek's model.py imports six kernels from kernel.py, which needs tilelang and a recent NVIDIA GPU. This module
is installed as `kernel` (sys.modules) before model.py is imported, so the release's own model code runs on the CPU.
Each function follows its tilelang kernel's arithmetic (read 2026-10-18 from the release's kernel.py):

  act_quant      per-row blocks along the last dim; amax = max(|x|, 1e-4); scale = amax * (1/448), or with
                 scale_fmt set the power of two 2^ceil(log2(amax * (1/448))); values clamp(x / scale, +-448) cast to
                 FP8 E4M3 (round to nearest even); inplace writes back FP8(x / scale) * scale in x's dtype
  fp4_act_quant  per-row blocks; E8M0 scales: amax = max(amax, 6 * 2^-126), scale = 2^ceil(log2(amax * (1/6)));
                 E4M3 scales: amax = max(amax, 6 * 2^-9), scale = FP8(amax / 6); values FP4 E2M1 (round to nearest
                 even, saturating at 6); inplace only (model.py never asks for the packed form)
  fp8_gemm       C = A @ B^T with A's per-row K-block scales and B's (N block, K block) scales, fp32 accumulate
  fp4_gemm       the same with an FP4 B (2 values per byte along K, low nibble first) and per-row 32-wide E8M0 scales
  sparse_attn    gathered keys (= values, MQA), fp32 scores, softmax with a sink in the denominator; the
                 probabilities are rounded to BF16 before the value product, as the kernel's acc_s_cast does
  hc_split_sinkhorn  pre / post / comb with 1 softmax + (iters - 1) row/column normalisations

Accumulation order differs from the GPU kernels (one fp32 matmul instead of per-block accumulators), so results
agree to fp32 rounding, not bitwise.

Mode switch (environment, read at call time):
  DSV41_REF_GEMM_ACT = fp8 (default)  activations into every FP8/FP4 GEMM rounded to FP8, as DeepSeek serves it
                     = fp32           GEMM inputs left unrounded - the phase 1 gate's "fp32-activation reference"
  In fp32 mode the in-place rounding of the attention output before wo_a (model.py: `act_quant(o, ...)`, a GEMM
  input that convert.py turned into a BF16 einsum) is skipped too. The storage roundings - window KV, compressed KV,
  index q/k, DSpark KV - stay in both modes: they define what the caches hold.
"""
from __future__ import annotations

import linecache
import os
import sys

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def gemm_act_mode() -> str:
    m = os.environ.get("DSV41_REF_GEMM_ACT", "fp8")
    if m not in ("fp8", "fp32"):
        raise ValueError(f"DSV41_REF_GEMM_ACT={m!r}: expected fp8 or fp32")
    return m


def _pow2_ceil_log2(v: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(v)) for positive normal fp32 v, as kernel.py's fast_log2_ceil / fast_pow2 do on the bits."""
    bits = v.float().contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    e = exp - 127 + (man != 0).to(torch.int32)
    return ((e + 127) << 23).view(torch.float32)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    n = x.size(-1)
    if n % block:
        raise ValueError(f"last dim {n} is not a multiple of {block}")
    return x.float().reshape(*x.shape[:-1], n // block, block)


def round_fp4(v: torch.Tensor) -> torch.Tensor:
    """E2M1 round to nearest, ties to even, for fp32 v already clamped to [-6, 6]."""
    a = v.abs()
    lv = _FP4_LEVELS.to(v.device)
    idx = torch.bucketize(a, lv)            # first level >= a
    idx = idx.clamp(1, len(lv) - 1)
    lo, hi = lv[idx - 1], lv[idx]
    d_lo, d_hi = a - lo, hi - a
    # at a tie keep the even code: the code index is idx-1 (lo) or idx (hi); even mantissa = even index
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (idx % 2 == 0))
    r = torch.where(pick_hi, hi, lo)
    r = torch.where(a == 0, torch.zeros_like(a), r)
    return torch.copysign(r, v)


def _caller_line() -> str:
    f = sys._getframe(2)
    return linecache.getline(f.f_code.co_filename, f.f_lineno)


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt=None, scale_dtype: torch.dtype = torch.float32,
              inplace: bool = False):
    if inplace and gemm_act_mode() == "fp32" and "act_quant(o," in _caller_line().replace(" ", ""):
        return x  # the wo_a input: a GEMM input, left unrounded in fp32 mode
    if not inplace and gemm_act_mode() == "fp32":
        return x.float(), None  # fp8_gemm / fp4_gemm take an unrounded fp32 activation
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    inv = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
    s = _pow2_ceil_log2(amax * inv) if scale_fmt is not None else amax * inv
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        y = (q.float() * s).reshape(x.shape).to(x.dtype)
        x.copy_(y)
        return x
    return q.reshape(x.shape), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False,
                  scale_dtype: torch.dtype = torch.float8_e8m0fnu):
    if not inplace:
        raise NotImplementedError("model.py only uses fp4_act_quant in place")
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    elif scale_dtype == torch.float8_e8m0fnu:
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -126)
        s = _pow2_ceil_log2(amax * torch.tensor(1.0 / FP4_MAX, dtype=torch.float32))
    else:
        raise ValueError(f"scale dtype {scale_dtype}")
    q = round_fp4((xb / s).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((q * s).reshape(x.shape).to(x.dtype))
    return x


def _dequant_act(a: torch.Tensor, a_s, block: int) -> torch.Tensor:
    if a_s is None:
        return a.float()
    k = a.size(-1)
    return (a.float().reshape(*a.shape[:-1], k // block, block) * a_s.float().unsqueeze(-1)).reshape(a.shape)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size: int = 128):
    n, k = b.shape
    A = _dequant_act(a, a_s, block_size).reshape(-1, k)
    sb = b_s.float().repeat_interleave(block_size, 0)[:n].repeat_interleave(block_size, 1)[:, :k]
    B = b.float() * sb
    c = A @ B.t()
    return c.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def dequant_fp4_weight(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """[N, K/2] float4_e2m1fn_x2 (element 2i = low nibble) x [N, K/32] E8M0 -> fp32 [N, K]."""
    u = b.view(torch.uint8)
    n = u.size(0)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         device=u.device)
    codes = torch.stack([u & 0x0F, u >> 4], dim=-1).reshape(n, -1).long()
    w = table[codes]
    return (w.reshape(n, -1, 32) * b_s.float().unsqueeze(-1)).reshape(n, -1)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size: int = 128):
    B = dequant_fp4_weight(b, b_s)
    k = B.size(1)
    A = _dequant_act(a, a_s, act_block_size).reshape(-1, k)
    c = A @ B.t()
    return c.reshape(*a.shape[:-1], B.size(0)).to(torch.get_default_dtype())


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor,
                softmax_scale: float, chunk: int = 64) -> torch.Tensor:
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.float()
    for bi in range(b):
        for s0 in range(0, m, chunk):
            s1 = min(m, s0 + chunk)
            idx = topk_idxs[bi, s0:s1].long()                     # [c, k]
            valid = idx >= 0
            g = kv[bi][idx.clamp_min(0)].float()                  # [c, k, d]
            qq = q[bi, s0:s1].float()                             # [c, h, d]
            sc = torch.einsum("chd,ckd->chk", qq, g) * softmax_scale
            sc = sc.masked_fill(~valid.unsqueeze(1), float("-inf"))
            mx = sc.amax(-1, keepdim=True).clamp_min(-1e30)       # the kernel's finite floor: empty rows -> 0
            e = torch.exp(sc - mx)
            den = e.sum(-1, keepdim=True) + torch.exp(sink.view(1, h, 1) - mx)
            o = torch.einsum("chk,ckd->chd", e.to(torch.bfloat16).float(), g) / den
            out[bi, s0:s1] = o.to(out.dtype)
    return out


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4,
                      sinkhorn_iters: int = 20, eps: float = 1e-6):
    hc = hc_mult
    m = mixes.float()
    sc, base = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * sc[1] + base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * sc[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
