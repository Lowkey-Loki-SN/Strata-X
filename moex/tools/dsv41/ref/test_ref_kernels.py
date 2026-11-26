"""moex/tools/dsv41/ref/test_ref_kernels.py - checks of the kernel.py stand-ins (CPU, no weights needed except one
real expert record from the pack for the FP4 GEMM check).

    python test_ref_kernels.py
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import os
import pathlib
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fp4_repack  # noqa: E402
import ref_kernels as K  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


torch.manual_seed(0)

# FP4 rounding: ties to even on the E2M1 grid
x = torch.tensor([0.0, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 6.0, -0.75, -2.5])
want = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -1.0, -2.0])
check("round_fp4 ties-to-even", torch.equal(K.round_fp4(x), want), str(K.round_fp4(x).tolist()))

# act_quant: power-of-two scales with scale_fmt, in-place == dequantized out-of-place, values on the FP8 grid
os.environ["DSV41_REF_GEMM_ACT"] = "fp8"
a = (torch.randn(5, 96) * torch.logspace(-3, 2, 96)).to(torch.bfloat16)
q, s = K.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
sf = s.float()
check("act_quant scales are powers of two", torch.equal(sf, torch.exp2(torch.log2(sf).round())))
b = a.clone()
K.act_quant(b, 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
deq = (q.float().reshape(5, 3, 32) * sf.unsqueeze(-1)).reshape(5, 96).to(torch.bfloat16)
check("act_quant in place == out of place dequantized", torch.equal(b, deq))
amax = a.float().reshape(5, 3, 32).abs().amax(-1)
check("act_quant scale >= amax/448 and < 2*amax/448", bool(((sf >= amax / 448 * (1 - 1e-6)) & (sf < 2 * amax / 448 + 1e-12)).all()))
os.environ["DSV41_REF_GEMM_ACT"] = "fp32"
q32, s32 = K.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
check("fp32 mode passes GEMM inputs through", s32 is None and torch.equal(q32, a.float()))
os.environ["DSV41_REF_GEMM_ACT"] = "fp8"

# fp4_act_quant: E8M0 (indexer) and E4M3 (compressed KV) scales
v = (torch.randn(4, 64) * 3).to(torch.bfloat16)
v8 = v.clone()
K.fp4_act_quant(v8, 32, True, torch.float8_e8m0fnu)
r = (v8.float().reshape(4, 2, 32))
amax = v.float().reshape(4, 2, 32).abs().amax(-1, keepdim=True)
sc = K._pow2_ceil_log2(amax / 6)
check("fp4_act_quant E8M0 values on the scaled grid", torch.equal(r, K.round_fp4((v.float().reshape(4, 2, 32) / sc).clamp(-6, 6)) * sc))
v4 = v.clone()
K.fp4_act_quant(v4, 16, True, torch.float8_e4m3fn)
sc4 = (v.float().reshape(4, 4, 16).abs().amax(-1, keepdim=True).clamp_min(6 * 2 ** -9) / 6).to(torch.float8_e4m3fn).float()
want4 = (K.round_fp4((v.float().reshape(4, 4, 16) / sc4).clamp(-6, 6)) * sc4).reshape(4, 64).to(torch.bfloat16)
check("fp4_act_quant E4M3 scales", torch.equal(v4, want4))

# fp8_gemm against plain math
w = (torch.randn(64, 96) * 0.1).to(torch.float8_e4m3fn)
ws = torch.tensor(np.exp2(np.random.default_rng(1).integers(-3, 3, size=(2, 3))).astype(np.float32)).to(torch.float8_e8m0fnu)
x = torch.randn(7, 96).to(torch.bfloat16)
xq, xs = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
got = K.fp8_gemm(xq, xs, w, ws, torch.float8_e8m0fnu, 32).float()
W = w.float() * ws.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
X = (xq.float().reshape(7, 3, 32) * xs.float().unsqueeze(-1)).reshape(7, 96)
check("fp8_gemm == dequantized matmul", torch.allclose(got, (X @ W.t()).to(torch.get_default_dtype()).float(), rtol=1e-5, atol=1e-6))

# fp4_gemm on a real expert from the pack against the phase 0 dequantizer
pack = pathlib.Path(_os.path.join(_MODELS, 'dsv41-1017/native/experts/L00.bin'))
if pack.exists():
    rec = open(pack, "rb").read(18_800_640)
    wq, wsc = fp4_repack.split_record(rec)["w1"]
    ref_w = torch.from_numpy(fp4_repack.dequant_reference(wq, wsc))
    got_w = K.dequant_fp4_weight(torch.from_numpy(wq).view(torch.float4_e2m1fn_x2), torch.from_numpy(wsc).view(torch.float8_e8m0fnu))
    check("fp4 weight dequant == phase 0 reference (real expert, layer 0 #0 w1)", torch.equal(got_w, ref_w))
else:
    print("  skip fp4 real-expert check (pack missing)")

# sparse_attn against dense masked attention with a sink
b_, m_, h_, d_, n_ = 1, 9, 4, 32, 20
q = torch.randn(b_, m_, h_, d_).to(torch.bfloat16)
kv = torch.randn(b_, n_, d_).to(torch.bfloat16)
sink = torch.randn(h_)
idx = torch.randint(0, n_, (b_, m_, 6)).int()
idx[0, 2, :] = -1
idx[0, 3, 1] = -1
o = K.sparse_attn(q, kv, sink, idx, 0.2).float()
ref = torch.zeros(m_, h_, d_)
for i in range(m_):
    sel = [j for j in idx[0, i].tolist() if j >= 0]
    if not sel:
        continue
    kk = kv[0, sel].float()
    sc = (q[0, i].float() @ kk.t()) * 0.2
    mx = sc.amax(-1, keepdim=True)
    e = torch.exp(sc - mx)
    ref[i] = (e.to(torch.bfloat16).float() @ kk) / (e.sum(-1, keepdim=True) + torch.exp(sink.view(h_, 1) - mx))
check("sparse_attn == dense masked attention with sink", torch.allclose(o[0], ref.to(torch.bfloat16).float(), atol=1e-2))
check("sparse_attn row with no keys is zero", bool((o[0, 2] == 0).all()))

# hc sinkhorn: comb close to doubly stochastic, pre/post ranges
mixes = torch.randn(2, 3, 24)
pre, post, comb = K.hc_split_sinkhorn(mixes, torch.tensor([1.0, 1.0, 1.0]), torch.zeros(24))
check("sinkhorn comb rows/cols sum to ~1",
      torch.allclose(comb.sum(-1), torch.ones(2, 3, 4), atol=1e-3) and torch.allclose(comb.sum(-2), torch.ones(2, 3, 4), atol=1e-3))
check("sinkhorn pre in (0,1+eps], post in (0,2)", bool((pre > 0).all() and (pre <= 1 + 1e-5).all() and (post > 0).all() and (post < 2).all()))

print("PASS" if not fails else f"FAIL: {fails}")
raise SystemExit(0 if not fails else 1)
