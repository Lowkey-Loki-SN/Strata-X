"""DSV41_BACKLOG #109's first step: one expert's rows against fp32, its activations rounded per 256 (ggml's Q8_K, the CPU
pool's) and per 32 (the GPU's Q8_1).  The weights: q3q2u's experts (L02: Q2_K, L39: Q3_K), dequantized; the activations:
layer 2's ffn_norm weight times N(0,1), then heavier-tailed variants."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import sys
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize

M = _os.path.join(_MODELS, 'dsv41-1017')
n, nf = 5120, 2304
rd = GGUFReader(f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf")
normw = None
for t in rd.tensors:
    if t.name == "blk.2.ffn_norm.weight":
        normw = np.array(t.data, dtype=np.float32).reshape(-1)
assert normw is not None and normw.size == n

def rows(layer, qt, bpb):
    rb = n // 256 * bpb
    raw = np.fromfile(f"{M}/q3q2u/experts/L{layer:02d}.bin", dtype=np.uint8, count=nf * rb)
    return dequantize(raw.reshape(nf, rb), qt).reshape(nf, n).astype(np.float64)

def q8k(x):   # ggml quantize_row_q8_K_ref: per 256, iscale = -127 / (the value of largest magnitude), nearest_int, min 127
    y = np.empty_like(x)
    for b in range(0, x.size, 256):
        v = x[b:b + 256].astype(np.float32)
        i = int(np.argmax(np.abs(v)))
        mx = v[i]
        if mx == 0:
            y[b:b + 256] = 0
            continue
        isc = np.float32(-127.0) / mx
        q = np.minimum(127, np.rint(isc * v))   # nearest_int: round half to even (ggml's float trick)
        y[b:b + 256] = q * np.float32(1.0 / isc)
    return y

def q81(x):   # ggml-cuda quantize_q8_1: per 32, d = amax / 127, q = roundf(x / d)
    y = np.empty_like(x)
    for b in range(0, x.size, 32):
        v = x[b:b + 32].astype(np.float32)
        am = np.float32(np.max(np.abs(v)))
        if am == 0:
            y[b:b + 32] = 0
            continue
        d = am / np.float32(127.0)
        q = np.sign(v / d) * np.floor(np.abs(v / d) + 0.5)   # roundf: half away from zero
        y[b:b + 32] = q * d
    return y

rng = np.random.default_rng(109)
for name, W in (("L02 gate, Q2_K", rows(2, GGMLQuantizationType.Q2_K, 84)), ("L39 gate, Q3_K", rows(39, GGMLQuantizationType.Q3_K, 110))):
    for dist in ("normal", "student-t 3", "4 outlier channels x30"):
        e256, e32 = [], []
        for _ in range(16):
            z = rng.standard_normal(n) if dist == "normal" else rng.standard_t(3, n)
            if dist.startswith("4 outlier"):
                z = rng.standard_normal(n)
                z[rng.choice(n, 4, replace=False)] *= 30
            x = (normw * z).astype(np.float32)
            ref = W @ x.astype(np.float64)
            s = np.sqrt(np.mean(ref ** 2))
            e256.append(np.sqrt(np.mean((W @ q8k(x).astype(np.float64) - ref) ** 2)) / s)
            e32.append(np.sqrt(np.mean((W @ q81(x).astype(np.float64) - ref) ** 2)) / s)
        print(f"{name}, x = ffn_norm * {dist}: relative RMS error of the row dots - per 256 (Q8_K, CPU) {np.mean(e256):.2e}, "
              f"per 32 (Q8_1, GPU) {np.mean(e32):.2e} (x{np.mean(e256) / np.mean(e32):.2f})", flush=True)
