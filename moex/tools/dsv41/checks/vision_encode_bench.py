"""vision_encode_bench.py [SIZES] - serve/vision_dsv41.py's image encoder (the release's ViT and aligner, on the CPU) four ways
on generated pictures: bf16 or fp32, the math path's attention (as served: STRATA_DSV41_VISION_ATTN_ROWS=1024) or the fused
CPU kernel (STRATA_DSV41_VISION_ATTN=fused).  For each picture: the patches, the rows, each way's encode time (after a warm-up
encode, as the server's) and its rows' error against fp32 with the math path (the most exact of the four): the relative
Frobenius error and the largest difference.  SIZES: WxH,... (default 640x480,1920x1080).  CPU-heavy (20 threads): it refuses
while strata-x runs.

    ~/AI/llama.cpp/.venv/bin/python -I moex/tools/dsv41/checks/vision_encode_bench.py
"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import io
import json
import subprocess
import sys
import time
from pathlib import Path

R = Path(_REPO)
sys.path.insert(0, str(R))
if subprocess.run(["pgrep", "-x", r"strata-(x|glm)(\..*)?"], capture_output=True).returncode == 0:
    sys.exit("vision_encode_bench: strata-x is running - stop the server first (this takes all 20 cores)")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from serve import vision_dsv41 as V  # noqa: E402

DENSE = Path(_os.path.join(_MODELS, 'dsv41-1017/dense'))
CFG = json.loads(Path(_os.path.join(_MODELS, 'dsv41-1017/reference/inference/config.json')).read_text())
sizes = [tuple(int(x) for x in s.split("x")) for s in (sys.argv[1] if len(sys.argv) > 1 else "640x480,1920x1080").split(",")]
torch.set_num_threads(20)


def picture(w, h, seed=7):
    """A screenshot-like picture: gradients, blocks of flat colour and some noise (the encoder's cost depends only on its size)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) % 256)], -1).astype(np.float32)
    for _ in range(40):
        x0, y0 = rng.integers(0, w - 8), rng.integers(0, h - 8)
        img[y0:y0 + rng.integers(4, h // 4), x0:x0 + rng.integers(4, w // 4)] = rng.integers(0, 256, 3)
    img += rng.normal(0, 6, img.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def encode(enc, data):
    patches, nvh, nvw, nlh, nlw = V.load_image(data, CFG, V.max_tokens_of(CFG, 0))
    t = time.time()
    rows = enc.span(enc.features(patches, nvh, nvw), nlh, nlw).float()
    return rows, time.time() - t, patches.shape[0]


encs = {}
for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    t = time.time()
    encs[name] = V.Encoder(DENSE, CFG, dt)
    print(f"{name} encoder loaded in {time.time() - t:.1f} s", flush=True)
warm = picture(640, 480, seed=1)
WAYS = (("bf16", "math"), ("bf16", "fused"), ("fp32", "math"), ("fp32", "fused"))
for w, h in sizes:
    data = picture(w, h)
    out = {}
    for dt, attn in WAYS:
        V.ATTN_FUSED, V.ATTN_ROWS = attn == "fused", 1024
        encode(encs[dt], warm)   # (a warm-up encode first, as the server's)
        out[(dt, attn)] = encode(encs[dt], data)
    ref = out[("fp32", "math")][0]
    print(f"{w}x{h}: {out[WAYS[0]][2]} patches, {ref.shape[0]} rows", flush=True)
    for way in WAYS:
        rows, s, _ = out[way]
        rel = float((rows - ref).norm() / ref.norm())
        print(f"  {way[0]} {way[1]:5}  {s:7.2f} s   rel. error {rel:.2e}   largest difference {float((rows - ref).abs().max()):.3e}",
              flush=True)
