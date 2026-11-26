"""vision_patches_parity.py [PICTURE ...] - DSV41_PLAN §54: the server's patches for the GPU encoder (serve/vision_dsv41_plan.py
patches_of: PIL and numpy) against the CPU encoder's (serve/vision_dsv41.py load_image: PIL and torch) - the same plan and the
same bf16 bits for every pixel.  CPU only, no model; run with llama.cpp's venv (it has torch):

    ~/AI/llama.cpp/.venv/bin/python -I moex/tools/dsv41/checks/vision_patches_parity.py

The pictures: the "7319" one (ref-runs/engine-checks/v43-number.png), generated ones of the sizes an agent sends (640x480 to
3840x2160, a tall and a wide one, one under the plan's minimum pixels), noise and gradients - plus any named on the line."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

R = Path(_REPO)
sys.path[:0] = [str(R)]
from serve import vision_dsv41 as cpu  # noqa: E402  (torch)
from serve import vision_dsv41_plan as plan  # noqa: E402

cfg = json.loads(Path(_os.path.join(_MODELS, 'dsv41-1017/reference/inference/config.json')).read_text())
max_n = plan.max_tokens_of(cfg, 0)


def png(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(a.astype(np.uint8), "RGB").save(b, "PNG")
    return b.getvalue()


rng = np.random.default_rng(54)
pictures = []
number = Path(_os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks/v43-number.png'))
if number.exists():
    pictures.append(("7319 (v43-number.png)", number.read_bytes()))
for w, h, kind in ((640, 480, "noise"), (1280, 720, "gradient"), (1920, 1080, "noise"), (2560, 1440, "gradient"),
                   (3840, 2160, "noise"), (500, 1600, "noise"), (2400, 300, "gradient"), (90, 60, "noise"), (37, 61, "noise")):
    if kind == "noise":
        a = rng.integers(0, 256, size=(h, w, 3))
    else:
        yy, xx = np.mgrid[0:h, 0:w]
        a = np.stack([xx * 255 // max(1, w - 1), yy * 255 // max(1, h - 1), (xx + yy) % 256], axis=-1)
    pictures.append((f"{w}x{h} {kind}", png(a)))
for name in sys.argv[1:]:
    pictures.append((name, Path(name).read_bytes()))

bad = 0
for name, data in pictures:
    t, vh, vw, lh, lw = cpu.load_image(data, cfg, max_n)
    ref_bits = t.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(vh * vw, -1)   # the bf16 bits
    got, gh, gw, glh, glw = plan.patches_of(data, cfg, max_n)
    same_plan = (vh, vw, lh, lw) == (gh, gw, glh, glw)
    same = same_plan and got.shape == ref_bits.shape and np.array_equal(got, ref_bits)
    bad += not same
    print(f"  {name}: {vh}x{vw} patches, {lh}x{lw} cells, {plan.num_image_tokens(lh, lw)} rows - "
          f"{'the same bits' if same else ('PLAN DIFFERS' if not same_plan else f'{int((got != ref_bits).sum())} values differ')}",
          flush=True)
print("ALL PASS" if bad == 0 else f"{bad} FAILED", flush=True)
sys.exit(1 if bad else 0)
