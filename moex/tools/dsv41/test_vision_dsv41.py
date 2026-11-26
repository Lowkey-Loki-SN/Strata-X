#!/usr/bin/env python3
"""test_vision_dsv41.py [--threads N] - serve/vision_dsv41.py against the release's own inference/vision.py and image_processor.py
(imported from $DSV41_REF, by default the model folder's reference/inference): for synthetic images of several shapes (one under
the minimum pixel count, a wide one, a tall one, a large one capped at 1,024 tokens) the plan (grids, token counts) must be the
release's, and the span rows - ViT, aligner, the image_start / image_newline / image_end rows in image_token_types' order - must be
the same numbers in bf16 (the release's dtype); fp32 is reported against it.  No GPU (torch on the CPU), no server."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import argparse
import importlib.util
import io
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import torch
from PIL import Image, ImageDraw

X = Path(__file__).resolve().parents[3]
REF = Path(os.environ.get("DSV41_REF", _os.path.join(_MODELS, 'dsv41-1017/reference/inference')))
DENSE = Path(os.environ.get("DSV41_DENSE", _os.path.join(_MODELS, 'dsv41-1017/dense')))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def images():
    out = {}
    for name, (w, h) in {"small_200x150": (200, 150), "wide_1600x300": (1600, 300), "tall_300x1400": (300, 1400),
                         "large_2400x1800": (2400, 1800), "photo_640x480": (640, 480)}.items():
        im = Image.new("RGB", (w, h), (30, 60, 200))
        d = ImageDraw.Draw(im)
        for i in range(0, w, 40):
            d.line([(i, 0), (w - i, h)], fill=(i % 255, (2 * i) % 255, 90), width=3)
        d.rectangle([w // 4, h // 4, w // 2, h // 2], fill=(240, 220, 20))
        d.text((10, 10), "STRATA 42", fill=(255, 255, 255))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out[name] = buf.getvalue()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, str(X))
    import serve.vision_dsv41 as V
    rv = module("dsv41_ref_vision", REF / "vision.py")
    rip = module("dsv41_ref_image_processor", REF / "image_processor.py")
    cfg = json.loads((REF / "config.json").read_text())
    args = SimpleNamespace(**{k: v for k, v in cfg.items() if k.startswith("vision_")}, dim=cfg["dim"])
    ours = V.Encoder(DENSE, cfg, torch.bfloat16)
    ours32 = V.Encoder(DENSE, cfg, torch.float32)
    # the release's modules with the same weights, in its default dtype (bfloat16)
    torch.set_default_dtype(torch.bfloat16)
    vit, ali = rv.ViT(args), rv.Aligner(args)
    torch.set_default_dtype(torch.float32)
    vit.load_state_dict({k[len("vision."):]: v for k, v in ours.w.items() if k.startswith("vision.")}, strict=True)
    ali.load_state_dict({k[len("aligner."):]: v for k, v in ours.w.items() if k.startswith("aligner.")}, strict=True)
    # a warm-up pass of both: torch's first multithreaded call on the CPU can take another kernel (the first run of an image
    # differed from later runs of it by one bf16 step, 2026-10-19), and the comparison is between steady states
    warm = images()["photo_640x480"]
    wp, wh, ww, wlh, wlw = V.load_image(warm, cfg, cfg["vision_max_n_token"])
    with torch.inference_mode():
        ali(vit(wp, wh, ww), wh, ww)
        ours.span(ours.features(wp, wh, ww), wlh, wlw)
    bad = 0
    for name, data in images().items():
        record = {"data": data}
        rp, rh, rw, rlh, rlw = rip.load_image(record, args)
        op, oh, ow, olh, olw = V.load_image(data, cfg, cfg["vision_max_n_token"])
        plan_ok = (rh, rw, rlh, rlw) == (oh, ow, olh, olw) and torch.equal(rp, op)
        with torch.inference_mode():
            t0 = time.time()
            feats = ali(vit(rp, rh, rw), rh, rw)
            t_ref = time.time() - t0
            types = rip.image_token_types(rlh, rlw)
            ref = torch.empty(types.numel(), cfg["dim"], dtype=torch.bfloat16)
            ref[types == rip.IMAGE_START] = ours.w["image_start"]
            ref[types == rip.IMAGE_END] = ours.w["image_end"]
            ref[types == rip.IMAGE_NEW_LINE] = ours.w["image_newline"]
            ref[types == rip.IMAGE] = feats.to(torch.bfloat16)
            t0 = time.time()
            got = ours.span(ours.features(op, oh, ow), olh, olw)
            t_ours = time.time() - t0
            got32 = ours32.span(ours32.features(op, oh, ow), olh, olw)
        same = torch.equal(got.float(), ref.float())
        rel32 = ((got32.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"  {name:16s} {rh}x{rw} patches -> {rlh}x{rlw}, {types.numel()} rows: plan {'equal' if plan_ok else 'DIFFERS'}; "
              f"bf16 rows {'bitwise equal' if same else 'DIFFER (max %.3g)' % (got.float() - ref.float()).abs().max().item()} to the "
              f"release's ({t_ours:.1f} s, release {t_ref:.1f} s); fp32 vs the release's bf16: relative L2 {rel32:.2e}", flush=True)
        bad += (not plan_ok) + (not same)
    print(f"test_vision_dsv41: {'PASS' if bad == 0 else 'FAIL (%d)' % bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
