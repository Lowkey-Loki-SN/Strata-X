"""serve/vision_hy4_plan.py - Hy4-flash's image plan without torch (moex; moex/docs/HY4_PORT.md): the size the release's
processor resizes a picture to, and the rows its encoder (serve/vision_hy4.py) gives it - so the server can count a picture's
tokens (count_tokens, FamilySpec.image_rows) without running the encoder.

The processor (the release's processor_config.json, image_processor_type HYV4VLImageProcessor; vLLM's Rust port, rust/.../
hy_v4_vl/processor.rs smart_resize): both sides to multiples of factor = patch_size x merge_size (32), Python's round (half
to even); over max_pixels both scaled down by sqrt(h w / max_pixels) and floored to the factor, under min_pixels scaled up and
ceiled.  The rows (vision_token_count): (height / 32) merged rows of (width / 32) cells and the newline column.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PATCH, MERGE = 16, 2
MIN_PIXELS, MAX_PIXELS = 4096, 4194304   # the release's processor_config.json image_processor


def processor_limits(path: str | Path | None) -> dict:
    """The release's processor_config.json image_processor values the plan reads (patch_size, merge_size, min_pixels,
    max_pixels, image_mean, image_std, rescale_factor) - the defaults above when `path` is None or not that file."""
    out = {"patch_size": PATCH, "merge_size": MERGE, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
           "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "rescale_factor": 1 / 255}
    if not path or not str(path).endswith(".json"):
        return out
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    ip = cfg.get("image_processor", cfg) if isinstance(cfg, dict) else {}
    for k in out:
        if k in ip:
            out[k] = ip[k]
    return out


def smart_resize(h: int, w: int, factor: int = PATCH * MERGE, lo: int = MIN_PIXELS, hi: int = MAX_PIXELS) -> tuple[int, int]:
    """(height, width) the processor resizes an h x w picture to."""
    if min(h, w) < 1:
        raise ValueError(f"an image of {w}x{h} pixels")
    if max(h, w) / min(h, w) > 200:
        raise ValueError(f"image aspect ratio over 200 ({w}x{h})")
    hh = round(h / factor) * factor   # (Python's round: half to even, as the reference)
    ww = round(w / factor) * factor
    if hh * ww > hi:
        scale = math.sqrt(h * w / hi)
        hh = max(factor, math.floor(h / scale / factor) * factor)
        ww = max(factor, math.floor(w / scale / factor) * factor)
    elif hh * ww < lo:
        scale = math.sqrt(lo / (h * w))
        hh = math.ceil(h * scale / factor) * factor
        ww = math.ceil(w * scale / factor) * factor
    hh, ww = max(factor, hh), max(factor, ww)
    if hh * ww > hi:
        raise ValueError("image aspect ratio and patch rounding exceed max_pixels")
    return hh, ww


def rows_of(hh: int, ww: int, factor: int = PATCH * MERGE) -> int:
    """The encoder's rows for a resized picture: (hh / factor) merged rows of (ww / factor) cells and the newline."""
    return (hh // factor) * (ww // factor + 1)


def plan(width: int, height: int, max_tokens: int = 0, limits: dict | None = None) -> tuple[int, int]:
    """(height, width) of the picture the encoder reads.  `max_tokens` > 0 (the vision block's "max_tokens"): max_pixels is
    lowered in steps of 10% until the picture's rows fit (or it reaches min_pixels) - an extension of the release's plan,
    which has none; 0 keeps the release's pixel limits."""
    lim = limits or processor_limits(None)
    f = int(lim["patch_size"]) * int(lim["merge_size"])
    lo, hi = int(lim["min_pixels"]), int(lim["max_pixels"])
    if max_tokens > 0:
        while True:
            hh, ww = smart_resize(height, width, f, lo, hi)
            if rows_of(hh, ww, f) <= max_tokens or hi <= lo:
                return hh, ww
            hi = max(lo, int(hi * 0.9))
    return smart_resize(height, width, f, lo, hi)


def image_rows(width: int, height: int, max_tokens: int = 0, limits: dict | None = None) -> int:
    """The rows (image tokens) the encoder gives a width x height picture."""
    lim = limits or processor_limits(None)
    f = int(lim["patch_size"]) * int(lim["merge_size"])
    hh, ww = plan(width, height, max_tokens, lim)
    return rows_of(hh, ww, f)


def max_rows(max_tokens: int = 0, limits: dict | None = None) -> int:
    """The most rows a picture may cost (a picture whose size is not known): the largest rows x (cols + 1) of a grid within
    max_pixels and the aspect limit (a tall picture beats a square: the newline column counts a row), at most max_tokens
    when that is set."""
    lim = limits or processor_limits(None)
    f = int(lim["patch_size"]) * int(lim["merge_size"])
    cells = int(lim["max_pixels"]) // (f * f)
    most = 0
    for r in range(1, cells + 1):
        c = cells // r
        if c < 1:
            break
        if max(r, c) <= 200 * min(r, c):
            most = max(most, r * (c + 1))
    return min(most, max_tokens) if max_tokens > 0 else most


# ------------------------------------------------------------------------------------------------ the resize, without torch
def _aa_bicubic(x):
    """torch's antialiased bicubic filter (UpSampleKernel.cpp HelperInterpCubic::aa_filter: a = -0.5, Pillow's)."""
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


def _int16_weights(n_in: int, n_out: int):
    """One axis of torch's uint8 antialiased bicubic (_compute_index_ranges_int16_weights): per output index the first input
    index and the weights - each window's filter values (double) normalized, then all scaled by 2^precision (the most bits
    that keep the largest weight under 2^15) and rounded half away from zero to int16 -> (xmin, weights (n_out, K), precision)."""
    import numpy as np
    scale = n_in / n_out
    support = 2.0 * scale if scale >= 1.0 else 2.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    rows, xmins, wmax = [], [], 0.0
    for i in range(n_out):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xsize = min(int(center + support + 0.5), n_in) - xmin
        w = [_aa_bicubic((j + xmin - center + 0.5) * invscale) for j in range(xsize)]
        tot = sum(w)   # (in order, as the reference)
        if tot != 0.0:
            w = [v / tot for v in w]
        wmax = max([wmax] + w)
        rows.append(w)
        xmins.append(xmin)
    prec = 0
    while prec < 22 and int(0.5 + wmax * (1 << (prec + 1))) < (1 << 15):
        prec += 1
    K = max(len(r) for r in rows)
    out = np.zeros((n_out, K), dtype=np.int64)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            s = v * (1 << prec)
            out[i, j] = int(s - 0.5) if s < 0 else int(s + 0.5)
    return np.asarray(xmins, dtype=np.int64), out, prec


def _resample_axis(img, n_out: int, axis: int):
    """One pass over `axis` (0: rows, 1: columns) of an (h, w, 3) uint8 array: per output index the int16 weights over its
    input window, summed in int, (sum + 2^(precision - 1)) >> precision, clamped to 0 .. 255 (Pillow-SIMD's, torch's AVX)."""
    import numpy as np
    n_in = img.shape[axis]
    xmin, w, prec = _int16_weights(n_in, n_out)
    K = w.shape[1]
    idx = np.minimum(xmin[:, None] + np.arange(K)[None, :], n_in - 1)   # (n_out, K): taps past the window have weight 0
    src = np.moveaxis(img, axis, 0).astype(np.int64)                    # (n_in, other, 3)
    out = np.empty((n_out,) + src.shape[1:], dtype=np.uint8)
    step = max(1, (1 << 24) // max(1, K * src.shape[1] * 3))           # output indices a block (bounded temporaries)
    for o0 in range(0, n_out, step):
        o1 = min(n_out, o0 + step)
        acc = np.einsum("ok,okcd->ocd", w[o0:o1], src[idx[o0:o1]]) + (1 << (prec - 1))
        out[o0:o1] = np.clip(acc >> prec, 0, 255).astype(np.uint8)
    return np.moveaxis(out, 0, axis)


def resize_like_torch(rgb, height: int, width: int):
    """An (h, w, 3) uint8 array resized as torch.nn.functional.interpolate(mode="bicubic", antialias=True) resizes a uint8
    tensor on the CPU (torchvision's resize; the release's processor; vLLM's Rust port resize_bicubic_torch_aa): the
    horizontal pass first, then the vertical, each only when its side changes, the intermediate rows uint8."""
    out = rgb
    if width != rgb.shape[1]:
        out = _resample_axis(out, width, 1)
    if height != rgb.shape[0]:
        out = _resample_axis(out, height, 0)
    return out


# ------------------------------------------------------------------------------------------------ the engine's patches
SVH_MAGIC = 0x31485653   # "SVH1": int32 {magic, patch rows gh, patch columns gw, values a patch (768)}, then gh x gw x 768 f32


def normalizer(limits: dict):
    """(offset, divisor) a channel, f32: the processor's rescale_and_normalize fused into byte units, (byte - mean / rescale)
    / (std / rescale) - vLLM's port (hy_v4_vl/processor.rs `normalized`), 127.5 and 127.5 for the release's 0.5 / 0.5."""
    import numpy as np
    into = np.float32(1.0 / float(limits["rescale_factor"]))
    off = np.asarray([np.float32(m) * into for m in limits["image_mean"]], dtype=np.float32)
    div = np.asarray([np.float32(s) * into for s in limits["image_std"]], dtype=np.float32)
    return off, div


def to_rgb_array(data: bytes):
    """The picture's (h, w, 3) uint8 pixels, upright (EXIF), a transparent one over white (vLLM's rgba_to_rgb)."""
    import io

    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            image = image.convert("RGBA")
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[-1])
            image = bg
        else:
            image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8).copy()


def patches_of(data: bytes, limits: dict | None = None, max_tokens: int = 0):
    """The picture as the engine's encoder reads it (serve/vision_hy4.py load_image's numbers, bit for bit): planned,
    resized as torch resizes a uint8 picture, normalized -> (float32 array (gh * gw, 768): each 16 x 16 patch's values in
    (channel, y, x) order, the patches in reading order; gh, gw)."""
    import numpy as np
    lim = limits or processor_limits(None)
    p = int(lim["patch_size"])
    rgb = to_rgb_array(data)
    hh, ww = plan(rgb.shape[1], rgb.shape[0], max_tokens, lim)
    if (hh, ww) != rgb.shape[:2]:
        rgb = resize_like_torch(rgb, hh, ww)
    off, div = normalizer(lim)
    x = (rgb.astype(np.float32) - off) / div                         # (hh, ww, 3)
    gh, gw = hh // p, ww // p
    patches = x.reshape(gh, p, gw, p, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, 3 * p * p)
    return np.ascontiguousarray(patches, dtype=np.float32), gh, gw


def write_patches(path, patches, gh: int, gw: int) -> None:
    """One picture's SVH1 file for the engine's VENC (patches_of's array)."""
    import struct

    import numpy as np
    a = np.ascontiguousarray(patches, dtype=np.float32)
    if a.shape != (gh * gw, 768):
        raise ValueError(f"patches of shape {a.shape} for {gh} x {gw}")
    with open(path, "wb") as f:
        f.write(struct.pack("<4i", SVH_MAGIC, gh, gw, a.shape[1]))
        f.write(a.tobytes())
