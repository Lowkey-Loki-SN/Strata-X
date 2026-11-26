"""serve/vision_dsv41_plan.py - DeepSeek-V4.1-Flash's image plan (moex): the release's inference/image_processor.py arithmetic,
without torch, for the image encoder (serve/vision_dsv41.py, which plans each picture's patches with it) and for the server,
which counts a picture's tokens with it without running the encoder (count_tokens, DSV41_BACKLOG #56).

A picture of n_llm_h x n_llm_w aligner cells is a span of

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

rows, at most the model's vision_max_n_token (1,024).  `cfg`: the release's inference/config.json (its vision_* keys)."""
from __future__ import annotations

import math


def num_image_tokens(n_llm_h, n_llm_w):
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height, best_width, patch_size, downsample_ratio):
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil((best_width // patch_size) / downsample_ratio)


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width, height, cfg, max_n_token):
    p = cfg["vision_patch_size"]
    if cfg.get("vision_max_wh_ratio") is not None and width > height * cfg["vision_max_wh_ratio"]:
        width = height * cfg["vision_max_wh_ratio"]
    if 0 < width * height < cfg["vision_min_pixels"]:
        ratio = (cfg["vision_min_pixels"] / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, cfg["vision_downsample_ratio"], max_n_token)


def max_tokens_of(cfg, max_tokens=0):
    """The most rows a picture may cost: the model's vision_max_n_token, or `max_tokens` (the encoder's --max-tokens, the
    config's vision "max_tokens") when that is set and smaller."""
    return min(cfg["vision_max_n_token"], max_tokens) if max_tokens > 0 else cfg["vision_max_n_token"]


def image_rows(width, height, cfg, max_tokens=0):
    """The rows the encoder gives a picture of this size (the span: its delimiter rows included)."""
    n_llm_h, n_llm_w, _, _ = plan_image_grid(width, height, cfg, max_tokens_of(cfg, max_tokens))
    return num_image_tokens(n_llm_h, n_llm_w)


# ------------------------------------------------------------------------------------------------ the GPU encoder's inputs
# moex (DSV41_PLAN §54): the engine's image encoder (strata-x VENC, src/x/dsv41_vision.cu) takes each picture's patches
# from the server and its weights through a manifest; numpy and Pillow are imported where they are used (count_tokens needs
# neither).

SVP_MAGIC = 0x31505653   # "SVP1": int32 {magic, n_vit_h, n_vit_w, n_llm_h, n_llm_w, 588}, then n_vit_h * n_vit_w x 588 bf16


def patches_of(data: bytes, cfg, max_n_token):
    """image_processor.load_image without torch (serve/vision_dsv41.py load_image's numbers, bit for bit): the picture planned,
    resized or padded with grey, its pixels as (x / 255 - 0.5) / 0.5 in float32 rounded to bfloat16 (nearest, ties to even, as
    torch's .to(bfloat16)) -> (uint16 array (patches, 588) of the bf16 bits in (channel, y, x) order, n_vit_h, n_vit_w,
    n_llm_h, n_llm_w)."""
    import io

    import numpy as np
    from PIL import Image, ImageOps

    p = cfg["vision_patch_size"]
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, cfg, max_n_token)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if cfg.get("vision_max_wh_ratio") is not None and image.width >= cfg["vision_max_wh_ratio"] * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    x = (x - np.float32(0.5)) / np.float32(0.5)
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bf = ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)   # (finite values)
    patches = bf.reshape(3, n_vit_h, p, n_vit_w, p).transpose(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3 * p * p)
    return np.ascontiguousarray(patches), n_vit_h, n_vit_w, n_llm_h, n_llm_w


def write_patches(path, patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w):
    """One picture's SVP1 file for the engine's VENC (patches_of's array)."""
    import struct

    import numpy as np

    a = np.ascontiguousarray(patches, dtype=np.uint16)
    if a.shape != (n_vit_h * n_vit_w, a.shape[1]) or a.shape[1] != 588:
        raise ValueError(f"patches of shape {a.shape} for {n_vit_h} x {n_vit_w}")
    with open(path, "wb") as f:
        f.write(struct.pack("<6i", SVP_MAGIC, n_vit_h, n_vit_w, n_llm_h, n_llm_w, a.shape[1]))
        f.write(a.tobytes())


def vision_tensor_names(cfg):
    """The tensors the encoder reads (serve/vision_dsv41.py Encoder's list, in its order)."""
    names = ["vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias", "vision.norm.weight",
             "aligner.w1.weight", "aligner.w1.bias", "aligner.w2.weight", "aligner.w2.bias",
             "image_start", "image_newline", "image_end"]
    for i in range(cfg["vision_n_layers"]):
        b = f"vision.blocks.{i}."
        names += [b + s for s in ("norm1.weight", "attn.wqkv.weight", "attn.wqkv.bias", "attn.wo.weight", "attn.wo.bias",
                                  "norm2.weight", "mlp.w1.weight", "mlp.w2.weight")]
    return names


def write_manifest(path, dense, cfg):
    """The engine's list of the encoder's weights in the release's safetensors (`dense`: the vision block's "mmproj"
    directory, with model.safetensors.index.json or the pack's index.json beside it): "SVM1", then per tensor
    "T <name> <file offset> <bytes> <shape> <file>" - BF16 each, read in place by offset (the engine reads no JSON).
    ValueError when a tensor is missing or not BF16."""
    import json
    import struct
    from pathlib import Path

    dense = Path(dense)
    idx = next((q for q in (dense / "model.safetensors.index.json", dense.parent / "index.json") if q.exists()), None)
    if idx is None:
        raise ValueError(f"no model.safetensors.index.json in {dense} (nor index.json beside it)")
    wmap = json.loads(idx.read_text())["weight_map"]
    heads: dict = {}
    lines = ["SVM1"]
    for name in vision_tensor_names(cfg):
        if name not in wmap:
            raise ValueError(f"{name}: not in {idx}")
        file = (dense / wmap[name]).resolve()
        if file not in heads:
            with open(file, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                heads[file] = (8 + hlen, json.loads(f.read(hlen)))
        base, head = heads[file]
        h = head.get(name)
        if h is None or h.get("dtype") != "BF16":
            raise ValueError(f"{name}: {'absent from' if h is None else h.get('dtype') + ' in'} {file}, BF16 expected")
        a, b = h["data_offsets"]
        lines.append(f"T {name} {base + a} {b - a} {'x'.join(str(d) for d in h['shape'])} {file}")
    Path(path).write_text("\n".join(lines) + "\n")
