#!/usr/bin/env python3
"""serve/vision_hy4.py - Hy4-flash's image encoder for the server (moex; moex/docs/HY4_PORT.md): the release's vision tower
(SigLIP2-So400m adapted: 27 blocks of 1,152, a learned 128 x 128 position grid) and its rmsnorm_mlp_adapter projector, from
AngelSlim's mmproj-Hy4-flash-bf16.gguf, with the server's encoder protocol (serve/server.py Vision): READY, then per line
`ENC <image file> <out file>` -> `OK <rows>` or `ERR <why>`; QUIT ends it.

The out file is one SVE1 record: int32 magic "SVE1", rows, nx, ny, width (3,072), then rows x width float32 - the picture's
rows in reading order: each of its ny merged rows' nx - 1 cells, then the projector's newline row (vLLM's
HYV4VisionPatchMerger with use_image_newline_token).  The prompt holds <image_start>, rows image-token placeholders and
<image_end>; the engine puts these rows at the placeholders and gives them M-RoPE positions from the grid (nx, ny).

What it computes, and where from:
  * the image plan (the release's processor_config.json, read by vLLM's port of the HF processor, rust/.../hy_v4_vl/
    processor.rs): smart_resize to multiples of 32 (patch 16 x merge 2, Python round: half to even) within min_pixels 4,096 ..
    max_pixels 4,194,304, a torch bicubic antialiased resize of the uint8 picture (torchvision's), rescale 1/255, then
    (x - 0.5) / 0.5 (the release's image_mean / image_std);
  * the tower (vLLM's vision.py HYV4VisionPatchEmbed / HYV4VisionBlock): the 16 x 16 patch conv, the position grid resampled
    by F.interpolate(scale_factor=((h + 0.1) / 128, (w + 0.1) / 128), bilinear, align_corners=False), 27 pre-LN blocks
    (LayerNorm eps 1e-5, 16 heads of 72, full attention, an MLP of 4,304 with GELU), no 2D RoPE (use_2d_rope's default);
  * the projector: RMS norm, Conv2d 2 x 2 stride 2 (1,152 -> 4,608), GELU (erf), Conv2d 1 x 1 (-> 3,072), the newline column.
The MLP's GELU: the tanh form by default - the config's bias_activation_fusion (the HF reference's fused bias + GELU, as
llama.cpp's mtmd runs it); STRATA_HY4_VISION_GELU=erf takes vLLM's (get_act_fn("gelu")): within ~1e-3 of each other.

    vision_hy4.py --mmproj FILE.gguf [--threads N] [--max-tokens N] [--dtype fp32|bf16]

CPU only (the engine holds both GPUs).
"""
from __future__ import annotations

import argparse
import io
import mmap
import os
import struct
import sys
import time
from pathlib import Path


class _Dropped:
    """STRATA_X_QUIET=1 (the server's "diagnostics": false): stderr goes nowhere - set before torch is imported."""

    def write(self, s):
        return len(s)

    def flush(self):
        pass


if __name__ == "__main__" and os.environ.get("STRATA_X_QUIET") == "1":
    sys.stderr = _Dropped()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))   # (vision_hy4_plan beside this file)
from vision_hy4_plan import PATCH, MERGE, normalizer, plan, processor_limits  # noqa: E402

MAGIC = 0x31455653   # "SVE1"


# ------------------------------------------------------------------------------------------------ the mmproj GGUF
def read_gguf(path: Path):
    """(metadata, {name: torch tensor}) of a GGUF, its tensors viewed in place in a read-only mapping (F32, BF16)."""
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    off = 0

    def rd(fmt):
        nonlocal off
        v = struct.unpack_from(fmt, mm, off)
        off += struct.calcsize(fmt)
        return v[0]

    def s():
        nonlocal off
        n = rd("<Q")
        v = bytes(mm[off:off + n]).decode("utf-8")
        off += n
        return v

    scal = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def val(t):
        if t in scal:
            return rd(scal[t])
        if t == 8:
            return s()
        if t == 9:
            et, n = rd("<I"), rd("<Q")
            return [val(et) for _ in range(n)]
        raise SystemExit(f"{path}: value type {t}")

    if bytes(mm[0:4]) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF")
    off = 4
    rd("<I")
    nt, nkv = rd("<Q"), rd("<Q")
    meta = {}
    for _ in range(nkv):
        k = s()
        meta[k] = val(rd("<I"))
    infos = []
    for _ in range(nt):
        name = s()
        nd = rd("<I")
        dims = [rd("<Q") for _ in range(nd)]
        infos.append((name, dims, rd("<I"), rd("<Q")))
    align = meta.get("general.alignment", 32)
    data = (off + align - 1) // align * align
    out = {}
    for name, dims, typ, toff in infos:
        n = 1
        for d in dims:
            n *= d
        shape = list(reversed(dims))
        if typ == 0:
            a = np.frombuffer(mm, dtype=np.float32, count=n, offset=data + toff)
            out[name] = torch.from_numpy(a.copy()).reshape(shape)
        elif typ == 30:
            a = np.frombuffer(mm, dtype=np.uint16, count=n, offset=data + toff)
            out[name] = torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy()).reshape(shape)
        else:
            raise SystemExit(f"{path}: {name} has ggml type {typ} (F32 / BF16 expected)")
    return meta, out


# ------------------------------------------------------------------------------------------------ the image plan
def to_rgb(image: Image.Image) -> Image.Image:
    """RGB, a transparent picture over white (vLLM's convert_image_mode / rgba_to_rgb, as serve/server.py normalize())."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        return bg
    return image.convert("RGB")


def load_image(data: bytes, max_tokens: int = 0, limits: dict | None = None) -> torch.Tensor:
    """The picture as the tower reads it: (3, H, W) float32, normalized, H and W multiples of 32 - the release's processor
    (HYV4VLImageProcessor): smart_resize (vision_hy4_plan.plan), torchvision's bicubic antialiased resize of the uint8
    picture (F.interpolate on uint8, torchvision's CPU path; vision_hy4_plan.resize_like_torch gives the same bytes without
    torch), then rescale and normalize fused as the processor does."""
    lim = limits or processor_limits(None)
    image = to_rgb(Image.open(io.BytesIO(data)))
    w, h = image.size
    hh, ww = plan(w, h, max_tokens, lim)
    x = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).unsqueeze(0)   # (1, 3, h, w) uint8
    if (hh, ww) != (h, w):
        x = F.interpolate(x, size=(hh, ww), mode="bicubic", antialias=True, align_corners=False)   # (uint8 in, uint8 out)
    off, div = normalizer(lim)   # (the processor's rescale_and_normalize, fused: (byte - 127.5) / 127.5)
    return (x[0].to(torch.float32) - torch.from_numpy(off).reshape(3, 1, 1)) / torch.from_numpy(div).reshape(3, 1, 1)


# ------------------------------------------------------------------------------------------------ the tower
class Encoder:
    def __init__(self, path: Path, dtype: torch.dtype = torch.float32):
        self.meta, t = read_gguf(path)
        if self.meta.get("clip.projector_type") != "hy_v4_flash":
            raise SystemExit(f"{path}: projector {self.meta.get('clip.projector_type')!r}, not hy_v4_flash")
        self.dtype = dtype
        cv = lambda k: t[k].to(dtype)   # noqa: E731
        self.L = int(self.meta["clip.vision.block_count"])
        self.H = int(self.meta["clip.vision.attention.head_count"])
        self.eps = float(self.meta.get("clip.vision.attention.layer_norm_epsilon", 1e-5))
        self.patch_w, self.patch_b = cv("v.patch_embd.weight"), cv("v.patch_embd.bias")
        pos = t["v.position_embd.weight"].float()                    # (128 * 128, 1152)
        self.edge = int(round(pos.shape[0] ** 0.5))
        self.pos = pos.reshape(1, self.edge, self.edge, -1).permute(0, 3, 1, 2).contiguous()
        self.blk = []
        for i in range(self.L):
            p = f"v.blk.{i}."
            self.blk.append({k: cv(p + k) for k in (
                "ln1.weight", "ln1.bias", "attn_q.weight", "attn_q.bias", "attn_k.weight", "attn_k.bias", "attn_v.weight",
                "attn_v.bias", "attn_out.weight", "attn_out.bias", "ln2.weight", "ln2.bias", "ffn_up.weight", "ffn_up.bias",
                "ffn_down.weight", "ffn_down.bias")})
        self.pre_norm = cv("mm.pre_norm.weight")
        self.m0_w, self.m0_b = cv("mm.0.weight"), cv("mm.0.bias")
        self.m2_w, self.m2_b = cv("mm.2.weight"), cv("mm.2.bias")
        self.newline = cv("v.image_newline")
        self.gelu_tanh = os.environ.get("STRATA_HY4_VISION_GELU", "tanh") != "erf"

    @torch.inference_mode()
    def rows(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """(rows, nx, ny): the picture's rows in reading order, nx = cells + 1 (the newline) a merged row, ny merged rows."""
        gh, gw = x.shape[1] // PATCH, x.shape[2] // PATCH
        h = F.conv2d(x.unsqueeze(0).to(self.dtype), self.patch_w, self.patch_b, stride=PATCH)      # (1, C, gh, gw)
        pos = F.interpolate(self.pos, scale_factor=((gh + 0.1) / self.edge, (gw + 0.1) / self.edge), mode="bilinear",
                            align_corners=False)
        if pos.shape[-2:] != (gh, gw):
            raise ValueError(f"the position grid resampled to {tuple(pos.shape[-2:])}, not {(gh, gw)}")
        h = (h + pos.to(self.dtype)).flatten(2).transpose(1, 2)[0]                                 # (N, C)
        C = h.shape[1]
        hd = C // self.H
        for b in self.blk:
            y = F.layer_norm(h, (C,), b["ln1.weight"], b["ln1.bias"], self.eps)
            q = F.linear(y, b["attn_q.weight"], b["attn_q.bias"]).reshape(-1, self.H, hd).transpose(0, 1)
            k = F.linear(y, b["attn_k.weight"], b["attn_k.bias"]).reshape(-1, self.H, hd).transpose(0, 1)
            v = F.linear(y, b["attn_v.weight"], b["attn_v.bias"]).reshape(-1, self.H, hd).transpose(0, 1)
            a = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(-1, C)
            h = h + F.linear(a, b["attn_out.weight"], b["attn_out.bias"])
            y = F.layer_norm(h, (C,), b["ln2.weight"], b["ln2.bias"], self.eps)
            y = F.gelu(F.linear(y, b["ffn_up.weight"], b["ffn_up.bias"]), approximate="tanh" if self.gelu_tanh else "none")
            h = h + F.linear(y, b["ffn_down.weight"], b["ffn_down.bias"])
        # the projector: RMS norm, 2 x 2 conv, GELU, 1 x 1 conv, then the newline column
        y = h.float() if self.dtype != torch.float64 else h   # (a check's float64 encode stays float64)
        y = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)).to(self.dtype) * self.pre_norm
        y = y.reshape(gh, gw, C).permute(2, 0, 1).unsqueeze(0)
        y = F.gelu(F.conv2d(y, self.m0_w, self.m0_b, stride=MERGE))
        y = F.conv2d(y, self.m2_w, self.m2_b)                                                      # (1, 3072, oh, ow)
        oh = y.shape[2]
        y = torch.cat((y, self.newline.reshape(1, -1, 1, 1).expand(1, -1, oh, 1)), dim=-1)          # (1, 3072, oh, ow + 1)
        nx = y.shape[3]
        return y.flatten(2).transpose(1, 2)[0].float(), nx, oh


def write_sve(path: Path, rows: torch.Tensor, nx: int, ny: int):
    a = rows.contiguous().numpy().astype(np.float32, copy=False)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<5i", MAGIC, a.shape[0], nx, ny, a.shape[1]))
        f.write(a.tobytes())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", required=True, help="AngelSlim's mmproj-Hy4-flash-bf16.gguf")
    ap.add_argument("--model", default="", help="the release's processor_config.json (its image_processor's limits; "
                    "without it the release's values)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="at most this many tokens an image (0: the release's pixel limits)")
    ap.add_argument("--min-tokens", type=int, default=0, help="(read for the protocol)")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default=os.environ.get("STRATA_HY4_VISION_DTYPE", "fp32"))
    ap.add_argument("--gpu", action="store_true", help="(refused: this encoder runs on the CPU)")
    a = ap.parse_args()
    if a.gpu:
        print("ERR the Hy4-flash image encoder runs on the CPU (the vision block's \"gpu\" must be false)", flush=True)
        return 1
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    enc = Encoder(Path(a.mmproj), torch.bfloat16 if a.dtype == "bf16" else torch.float32)
    limits = processor_limits(a.model or None)
    print(f"[vision-hy4] {enc.L} blocks loaded in {time.time() - t0:.1f} s ({a.dtype}, {torch.get_num_threads()} threads, GELU "
          f"{'tanh' if enc.gelu_tanh else 'erf'})", file=sys.stderr, flush=True)
    print("READY", flush=True)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] != "ENC" or len(parts) != 3:
            print("ERR expected ENC <image> <out>", flush=True)
            continue
        try:
            t1 = time.time()
            x = load_image(Path(parts[1]).read_bytes(), a.max_tokens, limits)
            rows, nx, ny = enc.rows(x)
            write_sve(Path(parts[2]), rows, nx, ny)
            print(f"[vision-hy4] {x.shape[2]}x{x.shape[1]} px -> {nx - 1}x{ny} cells + newlines = {rows.shape[0]} rows in "
                  f"{time.time() - t1:.1f} s", file=sys.stderr, flush=True)
            print(f"OK {rows.shape[0]}", flush=True)
        except Exception as e:  # noqa: BLE001 - one picture's failure is that request's error; the encoder stays up
            print(f"ERR {type(e).__name__}: {str(e)[:300]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
