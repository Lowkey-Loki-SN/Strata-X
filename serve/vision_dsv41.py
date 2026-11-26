#!/usr/bin/env python3
"""serve/vision_dsv41.py - DeepSeek-V4.1-Flash's image encoder for the server (moex): the release's ViT and aligner
(inference/vision.py) and its image plan (inference/image_processor.py), as a resident process with the protocol of the
server's other encoders (serve/server.py Vision): READY, then per line `ENC <image file> <out file>` -> `OK <rows>` (the out file
one SVE1 record: magic "SVE1", rows, nx, ny, width, then rows x width float32) or `ERR <why>`; QUIT ends it.

A record is the image's whole span as the model reads it - image_processor.image_token_types:

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

with the learned image_start / image_newline / image_end rows at the delimiters and the aligner's rows in reading order at the
IMAGE slots (model.py merge_image_embeddings).  The prompt holds <｜deepseek_image｜> (129264) at every one of those positions; the
engine reads the span causally, routes it with exp_probs_b_vl and gives it no engram (model.py forward).

    vision_dsv41.py --mmproj DENSE_DIR --model CONFIG_JSON [--threads N] [--max-tokens N] [--dtype bf16|fp32]

DENSE_DIR: the release's safetensors (vision.*, aligner.*, image_start/newline/end; model.safetensors.index.json or the pack's
index.json beside it); CONFIG_JSON: the release's inference/config.json (vision_*).  --dtype: bf16 computes as the release does
(torch's bfloat16 throughout, its default dtype in generate.py), fp32 keeps every activation in float32.  CPU only (no VRAM: the
engine holds both GPUs).  STRATA_DSV41_VISION_ATTN_ROWS=N (opt-in): the ViT's attention N queries at a time (see ATTN_ROWS)."""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time
from pathlib import Path


class _Dropped:
    """moex: STRATA_X_QUIET=1 (the server's "diagnostics": false) - sys.stderr, whose writes go nowhere: no diagnostic line
    (nor a warning, a usage error or a traceback) is sent anywhere.  Set before torch is imported, so its warnings too; the
    protocol on stdout is unchanged."""

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

MAGIC = 0x31455653   # "SVE1"


# ------------------------------------------------------------------------------------------------ the weights
def load_tensors(dense: Path, names: list[str]) -> dict[str, torch.Tensor]:
    """The named BF16 tensors from the release's safetensors (read by offset: no safetensors package needed)."""
    idx_file = next((p for p in (dense / "model.safetensors.index.json", dense.parent / "index.json") if p.exists()), None)
    if idx_file is None:
        raise SystemExit(f"no model.safetensors.index.json in {dense} (nor index.json beside it)")
    wmap = json.loads(idx_file.read_text())["weight_map"]
    by_file: dict[str, list[str]] = {}
    for n in names:
        if n not in wmap:
            raise SystemExit(f"{n}: not in {idx_file}")
        by_file.setdefault(wmap[n], []).append(n)
    out = {}
    for fname, ns in by_file.items():
        with open(dense / fname, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(hlen))
            base = 8 + hlen
            for n in ns:
                h = head[n]
                if h["dtype"] != "BF16":
                    raise SystemExit(f"{n}: {h['dtype']}, BF16 expected")
                a, b = h["data_offsets"]
                f.seek(base + a)
                raw = np.frombuffer(f.read(b - a), dtype=np.uint16).copy()
                out[n] = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16).reshape(h["shape"])
    return out


# ------------------------------------------------------------------------------------------------ the ViT (vision.py)
def vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def rotary(x, cos, sin):
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


def rms(x, w, eps=1e-6):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (w.float() * x).to(dtype)


# moex (DSV41_BACKLOG #56): vision.py's scaled_dot_product_attention on 3-D q/k/v takes torch's math path, which holds the
# (heads, n, n) scores in float32: for a 2400x1800 picture (8,748 patches) 4.6 GiB, and the call peaks 10.4 GiB above its
# inputs (torch 2.11 CPU, bf16).  STRATA_DSV41_VISION_ATTN_ROWS=N (opt-in) puts the queries through it N at a time, the same
# call on each block: 1.3 GiB at 1024.  Bitwise the same rows wherever compared (the call alone at 1,530-8,748 patches, 8
# and 20 threads, bf16 and fp32; the encoder on test_vision_dsv41.py's pictures but the 2400x1800 one), but at 20 threads
# the call alone took 3.30 s for 2.77 (8,748 patches, noisy).  0, the default: one call, as vision.py makes it.
ATTN_ROWS = int(os.environ.get("STRATA_DSV41_VISION_ATTN_ROWS", "0"))
# moex (the dsx audit, 2026-10-23), opt-in STRATA_DSV41_VISION_ATTN=fused: the same call on (1, heads, n, head_dim), which torch
# sends to its fused CPU kernel (blocks of queries and keys, softmax on the fly) instead of the math path - the same attention,
# summed in another order; ATTN_ROWS is moot there (no (heads, n, n) scores are held)
ATTN_FUSED = os.environ.get("STRATA_DSV41_VISION_ATTN") == "fused"


def attention(q, k, v):
    """vision.py's F.scaled_dot_product_attention(q, k, v) on (heads, n, head_dim), ATTN_ROWS queries at a time."""
    n = q.size(1)
    if ATTN_FUSED:
        return F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
    if ATTN_ROWS <= 0 or n <= ATTN_ROWS:
        return F.scaled_dot_product_attention(q, k, v)
    return torch.cat([F.scaled_dot_product_attention(q[:, i:i + ATTN_ROWS], k, v) for i in range(0, n, ATTN_ROWS)], dim=1)


class Encoder:
    def __init__(self, dense: Path, cfg: dict, dtype: torch.dtype):
        self.cfg, self.dtype = cfg, dtype
        self.L, self.heads = cfg["vision_n_layers"], cfg["vision_n_heads"]
        self.vdim, self.p, self.r = cfg["vision_dim"], cfg["vision_patch_size"], cfg["vision_downsample_ratio"]
        self.rope_dim, self.theta = self.vdim // self.heads // 2, float(cfg["vision_rope_theta"])
        names = ["vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias", "vision.norm.weight",
                 "aligner.w1.weight", "aligner.w1.bias", "aligner.w2.weight", "aligner.w2.bias",
                 "image_start", "image_newline", "image_end"]
        for i in range(self.L):
            b = f"vision.blocks.{i}."
            names += [b + s for s in ("norm1.weight", "attn.wqkv.weight", "attn.wqkv.bias", "attn.wo.weight", "attn.wo.bias",
                                      "norm2.weight", "mlp.w1.weight", "mlp.w2.weight")]
        t = load_tensors(dense, names)
        # RMSNorm weights stay float32 (vision.py RMSNorm's parameter), the rest in the compute dtype
        self.w = {k: (v.float() if k.endswith("norm1.weight") or k.endswith("norm2.weight") or k == "vision.norm.weight"
                      else v.to(dtype)) for k, v in t.items()}

    def lin(self, x, name, bias=True):
        return F.linear(x, self.w[name + ".weight"], self.w.get(name + ".bias") if bias else None)

    @torch.inference_mode()
    def features(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """ViT.forward then Aligner.forward: (n_vit_h * n_vit_w, 3, p, p) patches -> (n_llm_h * n_llm_w, dim) rows."""
        w = self.w
        x = self.lin(patches.to(self.dtype).flatten(1), "vision.patch_embed.proj")
        cos, sin = vision_cos_sin(n_h, n_w, self.rope_dim, self.theta)
        n, hd = x.size(0), self.vdim // self.heads
        for i in range(self.L):
            b = f"vision.blocks.{i}."
            h = rms(x, w[b + "norm1.weight"])
            q, k, v = (t.view(n, self.heads, hd) for t in self.lin(h, b + "attn.wqkv").chunk(3, dim=-1))
            q, k = rotary(q, cos, sin), rotary(k, cos, sin)
            o = attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
            x = x + self.lin(o.transpose(0, 1).reshape(n, -1), b + "attn.wo")
            h = rms(x, w[b + "norm2.weight"])
            gate, up = self.lin(h, b + "mlp.w1", bias=False).chunk(2, dim=-1)
            x = x + self.lin(F.silu(gate) * up, b + "mlp.w2", bias=False)
        x = rms(x, w["vision.norm.weight"])
        r = self.r
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.lin(F.gelu(self.lin(x, "aligner.w1")), "aligner.w2")

    def span(self, rows: torch.Tensor, n_llm_h: int, n_llm_w: int) -> torch.Tensor:
        """The image's span rows in the model's order (image_processor.image_token_types, merge_image_embeddings)."""
        w = self.w
        grid = rows.view(n_llm_h, n_llm_w, -1)
        nl = w["image_newline"].to(rows.dtype).view(1, 1, -1).expand(n_llm_h, 1, -1)
        body = torch.cat([grid, nl], dim=1).reshape(-1, rows.size(-1))
        return torch.cat([w["image_start"].to(rows.dtype)[None], body, w["image_end"].to(rows.dtype)[None]], dim=0)


# ------------------------------------------------------------------------------------------------ the plan (image_processor.py)
# moex (DSV41_BACKLOG #56): the arithmetic is serve/vision_dsv41_plan.py's, which the server's count_tokens shares (no torch)
try:
    from serve.vision_dsv41_plan import max_tokens_of, num_image_tokens, plan_image_grid   # imported as serve.vision_dsv41
except ImportError:
    from vision_dsv41_plan import max_tokens_of, num_image_tokens, plan_image_grid         # run as a script: its directory


def load_image(data: bytes, cfg, max_n_token):
    """image_processor.load_image: -> (patches in bfloat16, n_vit_h, n_vit_w, n_llm_h, n_llm_w)."""
    p = cfg["vision_patch_size"]
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, cfg, max_n_token)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if cfg.get("vision_max_wh_ratio") is not None and image.width >= cfg["vision_max_wh_ratio"] * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def write_sve(path: Path, rows: torch.Tensor):
    a = rows.float().contiguous().numpy()
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", MAGIC, a.shape[0], a.shape[0], 1, a.shape[1]))
        f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", required=True, help="the release's safetensors directory (the dense shards)")
    ap.add_argument("--model", required=True, help="the release's inference/config.json")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="at most this many tokens an image (the release's 1024 at most)")
    ap.add_argument("--min-tokens", type=int, default=0, help="(read for the protocol; the release's minimum is in pixels)")
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default=os.environ.get("STRATA_DSV41_VISION_DTYPE", "bf16"))
    ap.add_argument("--gpu", action="store_true", help="(refused: this encoder runs on the CPU; the config's \"gpu\": true has the "
                                                       "engine encode the pictures on its GPU instead, DSV41_PLAN §54)")
    a = ap.parse_args()
    if a.gpu:
        print("ERR this DeepSeek-V4.1 image encoder runs on the CPU; the server has the engine encode on its GPU when the "
              "config's vision block says \"gpu\": true (DSV41_PLAN §54)", flush=True)
        return 1
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    cfg = json.loads(Path(a.model).read_text())
    if not cfg.get("vision_n_layers"):
        print("ERR the model's config has no vision tower (vision_n_layers)", flush=True)
        return 1
    max_n = max_tokens_of(cfg, a.max_tokens)
    t0 = time.time()
    enc = Encoder(Path(a.mmproj), cfg, torch.bfloat16 if a.dtype == "bf16" else torch.float32)
    print(f"[vision-dsv41] {enc.L} ViT layers loaded in {time.time() - t0:.1f} s ({a.dtype}, {torch.get_num_threads()} threads, "
          f"at most {max_n} tokens an image)", file=sys.stderr, flush=True)
    # moex (DSV41_PLAN §22, opt-in STRATA_DSV41_VISION_EARLY=1): the weights are in memory - the server may start the engine
    # now (it sizes its RAM copy by what is free) and wait for READY, after the warm-up, before it serves anything
    if os.environ.get("STRATA_DSV41_VISION_EARLY") == "1":
        print("LOADED", flush=True)
    # moex (DSV41_BACKLOG #73, opt-in STRATA_DSV41_VISION_WARMUP=1): one pass of a fixed picture before READY, so the first
    # real picture is not the process's first pass - every start's first pass may round unlike the later ones (#58), and the
    # server keeps a picture's rows for its whole run (~7 s at the plan's smallest picture, 1,521 patches)
    if os.environ.get("STRATA_DSV41_VISION_WARMUP") == "1":
        t2 = time.time()
        g = np.add.outer(np.arange(544, dtype=np.uint16), np.arange(544, dtype=np.uint16)) % 256
        buf = io.BytesIO()
        Image.fromarray(np.stack([g, g[::-1], g.T], axis=-1).astype(np.uint8), "RGB").save(buf, "PNG")
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(buf.getvalue(), cfg, max_n)
        enc.span(enc.features(patches, n_vit_h, n_vit_w), n_llm_h, n_llm_w)
        print(f"[vision-dsv41] warmed up with a {n_vit_h}x{n_vit_w}-patch picture in {time.time() - t2:.1f} s", file=sys.stderr,
              flush=True)
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
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(Path(parts[1]).read_bytes(), cfg, max_n)
            rows = enc.span(enc.features(patches, n_vit_h, n_vit_w), n_llm_h, n_llm_w)
            assert rows.size(0) == num_image_tokens(n_llm_h, n_llm_w)
            write_sve(Path(parts[2]), rows)
            print(f"[vision-dsv41] {n_vit_h}x{n_vit_w} patches -> {n_llm_h}x{n_llm_w} + delimiters = {rows.size(0)} rows in "
                  f"{time.time() - t1:.1f} s", file=sys.stderr, flush=True)
            print(f"OK {rows.size(0)}", flush=True)
        except Exception as e:  # noqa: BLE001 - one image's failure is that request's error, the encoder stays up
            print(f"ERR {type(e).__name__}: {str(e)[:300]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
