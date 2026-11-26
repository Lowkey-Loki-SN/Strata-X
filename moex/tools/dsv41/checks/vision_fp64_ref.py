"""vision_fp64_ref.py WORK_DIR NAME [NAME ...] - DSV41_PLAN §54: the release's ViT and aligner in float64, the reference both fp32
encoders are measured against.  serve/vision_dsv41.py's steps with every weight and activation in float64 (the bf16 weights and
patches are exact there), the attention's softmax exact (1,024 queries at a time), the RoPE angles in float32 as vision.py forms
them (pos * inv_freq, both float32 - the model's own definition), their cos / sin in float64.  For each NAME of WORK_DIR (written
by vision_gpu_parity.py and dsv41_vision_parity's OUT): NAME.svp (the patches), NAME.cpu.sve (the CPU encoder: fp32, fused) and
NAME.gpu.sve (the GPU encoder) - each one's relative L2 error against the float64 span, and theirs against each other.
Runs on the vision block's interpreter (llama.cpp's venv: torch); CPU only."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R = Path(_REPO)
sys.path[:0] = [str(R)]
from serve import vision_dsv41 as cpu  # noqa: E402

work = Path(sys.argv[1])
cfg = json.loads((R / "strata-dsv41.json").read_text())
v = cfg["vision"]
mcfg = json.loads(Path(v["model"]).read_text())
torch.set_num_threads(int(v.get("threads", 20)))
L, D, H = mcfg["vision_n_layers"], mcfg["vision_dim"], mcfg["vision_n_heads"]
names = ["vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias", "vision.norm.weight", "aligner.w1.weight", "aligner.w1.bias",
         "aligner.w2.weight", "aligner.w2.bias", "image_start", "image_newline", "image_end"]
for i in range(L):
    names += [f"vision.blocks.{i}.{s}" for s in ("norm1.weight", "attn.wqkv.weight", "attn.wqkv.bias", "attn.wo.weight", "attn.wo.bias",
                                                  "norm2.weight", "mlp.w1.weight", "mlp.w2.weight")]
W = {k: t.to(torch.float64) for k, t in cpu.load_tensors(Path(v["mmproj"]), names).items()}


def read_svp(p):
    b = p.read_bytes()
    _, vh, vw, lh, lw, pin = struct.unpack("<6i", b[:24])
    bits = np.frombuffer(b[24:], dtype=np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(bits.view(np.float32).astype(np.float64)).reshape(vh * vw, pin), vh, vw, lh, lw


def read_sve(p):
    b = p.read_bytes()
    _, n, _, _, w = struct.unpack("<5i", b[:20])
    return np.frombuffer(b[20:], dtype=np.float32).reshape(n, w).astype(np.float64)


def rms(x, w, eps=1e-6):
    return w * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


@torch.inference_mode()
def span64(patches, vh, vw, lh, lw):
    x = F.linear(patches, W["vision.patch_embed.proj.weight"], W["vision.patch_embed.proj.bias"])
    inv = 1.0 / (float(mcfg["vision_rope_theta"]) ** (torch.arange(0, D // H // 2, 2, dtype=torch.float32) / (D // H // 2)))
    hp = torch.arange(vh).unsqueeze(1).expand(vh, vw)
    wp = torch.arange(vw).unsqueeze(0).expand(vh, vw)
    ang = (torch.stack([hp, wp], dim=-1).reshape(-1, 2, 1).float() * inv).flatten(1)        # vision_cos_sin's float32 angles
    cos, sin = ang.double().cos().unsqueeze(1), ang.double().sin().unsqueeze(1)
    n, hd = x.size(0), D // H

    def rope(t):
        a, b = t.chunk(2, dim=-1)
        return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

    for i in range(L):
        bk = f"vision.blocks.{i}."
        h = rms(x, W[bk + "norm1.weight"])
        q, k, vv = (t.view(n, H, hd) for t in F.linear(h, W[bk + "attn.wqkv.weight"], W[bk + "attn.wqkv.bias"]).chunk(3, dim=-1))
        q, k = rope(q).transpose(0, 1), rope(k).transpose(0, 1)
        vv = vv.transpose(0, 1)
        o = torch.empty_like(q)
        for q0 in range(0, n, 1024):
            s = torch.matmul(q[:, q0:q0 + 1024], k.transpose(1, 2)) * (1.0 / hd ** 0.5)
            o[:, q0:q0 + 1024] = torch.matmul(torch.softmax(s, dim=-1), vv)
        x = x + F.linear(o.transpose(0, 1).reshape(n, -1), W[bk + "attn.wo.weight"], W[bk + "attn.wo.bias"])
        gate, up = F.linear(rms(x, W[bk + "norm2.weight"]), W[bk + "mlp.w1.weight"]).chunk(2, dim=-1)
        x = x + F.linear(F.silu(gate) * up, W[bk + "mlp.w2.weight"])
    x = rms(x, W["vision.norm.weight"])
    r = mcfg["vision_downsample_ratio"]
    x = x.view(vh, vw, -1).permute(2, 0, 1)
    x = F.pad(x, (0, -vw % r, 0, -vh % r))
    x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
    rows = F.linear(F.gelu(F.linear(x, W["aligner.w1.weight"], W["aligner.w1.bias"])), W["aligner.w2.weight"], W["aligner.w2.bias"])
    grid = rows.view(lh, lw, -1)
    body = torch.cat([grid, W["image_newline"].view(1, 1, -1).expand(lh, 1, -1)], dim=1).reshape(-1, rows.size(-1))
    return torch.cat([W["image_start"][None], body, W["image_end"][None]], dim=0).numpy()


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


for name in sys.argv[2:]:
    cache = work / f"{name}.f64.npy"   # the float64 span, kept for the next comparison
    if cache.exists():
        ref = np.load(cache)
    else:
        ref = span64(*read_svp(work / f"{name}.svp"))
        np.save(cache, ref)
    c, g = read_sve(work / f"{name}.cpu.sve"), read_sve(work / f"{name}.gpu.sve")
    print(f"  {name}: {ref.shape[0]} rows - against float64: the CPU encoder {rel(c, ref):.3g}, the GPU encoder {rel(g, ref):.3g}; "
          f"GPU against CPU {rel(g, c):.3g}", flush=True)
