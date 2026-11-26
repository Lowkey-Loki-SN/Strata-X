"""moex/tools/dsv41/fp4_repack.py - DeepSeek-V4.1's FP4 experts to ggml block_mxfp4, losslessly.

The release stores a [rows, K] FP4 weight as int8 [rows, K/2]: element 2i is the low nibble of byte i, 2i+1 the high
nibble, values FP4_TABLE[code] (E2M1), with one E8M0 scale per 32 consecutive K elements in a separate [rows, K/32]
uint8 tensor (inference/convert.py, kernel.py fp4_gemm_kernel). Value = FP4_TABLE[code] * 2^(e - 127).

ggml's block_mxfp4 (17 bytes per 32 elements) is the scale byte e, then qs[16] with qs[j] = code(j) | code(j+16) << 4,
dequantized as kvalues[code] * e8m0_half(e), where kvalues is the doubled E2M1 table and e8m0_half(e) = 2^(e - 128).
So the codes and the scale byte are kept as they are; only the nibbles move inside each block, and every value is
the same float32. Same 4.25 bits per weight. (llama.cpp does the same for V4: conversion/base.py repack_mxfp4_blocks.)

An expert record in the pack is [w1 | w3 | w2], each as rows x (K/32) blocks in row-major order.
"""
from __future__ import annotations

import numpy as np

BLOCK = 32
BLOCK_BYTES = 17
FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                     dtype=np.float32)


def repack(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """weight: uint8/int8 [rows, K/2] as released; scale: uint8 [rows, K/32] (E8M0). Returns uint8 [rows, K/32*17]."""
    w = np.ascontiguousarray(weight).view(np.uint8)
    s = np.ascontiguousarray(scale).view(np.uint8)
    rows, half_k = w.shape
    nb = half_k * 2 // BLOCK
    if s.shape != (rows, nb) or half_k * 2 % BLOCK:
        raise ValueError(f"shape mismatch: weight {w.shape}, scale {s.shape}")
    b = w.reshape(rows, nb, 16)
    codes = np.empty((rows, nb, BLOCK), dtype=np.uint8)
    codes[..., 0::2] = b & 0x0F
    codes[..., 1::2] = b >> 4
    out = np.empty((rows, nb, BLOCK_BYTES), dtype=np.uint8)
    out[..., 0] = s
    out[..., 1:] = codes[..., :16] | (codes[..., 16:] << 4)
    return out.reshape(rows, nb * BLOCK_BYTES)


def unrepack(blocks: np.ndarray, rows: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of repack: ggml block_mxfp4 bytes [rows, K/32*17] -> (weight uint8 [rows, K/2], scale uint8
    [rows, K/32]) in the release's layout (element 2i = low nibble of byte i)."""
    nb = k // BLOCK
    b = np.ascontiguousarray(blocks).view(np.uint8).reshape(rows, nb, BLOCK_BYTES)
    scale = b[..., 0].copy()
    qs = b[..., 1:]
    codes = np.empty((rows, nb, BLOCK), dtype=np.uint8)
    codes[..., :16] = qs & 0x0F
    codes[..., 16:] = qs >> 4
    weight = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(rows, k // 2)
    return weight, scale


EXPERT_SHAPES = (("w1", 2304, 5120), ("w3", 2304, 5120), ("w2", 5120, 2304))  # pack record order


def split_record(rec: bytes | np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One pack record [w1 | w3 | w2] (ggml MXFP4) -> {name: (weight uint8 [rows, K/2], scale uint8 [rows, K/32])}."""
    buf = np.frombuffer(rec, dtype=np.uint8) if isinstance(rec, (bytes, bytearray, memoryview)) else rec
    out, at = {}, 0
    for name, rows, k in EXPERT_SHAPES:
        n = rows * (k // BLOCK) * BLOCK_BYTES
        out[name] = unrepack(buf[at:at + n], rows, k)
        at += n
    if at != buf.size:
        raise ValueError(f"record is {buf.size} bytes, the shapes need {at}")
    return out


def dequant_reference(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The release's own decoding (FP4_TABLE x 2^(e-127)), float32 [rows, K]."""
    w = np.ascontiguousarray(weight).view(np.uint8)
    s = np.ascontiguousarray(scale).view(np.uint8)
    rows, half_k = w.shape
    codes = np.empty((rows, half_k * 2), dtype=np.uint8)
    codes[:, 0::2] = w & 0x0F
    codes[:, 1::2] = w >> 4
    vals = FP4_TABLE[codes].reshape(rows, -1, BLOCK)
    e = s.astype(np.int32)
    if (e == 255).any():
        raise ValueError("E8M0 NaN scale (255) in the release")
    # 2^(e-127) exactly, including the subnormal range e < 1 (float32 cannot hold 2^-127 as a normal number)
    pw = np.ldexp(np.float32(1.0), e - 127).astype(np.float32)
    return (vals * pw[..., None]).reshape(rows, -1)


def expert_record(w1, s1, w3, s3, w2, s2) -> bytes:
    """One pack record: [w1 | w3 | w2], each repacked. 18,800,640 bytes for V4.1 (4,590 pages of 4 KiB)."""
    return b"".join(repack(w, s).tobytes() for w, s in ((w1, s1), (w3, s3), (w2, s2)))
