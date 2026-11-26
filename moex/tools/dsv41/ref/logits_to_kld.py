"""moex/tools/dsv41/ref/logits_to_kld.py - a reference run's logits as a llama-perplexity --kl-divergence-base file.

    python logits_to_kld.py --run REF_RUN_DIR --out FILE [--n-ctx N]

REF_RUN_DIR is ref_run.py's output (ids.npy, logits.npy: [S, vocab] fp32 from one prefill). The file holds one chunk of
n_ctx tokens (default: S rounded down to even) and the log-probabilities of positions [n_ctx/2, n_ctx - 1), encoded as
strata-x's kld_encode does (src/x/main.cpp: scale and min log-probability as two floats, then a uint16 per vocab
entry, the floor at max - 16). `strata-x --kld FILE` then reads the same tokens - the first half as a prompt, the scored
half through decode windows - and prints llama-perplexity's numbers (mean KLD, same top token, ...) against the reference.
"""
from __future__ import annotations

import argparse
import pathlib
import struct

import numpy as np


def encode_row(logits: np.ndarray) -> np.ndarray:
    lg = logits.astype(np.float32)
    mx, mn = float(lg.max()), float(lg.min())
    lse = float(np.log(np.exp((lg - mx).astype(np.float64)).sum()))
    mn = max(mn, mx - 16.0)
    scale = max(1e-12, (mx - mn) / 65535.0)
    q = np.clip(np.rint((np.maximum(lg, mn) - mn) / scale), 0, 65535).astype(np.uint16)
    nv = 2 * ((lg.size + 1) // 2) + 4
    out = np.zeros(nv, dtype=np.uint16)
    out[:4] = np.frombuffer(struct.pack("<ff", scale, mn - mx - lse), dtype=np.uint16)
    out[4:4 + lg.size] = q
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--n-ctx", type=int, default=0)
    a = ap.parse_args()
    ids = np.load(a.run / "ids.npy").astype(np.int32)
    logits = np.load(a.run / "logits.npy", mmap_mode="r")
    n_ctx = a.n_ctx or (len(ids) // 2 * 2)
    if n_ctx > len(ids) or n_ctx > logits.shape[0] or n_ctx < 4:
        raise SystemExit(f"n_ctx {n_ctx} does not fit the run ({len(ids)} tokens, {logits.shape[0]} logit rows)")
    n_vocab = logits.shape[1]
    with open(a.out, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<iii", n_ctx, n_vocab, 1))
        f.write(ids[:n_ctx].tobytes())
        for p in range(n_ctx // 2, n_ctx - 1):
            f.write(encode_row(np.asarray(logits[p])).tobytes())
    print(f"{a.out}: 1 chunk of {n_ctx} tokens, {n_ctx - 1 - n_ctx // 2} scored rows, vocab {n_vocab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
