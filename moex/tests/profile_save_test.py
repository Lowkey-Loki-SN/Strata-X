#!/usr/bin/env python3
"""moex/tests/profile_save_test.py - strata-x's learned expert profile (--expert-profile-save) in Strata's #477 order, on
the 4-layer GLM slice through the engine protocol (the GPUs must be free).

    python3 moex/tests/profile_save_test.py [--model SLICE.gguf] [--tokens FILE] [--engine build/strata-x]

Checks (each prints PASS / FAIL):
  1. the saved file: a header with the model's shape (serve/server.py profile_shape reads it), every expert of the MoE
     layers once, the VRAM tier's experts first, and the counts this run's decode routings (a prompt-only request adds
     none: a prompt read is not counted, as Strata's)
  2. a start from it begins where the run ended: an engine loaded from the saved file and saved again without a request
     writes the same order, every count halved (no routing this run, half the loaded profile's)
"""
import argparse
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_protocol_test import DATA, X, Engine  # noqa: E402

sys.path.insert(0, X)
sys.path.insert(0, os.path.join(X, "tools"))
from serve.server import profile_shape  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(name)


def read(path):
    with open(path) as f:
        head = f.readline()
        rows = [l.split() for l in f if l.strip()]
    return head, [(int(a), int(b), float(c)) for a, b, c in rows]


def run(exe, model, extra, requests):
    e = Engine(exe, model, extra)
    try:
        out = [e.gen(ids, n, "temperature=0") for ids, n in requests]
        e.send("QUIT")
        e.p.wait(timeout=120)
        return out
    finally:
        if e.p.poll() is None:
            e.p.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{DATA}/glm-mini/glm53-mini-L0-3.gguf")
    ap.add_argument("--tokens", default=f"{DATA}/glm-mini/dump-long/tokens.txt")
    ap.add_argument("--engine", default=os.path.join(X, "build/strata-x"))
    a = ap.parse_args()
    ids = [int(t) for t in open(a.tokens).read().replace(",", " ").split()][:600]
    tmp = tempfile.mkdtemp(prefix="glm-profile-")
    p1, p2, p0 = (os.path.join(tmp, n) for n in ("learned.txt", "again.txt", "prompt-only.txt"))
    slots = ["--expert-slots", "64", "--suffix-draft", "0"]   # a tier smaller than the layer: residency matters

    # a prompt-only request (max_new 1: the first window is the only decode) counts at most one window's routings
    run(a.engine, a.model, slots + ["--expert-profile-save", p0], [(ids, 1)])
    _, r0 = read(p0)
    check("1a. a prompt read is not counted (one window's routings at most)", sum(c for _, _, c in r0) <= 8,
          f"{sum(c for _, _, c in r0)} routings")

    run(a.engine, a.model, slots + ["--expert-profile-save", p1], [(ids, 48), (ids[:300], 48)])
    head, rows = read(p1)
    check("1b. the header names the model's shape", profile_shape(p1) is not None and "n_layer" in head, head.strip())
    keys = [(l, e) for l, e, _ in rows]
    n_exp = profile_shape(p1)[1] if profile_shape(p1) else 0
    layers = sorted({l for l, _ in keys})
    check("1c. every expert of the MoE layers once", len(set(keys)) == len(keys) == len(layers) * n_exp,
          f"{len(keys)} rows, {len(layers)} layers x {n_exp}")
    total = sum(c for _, _, c in rows)
    check("1d. the counts are decode routings (2 x 48 tokens x top-8 per MoE layer, at most)",
          0 < total <= 2 * 49 * 8 * len(layers), f"{total}")
    m = re.search(r"(\d+) in VRAM", head)
    in_vram = int(m.group(1)) if m else -1
    rises = [i for i in range(1, len(rows)) if rows[i][2] > rows[i - 1][2]]
    check("1e. the VRAM tier first, each part hottest first (the count rises only where the tier ends)",
          0 < in_vram <= 64 and set(rises) <= {in_vram}, f"{in_vram} in VRAM, rises at {rises[:5]}")

    run(a.engine, a.model, slots + ["--expert-profile", p1, "--expert-profile-save", p2], [])
    _, rows2 = read(p2)
    same = [(l, e) for l, e, _ in rows2] == keys
    halved = all(abs(c2 - 0.5 * c1) < 0.051 for (_, _, c1), (_, _, c2) in zip(rows, rows2))
    check("2. a start from it saves the same order, the counts halved", same and halved,
          f"order {'same' if same else 'differs'}, counts {'halved' if halved else 'not halved'}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}", flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
