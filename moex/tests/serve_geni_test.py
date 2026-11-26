#!/usr/bin/env python3
"""moex/tests/serve_geni_test.py - strata-x --serve's GENI (images, moex vision) on the 4-layer GLM slice.

    python3 -B moex/tests/serve_geni_test.py [--model SLICE.gguf] [--tokens FILE] [--engine build/strata-x]
                                             [--gguf-py LLAMA_CPP/gguf-py]

Checks (each prints PASS / FAIL):
  1. GENI is refused (ERR, the engine keeps serving) by an engine started without --image-pad
  2. IDENTITY: image rows equal to the pad token's own embedding (dequantized from the GGUF with gguf-py) give the very
     tokens GEN gives for the same ids - on the window path and on the prompt path: the substitution changes nothing else
  3. image rows matter: other rows, other tokens; the same rows twice, the same tokens
  4. reuse: the same images extend the live sequence (RESUME = what the engine held); the same ids with other images are
     not reused past the image start (RESUME <= image start)
  5. checkpoints keyed by images: a long prompt (prompt path, a checkpoint at the turn token) with image A, then the same
     ids with image A (RESUME = the checkpoint) and with image B (RESUME = 0)
  6. malformed GENI: a missing file, rows too narrow, more pads than rows, rows left over, a pad on the last token -> ERR
Nothing else is started: one engine process at a time (the GPUs must be free; take moex/.gpu.lock).
"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
import argparse
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_protocol_test import DATA, X, Engine  # noqa: E402

PAD = 154854          # GLM's <|image|>
N_EMBD = 4096

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""), flush=True)
    fails += 0 if ok else 1


def write_sve(path, images):
    """images: list of float32 arrays (n, N_EMBD) -> concatenated SVE1 records (nx = n, ny = 1: 1-D positions)."""
    with open(path, "wb") as f:
        for rows in images:
            n = rows.shape[0]
            f.write(struct.pack("<5i", 0x31455653, n, n, 1, rows.shape[1]))
            f.write(np.ascontiguousarray(rows, dtype=np.float32).tobytes())


def pad_embedding(model, gguf_py):
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(model)
    t = next(t for t in r.tensors if t.name == "token_embd.weight")
    row_bytes = t.data.shape[-1] if t.data.ndim == 2 else None
    raw = np.asarray(t.data)
    row = dequantize(raw[PAD:PAD + 1], t.tensor_type)
    assert row.shape[-1] == N_EMBD, (row.shape, row_bytes)
    return row.reshape(N_EMBD).astype(np.float32)


def geni(eng, path, ids, max_new, keys=""):
    """GEN's reader with a GENI line."""
    eng.send(f"GENI {max_new}{(' ' + keys) if keys else ''} {path} {','.join(map(str, ids))}")
    r = dict(tokens=[], resume=None, reused=None, done=None, err=None)
    while True:
        line = eng.q.get(timeout=600)
        if line is None:
            r["err"] = "engine ended"
            return r
        f = line.split()
        if line.startswith("T "):
            r["tokens"].append(int(f[1]))
        elif line.startswith("RESUME "):
            r["resume"] = int(f[1])
        elif line.startswith("REUSED "):
            r["reused"] = int(f[1])
        elif line.startswith("DONE"):
            r["done"] = f[1:]
            return r
        elif line.startswith("ERR"):
            r["err"] = line[4:]
            return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{DATA}/glm-mini/glm53-mini-L0-3.gguf")
    ap.add_argument("--tokens", default=f"{DATA}/glm-mini/dump-long/tokens.txt")
    ap.add_argument("--engine", default=os.path.join(X, "build/strata-x"))
    ap.add_argument("--gguf-py", default=_os.path.join(_DATA, 'llama.cpp-mainline/gguf-py'))
    a = ap.parse_args()
    toks = [int(x) for x in open(a.tokens).read().strip().split(",")]
    tmp = tempfile.mkdtemp(prefix="geni-test-", dir=os.environ.get("TMPDIR"))
    rng = np.random.default_rng(7)
    pad_row = pad_embedding(a.model, a.gguf_py)
    scale = float(np.sqrt((pad_row.astype(np.float64) ** 2).mean()))

    # 1. no --image-pad: refused, and the engine keeps serving
    eng = Engine(a.engine, a.model, ["--prefill-chunk", "0", "--short-read", "64"])
    p1 = os.path.join(tmp, "one.sve")
    k = 12
    ids = toks[:20] + [PAD] * k + toks[20:30]
    write_sve(p1, [np.tile(pad_row, (k, 1))])
    r = geni(eng, p1, ids, 4)
    check("1a. GENI without --image-pad is an ERR", r["err"] is not None and "image-pad" in r["err"], str(r["err"]))
    r = eng.gen(toks[:30], 4)
    check("1b. ... and the engine keeps serving", r["err"] is None and len(r["tokens"]) == 4, str(r["err"]))
    eng.close()

    turn = 154827   # <|user|>: a turn token for the checkpoint policy (placed into the long prompt below)
    # --no-adaptive: a frozen expert tier, so two runs of the same input round the same (GPU vs CPU experts)
    eng = Engine(a.engine, a.model, ["--prefill-chunk", "0", "--short-read", "64", "--image-pad", str(PAD),
                                     "--turn-tokens", str(turn), "--no-adaptive"])
    # 2. identity, window path (a 42-token prompt <= short read 64) and prompt path (a 400-token one)
    for label, pre, post in (("window path", toks[:20], toks[20:30]), ("prompt path", toks[:200], toks[200:388])):
        ids = pre + [PAD] * k + post
        base = eng.gen(ids, 12, "ckpt=0")
        same = geni(eng, p1, ids, 12, "ckpt=0")
        check(f"2. identity ({label}): pad-embedding rows give GEN's tokens", base["err"] is None and same["err"] is None and
              same["tokens"] == base["tokens"] and len(base["tokens"]) == 12, f"{same['tokens'][:6]} vs {base['tokens'][:6]} {same['err']}")
    # 3. rows matter, and are deterministic
    ids = toks[:20] + [PAD] * k + toks[20:30]
    pA, pB = os.path.join(tmp, "a.sve"), os.path.join(tmp, "b.sve")
    write_sve(pA, [(rng.standard_normal((k, N_EMBD)) * scale).astype(np.float32)])
    write_sve(pB, [(rng.standard_normal((k, N_EMBD)) * scale).astype(np.float32)])
    ra1, ra2, rb = geni(eng, pA, ids, 12, "ckpt=0"), geni(eng, pA, ids, 12, "ckpt=0"), geni(eng, pB, ids, 12, "ckpt=0")
    base = eng.gen(ids, 12, "ckpt=0")
    check("3a. the same rows twice: the same tokens", ra1["tokens"] == ra2["tokens"] and len(ra1["tokens"]) == 12, f"{ra1['tokens'][:6]} {ra2['tokens'][:6]}")
    check("3b. other rows: other tokens (rows reach the model)", ra1["tokens"] != rb["tokens"] or ra1["tokens"] != base["tokens"],
          f"A {ra1['tokens'][:6]} B {rb['tokens'][:6]} pad {base['tokens'][:6]}")
    # 4. reuse of the live sequence: a prompt that extends the live ids with OTHER image rows is not reused past the image
    #    start; with the same image it is (RESUME = what the engine held)
    r1 = geni(eng, pA, ids, 8)
    ext = ids + r1["tokens"] + toks[30:36]
    held = len(ids) + len(r1["tokens"]) - 1
    r3 = geni(eng, pB, ext, 6)
    check("4a. live ids extended, other image: not reused past the image start", r3["resume"] is not None and r3["resume"] <= 20,
          f"resume {r3['resume']} (image at 20, the live ids {held} long)")
    r1b = geni(eng, pA, ids, 8)
    r2 = geni(eng, pA, ids + r1b["tokens"] + toks[30:36], 6)
    check("4b. live ids extended, same image: RESUME = what the engine held", r1b["tokens"] == r1["tokens"] and r2["resume"] == held,
          f"resume {r2['resume']} held {held}")
    eng2 = Engine(a.engine, a.model, ["--prefill-chunk", "0", "--short-read", "64", "--image-pad", str(PAD), "--no-adaptive"])
    fresh = geni(eng2, pB, ext, 6)
    check("4c. ... and continues as a from-scratch run with that image", fresh["tokens"] == r3["tokens"], f"{r3['tokens']} vs {fresh['tokens']}")
    eng2.close()
    # 5. checkpoints: long prompt (the prompt path) with a turn token after the image
    longp = toks[:100] + [PAD] * k + toks[100:300] + [turn] + toks[300:340]
    pos_turn = max(i for i, t in enumerate(longp) if t == turn)   # the checkpoint goes in front of the last turn token
    ra = geni(eng, pA, longp, 4)
    rb2 = geni(eng, pA, longp, 4)
    check("5a. same image: resumed from the checkpoint at the turn token", rb2["resume"] == pos_turn and rb2["tokens"] == ra["tokens"],
          f"resume {rb2['resume']} (turn at {pos_turn}) tokens {rb2['tokens']} vs {ra['tokens']}")
    rc = geni(eng, pB, longp, 4)
    check("5b. same ids, other image: nothing reused", rc["resume"] == 0, f"resume {rc['resume']}")
    # 6. malformed requests: ERR and the engine keeps serving
    narrow = os.path.join(tmp, "narrow.sve")
    with open(narrow, "wb") as f:
        f.write(struct.pack("<5i", 0x31455653, k, k, 1, 2048))
        f.write(np.zeros((k, 2048), np.float32).tobytes())
    two = os.path.join(tmp, "two.sve")
    write_sve(two, [np.tile(pad_row, (k, 1)), np.tile(pad_row, (3, 1))])
    cases = (("a missing file", os.path.join(tmp, "nope.sve"), ids),
             ("rows of the wrong width", narrow, ids),
             ("more pads than rows", p1, toks[:20] + [PAD] * (k + 2) + toks[20:30]),
             ("rows left over", two, ids),
             ("a pad on the last token", p1, toks[:20] + [PAD] * k))
    for name, path, cids in cases:
        r = geni(eng, path, cids, 4)
        check(f"6. {name}: ERR", r["err"] is not None, str(r["err"]))
    r = eng.gen(toks[:30], 4)
    check("6. ... the engine keeps serving", r["err"] is None and len(r["tokens"]) == 4, str(r["err"]))
    eng.close()
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
