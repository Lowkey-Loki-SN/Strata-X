#!/usr/bin/env python3
"""moex/tests/serve_nan_guard_test.py - strata-x's non-finite logits guard (upstream #879) and the pageable fallback of
the prompt path's host buffers, on the 4-layer GLM slice through the engine protocol.

    python3 moex/tests/serve_nan_guard_test.py [--model SLICE.gguf] [--tokens FILE] [--engine build/strata-x]

Checks (each prints PASS / FAIL); the GPUs must be free:
  1. STRATA_NAN_INJECT=1,5: the request's first window is flagged: DONE ... nonfinite with no token, and the engine forgot
     its state - the next request reads from token 0 (RESUME 0), is flagged at its 4th window (the 5th checked) after
     the 3 tokens of the windows before it, and the one after reads from 0 again and gives the full greedy reply, whose
     start equals the flagged request's tokens and an unguarded engine's reply (STRATA_NAN_GUARD=0) token for token.
  2. STRATA_X_TEST_PAGEABLE=1 (every prompt-path host buffer takes the pageable fallback, as if each pin were refused):
     the same greedy reply as the pinned engine's, on one GPU and on a two-stage split (the hand-off buffers).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_protocol_test import DATA, X, Engine  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(name)


def with_env(env, fn):
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def reply(exe, model, ids, n, extra=(), env=None):
    """One engine: the greedy reply of n tokens (and the DONE fields)."""
    def run():
        e = Engine(exe, model, list(extra))
        try:
            r = e.gen(ids, n, "temperature=0")
            return r["tokens"], r["done"]
        finally:
            e.p.kill()
    return with_env(env or {}, run)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{DATA}/glm-mini/glm53-mini-L0-3.gguf")
    ap.add_argument("--tokens", default=f"{DATA}/glm-mini/dump-long/tokens.txt")
    ap.add_argument("--engine", default=os.path.join(X, "build/strata-x"))
    a = ap.parse_args()
    ids = [int(t) for t in open(a.tokens).read().replace(",", " ").split()][:600]
    N = 16

    # 1. the guard
    one = ["--suffix-draft", "0"]   # one token a window (no prompt-lookup windows): the flagged window's place is known
    want, _ = reply(a.engine, a.model, ids, N, one, env={"STRATA_NAN_GUARD": "0"})
    check("1a. the unguarded engine gives a full greedy reply", len(want) == N, f"{len(want)} tokens")

    def guarded():
        e = Engine(a.engine, a.model, one)
        try:
            r1 = e.gen(ids, N, "temperature=0")
            r2 = e.gen(ids, N, "temperature=0")
            r3 = e.gen(ids, N, "temperature=0")
            return r1, r2, r3
        finally:
            e.p.kill()
    r1, r2, r3 = with_env({"STRATA_NAN_INJECT": "1,5"}, guarded)
    check("1b. a flagged first window: DONE nonfinite, no token", r1["done"] is not None and r1["done"][4] == "nonfinite"
          and r1["tokens"] == [], f"done={r1['done']} tokens={r1['tokens']}")
    check("1c. the next request reads from token 0", r2["resume"] == 0, f"resume={r2['resume']}")
    check("1d. flagged at its 4th window after 3 tokens", r2["done"] is not None and r2["done"][4] == "nonfinite"
          and len(r2["tokens"]) == 3, f"done={r2['done']} tokens={r2['tokens']}")
    check("1e. the request after it reads from 0 again and finishes", r3["resume"] == 0 and r3["done"] is not None
          and r3["done"][4] == "length" and len(r3["tokens"]) == N, f"resume={r3['resume']} done={r3['done']}")
    check("1f. the guard changes no token", r3["tokens"] == want and r2["tokens"] == want[:3],
          f"r3={r3['tokens']} r2={r2['tokens']} want={want}")

    # 2. the pageable fallback (the prompt path's tables, rows, hand-offs and embedding rows in heap memory)
    for name, extra in (("one GPU", ()), ("a two-stage split", ("--gpus", "0,1", "--layer-split", "2"))):
        pinned, _ = reply(a.engine, a.model, ids, N, extra)
        heap, _ = reply(a.engine, a.model, ids, N, extra, env={"STRATA_X_TEST_PAGEABLE": "1"})
        check(f"2. pageable buffers, {name}: the same reply", len(pinned) == N and heap == pinned, f"{heap} vs {pinned}")

    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}", flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
