#!/usr/bin/env python3
"""moex/ref/compare_dumps.py - line up two dumps (llama_dump / glm_ref format) and report where they differ.

    compare_dumps.py <candidate_dir> <oracle_dir> [--tol 1e-3] [--steps 0-19] [--verbose]

Records are matched by (step, name, occurrence): a name that is recorded twice in one step (hc_mixes-<l> for
the attention and the FFN halves) is matched in order.  For every candidate record:

    rel = ||cand - oracle|| / ||oracle||       (over the shared extent; the oracle's indexer_score is padded)

and per name the worst step is reported, in the order the candidate recorded them, i.e. graph order.  The
first name whose worst rel exceeds --tol is where a semantic difference enters; everything after it inherits it.
I32 records (expert ids) must agree exactly, as sets.
"""
import argparse, collections, os, sys

import numpy as np


def load(d):
    idx = []
    with open(os.path.join(d, "index.tsv")) as f:
        for line in f:
            step, name, typ, n0, n1, n2, n3, off = line.rstrip("\n").split("\t")
            idx.append((int(step), name, typ, (int(n0), int(n1), int(n2), int(n3)), int(off)))
    data = np.memmap(os.path.join(d, "dump.bin"), dtype=np.uint8, mode="r")
    recs = collections.OrderedDict()
    occ = collections.Counter()
    for step, name, typ, ne, off in idx:
        key = (step, name)
        k = (step, name, occ[key])
        occ[key] += 1
        n = ne[0] * ne[1] * ne[2] * ne[3]
        dt = np.float32 if typ == "f32" else np.int32
        recs[k] = (typ, ne, np.frombuffer(data, dtype=dt, count=n, offset=off))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("oracle")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--steps", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    A, B = load(a.cand), load(a.oracle)
    steps = None
    if a.steps:
        lo, hi = a.steps.split("-") if "-" in a.steps else (a.steps, a.steps)
        steps = set(range(int(lo), int(hi) + 1))

    worst = collections.OrderedDict()   # name#occ -> (rel, step, detail)
    missing = collections.Counter()
    for (step, name, o), (typ, ne, x) in A.items():
        if steps is not None and step not in steps:
            continue
        key = f"{name}#{o}" if o else name
        if (step, name, o) not in B:
            missing[key] += 1
            continue
        _, ne_b, y = B[(step, name, o)]
        if typ == "i32":
            ok = sorted(x.tolist()) == sorted(y[: len(x)].tolist())
            rel = 0.0 if ok else 1.0
            detail = "" if ok else f"{x.tolist()} vs {y[:len(x)].tolist()}"
        else:
            n = min(len(x), len(y))
            if len(x) != len(y) and not name.startswith("indexer_score"):
                rel, detail = float("inf"), f"shape {ne} vs {ne_b}"
            else:
                xa, ya = x[:n].astype(np.float64), y[:n].astype(np.float64)
                if not np.all(np.isfinite(ya)) or not np.all(np.isfinite(xa)):
                    fin = np.isfinite(ya) & np.isfinite(xa)
                    same_inf = np.all((np.isfinite(ya) == np.isfinite(xa)))
                    xa, ya = xa[fin], ya[fin]
                    detail = "" if same_inf else "non-finite mismatch"
                else:
                    detail = ""
                den = np.linalg.norm(ya)
                rel = float(np.linalg.norm(xa - ya) / den) if den > 0 else float(np.linalg.norm(xa - ya))
                if detail:
                    rel = max(rel, 1.0)
                mad = float(np.max(np.abs(xa - ya))) if len(xa) else 0.0
                detail = (detail + f" maxabs {mad:.3g} |ref| {float(np.max(np.abs(ya))) if len(ya) else 0:.3g}").strip()
        if key not in worst or rel > worst[key][0]:
            worst[key] = (rel, step, detail)

    first_bad = None
    print(f"{'record':34s} {'worst rel':>10s} {'step':>5s}  detail")
    for key, (rel, step, detail) in worst.items():
        bad = rel > a.tol
        if bad and first_bad is None:
            first_bad = key
        if a.verbose or bad or key.startswith(("l_out", "logits", "result", "hc_head", "kda_out", "attn_out", "ffn_out")):
            print(f"{key:34s} {rel:10.3e} {step:5d}  {detail}{'   <-- over tol' if bad else ''}")
    if missing:
        print("not in the oracle:", ", ".join(f"{k} x{v}" for k, v in missing.items()))
    # logits: arg-max agreement per step
    agree = tot = 0
    for (step, name, o), (_, _, x) in A.items():
        if name != "logits" or (step, name, o) not in B:
            continue
        y = B[(step, name, o)][2]
        tot += 1
        agree += int(np.argmax(x) == np.argmax(y))
    if tot:
        print(f"logits arg-max agreement: {agree}/{tot}")
    print("FIRST OVER TOL:", first_bad if first_bad else "none")
    return 0 if first_bad is None else 1


if __name__ == "__main__":
    sys.exit(main())
