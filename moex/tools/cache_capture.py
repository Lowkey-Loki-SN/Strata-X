#!/usr/bin/env python3
"""moex/tools/cache_capture.py - how much of a model's routing a Strata-style GPU expert cache would catch.

Input: a per-prompt routing capture (npz with counts_prefill / counts_decode [prompts, moe_layers, experts], the format
of ~/AI/exl3-profiles/util/moe_profile_build.py).  The cache is ranked over (layer, expert) PAIRS globally, as Strata's
expert-profile placement is (its log: "the caches hold 8618 of 24576 profiled pairs").

    static   one global ranking from the OTHER prompts' decode routing (leave-one-out: no prompt grades itself)
    adaptive the prompt's own prefill routing ranks the pairs (ties: the global ranking) - a cache that refills from
             the current conversation, the effect Strata's adaptive cache exploits.  A floor, not a ceiling: on
             Qwen3.8-Flash-Next this proxy gives 77% at 34% of the pairs where Strata measured 86-90% live.

Budgets: --frac 0.15 0.225 0.34   or   --gib 24.3 --bpw 3.06 4.5 --model moex/models/<type>.json
    python moex/tools/cache_capture.py <capture.npz> [--frac ...] [--gib G --bpw B ... --model M] [--json out.json]
"""
import argparse, json
import numpy as np

def capture(npz, fracs):
    z = np.load(npz)
    pre, dec = z["counts_prefill"].astype(np.float64), z["counts_decode"].astype(np.float64)
    P, L, E = dec.shape
    fp, fd = pre.reshape(P, -1), dec.reshape(P, -1)
    tot = fd.sum(0)
    out = {f: {"static": [], "adaptive": []} for f in fracs}
    for p in range(P):
        loo = tot - fd[p]
        o_s = np.argsort(-loo, kind="stable")
        o_a = np.argsort(-(fp[p] * 1e6 + loo / (loo.max() + 1)), kind="stable")
        d = fd[p]; dt = d.sum()
        for f in fracs:
            b = int(round(f * L * E))
            out[f]["static"].append(d[o_s[:b]].sum() / dt)
            out[f]["adaptive"].append(d[o_a[:b]].sum() / dt)
    return (P, L, E), {f: {k: float(np.mean(v)) for k, v in r.items()} for f, r in out.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("--frac", type=float, nargs="*", default=[])
    ap.add_argument("--gib", type=float); ap.add_argument("--bpw", type=float, nargs="*", default=[])
    ap.add_argument("--model"); ap.add_argument("--json")
    a = ap.parse_args()
    fracs = list(a.frac); labels = {f: f"{f:.1%} of pairs" for f in fracs}
    if a.gib and a.bpw and a.model:
        g = json.load(open(a.model))["derived"]
        for b in a.bpw:
            f = min(1.0, a.gib * 2**30 / (g["expert_params"] * b / 8) / g["experts_total"])
            fracs.append(f); labels[f] = f"{a.gib:g} GiB @ {b:g} bpw = {f:.1%} of pairs"
    shape, res = capture(a.npz, fracs)
    print(f"{a.npz}: {shape[0]} prompts x {shape[1]} MoE layers x {shape[2]} experts")
    for f in fracs:
        r = res[f]
        print(f"  {labels[f]:34s} static {r['static']:6.1%}   adaptive {r['adaptive']:6.1%}")
    if a.json:
        json.dump({"shape": shape, "capture": {labels[f]: {"frac": f, **res[f]} for f in fracs}}, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
