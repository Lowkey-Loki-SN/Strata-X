#!/usr/bin/env python3
"""moex/tools/nsys_decode.py - where a decode step's time goes, from an nsys sqlite export.

    nsys_decode.py <report.sqlite> --tokens N [--window-s S]

The decode phase is the tail of the run: the last S seconds of GPU activity (default: enough for N tokens at the run's
own rate: the time from the last kernel back to where the per-token kernel rate becomes steady).  Prints, per GPU: the
kernels per token, the busy fraction (union of kernel intervals), the gaps; and per CUDA runtime API call on the
launching threads: calls per token and microseconds per token - the host-side cost of launching, which is what CUDA
graphs and a doorbell remove.
"""
import argparse, collections, sqlite3, sys


def union_busy(ivs):
    ivs = sorted(ivs)
    if not ivs:
        return 0
    busy, (ms, me) = 0, ivs[0]
    for a, b in ivs[1:]:
        if a <= me:
            me = max(me, b)
        else:
            busy += me - ms
            ms, me = a, b
    return busy + (me - ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--window-s", type=float, default=0.0)
    a = ap.parse_args()
    cur = sqlite3.connect(a.db).cursor()
    names = {r[0]: r[1] for r in cur.execute("select id, value from StringIds")}
    kern = cur.execute("select deviceId, start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    t_end = max(k[2] for k in kern)
    if a.window_s > 0:
        t0 = t_end - int(a.window_s * 1e9)
    else:   # the steady tail: walk back in 100 ms bins until the kernel rate falls to a quarter of the final second's
        bins = collections.Counter(int((t_end - k[1]) / 1e8) for k in kern)
        ref = sum(bins[i] for i in range(0, 10)) / 10.0
        n = 10
        while bins.get(n, 0) > ref / 4 and n < 600:
            n += 1
        t0 = t_end - n * int(1e8)
    win = [k for k in kern if k[1] >= t0]
    span = (t_end - t0) / 1e9
    print("decode window: last %.2f s, %d kernels, %d tokens -> %.1f ms per token" % (span, len(win), a.tokens, 1e3 * span / a.tokens))
    for d in sorted({k[0] for k in win}):
        ks = [k for k in win if k[0] == d]
        busy = union_busy([(k[1], k[2]) for k in ks])
        tiny = sum(1 for k in ks if k[2] - k[1] < 5000)
        print("  GPU %d: %d kernels (%.0f per token), busy %.1f ms per token (%.0f%% of the window), %d kernels < 5 us (%.0f%%)"
              % (d, len(ks), len(ks) / a.tokens, 1e3 * busy / 1e9 / a.tokens, 100.0 * busy / (t_end - t0), tiny, 100.0 * tiny / len(ks)))
    by = collections.defaultdict(lambda: [0, 0.0])
    for nid, s, e, tid in cur.execute("select nameId, start, end, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start >= %d" % t0):
        nm = names.get(nid, str(nid)).split("_v")[0]
        x = by[(nm, tid)]
        x[0] += 1
        x[1] += e - s
    tot_by_tid = collections.Counter()
    for (nm, tid), (c, tt) in by.items():
        tot_by_tid[tid] += tt
    print("runtime API per thread (the busiest first):")
    for tid, tt in tot_by_tid.most_common(3):
        print("  thread %d: %.1f ms per token inside CUDA calls" % (tid, tt / 1e6 / a.tokens))
        rows = sorted(((nm, v) for (nm, t), v in by.items() if t == tid), key=lambda r: -r[1][1])[:8]
        for nm, (c, t) in rows:
            print("     %-30s %6.1f calls/token %7.1f us/token  (%.2f us per call)" % (nm, c / a.tokens, t / 1e3 / a.tokens, t / 1e3 / c))


if __name__ == "__main__":
    sys.exit(main())
