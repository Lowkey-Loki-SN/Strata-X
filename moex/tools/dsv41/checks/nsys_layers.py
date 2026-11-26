"""nsys_layers.py SQLITE - DSV41_BACKLOG #55: each decode layer split at its routing hand-off (doorbell_publish) and at the
host's rows (doorbell_wait), per GPU: 'pre' = from the previous layer's wait end to this layer's publish (the previous combine,
the mixers, the attention, the router), 'mid' = publish end to wait start (the VRAM experts), 'wait' = the wait kernel (the
GPU idles for the host's pool).  Busy = the kernels' summed time in 'pre' (the rest are gaps).  Per token and per kernel."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select start, end, deviceId, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by deviceId, start").fetchall()
by_dev = collections.defaultdict(list)
for s, e, d, n in rows:
    by_dev[d].append((s, e, names.get(n, "?")))
tot = collections.Counter()
pre_k = collections.Counter()
layers = collections.Counter()
for d, ks in by_dev.items():
    prev_wait_end, i = None, 0
    seg = []   # kernels since the previous wait's end
    pub = None
    for s, e, n in ks:
        if n == "doorbell_publish_kernel":
            if prev_wait_end is not None:
                tot[(d, "pre")] += s - prev_wait_end
                tot[(d, "pre_busy")] += sum(min(ke, s) - max(ks_, prev_wait_end) for ks_, ke, kn in seg if ke > prev_wait_end)
                for ks_, ke, kn in seg:
                    pre_k[(d, kn)] += ke - ks_
                layers[d] += 1
            pub = (s, e)
            seg = []
        elif n == "doorbell_wait_kernel":
            if pub is not None and prev_wait_end is not None:
                tot[(d, "mid")] += s - pub[1]
            tot[(d, "wait")] += e - s
            prev_wait_end = e
            seg = []
        else:
            seg.append((s, e, n))
for d in sorted(by_dev):
    L = max(1, layers[d])
    toks = L / 20.0
    print(f"GPU {d}: {L} layers measured ({toks:.0f} tokens x 20)")
    for k in ("pre", "pre_busy", "mid", "wait"):
        print(f"  {k:9s} {tot[(d, k)] / 1e6 / toks:7.2f} ms a token  ({tot[(d, k)] / 1e3 / L:7.1f} us a layer)")
    top = sorted(((v, kn) for (dd, kn), v in pre_k.items() if dd == d), reverse=True)[:12]
    print("  'pre' kernels, us a layer:", ", ".join(f"{kn.replace('_kernel', '')} {v / 1e3 / L:.1f}" for v, kn in top))
