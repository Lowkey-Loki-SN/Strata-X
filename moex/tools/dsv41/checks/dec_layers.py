"""dec_layers.py SQLITE - the decode windows' kernels per GPU (route_publish_kernel marks a layer's routing hand-off, since #112;
doorbell_wait_kernel the wait for the host's rows): per layer 'pre' = previous wait's end -> this publish's start (busy = its
kernels' time), 'mid' = publish end -> wait start, 'wait'; and every kernel's time a window in the decode region (the first
publish to the last wait), the ones between a GPU's last layer and its next first layer ('tail': the head, the hand-off) apart."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select start, end, deviceId, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by deviceId, start").fetchall()
by = collections.defaultdict(list)
for s, e, d, n in rows:
    by[d].append((s, e, names.get(n, "?")))
for d, ks in sorted(by.items()):
    pubs = [i for i, k in enumerate(ks) if k[2] == "route_publish_kernel"]
    if not pubs:
        continue
    last_mmq = max((i for i, k in enumerate(ks) if k[2] in ("mul_mat_q", "copy16_group_kernel")), default=-1)   # (the prompt reads)
    pubs = [i for i in pubs if i > last_mmq]
    i0, i1 = pubs[0], max(i for i, k in enumerate(ks) if k[2] == "doorbell_wait_kernel")
    seg = ks[i0:i1 + 1]
    wins = len(pubs) / 20.0
    tot = collections.Counter()
    per = collections.Counter()
    prev_wait_end, pub, cur = None, None, []
    for s, e, n in seg:
        per[n] += e - s
        if n == "route_publish_kernel":
            if prev_wait_end is not None:
                tot["pre"] += s - prev_wait_end
                tot["pre_busy"] += sum(min(ke, s) - max(ks_, prev_wait_end) for ks_, ke, _ in cur if ke > prev_wait_end)
                tot["n_pre"] += 1
            pub = e
            cur = []
        elif n == "doorbell_wait_kernel":
            if pub is not None:
                tot["mid"] += s - pub
                tot["n_mid"] += 1
            tot["wait"] += e - s
            prev_wait_end = e
            cur = []
        else:
            cur.append((s, e, n))
    span = seg[-1][1] - seg[0][0]
    busy = sum(e - s for s, e, n in seg if n != "doorbell_wait_kernel")
    print(f"GPU {d}: {len(pubs)} layer hand-offs = {wins:.1f} windows; the region {span / 1e6:.0f} ms, its kernels (not the waits) "
          f"{busy / 1e6 / wins:.2f} ms a window")
    for k in ("pre", "pre_busy", "mid", "wait"):
        n = tot["n_pre"] if k.startswith("pre") else (tot["n_mid"] if k == "mid" else len(pubs))
        print(f"  {k:9s} {tot[k] / 1e3 / max(1, n):8.1f} us a layer")
    top = sorted(((v, n) for n, v in per.items() if n != "doorbell_wait_kernel"), reverse=True)[:22]
    print("  a window's kernels, us: " + ", ".join(f"{n.replace('_kernel', '')} {v / 1e3 / wins:.0f}" for v, n in top))
