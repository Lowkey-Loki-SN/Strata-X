"""nsys_uploads.py (DSV41_PLAN §64) SQLITE T0_S T1_S ROUTE_BYTES - inside [T0, T1] (seconds from the trace's start): each GPU's layers, cut at the
routing's readback (a device-to-host copy of ROUTE_BYTES = tokens x 6 x 4), with each layer's host-to-device copies (GB, the
copy engine's busy time), its kernels' busy time, and the gap from its last copy's end to the next routing's readback."""
import sqlite3, sys

db = sqlite3.connect(sys.argv[1])
a_s, b_s, rb = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
mc = db.execute("select start, end, deviceId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
kn = db.execute("select start, end, deviceId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
t0 = min(min(r[0] for r in mc), min(r[0] for r in kn))
A, B = t0 + a_s * 1e9, t0 + b_s * 1e9


def busy(iv, a, b):
    iv = sorted((max(x, a), min(y, b)) for x, y in iv if y > a and x < b)
    tot, cur = 0, None
    for x, y in iv:
        if cur and x <= cur[1]:
            cur[1] = max(cur[1], y)
        else:
            if cur: tot += cur[1] - cur[0]
            cur = [x, y]
    if cur: tot += cur[1] - cur[0]
    return tot / 1e9


for d in sorted({r[2] for r in mc}):
    rt = sorted(r[1] for r in mc if r[2] == d and r[4] == 2 and r[3] == rb and A <= r[0] <= B)
    up = [(r[0], r[1], r[3]) for r in mc if r[2] == d and r[4] == 1 and A <= r[0] <= B]
    ks = [(r[0], r[1]) for r in kn if r[2] == d and A <= r[0] <= B]
    if len(rt) < 2:
        continue
    print(f"GPU {d}: {len(rt)} routings")
    tot_gap = tot_up = tot_k = 0
    for i in range(len(rt) - 1):
        a, b = rt[i], rt[i + 1]
        u = [x for x in up if a <= x[0] < b]
        gb = sum(x[2] for x in u) / 1e9
        cb = busy([(x[0], x[1]) for x in u], a, b)
        kb = busy(ks, a, b)
        last = max((x[1] for x in u), default=a)
        gap = (b - last) / 1e9
        tot_gap += gap; tot_up += cb; tot_k += kb
        print(f"  layer {i:2d}: {(b - a) / 1e9 * 1e3:6.0f} ms, {gb:5.2f} GB up, copies busy {cb * 1e3:5.0f} ms, kernels {kb * 1e3:4.0f} ms, "
              f"last copy to next routing {gap * 1e3:5.0f} ms")
    print(f"  in all: copies busy {tot_up:.2f} s, kernels {tot_k:.2f} s, last copy to next routing {tot_gap:.2f} s")
