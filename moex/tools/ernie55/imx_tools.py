"""moex/tools/ernie55/imx_tools.py - ERNIE-5.5's importance matrices: merge several into one, or compare two.

    python3 -I moex/tools/ernie55/imx_tools.py merge OUT.imx A.imx[:WEIGHT] B.imx[:WEIGHT] ...
    python3 -I moex/tools/ernie55/imx_tools.py compare A.imx B.imx [MIN_ROWS]

Both read ERN55IMX (the reference model's, ref/imatrix.py: f64 sums, u64 rows; convert_gguf_ernie55 --imatrix reads it) and
DSV41IMX (strata-x --imatrix over --tokens-file ids: f32 sums, u32 rows, the experts' input width). A matrix holds, per layer
and routed expert, the sums of squared inputs per column of gate/up and of down and the rows routed, so `merge` writes the
matrix of the joined calibration sets (WEIGHT scales a file's sums and rows; 1 when absent) as ERN55IMX. `compare` prints, at
a few layers, the per-expert row counts' correlation and, over the experts both saw at least MIN_ROWS times (100), the cosine
and relative L2 difference of the per-column mean squares.

Measured with it (2026-11-27, moex/docs/ERNIE55_PLAN.md section 10): the engine's matrix over the reference's 16 calibration
chunks matched the reference's (gate/up cosine 0.9993-1.0000, down 0.993-0.998); a 64-chunk matrix of the same files moved
down's column importance by 35-60% (relative L2) against the 16-chunk one - and the quantization error by 0.1-2%.
"""
import struct
import sys

import numpy as np


def load(p):
    b = open(p, "rb").read()
    ver, L, E, di, df = struct.unpack("<5I", b[8:28])
    if ver != 1:
        raise SystemExit(f"{p}: version {ver}")
    if b[:8] == b"ERN55IMX":
        rec = np.dtype([("gu", "<f8", di), ("dn", "<f8", df), ("cnt", "<u8")])
    elif b[:8] == b"DSV41IMX":
        rec = np.dtype([("gu", "<f4", di), ("dn", "<f4", df), ("cnt", "<u4")])
    else:
        raise SystemExit(f"{p}: not an importance matrix")
    if len(b) != 28 + rec.itemsize * L * E:
        raise SystemExit(f"{p}: {len(b)} bytes, not the header's geometry")
    return (L, E, di, df), np.frombuffer(b, dtype=rec, offset=28, count=L * E)


def merge(out, specs):
    geo, acc = None, None
    for spec in specs:
        p, w = (spec.rsplit(":", 1)[0], float(spec.rsplit(":", 1)[1])) if ":" in spec else (spec, 1.0)
        g, a = load(p)
        if geo is None:
            geo = g
            acc = np.zeros(g[0] * g[1], dtype=np.dtype([("gu", "<f8", g[2]), ("dn", "<f8", g[3]), ("cnt", "<u8")]))
        if g != geo:
            raise SystemExit(f"{p}: geometry {g}, not {geo}")
        acc["gu"] += w * a["gu"].astype(np.float64)
        acc["dn"] += w * a["dn"].astype(np.float64)
        acc["cnt"] += np.round(w * a["cnt"].astype(np.float64)).astype(np.uint64)
        print(f"{p} (x{w:g}): rows {int(a['cnt'].astype(np.int64).sum())}, experts seen {int((a['cnt'] > 0).sum())} of {g[0] * g[1]}")
    with open(out + ".tmp", "wb") as f:
        f.write(b"ERN55IMX" + struct.pack("<5I", 1, *geo))
        f.write(acc.tobytes())
    import os
    os.replace(out + ".tmp", out)
    print(f"{out}: rows {int(acc['cnt'].sum())}, experts seen {int((acc['cnt'] > 0).sum())} of {geo[0] * geo[1]}")


def compare(pa, pb, min_rows=100.0):
    ga, a = load(pa)
    gb, b = load(pb)
    if ga != gb:
        raise SystemExit(f"geometries differ: {ga} {gb}")
    L, E, di, df = ga
    ca, cb = a["cnt"].astype(np.float64).reshape(L, E), b["cnt"].astype(np.float64).reshape(L, E)
    print(f"rows: A {ca.sum():.0f}, B {cb.sum():.0f}")
    for il in range(L):
        if ca[il].sum() == 0 and cb[il].sum() == 0:
            continue
        if il not in (1, 2, 7, 12, 21, 22, 32, 41, L - 1):
            continue
        m = (ca[il] >= min_rows) & (cb[il] >= min_rows)
        out = []
        for nm, w in (("gu", di), ("dn", df)):
            A = a[nm].astype(np.float64).reshape(L, E, w)[il][m] / ca[il][m][:, None]
            B = b[nm].astype(np.float64).reshape(L, E, w)[il][m] / cb[il][m][:, None]
            cos = (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1))
            rel = np.sqrt(((A - B) ** 2).sum(1) / (A * A).sum(1))
            out.append(f"{nm}: cos median {np.median(cos):.5f} min {cos.min():.4f}, rel L2 median {np.median(rel):.4f}")
        print(f"layer {il:2d}: rows corr {np.corrcoef(ca[il], cb[il])[0, 1]:.4f}, experts compared {int(m.sum())}; " + "; ".join(out))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) in (4, 5) and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) == 5 else 100.0)
    else:
        raise SystemExit(__doc__)
