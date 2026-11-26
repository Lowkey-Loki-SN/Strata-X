"""moex/tools/ernie55/profile_from_imatrix.py - an expert profile for strata-x (--expert-profile: "<layer> <expert> <count>"
lines, hottest first) from an ERN55IMX importance matrix's row counts: how often the reference model routed each expert on
the calibration text - the VRAM tier's first fill before a server has learned its own (expert_profile_save).

    python3 -I profile_from_imatrix.py ernie55-16x4096.imx OUT.txt
"""
import struct
import sys


def main() -> int:
    src, out = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        if f.read(8) != b"ERN55IMX":
            raise SystemExit(f"{src}: not an ERN55IMX file")
        ver, n_layer, n_expert, d_in, d_ff = struct.unpack("<5I", f.read(20))
        if ver != 1:
            raise SystemExit(f"{src}: version {ver}")
        rows = []
        skip = 8 * (d_in + d_ff)
        for il in range(n_layer):
            for e in range(n_expert):
                f.seek(skip, 1)
                (cnt,) = struct.unpack("<Q", f.read(8))
                if cnt:
                    rows.append((cnt, il, e))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    with open(out, "w") as f:
        f.write(f"# from {src}: the reference model's routing counts on its calibration chunks (hottest first)\n")
        for cnt, il, e in rows:
            f.write(f"{il} {e} {cnt}\n")
    print(f"{out}: {len(rows)} experts with a count, of {n_layer * n_expert}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
