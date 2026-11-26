"""moex/tools/dsv41/verify_pack.py - check a converted pack against the release, independently of convert_pack.py.

    python verify_pack.py --fast DIR --index model.safetensors.index.json [--samples 3] [--all-crc]

Per converted shard (manifest.json):
  - expert records: `--samples` records per file (first, last, random) rebuilt from fresh range reads of the release
    on Hugging Face (test_fp4_repack.py's path) and compared byte for byte; with --all-crc, every record's crc32 is
    recomputed from the file (O_DIRECT-free, page cache dropped afterwards);
  - dense safetensors: every tensor's dtype/shape against the release header, and the bytes of up to 4 tensors
    against range reads;
  - engram files: size against the header, and 2 sampled 4 KiB spans against range reads.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import struct
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fp4_repack  # noqa: E402
import test_fp4_repack as T  # noqa: E402

RECORD = 18_800_640


def read_at(path: str, at: int, n: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(at)
        data = f.read(n)
        os.posix_fadvise(f.fileno(), at, n, os.POSIX_FADV_DONTNEED)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", required=True, type=pathlib.Path)
    ap.add_argument("--index", required=True, type=pathlib.Path)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--all-crc", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--local", action="store_true",
                    help="no network: every record's crc32, file sizes and headers only (implies --all-crc)")
    a = ap.parse_args()
    if a.local:
        a.all_crc = True
    index = json.loads(a.index.read_text())
    man = json.loads((a.fast / "manifest.json").read_text())
    rng = random.Random(a.seed)
    fails = []

    for shard, r in sorted(man["shards"].items()):
        base, hdr = (None, None) if a.local else T.header(shard)
        if r.get("experts"):
            ex = r["experts"]
            path, n = ex["path"], ex["records"]
            if os.path.getsize(path) != n * RECORD:
                fails.append(f"{path}: size")
            stem = pathlib.Path(path).stem
            prefix = f"layers.{int(stem[1:])}" if stem[0] == "L" else f"mtp.{int(stem[1:])}"
            picks = [] if a.local else sorted({0, n - 1, *rng.sample(range(n), min(n, max(0, a.samples - 2)))})
            for e in picks:
                p = {w: (T.tensor(index, f"{prefix}.ffn.experts.{e}.{w}.weight"),
                         T.tensor(index, f"{prefix}.ffn.experts.{e}.{w}.scale")) for w in ("w1", "w3", "w2")}
                want = fp4_repack.expert_record(*p["w1"], *p["w3"], *p["w2"])
                got = read_at(path, e * RECORD, RECORD)
                ok = got == want and zlib.crc32(got) == ex["crc32"][e]
                print(f"  {stem} record {e:3d}: {'same as the release' if ok else 'DIFFERENT'}")
                if not ok:
                    fails.append(f"{stem}:{e}")
            if a.all_crc:
                bad = [e for e in range(n) if zlib.crc32(read_at(path, e * RECORD, RECORD)) != ex["crc32"][e]]
                print(f"  {stem}: crc32 of all {n} records: {len(bad)} bad")
                if bad:
                    fails.append(f"{stem} crc {bad[:5]}")
        if r.get("dense"):
            dp = r["dense"]["path"]
            with open(dp, "rb") as f:
                nh = struct.unpack("<Q", f.read(8))[0]
                dh = json.loads(f.read(nh))
            dh.pop("__metadata__", None)
            end = 8 + nh + max((t["data_offsets"][1] for t in dh.values()), default=0)
            if os.path.getsize(dp) != end:
                fails.append(f"{dp}: size {os.path.getsize(dp):,} != {end:,}")
            if a.local:
                print(f"  {pathlib.Path(dp).name}: {len(dh)} tensors, size {'ok' if os.path.getsize(dp) == end else 'WRONG'}")
                continue
            for k, t in dh.items():
                if hdr[k]["dtype"] != t["dtype"] or hdr[k]["shape"] != t["shape"]:
                    fails.append(f"{dp}:{k} meta")
            for k in rng.sample(sorted(dh), min(4, len(dh))):
                s0, s1 = dh[k]["data_offsets"]
                n = min(s1 - s0, 32 << 20)
                got = read_at(dp, 8 + nh + s0, n)
                a0 = hdr[k]["data_offsets"][0]
                want = T.fetch(T.REPO + shard, base + a0, base + a0 + n)
                ok = got == want
                print(f"  {pathlib.Path(dp).name} {k}: {'same as the release' if ok else 'DIFFERENT'} ({n:,} B)")
                if not ok:
                    fails.append(f"{dp}:{k}")
            print(f"  {pathlib.Path(dp).name}: {len(dh)} tensors, dtypes and shapes "
                  f"{'match' if not any(f.startswith(dp) for f in fails) else 'DIFFER'}")
        for part, e in (r.get("engram") or {}).items():
            if os.path.getsize(e["path"]) != e["bytes"]:
                fails.append(f"{e['path']}: size")
            if a.local:
                continue
            k = next(k for k in hdr if k.endswith(f"engram.embed.{part}"))
            a0 = hdr[k]["data_offsets"][0]
            for at in (0, rng.randrange(0, e["bytes"] - 4096) // 4096 * 4096):
                ok = read_at(e["path"], at, 4096) == T.fetch(T.REPO + shard, base + a0 + at, base + a0 + at + 4096)
                print(f"  {pathlib.Path(e['path']).name} @ {at:,}: {'same as the release' if ok else 'DIFFERENT'}")
                if not ok:
                    fails.append(f"{e['path']}@{at}")
    print("PASS" if not fails else f"FAIL: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
