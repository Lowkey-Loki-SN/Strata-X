"""refresh_pack.py PACK [--engram | --engram-only] [--verify] - write a pack's expert records back where they are, with O_DIRECT, so the drive
holds them in fresh NAND blocks (DSV41_PLAN §14, DSV41_BACKLOG #59).

The engine reads the pack's records over and over (~150 TB in two days on a 2 TB drive).  A block read that often must be
rewritten by the drive to stay readable (read disturb), and this drive does that in the second-long lulls of a session -
for ~64 s it then serves host reads at ~490 a second (63 MB/s), the "slow mode" of the server's log.  Written back here in
one pass, with the server stopped, the records land in new blocks and the drive has nothing left to refresh until they have
been read that often again.  The bytes are the file's own (each 16 MiB piece read, then written to the same offset), so the
files do not change and no free space is needed; an LBA written again frees its old pages without a trim.

--engram also rewrites the engram tables (the pack's engram/ directory, 196 GB, read a 4 KB row at a time all over);
--engram-only rewrites only them.  --verify reads every piece back and compares it.
Refuses while a strata-x process runs.  Prints each file's time and the drive's write rate."""
import mmap
import os
import subprocess
import sys
import time
from pathlib import Path

PIECE = 16 << 20


def refresh_file(f, buf, chk, verify):
    """one file written back in place (its O_DIRECT part, then a partial last 4 KiB buffered); returns its size"""
    size = f.stat().st_size
    aligned = size - size % 4096   # O_DIRECT's part; the last partial 4 KiB (the engram tables have one) goes buffered
    t0 = time.time()
    fd = os.open(f, os.O_RDWR | os.O_DIRECT)
    try:
        off = 0
        while off < aligned:
            n = min(PIECE, aligned - off)
            v = memoryview(buf)[:n]
            if os.preadv(fd, [v], off) != n:
                raise SystemExit(f"refresh_pack: a short read in {f} at {off}")
            if os.pwritev(fd, [v], off) != n:
                raise SystemExit(f"refresh_pack: a short write in {f} at {off} (the piece's bytes are the file's own)")
            if verify:
                w = memoryview(chk)[:n]
                if os.preadv(fd, [w], off) != n or w != v:
                    raise SystemExit(f"refresh_pack: {f} at {off} reads back unlike what was written")
            off += n
        os.fsync(fd)
    finally:
        os.close(fd)
    if aligned < size:   # the tail through the page cache, written back with fsync
        fb = os.open(f, os.O_RDWR)
        try:
            tail = os.pread(fb, size - aligned, aligned)
            if len(tail) != size - aligned or os.pwrite(fb, tail, aligned) != len(tail):
                raise SystemExit(f"refresh_pack: the last {size - aligned} bytes of {f}")
            os.fsync(fb)
            if verify and os.pread(fb, len(tail), aligned) != tail:
                raise SystemExit(f"refresh_pack: the last {size - aligned} bytes of {f} read back unlike what was written")
        finally:
            os.close(fb)
    return size


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        raise SystemExit(__doc__)
    pack = Path(args[0])
    if subprocess.run(["pgrep", "-x", r"strata-(x|glm)(\..*)?"], capture_output=True).returncode == 0:
        raise SystemExit("refresh_pack: a strata-x process runs - stop the server first")
    files = [] if "--engram-only" in sys.argv else sorted((pack / "experts").glob("L*.bin"))
    if "--engram" in sys.argv or "--engram-only" in sys.argv:
        files += sorted((pack / "engram").resolve().glob("*.bin"))
    if not files:
        raise SystemExit(f"refresh_pack: no record files under {pack}")
    verify = "--verify" in sys.argv
    buf = mmap.mmap(-1, PIECE)         # page-aligned: O_DIRECT's buffer
    chk = mmap.mmap(-1, PIECE) if verify else None
    total = sum(f.stat().st_size for f in files)
    print(f"refresh_pack: {len(files)} files, {total / 1e9:.1f} GB{' (verified)' if verify else ''}", flush=True)
    done, t_all = 0, time.time()
    for f in files:
        t0 = time.time()
        size = refresh_file(f, buf, chk, verify)
        done += size
        dt = time.time() - t0
        print(f"  {f.name}: {size / 1e9:.2f} GB in {dt:.1f} s ({size / 1e9 / dt:.2f} GB/s); {100 * done / total:.0f}% of the pack",
              flush=True)
    print(f"refresh_pack: {done / 1e9:.1f} GB written back in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()
