"""eg_fetch_bench.py TOKENS MODE [SCALES] [RECORDS] - the engram fetch of a prompt chunk replayed without the engine, as
eg_rows' mapped path does it: the tables mapped MADV_RANDOM, each row's pages asked for (MADV_WILLNEED) and then read
(touched), for TOKENS random tokens x 2 layers x 24 rows (uniform random rows: the drive's cost per page, not the n-gram
statistics).  MODE "all": every row asked for first, then every row read (today's code); "window:W": W rows asked for
ahead of the ones being read.  SCALES 1 (default) reads the scale pages too, 0 the weight pages only.  RECORDS N > 0:
N threads read the pack's records with O_DIRECT meanwhile (the record stream, as drivetest.py).  The tables' pages are
dropped from the page cache first (posix_fadvise DONTNEED), and a new seed each run.  Reads only.

Prints the fetch's time and, from the drive's counters every 250 ms (the drive guard's sampling), the mean time per
request of each sample: median, 99th percentile and max."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import mmap
import os
import random
import sys
import threading
import time
from pathlib import Path

ntok, mode = int(sys.argv[1]), sys.argv[2]
scales = len(sys.argv) <= 3 or sys.argv[3] != "0"
nrec = int(sys.argv[4]) if len(sys.argv) > 4 else 0
M = Path(_os.path.join(_MODELS, 'dsv41-1017'))
EG = M / "native/engram"
tables = [(EG / "L01.weight.bin", EG / "L01.scale.bin"), (EG / "L14.weight.bin", EG / "L14.scale.bin")]
D, SB, C = 256, 8, 24   # the engine's log: 24 rows of 264 bytes (256 weight bytes + 8 scale bytes) a token, each layer
STAT = "/sys/block/nvme0n1/stat"


def stat():
    v = open(STAT).read().split()
    return int(v[0]), int(v[3]), int(v[9])   # reads completed, ms spent reading, ms busy


maps = []
for w, s in tables:
    for f in (w, s):
        fd = os.open(f, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)   # cold: as after a reboot
        m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        m.madvise(mmap.MADV_RANDOM)
        maps.append(m)
        os.close(fd)
rows_w = [maps[0].size() // D, maps[2].size() // D]
rnd = random.Random(time.time_ns())
plan = []   # (map index, byte offset) of each read, a row's weight page then its scale page
for _ in range(ntok):
    for j in range(2):
        for _ in range(C):
            r = rnd.randrange(rows_w[j])
            plan.append((2 * j, r * D))
            if scales:
                plan.append((2 * j + 1, r * SB))

stop = threading.Event()
samples = []


def sampler():
    a = stat()
    while not stop.is_set():
        time.sleep(0.25)
        b = stat()
        if b[0] > a[0]:
            samples.append(((b[1] - a[1]) / (b[0] - a[0]), (b[0] - a[0]) / 0.25))
        a = b


def records():   # the record stream: whole random records, O_DIRECT (drivetest.py's way)
    files = sorted((M / "q3q2u/experts").glob("L*.bin"))
    piece = 4 << 20
    buf = mmap.mmap(-1, piece)
    rr = random.Random(os.getpid() ^ threading.get_ident())
    while not stop.is_set():
        f = rr.choice(files)
        fd = os.open(f, os.O_RDONLY | os.O_DIRECT)
        try:
            size = os.fstat(fd).st_size
            off = rr.randrange(size // piece) * piece
            os.preadv(fd, [buf], off)
        finally:
            os.close(fd)


th = [threading.Thread(target=sampler, daemon=True)] + [threading.Thread(target=records, daemon=True) for _ in range(nrec)]
for t in th:
    t.start()
time.sleep(1.0)   # the record stream under way
t0 = time.time()
if mode == "all":
    for mi, off in plan:
        maps[mi].madvise(mmap.MADV_WILLNEED, off - off % 4096, 4096)
    x = 0
    for mi, off in plan:
        x ^= maps[mi][off]
else:
    W = int(mode.split(":")[1])
    x = 0
    for i, (mi, off) in enumerate(plan):
        if i == 0:
            for mj, oj in plan[:W]:
                maps[mj].madvise(mmap.MADV_WILLNEED, oj - oj % 4096, 4096)
        elif i + W - 1 < len(plan):
            mj, oj = plan[i + W - 1]
            maps[mj].madvise(mmap.MADV_WILLNEED, oj - oj % 4096, 4096)
        x ^= maps[mi][off]
dt = time.time() - t0
time.sleep(0.5)
stop.set()
lat = sorted(s[0] for s in samples)
rate = sorted(s[1] for s in samples)
q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
print(f"{ntok} tokens, {len(plan)} page reads ({'weights + scales' if scales else 'weights only'}), {mode}, {nrec} record "
      f"threads: the fetch {dt:.2f} s ({len(plan) / dt:,.0f} pages/s); the drive's 250 ms samples ({len(lat)}): ms a request median "
      f"{q(lat, 0.5):.1f}, p99 {q(lat, 0.99):.1f}, max {max(lat) if lat else float('nan'):.1f}; completions/s median {q(rate, 0.5):,.0f}, "
      f"min {min(rate) if rate else float('nan'):,.0f}", flush=True)
