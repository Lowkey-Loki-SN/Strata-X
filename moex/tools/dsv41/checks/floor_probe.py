"""floor_probe.py [SECONDS] - the drive guard's floor measured on the drive in its normal state, without the engine: random
reads at one and at two in flight, SECONDS each (default 5): 4 KB O_DIRECT reads of the engram weight tables, 128 KB
O_DIRECT reads of the pack's records, and the mapped path at a window of one (the tables mapped MADV_RANDOM, one page asked
for with MADV_WILLNEED, then read - eg_rows' way, one page in flight).  Each read timed; meanwhile the drive's in-flight
count (/sys/block/nvme0n1/inflight) sampled every 2 ms by a separate process (in this one the mapped reads hold the
interpreter's lock and starve a sampler thread).  Prints each arm's latency (median, 90th and 99th percentile, max)
and the in-flight samples (the counter's mean and max): a lone read's normal time, for the guard's release threshold, and
how many requests a floor of one read puts on the drive.  At most two reads in flight: no load another program would see.
Reads only."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import mmap
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
M = Path(_os.path.join(_MODELS, 'dsv41-1017'))
EGW = sorted((M / "native/engram").glob("*.weight.bin"))
REC = sorted((M / "q3q2u/experts").glob("L*.bin"))
INFL = "/sys/block/nvme0n1/inflight"


SAMPLER = ("import sys, time\nv = []\nt = time.time()\nwhile time.time() - t < float(sys.argv[1]):\n"
           "    v.append(int(open('/sys/block/nvme0n1/inflight').read().split()[0]))\n    time.sleep(0.002)\n"
           "print(' '.join(map(str, v)))")


def direct_reader(files, size, stop, lat, seed):
    fds = [os.open(f, os.O_RDONLY | os.O_DIRECT) for f in files]
    n = [os.fstat(fd).st_size // size for fd in fds]
    buf = mmap.mmap(-1, size)
    rr = random.Random(seed)
    while not stop.is_set():
        i = rr.randrange(len(fds))
        off = rr.randrange(n[i]) * size
        t = time.perf_counter()
        os.preadv(fds[i], [buf], off)
        lat.append((time.perf_counter() - t) * 1e3)
    for fd in fds:
        os.close(fd)


def mapped_reader(stop, lat, seed):
    maps = []
    for f in EGW:
        fd = os.open(f, os.O_RDONLY)
        m = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        m.madvise(mmap.MADV_RANDOM)
        maps.append(m)
        os.close(fd)
    rr = random.Random(seed)
    x = 0
    while not stop.is_set():
        m = maps[rr.randrange(len(maps))]
        off = rr.randrange(m.size() // 4096) * 4096
        t = time.perf_counter()
        m.madvise(mmap.MADV_WILLNEED, off, 4096)
        x ^= m[off]
        lat.append((time.perf_counter() - t) * 1e3)


def arm(name, target, args_of, threads):
    stop, lat = threading.Event(), []
    sampler = subprocess.Popen([sys.executable, "-I", "-c", SAMPLER, str(secs - 0.4)], stdout=subprocess.PIPE, text=True)
    th = [threading.Thread(target=target, args=args_of(stop, lat, k), daemon=True) for k in range(threads)]
    for t in th:
        t.start()
    time.sleep(secs)
    stop.set()
    for t in th:
        t.join()
    infl = [int(x) for x in sampler.communicate()[0].split()]
    v = sorted(lat)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    busy = [x for x in infl if x > 0]
    print(f"{name:34s} {threads} in flight: {len(v):7,} reads, ms median {q(0.5):6.3f}, p90 {q(0.9):6.3f}, p99 {q(0.99):6.3f}, "
          f"max {v[-1]:7.2f}; the drive's in-flight count: mean {sum(infl) / len(infl):5.2f}, max {max(infl)} "
          f"({len(busy)} of {len(infl)} samples above 0)", flush=True)


seed = time.time_ns()
for threads in (1, 2):
    arm("4 KB O_DIRECT, the engram tables", direct_reader, lambda s, l, k: (EGW, 4096, s, l, seed + k), threads)
    arm("128 KB O_DIRECT, the pack's records", direct_reader, lambda s, l, k: (REC, 128 << 10, s, l, seed + 99 + k), threads)
arm("mapped, a window of one page", mapped_reader, lambda s, l, k: (s, l, seed + 7), 1)
