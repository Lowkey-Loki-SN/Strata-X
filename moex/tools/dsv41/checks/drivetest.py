"""drivetest.py SECONDS THREADS ON OFF [CHUNK_KB [KEEP_MS]] - read the pack's expert records with O_DIRECT, as the engine's stream does,
without the engine: THREADS threads each read whole random records (15.2 / 11.6 MB) of experts/L*.bin; with OFF > 0 the
load runs ON seconds, then stops for OFF seconds (the drive idle: its power states), and again, for SECONDS in all.
CHUNK_KB > 0: each record read in CHUNK_KB pieces one after another (fewer requests in flight).  KEEP_MS > 0: while the
load is off, one 4 KB read every KEEP_MS ms (a keep-alive: the drive never idles long enough for its power states).
DRIVETEST_FILES=a:b:...: those files instead of the pack, in pieces of DRIVETEST_PIECE bytes (15,208,448) at their
multiples.  Reads only.  DSV41_PLAN §14 has what it showed (the drive's slow mode, DSV41_BACKLOG #59).

Prints one line a second: MB/s, records done, and SLOW when the second delivered under 150 MB/s while the load was on;
the summary counts the slow seconds and their runs."""
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

secs, nth, on, off = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
chunk = int(sys.argv[5]) * 1024 if len(sys.argv) > 5 else 0
keep_ms = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
P = _os.path.join(_MODELS, 'dsv41-1017/ud-iq3xxs')
recs = []
for line in open(f"{P}/experts.tsv"):
    il, _t, size, rel = line.split()
    recs.append((int(il), int(size), rel))
last = {}
for il, size, rel in recs:   # experts.tsv: the last line of a layer counts
    last[il] = (size, rel)
files = []
if os.environ.get("DRIVETEST_FILES"):   # other files instead of the pack: pieces of a record's size at aligned offsets
    for path in os.environ["DRIVETEST_FILES"].split(":"):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        piece = int(os.environ.get("DRIVETEST_PIECE", "15208448"))
        files.append((fd, piece, os.fstat(fd).st_size // piece))
else:
    for il, (size, rel) in sorted(last.items()):
        fd = os.open(f"{P}/{rel}", os.O_RDONLY | os.O_DIRECT)
        n = os.fstat(fd).st_size // size
        files.append((fd, size, n))
done = [0, 0]   # bytes, records
lock = threading.Lock()
load_on = threading.Event()
stop = threading.Event()


def worker(seed):
    r = random.Random(seed)
    buf = mmap.mmap(-1, 16 << 20)   # page-aligned: O_DIRECT's buffer
    view = memoryview(buf)
    while not stop.is_set():
        if not load_on.wait(0.05):
            continue
        fd, size, n = files[r.randrange(len(files))]
        off0 = r.randrange(n) * size
        if chunk:
            got = 0
            while got < size and not stop.is_set():
                ln = min(chunk, size - got)
                got += os.preadv(fd, [view[:ln]], off0 + got)
        else:
            got = os.preadv(fd, [view[:size]], off0)
        with lock:
            done[0] += got
            done[1] += 1


def keeper():   # the keep-alive: a 4 KB read every keep_ms while the load is off
    buf = mmap.mmap(-1, 4096)
    fd, size, n = files[0]
    k = 0
    while not stop.is_set():
        time.sleep(keep_ms / 1000.0)
        if not load_on.is_set():
            os.preadv(fd, [memoryview(buf)], (k % (n * size // 4096)) * 4096)
            k += 1


th = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(nth)]
for t in th:
    t.start()
if keep_ms > 0:
    threading.Thread(target=keeper, daemon=True).start()
t0 = time.time()
slow, runs, cur, prev_b = 0, [], 0, 0
next_tick = t0 + 1
phase_end = t0 + on
load_on.set()
loaded = True   # the second had load in it
while time.time() - t0 < secs:
    while time.time() < next_tick:   # the phases switch at 20 ms steps (gaps under a second)
        time.sleep(0.02)
        now = time.time()
        if off > 0 and now >= phase_end:   # switch between load and idle
            if load_on.is_set():
                load_on.clear()
                phase_end = now + off
            else:
                load_on.set()
                phase_end = now + on
                loaded = True
    with lock:
        b, nrec = done
    mb = (b - prev_b) / 1e6
    prev_b = b
    busy = load_on.is_set() and loaded   # a second with an idle gap in it counts as load only if the gap was short
    loaded = load_on.is_set() or off < 1
    is_slow = busy and mb < 150 * (1 if off >= 1 else on / (on + off))
    slow += is_slow
    if is_slow:
        cur += 1
    elif cur:
        runs.append(cur)
        cur = 0
    print(f"{time.strftime('%H:%M:%S')} {mb:7.0f} MB/s {nrec:6d} records {'load' if busy else 'idle'}{' SLOW' if is_slow else ''}",
          flush=True)
    next_tick += 1
stop.set()
load_on.set()
if cur:
    runs.append(cur)
print(f"summary: {secs:.0f} s, {nth} threads, on {on} s / off {off} s, chunk {chunk // 1024 or 'whole'}, keep-alive {keep_ms or 'off'}: {done[0] / 1e9:.1f} GB, "
      f"{slow} slow seconds in {len(runs)} runs {runs}", flush=True)
