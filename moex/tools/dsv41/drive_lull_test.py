"""moex/tools/dsv41/drive_lull_test.py - does a filler in the drive's lulls hold its slow mode off? (DSV41_PLAN §56 B2)

    python -I drive_lull_test.py --pack PACK_DIR --scratch DIR [--arms plain,fill:10:4096,plain,...] [--seconds 300]

The slow mode (§14, §18, §55): the drive rewrites the blocks it read most, starting in a lull of its own - §14's loads met
it with 2 s pauses and never with 0.4 s ones or none.  Each arm reads the pack's record files as the engine's stream does
(12 readers, 4 MiB O_DIRECT calls at random 4 MiB offsets) 8 s on / 2 s off, and counts the slow seconds - a second of
the load's under 150 MB/s, as §14 counted them.  Arms:
  plain              the load alone (§14's E2)
  fill:T:BYTES       the load, and a filler: one O_DIRECT read of BYTES from a 256 MiB random file whenever the drive's
                     /sys/block/<disk>/inflight has been 0 for T ms (polled every millisecond)
  episode:silent     the load until a slow second, then nothing but a probe every 2 s - 64 back-to-back 128 KB reads, one
                     in flight: healthy ~4,400 a second here, the slow mode ~500 (2026-10-27); a single read
                     cannot tell them apart (a read after a pause waits ~1 ms for the drive's power state, §44)
  episode:trickle    the same with one 128 KB reader nonstop (the guard's floor) instead: healthy again once it completes
                     >= 2,000 a second
Plain arms between the others show the rewrites a filler only put off.  Nothing is written but the filler file (once)
and the CSV of seconds; the server must be stopped (its reads would mix in): the tool refuses while strata-x runs.
"""
from __future__ import annotations

import argparse
import mmap
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

MiB = 1 << 20
CALL = 4 * MiB
READERS = 12
SLOW_MBS = 150.0


def disk_of(path: Path) -> str:
    """The whole disk under `path` (nvme0n1 for nvme0n1p1)."""
    st = os.stat(path)
    dev = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
    return dev.parent.name if (dev / "partition").exists() else dev.name


class Stat:
    def __init__(self, disk: str):
        self.stat = f"/sys/block/{disk}/stat"
        self.inflight = os.open(f"/sys/block/{disk}/inflight", os.O_RDONLY)

    def read(self):
        f = open(self.stat).read().split()
        return int(f[0]), int(f[2]), int(f[3])      # reads completed, sectors read, ms reading

    def busy(self) -> bool:
        r, w = os.pread(self.inflight, 64, 0).split()
        return int(r) + int(w) > 0


class Load:
    """READERS threads of CALL-byte O_DIRECT reads over the pack's record files while `on` is set."""

    def __init__(self, files: list[Path]):
        self.fds = [(os.open(f, os.O_RDONLY | os.O_DIRECT), os.path.getsize(f)) for f in files]
        self.on = threading.Event()
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=self.run, args=(i,), daemon=True) for i in range(READERS)]
        for t in self.threads:
            t.start()

    def run(self, i: int):
        rng = random.Random(1000 + i)
        buf = mmap.mmap(-1, CALL)
        while not self.stop.is_set():
            if not self.on.wait(0.05):
                continue
            fd, size = self.fds[rng.randrange(len(self.fds))]
            off = rng.randrange(max(1, size // CALL)) * CALL
            try:
                os.preadv(fd, [buf], off)
            except OSError:
                pass

    def close(self):
        self.stop.set()
        self.on.set()
        for t in self.threads:
            t.join(2)


class Filler:
    """One O_DIRECT read of `size` bytes at a random offset of `path` whenever the disk has had nothing in flight for
    `idle_ms` milliseconds (polled every millisecond)."""

    def __init__(self, st: Stat, path: Path, idle_ms: float, size: int):
        self.st, self.idle, self.size = st, idle_ms / 1000.0, size
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.blocks = os.path.getsize(path) // size
        self.reads = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        rng = random.Random(7)
        buf = mmap.mmap(-1, max(self.size, mmap.PAGESIZE))
        view = memoryview(buf)[: self.size]
        last_busy = time.monotonic()
        while not self.stop.is_set():
            now = time.monotonic()
            if self.st.busy():
                last_busy = now
            elif now - last_busy >= self.idle:
                os.preadv(self.fd, [view], rng.randrange(self.blocks) * self.size)
                self.reads += 1
                last_busy = time.monotonic()
            time.sleep(0.001)

    def close(self):
        self.stop.set()
        self.thread.join(2)


class Trickle:
    """One reader of 128 KB O_DIRECT calls nonstop over the pack (the drive guard's floor)."""

    def __init__(self, fds):
        self.fds, self.stop = fds, threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        buf = mmap.mmap(-1, 128 * 1024)
        rng = random.Random(99)
        while not self.stop.is_set():
            fd, size = self.fds[rng.randrange(len(self.fds))]
            os.preadv(fd, [buf], rng.randrange(max(1, size // (128 * 1024))) * 128 * 1024)

    def close(self):
        self.stop.set()
        self.thread.join(2)


def probe_rate(fds, n: int = 64) -> float:
    """n back-to-back 128 KB O_DIRECT reads, one in flight -> reads a second (the first, which may wake the drive, left out)."""
    buf = mmap.mmap(-1, 128 * 1024)
    rng = random.Random(5)
    t1 = None
    for i in range(n + 1):
        fd, size = fds[rng.randrange(len(fds))]
        os.preadv(fd, [buf], rng.randrange(max(1, size // (128 * 1024))) * 128 * 1024)
        if i == 0:
            t1 = time.monotonic()
    return n / max(1e-9, time.monotonic() - t1)


def filler_file(scratch: Path) -> Path:
    p = scratch / "drive-lull-filler.bin"
    if not p.exists() or p.stat().st_size != 256 * MiB:
        with open(p, "wb") as f:
            for _ in range(256):
                f.write(os.urandom(MiB))
            f.flush()
            os.fsync(f.fileno())
    return p


def run_arm(arm: str, seconds: int, load: Load, st: Stat, scratch: Path, log) -> dict:
    """One arm: per second the load's phase, MB/s, reads, ms a read; -> its slow seconds and episodes."""
    filler = None
    if arm.startswith("fill:"):
        _, t_ms, size = arm.split(":")
        filler = Filler(st, filler_file(scratch), float(t_ms), int(size))
    slow, runs, cur, on_mb, on_s = 0, [], 0, 0.0, 0
    prev, t0 = st.read(), time.monotonic()
    episode = arm.startswith("episode:")
    ep_start, ep_end, mode = None, None, "load"
    trickle = None
    for sec in range(seconds):
        phase_on = (sec % 10) < 8 if mode == "load" else False
        load.on.set() if phase_on else load.on.clear()
        probe = probe_rate(load.fds) if mode == "probe" and sec % 2 == 0 else None   # episode:silent, every 2 s
        time.sleep(max(0.0, t0 + sec + 1 - time.monotonic()))
        cur_s = st.read()
        d_ios, d_sec, d_ms = (cur_s[i] - prev[i] for i in range(3))
        prev = cur_s
        mbs = d_sec * 512 / 1e6
        ms_read = d_ms / d_ios if d_ios else 0.0
        is_slow = phase_on and mbs < SLOW_MBS
        if phase_on:
            on_mb += mbs
            on_s += 1
        if is_slow:
            slow += 1
            cur += 1
        elif phase_on and cur:
            runs.append(cur)
            cur = 0
        log.write(f"{arm},{sec},{int(phase_on)},{mode},{mbs:.1f},{d_ios},{ms_read:.2f},"
                  f"{'' if probe is None else f'{probe:.0f}'},{filler.reads if filler else 0}\n")
        log.flush()
        if episode:
            if mode == "load" and is_slow:                  # the episode began: stop the load
                ep_start, mode = sec, ("probe" if arm == "episode:silent" else "trickle")
                load.on.clear()
                if mode == "trickle":
                    trickle = Trickle(load.fds)
            elif mode == "probe" and probe is not None and probe >= 2000:   # healthy again (~4,400 healthy, ~500 slow)
                ep_end = sec
                break
            elif mode == "trickle" and d_ios >= 2000:          # healthy: >= 2,000 reads a second at one in flight
                ep_end = sec
                break
    if cur:
        runs.append(cur)
    load.on.clear()
    if filler:
        filler.close()
    if trickle:
        trickle.close()
    out = {"arm": arm, "slow_s": slow, "episodes": len(runs), "longest_s": max(runs) if runs else 0,
           "on_mbs": on_mb / on_s if on_s else 0.0, "filler_reads": filler.reads if filler else 0}
    if episode:
        out["episode_s"] = None if ep_start is None or ep_end is None else ep_end - ep_start
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True, type=Path, help="the pack directory (its experts/*.bin are read)")
    ap.add_argument("--scratch", required=True, type=Path, help="where the filler file and the CSV go (on the same drive)")
    ap.add_argument("--arms", default="plain,fill:10:4096,plain,fill:2:4096,plain,fill:10:131072,plain")
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--force", action="store_true", help="run even if strata-x runs (its reads mix in)")
    a = ap.parse_args()
    if not a.force and subprocess.run(["pgrep", "-x", r"strata-(x|glm)(\..*)?"], capture_output=True).returncode == 0:
        print("strata-x is running: stop the server first (or --force)", file=sys.stderr)
        return 2
    files = sorted((a.pack / "experts").glob("L*.bin"))
    if not files:
        print(f"no record files in {a.pack / 'experts'}", file=sys.stderr)
        return 2
    a.scratch.mkdir(parents=True, exist_ok=True)
    disk = disk_of(files[0])
    st, load = Stat(disk), Load(files)
    csv = a.scratch / f"drive_lull_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    results = []
    with open(csv, "w") as log:
        log.write("arm,second,load_on,mode,mb_s,reads,ms_a_read,probe_reads_s,filler_reads\n")
        for arm in [x for x in a.arms.split(",") if x]:
            print(time.strftime("%H:%M:%S"), f"arm {arm} ({a.seconds} s)", flush=True)
            r = run_arm(arm, a.seconds, load, st, a.scratch, log)
            results.append(r)
            print("   ", r, flush=True)
    load.close()
    print(f"seconds: {csv}")
    for r in results:
        print(f"  {r['arm']:18s} slow seconds {r['slow_s']:4d} in {r['episodes']} episode(s), longest {r['longest_s']} s; "
              f"the load {r['on_mbs']:.0f} MB/s; filler reads {r['filler_reads']}"
              + (f"; episode length {r.get('episode_s')} s" if 'episode_s' in r else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
