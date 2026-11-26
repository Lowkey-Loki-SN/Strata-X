"""serve_cold_start.py OUT_DIR - DSV41_PLAN §18: the user's failing scenario of 2026-10-21 (a reboot, the server started, an
84,524-token request read from 0) through the whole server, as near as a running box allows: the page cache emptied of
the engram tables and the dense weights first (a reboot's cold start; the records are read with O_DIRECT anyway and the
SSD cache starts empty with every engine), strata-dsv41.json as it stands (port 1235 and its own log in OUT_DIR, so the
live server's log is untouched), one request of ~84.5K tokens of this repo's own docs (no conversation of the user's),
streamed.  Meanwhile, each in its own process: another program's read - one 4 KB O_DIRECT read of a file the engine never
reads (GLM-5.3's GGUF) every 100 ms, timed - and the drive's in-flight count every 2 ms.

Pass: a progress line in the server's window at least every 30 s while the prompt is read, the first token arrives (no
silence limit), and another program's reads: none waits over 2.5 s (the requests queued when a slow mode starts drain at
its rate: up to 541 ms in serve_guard_episode.py's run) and, while the guard is at its floor, they wait little for the
engine's: their median beside the engine's reads within 3 ms of their median with nothing else on the drive, in the same
floor periods (DSV41_PLAN §29: in a deeper episode the drive served the probe's lone reads in up to 100 ms, p99 ~80 ms, so
the earlier "p99 under 10 ms" measured the drive, not the guard; with fewer than 10 lone reads: their p90 under 10 ms)."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
R = _REPO
M = _os.path.join(_MODELS, 'dsv41-1017')
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
cfg.update(port=1235, log=str(out / "engine.log"), open_browser=False)
(out / "config.json").write_text(json.dumps(cfg, indent=1))
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
text = "".join(subprocess.run(["git", "-C", R, "show", f"645844e9:{f}"], capture_output=True, text=True, check=True).stdout
               for f in ("docs/DETAILS.md", "moex/docs/DSV41_PLAN.md", "moex/docs/DSV41_PORT.md", "docs/AI_SETUP.md"))
ids = tok.encode(text)[:84400]
body = tok.decode(ids)
PY = _os.path.join(_AI, 'Strata/.venv/bin/python')
cold = sorted(str(f) for f in Path(f"{M}/native/engram").glob("*.bin")) + [f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf"]
print(subprocess.run([PY, "-I", f"{R}/moex/tools/dsv41/checks/pagecache.py", "--evict"] + cold, capture_output=True,
                     text=True).stdout.strip().splitlines()[-1:], flush=True)
OTHER = _os.path.join(_MODELS, 'GGUFs/glm53-flash/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00002-of-00004.gguf')
SAMPLER = ("import sys, time\nout = open(sys.argv[1], 'w')\nwhile True:\n"
           "    out.write(f\"{time.monotonic():.4f} {open('/sys/block/nvme0n1/inflight').read().split()[0]}\\n\")\n"
           "    out.flush()\n    time.sleep(0.002)\n")
PROBE = ("import mmap, os, random, sys, time\nfd = os.open(sys.argv[2], os.O_RDONLY | os.O_DIRECT)\n"
         "n = os.fstat(fd).st_size // 4096\nbuf = mmap.mmap(-1, 4096)\nout = open(sys.argv[1], 'w')\nr = random.Random()\n"
         "while True:\n    t = time.monotonic()\n    os.preadv(fd, [buf], r.randrange(n) * 4096)\n"
         "    out.write(f'{t:.4f} {(time.monotonic() - t) * 1e3:.3f}\\n')\n    out.flush()\n    time.sleep(0.1)\n")
smp = subprocess.Popen([sys.executable, "-I", "-c", SAMPLER, str(out / "inflight.txt")])
probe = subprocess.Popen([sys.executable, "-I", "-c", PROBE, str(out / "probe.txt"), OTHER])
t0 = time.monotonic()
srv = subprocess.Popen([PY, f"{R}/serve/server.py", "--engine", "strata", "--config", str(out / "config.json"), "--port", "1235"],
                       cwd=R, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
console, ready = [], threading.Event()


def read_console():
    for line in srv.stdout:
        console.append((time.monotonic(), line.rstrip("\n")))
        if line.startswith("ready:"):
            ready.set()


threading.Thread(target=read_console, daemon=True).start()
guard_seen, done = [], threading.Event()


def tail_log():   # the engine's lines (diagnostics on) as they reach its log: the guard's, timed
    pos, path = 0, Path(cfg["log"])
    while not done.is_set():
        time.sleep(0.01)
        if not path.exists():
            continue
        s = path.read_text(errors="replace")
        for line in s[pos:].splitlines(keepends=True):
            if not line.endswith("\n"):
                break
            pos += len(line)
            if "the drive guard" in line or "scale tables read" in line or "prompt heartbeat" in line:
                guard_seen.append((time.monotonic(), line.strip()))


threading.Thread(target=tail_log, daemon=True).start()
if not ready.wait(600):
    print("the server did not get ready in 600 s", flush=True)
    srv.send_signal(signal.SIGINT)
    sys.exit(1)
t_ready = time.monotonic()
req = urllib.request.Request("http://127.0.0.1:1235/v1/chat/completions", headers={"Content-Type": "application/json"},
                             data=json.dumps({"model": cfg["model_name"], "stream": True, "max_tokens": 16, "messages": [
                                 {"role": "user", "content": "Summarize these documents in one paragraph.\n\n" + body}]}).encode())
t_req = time.monotonic()
t_first, n_chunks = None, 0
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode(errors="replace").strip()
        if line.startswith("data: {"):
            d = json.loads(line[6:])
            ch = (d.get("choices") or [{}])[0].get("delta") or {}
            if (ch.get("content") or ch.get("reasoning_content")) and t_first is None:
                t_first = time.monotonic()
            n_chunks += 1
t_end = time.monotonic()
time.sleep(2)
srv.send_signal(signal.SIGINT)
try:
    srv.wait(timeout=120)
except subprocess.TimeoutExpired:
    srv.kill()
for q in (smp, probe):
    q.terminate()
done.set()

(out / "console.txt").write_text("".join(f"{ts - t_req:+9.2f} {l}\n" for ts, l in console))
reading = [ts for ts, l in console if "reading the prompt" in l and t_req <= ts <= (t_first or t_end)]
marks = [t_req] + reading + [t_first or t_end]
gaps = [b - a for a, b in zip(marks, marks[1:])]
pv = [(float(a), float(b)) for a, b in (l.split() for l in (out / "probe.txt").read_text().splitlines() if l.count(" ") == 1)]
iv = [(float(a), int(b)) for a, b in (l.split() for l in (out / "inflight.txt").read_text().splitlines() if l.count(" ") == 1)]


def stats(v):
    v = sorted(v)
    q = lambda f: v[min(len(v) - 1, int(f * len(v)))]
    return f"{len(v)} reads, median {q(0.5):.2f} ms, p90 {q(0.9):.2f}, p99 {q(0.99):.2f}, max {v[-1]:.1f}" if v else "none"


print(f"  the server ready {t_ready - t0:.0f} s after its start; the first token {((t_first or t_end) - t_req):.1f} s after the "
      f"request ({len(ids):,} tokens of text); {n_chunks} stream events", flush=True)
print(f"  the window's progress lines while reading: {len(reading)}, the longest gap {max(gaps):.1f} s", flush=True)
for l in [l for ts, l in console if "reading the prompt" in l][:3] + [l for ts, l in console if "reading the prompt" in l][-2:]:
    print(f"      {l[:160]}", flush=True)
for ts, l in guard_seen:
    print(f"      {ts - t_req:+8.2f} s  {l[:300]}", flush=True)
lows, lo = [], None   # the guard's spans at its floor, from its lines' arrival
for ts, l in guard_seen:
    if "the drive guard: the drive's slow mode" in l and lo is None:
        lo = ts
    elif "full depth again" in l and lo is not None:
        lows.append((lo, ts))
        lo = None
if lo is not None:
    lows.append((lo, t_end))
floor_reads = [x for ts, x in pv if any(a <= ts <= b for a, b in lows)]
its = [t for t, _ in iv]


def others(t, ms):   # the most requests besides the probe's on the drive while it ran (the 2 ms samples)
    import bisect
    v = [n for _, n in iv[bisect.bisect_left(its, t):bisect.bisect_right(its, t + ms / 1000)]]
    return max(v) - 1 if v else -1


floor_pairs = [(x, others(ts, x)) for ts, x in pv if any(a <= ts <= b for a, b in lows)]
alone = sorted(x for x, o in floor_pairs if o == 0)
beside = sorted(x for x, o in floor_pairs if o > 0)
med = lambda v: v[len(v) // 2] if v else None
print(f"  another program's lone 4 KB read during the request: {stats([x for ts, x in pv if t_req <= ts <= t_end])}", flush=True)
print(f"     ... while the guard was at its floor ({[(round(a - t_req, 1), round(b - t_req, 1)) for a, b in lows]} s): "
      f"{stats(floor_reads)}", flush=True)
print(f"     ... of them with nothing else on the drive: {stats(alone)}; beside the engine's: {stats(beside)}", flush=True)
slow = [(round(ts - t_req, 1), round(x)) for ts, x in pv if t_req <= ts <= t_end and x > 100]
print(f"     waits over 100 ms (s into the request, ms): {slow[:40]}", flush=True)
over = [x for ts, x in pv if t_req <= ts <= t_end and x > 2500]
mx = max((n for ts, n in iv if t_req <= ts <= t_end), default=None)
print(f"  the drive's in-flight count during the request: max {mx}", flush=True)
ok = {
    "a progress line at least every 30 s while the prompt was read": bool(gaps) and max(gaps) <= 30.5,
    "the first token came (no silence limit)": t_first is not None,
    "another program's reads: none over 2.5 s": not over,
    "... and at the guard's floor they wait little for the engine's (or no slow mode met)": not floor_reads or
    (med(beside) - med(alone) < 3.0 if len(alone) >= 10 and beside else
     sorted(floor_reads)[min(len(floor_reads) - 1, int(0.9 * len(floor_reads)))] < 10.0),
}
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
