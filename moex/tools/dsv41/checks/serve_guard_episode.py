"""serve_guard_episode.py LOG_PREFIX NEW_EXE [HASH TOKENS] - DSV41_PLAN §18: the drive guard against the drive itself.  The new
build with the config's §18 switches (the guard on the drive's own counters, the bounded engram fetch, the scales' pre-read,
layer-major), the engram tables evicted from the page cache first - the cold engram fetch the slow mode met three times on
2026-10-21/22 - the 16,398-token text of serve_guard.py read from 0 and 8 greedy tokens, one tier.  Meanwhile, each in its
own process: the drive's in-flight count every 2 ms, and another program's read - one 4 KB O_DIRECT read of a file the
engine never reads (GLM-5.3's GGUF) every 100 ms, timed.  Prints the guard's lines as they came, the prompt's time, and the
lone reads' times while the guard was at its floor and outside it; HASH TOKENS (serve_guard.py's arm A): the output must
be the same bit for bit."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, subprocess, sys, threading, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, new_exe = sys.argv[1], sys.argv[2]
want_hash = sys.argv[3] if len(sys.argv) > 3 else None
want_tokens = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
M = _os.path.join(_MODELS, 'dsv41-1017')
cfg = json.loads(Path(_os.path.join(_REPO, 'strata-dsv41.json')).read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
text = subprocess.run(["git", "-C", _REPO, "show", "645844e9:moex/docs/DSV41_PLAN.md"], capture_output=True,
                      text=True, check=True).stdout
P = tok.encode(text)[:16384] + tok.encode("\n\nIn one sentence: what does the plan above do?<｜Assistant｜></think>",
                                          parse_special=True)
ARGS = ["--serve", "-m", f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf", "--pack", f"{M}/ud-iq3xxs", "--ssd-cache-gib", "30",
        "--max-context", "65536", "--threads", "20", "--short-read", "96", "--layer-split", "20", "--eos-ids", "1",
        "--turn-tokens", "128799,128803,128804", "--no-adaptive", "--expert-slots", "1800",
        "--expert-profile", _os.path.join(_DATA, 'dsv41-expert-profile-learned.txt')]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4"):
        ENV.pop(k)
ENV.update({k: v for k, v in cfg["env"].items() if k != "STRATA_DSV41_EG_READER"})
ENV.update(STRATA_DSV41_LAYER_MAJOR="1", STRATA_DSV41_BOUNCE="96,32", STRATA_DSV41_DRIVE_GUARD="1",
           STRATA_DSV41_EG_WINDOW="1024", STRATA_DSV41_EG_SCALES_PRELOAD="1")
PY = _os.path.join(_AI, 'Strata/.venv/bin/python')
EVICT = [_os.path.join(_REPO, 'moex/tools/dsv41/checks/pagecache.py'), "--evict"] + \
    sorted(str(f) for f in Path(f"{M}/native/engram").glob("*.bin"))
OTHER = _os.path.join(_MODELS, 'GGUFs/glm53-flash/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00002-of-00004.gguf')
SAMPLER = ("import sys, time\nout = open(sys.argv[1], 'w')\nwhile True:\n"
           "    out.write(f\"{time.monotonic():.4f} {open('/sys/block/nvme0n1/inflight').read().split()[0]}\\n\")\n"
           "    out.flush()\n    time.sleep(0.002)\n")
PROBE = ("import mmap, os, random, sys, time\nfd = os.open(sys.argv[2], os.O_RDONLY | os.O_DIRECT)\n"
         "n = os.fstat(fd).st_size // 4096\nbuf = mmap.mmap(-1, 4096)\nout = open(sys.argv[1], 'w')\nr = random.Random()\n"
         "while True:\n    t = time.monotonic()\n    os.preadv(fd, [buf], r.randrange(n) * 4096)\n"
         "    out.write(f'{t:.4f} {(time.monotonic() - t) * 1e3:.3f}\\n')\n    out.flush()\n    time.sleep(0.1)\n")

print(subprocess.run([PY, "-I"] + EVICT, capture_output=True, text=True).stdout.strip().splitlines()[-1:], flush=True)
log, infl, prb = f"{prefix}-d.log", f"{prefix}-d-inflight.txt", f"{prefix}-d-probe.txt"
err = open(log, "w")
t_start = time.monotonic()
p = subprocess.Popen([new_exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=ENV)
seen, done = [], threading.Event()


def tail():
    pos = 0
    while not done.is_set():
        time.sleep(0.01)
        s = Path(log).read_text(errors="replace")
        for line in s[pos:].splitlines(keepends=True):
            if not line.endswith("\n"):
                break
            pos += len(line)
            if "the drive guard" in line or "scale tables read" in line or "prompt heartbeat" in line:
                seen.append((time.monotonic(), line.strip()))


threading.Thread(target=tail, daemon=True).start()
smp = subprocess.Popen([sys.executable, "-I", "-c", SAMPLER, infl])
probe = subprocess.Popen([sys.executable, "-I", "-c", PROBE, prb, OTHER])
for line in p.stdout:
    if line.startswith("READY"):
        break
t_gen = time.monotonic()
r = {"tokens": [], "pl": 0, "pp": [], "finish": None, "prompt_s": None}
p.stdin.write("GEN 8 temperature=0 " + ",".join(map(str, P)) + "\n")
p.stdin.flush()
for line in p.stdout:
    if line.startswith("PL "):
        r["pl"] += 1
    elif line.startswith("PP "):
        r["pp"].append(int(line.split()[1]))
    elif line.startswith("T "):
        r["tokens"].append(int(line.split()[1]))
    elif line.startswith("DONE") or line.startswith("ERR"):
        f = line.split()
        r["finish"] = f[5] if line.startswith("DONE") else line.strip()
        r["prompt_s"] = float(f[3]) / 1000 if line.startswith("DONE") else None
        break
t_done = time.monotonic()
time.sleep(2)
for q in (smp, probe):
    q.terminate()
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=180)
done.set()
err.close()
s = Path(log).read_text(errors="replace")
t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
r["hash"] = t[-1][2] if t else None
print(f"  d: load {t_gen - t_start:.0f} s, prompt {r['prompt_s']} s, finish {r['finish']}, PP {r['pp']}, {r['pl']} PL lines, tokens "
      f"{r['tokens']}, logits {r['hash']}", flush=True)
for ts, l in seen:
    print(f"      {ts - t_gen:+8.2f} s  {l[:330]}", flush=True)
iv = [(float(a), int(b)) for a, b in (l.split() for l in Path(infl).read_text().splitlines() if l.count(" ") == 1)]
pv = [(float(a), float(b)) for a, b in (l.split() for l in Path(prb).read_text().splitlines() if l.count(" ") == 1)]
lows, lo = [], None   # the guard's spans below full depth: from its floor line to its full-depth line
for ts, l in seen:
    if "the drive guard: the drive's slow mode" in l and lo is None:
        lo = ts
    elif "full depth again" in l and lo is not None:
        lows.append((lo, ts))
        lo = None
if lo is not None:
    lows.append((lo, t_done))


def stats(v):
    v = sorted(v)
    q = lambda f: v[min(len(v) - 1, int(f * len(v)))]
    return f"{len(v)} reads, median {q(0.5):.2f} ms, p90 {q(0.9):.2f}, p99 {q(0.99):.2f}, max {v[-1]:.1f}" if v else "none"


inside = lambda ts: any(a <= ts <= b for a, b in lows)
print(f"  the guard below full depth: {[(round(a - t_gen, 2), round(b - t_gen, 2)) for a, b in lows]} s (from the request)", flush=True)
print(f"  another program's lone 4 KB read, the request's whole time: {stats([x for ts, x in pv if t_gen <= ts <= t_done])}", flush=True)
print(f"     ... while the guard was below full depth: {stats([x for ts, x in pv if inside(ts)])}", flush=True)
print(f"     ... after its first 2.5 s there: {stats([x for ts, x in pv if any(a + 2.5 <= ts <= b for a, b in lows)])}", flush=True)
print(f"     ... with the guard at full depth: {stats([x for ts, x in pv if t_gen <= ts <= t_done and not inside(ts)])}", flush=True)
w = [n for ts, n in iv if any(a + 2.5 <= ts <= b for a, b in lows)]
if w:
    print(f"  the drive's in-flight count from 2.5 s after the floor: max {max(w)}, "
          f"{100 * sum(1 for n in w if n > 2) / len(w):.2f}% of {len(w)} samples above 2", flush=True)
if want_hash:
    same = r["hash"] == want_hash and (want_tokens is None or r["tokens"] == want_tokens)
    what = f"logits {want_hash}" + (", the same 8 tokens" if want_tokens is not None else " - the tokens after it are the route bias's")
    print(f"  {'PASS' if same else 'FAIL'}  bit for bit with today's build ({what})", flush=True)
