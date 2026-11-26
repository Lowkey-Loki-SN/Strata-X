"""serve_guard.py LOG_PREFIX TODAY_EXE NEW_EXE - DSV41_PLAN §18: the drive guard, the bounded engram fetch and the scale tables'
pre-read against today's build, through the engine's serve protocol, one tier (--no-adaptive --expert-slots 1800: see
serve_stop_layers.py), the config's switches with layer-major prompt reads: a 16,384-token text (moex/docs/DSV41_PLAN.md at
645844e9, not the 44K report prompt the other checks read) and a question, read from 0, then 8 greedy tokens.  Bit for
bit: the probe's hash of the whole first-token logits row, and the tokens.

 A  today's build (GATE_A=HASH:T1,T2,... takes its result from an earlier run instead)
 B  the new one with STRATA_DSV41_DRIVE_GUARD=1 on counters this script writes (STRATA_DSV41_DRIVE_GUARD_STAT): a healthy
    drive's through the load, the slow mode's from the request on - the whole request at the guard's floor (one record
    reader, 128 KB a call, the engram rows' pages one at a time; the pre-read waits) - STRATA_DSV41_EG_WINDOW=1024, the
    engram tables evicted from the page cache first; the drive's in-flight count (/sys/block/nvme0n1/inflight) sampled
    every 2 ms by a separate process
 C  the new one with STRATA_DSV41_DRIVE_GUARD=1 on counters this script writes (STRATA_DSV41_DRIVE_GUARD_STAT): a healthy
    drive's (8,000 reads a second at 0.2 ms), then from 20 s into the read 8 s of the slow mode's (1,000 a second at
    1,000 ms), then healthy again - when the guard went to its floor and back to full depth (its lines, timed as they
    arrive), and the drive's in-flight count meanwhile"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, subprocess, sys, threading, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, today_exe, new_exe = sys.argv[1], sys.argv[2], sys.argv[3]
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
ENV.update(STRATA_DSV41_LAYER_MAJOR="1", STRATA_DSV41_BOUNCE="96,32")
PY = _os.path.join(_AI, 'Strata/.venv/bin/python')
EVICT = [_os.path.join(_REPO, 'moex/tools/dsv41/checks/pagecache.py'), "--evict"] + \
    sorted(str(f) for f in Path(f"{M}/native/engram").glob("*.bin"))
SAMPLER = ("import sys, time\nout = open(sys.argv[1], 'w')\nwhile True:\n"
           "    out.write(f\"{time.monotonic():.4f} {open('/sys/block/nvme0n1/inflight').read().split()[0]}\\n\")\n"
           "    out.flush()\n    time.sleep(0.002)\n")


class Counters:
    """The counters file the guard reads in arm C: reads and read milliseconds, advanced every 100 ms."""
    def __init__(self, path):
        self.path, self.slow, self.reads, self.ms, self.stop = path, False, 0, 0, False
        self.write()
        threading.Thread(target=self.run, daemon=True).start()

    def write(self):
        Path(self.path + ".tmp").write_text(f"{self.reads} 0 0 {self.ms} 0 0 0 0 0 0 0\n")
        os.replace(self.path + ".tmp", self.path)

    def run(self):
        while not self.stop:
            time.sleep(0.1)
            n = 100 if self.slow else 800               # 1,000 or 8,000 reads a second
            self.reads += n
            self.ms += n * (1000 if self.slow else 0.2)  # 1,000 ms or 0.2 ms a read
            self.ms = int(self.ms)
            self.write()


def arm(tag, exe, extra, evict=False, counters=None, episode_at=None):
    if evict:
        print(subprocess.run([PY, "-I"] + EVICT, capture_output=True, text=True).stdout.strip().splitlines()[-1:], flush=True)
    log = f"{prefix}-{tag}.log"
    infl = f"{prefix}-{tag}-inflight.txt"
    err = open(log, "w")
    t_start = time.monotonic()
    p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                         env=dict(ENV, **extra))
    seen = []   # (monotonic time, line) of the guard's lines, as they reach the log
    done = threading.Event()

    def tail():
        pos = 0
        while not done.is_set():
            time.sleep(0.01)
            s = Path(log).read_text(errors="replace")
            for line in s[pos:].splitlines(keepends=True):
                if not line.endswith("\n"):
                    break
                pos += len(line)
                if "the drive guard" in line or "scale tables read" in line:
                    seen.append((time.monotonic(), line.strip()))

    threading.Thread(target=tail, daemon=True).start()
    for line in p.stdout:
        if line.startswith("READY"):
            break
    t_ready = time.monotonic()
    smp = subprocess.Popen([sys.executable, "-I", "-c", SAMPLER, infl])
    r = {"tokens": [], "pl": 0, "pp": [], "finish": None, "prompt_s": None, "load_s": t_ready - t_start}
    t_gen = time.monotonic()
    if counters and episode_at is None:   # the whole request in the slow mode's counters
        counters.slow, r["t_slow"] = True, time.monotonic()
    elif counters:
        def episode():
            time.sleep(episode_at[0])
            counters.slow, r["t_slow"] = True, time.monotonic()
            time.sleep(episode_at[1])
            counters.slow, r["t_healthy"] = False, time.monotonic()
        threading.Thread(target=episode, daemon=True).start()
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
    r["t_done"] = time.monotonic()
    if counters and episode_at is None:
        counters.slow = False
    smp.terminate()
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    done.set()
    err.close()
    s = Path(log).read_text(errors="replace")
    t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
    r["hash"] = t[-1][2] if t else None
    r["seen"] = seen
    r["inflight"] = [(float(a), int(b)) for a, b in (l.split() for l in Path(infl).read_text().splitlines() if l.count(" ") == 1)]
    r["t_gen"] = t_gen
    print(f"  {tag}: load {r['load_s']:.0f} s, prompt {r['prompt_s']} s, finish {r['finish']}, PP {r['pp']}, {r['pl']} PL lines, "
          f"tokens {r['tokens']}, logits {r['hash']}", flush=True)
    for ts, l in seen:
        print(f"      +{ts - t_gen:7.2f} s  {l[:300]}", flush=True)
    return r


def share(v, lo, hi, cap):
    w = [n for ts, n in v if lo <= ts <= hi]
    return (max(w) if w else None, sum(1 for n in w if n > cap) / len(w) if w else None, len(w))


print(f"=== the prompt: {len(P)} tokens; the switches: " + " ".join(f"{k[7:]}={v}" for k, v in sorted(ENV.items())
                                                              if k.startswith("STRATA_DSV4")), flush=True)
if os.environ.get("GATE_A"):
    h, ts = os.environ["GATE_A"].split(":")
    A = {"hash": h, "tokens": [int(x) for x in ts.split(",")]}
    print(f"  a-today (an earlier run): tokens {A['tokens']}, logits {A['hash']}", flush=True)
else:
    A = arm("a-today", today_exe, {})
if os.environ.get("GATE_SKIP_B"):   # an earlier run's arm B stands (GATE_SKIP_B=1): its checks are left out below
    B = None
else:
    bpath = f"{prefix}-b-counters.txt"
    bcnt = Counters(bpath)
    B = arm("b-floor", new_exe, {"STRATA_DSV41_DRIVE_GUARD": "1", "STRATA_DSV41_DRIVE_GUARD_STAT": bpath,
                                 "STRATA_DSV41_EG_WINDOW": "1024", "STRATA_DSV41_EG_SCALES_PRELOAD": "1"}, evict=True, counters=bcnt)
    bcnt.stop = True
cpath = f"{prefix}-c-counters.txt"
cnt = Counters(cpath)
C = arm("c-episode", new_exe, {"STRATA_DSV41_DRIVE_GUARD": "1", "STRATA_DSV41_DRIVE_GUARD_STAT": cpath,
                               "STRATA_DSV41_EG_WINDOW": "1024", "STRATA_DSV41_EG_SCALES_PRELOAD": "1"}, counters=cnt,
        episode_at=(20, 8))
cnt.stop = True

print("=== results", flush=True)
if B:
    beng = next((ts for ts, l in B["seen"] if "the drive guard: the drive's slow mode" in l), None)
    bmax, bover, bn = share(B["inflight"], (beng or B["t_gen"]) + 2.5, B["t_done"], 2)
    print(f"  B, the request at the floor (from {((beng or 0) - B['t_gen']):.2f} s): the drive's in-flight count from 2.5 s after that "
          f"to its end: max {bmax}, {100 * (bover or 0):.2f}% of {bn} samples above 2 (the engine's floor: one record read and one "
          f"engram page; other programs' reads count too)", flush=True)
eng = next((ts for ts, l in C["seen"] if "the drive guard: the drive's slow mode" in l), None)
full = next((ts for ts, l in C["seen"] if "full depth again" in l), None)
c_ok = "t_slow" in C and "t_healthy" in C and eng is not None and full is not None
if c_ok:
    print(f"  C: the slow mode's counters from +{C['t_slow'] - C['t_gen']:.2f} s to +{C['t_healthy'] - C['t_gen']:.2f} s; the guard "
          f"at its floor {eng - C['t_slow']:.2f} s after they began, at full depth {full - C['t_healthy']:.2f} s after they ended",
          flush=True)
    for lag in (0.0, 0.5, 1.0, 2.0):
        mx, ov, n = share(C["inflight"], eng + lag, C["t_healthy"], 2)
        print(f"     the in-flight count from {lag:.1f} s after the floor to the counters' end: max {mx}, "
              f"{100 * (ov or 0):.2f}% of {n} samples above 2", flush=True)
    mx, ov, n = share(C["inflight"], C["t_slow"] - 8, C["t_slow"], 2)
    print(f"     ... and in the 8 s before the slow mode's counters (full depth): max {mx}, {100 * (ov or 0):.2f}% of {n} above 2",
          flush=True)
# the first token's logits: the prompt path bit for bit (the tokens after it are the route bias's, which picks among the
# experts resident at the time - the RAM copy is sized at each start: serve_guard_decode.py checks the decode with it off)
ok = {} if not B else {
    "B (the request at the floor; the pre-read waits there): the first token's logits are today's": B["hash"] == A["hash"],
    "B: at the floor within 0.5 s, then at most 2 reads in flight in 99% of the samples": beng is not None and
    beng - B["t_slow"] <= 0.5 and bmax is not None and (bover or 0) <= 0.01,
}
ok |= {
    "C (the window, the pre-read, the guard through a slow mode): the first token's logits are today's": C["hash"] == A["hash"],
    "C: the guard went to its floor within 0.5 s and back to full depth within 2 s": c_ok and eng - C["t_slow"] <= 0.5 and
    full - C["t_healthy"] <= 2.0,
    "C: from 1 s after the floor to the slow mode's end at most 2 reads in flight in 99% of the samples": c_ok and
    (share(C["inflight"], eng + 1.0, C["t_healthy"], 2)[1] or 0) <= 0.01,
    "the reads sent PL lines": (not B or B["pl"] > 0) and C["pl"] > 0,
}
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
