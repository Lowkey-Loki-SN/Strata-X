"""serve_stop_layers.py LOG_PREFIX - DSV41_BACKLOG #78 (STRATA_DSV41_STOP_LAYERS=1): a stop between a chunk's layers, through
the engine's serve protocol (the dev build), against references read by the same build with the switch off.  The tier is
frozen (--no-adaptive) and of one size in both engines (--expert-slots 1800: the slots fill what a GPU has free, and GPU 1's
free memory moves with the display - a decode window computes a VRAM expert on the GPU, the others on the CPU, rounding
apart), so equal states give equal first-token logits: the probe's hash of the whole row, bit for bit.

 ref  (the switch off)  R1 = P[:16384]; R2 = P (resumes at 16384); a reset; R4 = P (read from 0)
 (A)  T1 = P[:16384] (must equal R1: the switch changes nothing without a stop); T2 = P, stopped ~8 s into its first chunk -
      GPU 0 stops before a layer > 0 and goes back to 16384, the read's start, so the live tokens stay; T3 = P resumes at
      16384 and must equal R2
 (B)  a reset; T4 = P, stopped ~8 s after PP 16384 (a checkpoint): GPU 0, inside the chunk at 16384, goes back; T5 = P
      resumes at 16384 (the live tokens win a tie with the checkpoint) and must equal R4
 (C)  a reset; T6 = P, stopped once GPU 0 has read its last chunk (the log's "prompt path: N tokens waited" line): GPU 1
      stops inside the last chunk and both go back to its start; T7 = P resumes there and must equal R4
Each stop's latency (STOP sent -> DONE) is printed."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, subprocess, sys, threading, time
from pathlib import Path

prefix = sys.argv[1]
S = os.environ.get("STRATA_CHECK_DIR", _os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks'))
P = json.load(open(f"{S}/long-44k-prompt.json"))["ids"]
M = _os.path.join(_MODELS, 'dsv41-1017')
CMD = [_os.path.join(_REPO, 'build-dsv41-dev/strata-x'), "--serve", "-m", f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf", "--pack",
       f"{M}/ud-iq3xxs", "--ssd-cache-gib", "30", "--max-context", "65536", "--threads", "20", "--short-read", "96", "--layer-split",
       "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--no-adaptive", "--expert-slots", "1800",
       "--expert-profile", _os.path.join(_DATA, 'dsv41-expert-profile-learned.txt')]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_DSV41_HEADS_MMVQ="1",
           STRATA_DSV41_HC_SPLIT="1", STRATA_DSV41_PROMPT_KEEP="1", STRATA_X_FIRST_TOP="8")
ENV.pop("STRATA_DSV41_STOP_LAYERS", None)


class Engine:
    def __init__(self, tag, stop_layers):
        self.log = f"{prefix}-{tag}.log"
        env = dict(ENV, STRATA_DSV41_STOP_LAYERS="1") if stop_layers else ENV
        self.err = open(self.log, "w")
        self.p = subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, env=env)
        self.lock = threading.Lock()
        for line in self.p.stdout:
            if line.startswith("READY"):
                break

    def text(self):
        self.err.flush()
        return Path(self.log).read_text(errors="replace")

    def send(self, s):
        with self.lock:
            self.p.stdin.write(s)
            self.p.stdin.flush()

    def gen(self, ids, n=4, stop=None):
        """One request.  stop: None, ("resume", seconds), ("pp", position, seconds) or ("log", regex) - when to send STOP.
        Returns a dict: resume, pp (reached list), finish, tokens, top (the probe's line), hash, stop_s (STOP -> DONE), log
        (the engine's lines of this request)."""
        start = len(self.text())
        self.send(f"GEN {n} temperature=0 " + ",".join(map(str, ids)) + "\n")
        r = {"resume": None, "pp": [], "tokens": [], "finish": None, "t_stop": None}
        ev_resume, ev_pp = threading.Event(), threading.Event()

        def stopper():
            if stop[0] == "resume":
                ev_resume.wait()
                time.sleep(stop[1])
            elif stop[0] == "pp":
                while not (r["pp"] and r["pp"][-1] >= stop[1]):
                    ev_pp.wait(0.2)
                    if r["finish"]:
                        return
                time.sleep(stop[2])
            else:
                while not re.search(stop[1], self.text()[start:]):
                    time.sleep(0.2)
                    if r["finish"]:
                        return
            if not r["finish"]:
                r["t_stop"] = time.time()
                self.send("STOP\n")

        th = threading.Thread(target=stopper, daemon=True) if stop else None
        if th:
            th.start()
        for line in self.p.stdout:
            if line.startswith("RESUME"):
                r["resume"] = int(line.split()[1])
                ev_resume.set()
            elif line.startswith("PP "):
                r["pp"].append(int(line.split()[1]))
                ev_pp.set()
            elif line.startswith("T "):
                r["tokens"].append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                f = line.split()
                r["finish"] = f[5] if line.startswith("DONE") else "ERR " + line.strip()
                r["stop_s"] = time.time() - r["t_stop"] if r["t_stop"] else None
                break
        time.sleep(0.5)
        mine = self.text()[start:]
        t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", mine)
        r["top"], r["hash"] = (t[-1][1].strip(), t[-1][2]) if t else (None, None)
        r["log"] = [l for l in mine.splitlines() if "a stop before layer" in l or "running state is back" in l or "exact nowhere" in l]
        return r

    def quit(self):
        self.send("QUIT\n")
        self.p.wait(timeout=180)


def reset(e):   # an unrelated prompt read from 0: no prefix of P stays (the live tokens, the checkpoints)
    e.gen(P[30000:30200], 1)


def show(name, r):
    print(f"  {name}: resume {r['resume']}, PP {r['pp']}, finish {r['finish']}, tokens {r['tokens']}, logits {r['hash']}"
          + (f", STOP -> DONE {r['stop_s']:.1f} s" if r.get("stop_s") is not None else ""), flush=True)
    for l in r["log"]:
        print(f"      {l.strip()}", flush=True)


print("=== the reference (the switch off)", flush=True)
e = Engine("ref", False)
R1 = e.gen(P[:16384], 1); show("R1 P[:16384]", R1)   # one token: the live tokens are exactly P[:16384] after it
R2 = e.gen(P); show("R2 P", R2)
reset(e)
R4 = e.gen(P); show("R4 P from 0", R4)
e.quit()

print("=== the switch on", flush=True)
e = Engine("test", True)
ok = {}
T1 = e.gen(P[:16384], 1); show("T1 P[:16384]", T1)
ok["switch on, no stop = off"] = T1["hash"] == R1["hash"] and T1["tokens"] == R1["tokens"]

T2 = e.gen(P, stop=("resume", 8.0)); show("(A) T2 P, stopped 8 s after RESUME", T2)
T3 = e.gen(P); show("(A) T3 P", T3)
a_stop = [l for l in T2["log"] if "GPU 0" in l and "a stop before layer" in l and "goes back" in l]
ok["(A) stopped inside the first chunk, back at the read's start"] = T2["finish"] == "cancel" and bool(a_stop) and \
    any("back at 16384" in l for l in T2["log"])
ok["(A) resumed at 16384, equal to R2"] = T3["resume"] == 16384 and T3["hash"] == R2["hash"] and T3["tokens"] == R2["tokens"]

reset(e)
T4 = e.gen(P, stop=("pp", 16384, 8.0)); show("(B) T4 P, stopped 8 s after PP 16384", T4)
T5 = e.gen(P); show("(B) T5 P", T5)
# (the stop lands in the chunk GPU 0 has reached by then - 2026-10-21's run: the one at 24576, its layer 6: the state went back
# there, where the last stage had got, not a checkpoint - so the next read resumed at the one below, 16384)
b_at = T4["pp"][-1] if T4["pp"] else 0
ok["(B) GPU 0 stopped inside a chunk after PP 16384, back where the last stage got"] = T4["finish"] == "cancel" and \
    any("(GPU 0): its running state goes back" in l for l in T4["log"]) and \
    any(f"GPU 0's running state is back at {b_at}" in l for l in T4["log"])
ok["(B) resumed at the checkpoint at or below it, equal to R4"] = T5["resume"] in (16384, b_at) and T5["hash"] == R4["hash"] and \
    T5["tokens"] == R4["tokens"]

reset(e)
T6 = e.gen(P, stop=("log", r"prompt path: \d+ tokens waited")); show("(C) T6 P, stopped after GPU 0's last chunk", T6)
T7 = e.gen(P); show("(C) T7 P", T7)
c_back = [int(m) for l in T6["log"] for m in re.findall(r"GPU 1's running state is back at (\d+)", l)]
ok["(C) GPU 1 stopped inside its last chunk, both back at its start"] = T6["finish"] == "cancel" and \
    any("(GPU 1): its running state goes back" in l for l in T6["log"]) and bool(c_back) and \
    any(f"GPU 0's running state is back at {c_back[0]}" in l for l in T6["log"])
ok["(C) resumed at that start, equal to R4"] = bool(c_back) and T7["resume"] == c_back[0] and T7["hash"] == R4["hash"] and \
    T7["tokens"] == R4["tokens"]
e.quit()

print("=== results", flush=True)
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
