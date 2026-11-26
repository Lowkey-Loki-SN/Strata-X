"""serve_layer_major.py LOG_PREFIX - DSV41_BACKLOG #76 (STRATA_DSV41_LAYER_MAJOR=1): the first stage reads windows of chunks
layer by layer.  Through the engine's serve protocol (the dev build), the chunk loop's engine against the layer-major one,
each fresh, with one tier (--no-adaptive --expert-slots 1800, LM_SLOTS=N for another count: see serve_stop_layers.py) and the engram tables evicted from
the page cache before each starts - so the first reads' times compare too.  Bit for bit: the probe's hash of the whole
first-token logits row.

 X1  P (the 44K report prompt) from 0: equal in both engines (the core claim), and its prompt time in each
 X2  P[:root] + a question: resumes at the root checkpoint (34859) the first read made - its last window's copy in the
     layer-major engine - and must be equal in both
 X3  P[:16384] + the question: resumes at the checkpoint at 16384, made inside the layer-major engine's first window
 X4  (layer-major only) a reset, then P stopped 8 s after PP 16384: inside a window; X5 = P resumes at 16384 and must equal
     X1 (the same chunks, read in two calls)"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, subprocess, sys, threading, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix = sys.argv[1]
S = os.environ.get("STRATA_CHECK_DIR", _os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks'))
P = json.load(open(f"{S}/long-44k-prompt.json"))["ids"]
M = _os.path.join(_MODELS, 'dsv41-1017')
cfg = json.loads(Path(_os.path.join(_REPO, 'strata-dsv41.json')).read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
Q = tok.encode("\n\nIn one sentence: what does the code above do?<｜Assistant｜></think>", parse_special=True)
ROOT = next(i for i in range(2048, len(P)) if P[i] in (128799, 128803, 128804))
CMD = [os.environ.get("LM_EXE", _os.path.join(_REPO, 'build-dsv41-dev/strata-x')), "--serve", "-m", f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf", "--pack",
       f"{M}/ud-iq3xxs", "--ssd-cache-gib", "30", "--max-context", "65536", "--threads", "20", "--short-read", "96", "--layer-split",
       "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--no-adaptive", "--expert-slots", os.environ.get("LM_SLOTS", "1800"),
       "--expert-profile", _os.path.join(_DATA, 'dsv41-expert-profile-learned.txt')]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_DSV41_HEADS_MMVQ="1",
           STRATA_DSV41_HC_SPLIT="1", STRATA_DSV41_PROMPT_KEEP="1", STRATA_X_FIRST_TOP="8")
for k in ("STRATA_DSV41_LAYER_MAJOR", "STRATA_DSV41_STOP_LAYERS", "STRATA_DSV41_BOUNCE"):
    ENV.pop(k, None)
PY = _os.path.join(_AI, 'Strata/.venv/bin/python')
EVICT = [_os.path.join(_REPO, 'moex/tools/dsv41/checks/pagecache.py'), "--evict"] + \
    sorted(str(f) for f in Path(f"{M}/native/engram").glob("*.bin"))


class Engine:
    def __init__(self, tag, lm):
        print(subprocess.run([PY, "-I"] + EVICT, capture_output=True, text=True).stdout.strip().splitlines()[-1:], flush=True)
        self.log = f"{prefix}-{tag}.log"
        env = dict(ENV, STRATA_DSV41_LAYER_MAJOR="1", STRATA_DSV41_BOUNCE=os.environ.get("LM_BOUNCE", "96,32")) if lm else ENV
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

    def gen(self, ids, n=4, stop_pp=None):
        """One request; stop_pp = (position, seconds): STOP that long after the PP line reaching position."""
        start = len(self.text())
        self.send(f"GEN {n} temperature=0 " + ",".join(map(str, ids)) + "\n")
        r = {"resume": None, "pp": [], "tokens": [], "finish": None, "t_stop": None, "prompt_s": None}

        def stopper():
            while not (r["pp"] and r["pp"][-1] >= stop_pp[0]):
                time.sleep(0.2)
                if r["finish"]:
                    return
            time.sleep(stop_pp[1])
            if not r["finish"]:
                r["t_stop"] = time.time()
                self.send("STOP\n")

        if stop_pp:
            threading.Thread(target=stopper, daemon=True).start()
        for line in self.p.stdout:
            if line.startswith("RESUME"):
                r["resume"] = int(line.split()[1])
            elif line.startswith("PP "):
                r["pp"].append(int(line.split()[1]))
            elif line.startswith("T "):
                r["tokens"].append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                f = line.split()
                r["finish"] = f[5] if line.startswith("DONE") else "ERR " + line.strip()
                r["prompt_s"] = float(f[3]) / 1000 if line.startswith("DONE") else None
                r["stop_s"] = time.time() - r["t_stop"] if r["t_stop"] else None
                break
        time.sleep(0.5)
        mine = self.text()[start:]
        t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", mine)
        r["hash"] = t[-1][2] if t else None
        r["log"] = [l for l in mine.splitlines() if "layer-major" in l or "running state is back" in l or "exact nowhere" in l
                    or "serve: prompt path:" in l]
        return r

    def quit(self):
        self.send("QUIT\n")
        self.p.wait(timeout=180)


def show(name, r):
    print(f"  {name}: resume {r['resume']}, PP {r['pp']}, finish {r['finish']}, prompt {r['prompt_s']} s, tokens {r['tokens']}, "
          f"logits {r['hash']}" + (f", STOP -> DONE {r['stop_s']:.1f} s" if r.get("stop_s") is not None else ""), flush=True)
    for l in r["log"]:
        print(f"      {l.strip()[:420]}", flush=True)


res = {}
for tag, lm in (("cm", False), ("lm", True)):
    print(f"=== {'the layer-major engine' if lm else 'the chunk loop'} (root {ROOT})", flush=True)
    e = Engine(tag, lm)
    x = {}
    x["X1"] = e.gen(P); show("X1 P from 0", x["X1"])
    x["X2"] = e.gen(P[:ROOT] + Q); show(f"X2 P[:{ROOT}] + Q", x["X2"])
    x["X3"] = e.gen(P[:16384] + Q); show("X3 P[:16384] + Q", x["X3"])
    if lm:
        e.gen(P[30000:30200], 1)   # a reset: no prefix of P stays
        x["X4"] = e.gen(P, stop_pp=(16384, 8.0)); show("X4 P, stopped 8 s after PP 16384", x["X4"])
        x["X5"] = e.gen(P); show("X5 P", x["X5"])
    e.quit()
    res[tag] = x

cm, lm = res["cm"], res["lm"]
ok = {
    "X1: the 44K read from 0, equal": cm["X1"]["hash"] == lm["X1"]["hash"] and cm["X1"]["tokens"] == lm["X1"]["tokens"],
    f"X2: resumed at the root ({ROOT}), equal": lm["X2"]["resume"] == ROOT == cm["X2"]["resume"] and cm["X2"]["hash"] == lm["X2"]["hash"],
    "X3: resumed at 16384 (a window's copy), equal": lm["X3"]["resume"] == 16384 == cm["X3"]["resume"] and
    cm["X3"]["hash"] == lm["X3"]["hash"],
    "X4/X5: stopped inside a window, resumed at 16384, equal to X1": lm["X4"]["finish"] == "cancel" and lm["X5"]["resume"] == 16384
    and lm["X5"]["hash"] == lm["X1"]["hash"] and lm["X5"]["tokens"] == lm["X1"]["tokens"],
}
print("=== results", flush=True)
print(f"  the 44K read from 0: {cm['X1']['prompt_s']:.1f} s in the chunk loop, {lm['X1']['prompt_s']:.1f} s layer-major", flush=True)
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
