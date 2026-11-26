"""serve_lm_speed.py LOG_PREFIX ARM... - DSV41_BACKLOG #76's speed: each ARM (a name, then NAME=VALUE environment settings, e.g.
"lm STRATA_DSV41_LAYER_MAJOR=1") a fresh engine of the dev build, the engram tables evicted from the page cache first, one tier
(--no-adaptive --expert-slots 1800), reading the 44K report prompt from 0 (4 greedy tokens).  Prints each arm's prompt time,
its prompt-path line and the first token's logits hash - equal in every arm (the switches are exact).  EXTRA_ARGS adds engine
flags to every arm; CONFIG_ENV=1 starts every arm from strata-dsv41.json's switches."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, subprocess, sys
from pathlib import Path

prefix = sys.argv[1]
arms = []
for a in sys.argv[2:]:
    if "=" in a:
        k, v = a.split("=", 1)
        arms[-1][1][k] = v
    else:
        arms.append((a, {}))
S = os.environ.get("STRATA_CHECK_DIR", _os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks'))
P = json.load(open(f"{S}/long-44k-prompt.json"))["ids"]
M = _os.path.join(_MODELS, 'dsv41-1017')
CMD = [_os.path.join(_REPO, 'build-dsv41-dev/strata-x'), "--serve", "-m", f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf", "--pack",
       f"{M}/ud-iq3xxs", "--ssd-cache-gib", "30", "--max-context", "65536", "--threads", "20", "--short-read", "96", "--layer-split",
       "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--no-adaptive", "--expert-slots", "1800",
       "--expert-profile", _os.path.join(_DATA, 'dsv41-expert-profile-learned.txt')] + os.environ.get("EXTRA_ARGS", "").split()
BASE = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_DSV41_HEADS_MMVQ="1",
            STRATA_DSV41_HC_SPLIT="1", STRATA_DSV41_PROMPT_KEEP="1", STRATA_X_FIRST_TOP="8")
for k in ("STRATA_DSV41_LAYER_MAJOR", "STRATA_DSV41_STOP_LAYERS", "STRATA_DSV41_BOUNCE", "STRATA_DSV41_LM_CHUNKS"):
    BASE.pop(k, None)
if os.environ.get("CONFIG_ENV") == "1":   # (DSV41_PLAN §59) the arms on strata-dsv41.json's switches, as the server reads
    BASE.update(json.load(open(_os.path.join(_REPO, 'strata-dsv41.json')))["env"])
PY = _os.path.join(_AI, 'Strata/.venv/bin/python')
EVICT = [PY, "-I", _os.path.join(_REPO, 'moex/tools/dsv41/checks/pagecache.py'), "--evict"] + \
    sorted(str(f) for f in Path(f"{M}/native/engram").glob("*.bin"))
hashes = {}
for name, env in arms:
    print(f"=== {name} {env}: " + subprocess.run(EVICT, capture_output=True, text=True).stdout.strip().splitlines()[-1], flush=True)
    log = f"{prefix}-{name}.log"
    with open(log, "w") as err:
        p = subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                             env=dict(BASE, **env))
        for line in p.stdout:
            if line.startswith("READY"):
                break
        p.stdin.write("GEN 4 temperature=0 " + ",".join(map(str, P)) + "\n")
        p.stdin.flush()
        toks, done = [], None
        for line in p.stdout:
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                done = line.split()
                break
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=180)
    text = Path(log).read_text(errors="replace")
    h = re.findall(r"\| logits ([0-9a-f]{16})", text)
    hashes[name] = (h[-1] if h else None, tuple(toks))
    print(f"  prompt {float(done[3]) / 1000:.1f} s, tokens {toks}, logits {hashes[name][0]}" if done and done[0] == "DONE"
          else f"  {done}", flush=True)
    for l in text.splitlines():
        if "layer-major:" in l or "serve: prompt path:" in l or "tokens waited" in l:
            print("    " + l.strip()[:1000], flush=True)
same = len({v for v in hashes.values()}) == 1
print(f"{'PASS' if same else 'FAIL'}: every arm's first-token logits and tokens equal", flush=True)
