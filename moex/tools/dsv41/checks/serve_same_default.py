"""serve_same_default.py LOG_PREFIX OLD_EXE NEW_EXE - a new build with its opt-in switches off against the build it replaces,
through the serve protocol, one tier (--no-adaptive --expert-slots 1800: see serve_stop_layers.py): the 44K report prompt
read from 0, then 32 greedy tokens; the first token's top-8 (5 decimals: the old build prints no logits hash) and the
tokens must be the same.  Then a follow-up from the checkpoint at 16384."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, subprocess, sys, threading
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, old_exe, new_exe = sys.argv[1], sys.argv[2], sys.argv[3]
S = os.environ.get("STRATA_CHECK_DIR", _os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks'))
P = json.load(open(f"{S}/long-44k-prompt.json"))["ids"]
M = _os.path.join(_MODELS, 'dsv41-1017')
cfg = json.loads(Path(_os.path.join(_REPO, 'strata-dsv41.json')).read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
Q = tok.encode("\n\nIn one sentence: what does the code above do?<｜Assistant｜></think>", parse_special=True)
ARGS = ["--serve", "-m", f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf", "--pack", f"{M}/ud-iq3xxs", "--ssd-cache-gib", "30",
        "--max-context", "65536", "--threads", "20", "--short-read", "96", "--layer-split", "20", "--eos-ids", "1",
        "--turn-tokens", "128799,128803,128804", "--no-adaptive", "--expert-slots", "1800",
        "--expert-profile", _os.path.join(_DATA, 'dsv41-expert-profile-learned.txt')]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_DSV41_HEADS_MMVQ="1",
           STRATA_DSV41_HC_SPLIT="1", STRATA_DSV41_PROMPT_KEEP="1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k in ("STRATA_DSV41_LAYER_MAJOR", "STRATA_DSV41_STOP_LAYERS"):
        ENV.pop(k)


def run(tag, exe):
    log = f"{prefix}-{tag}.log"
    err = open(log, "w")
    p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=ENV)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    out = []
    for ids, n in ((P, 32), (P[:16384] + Q, 32)):
        p.stdin.write(f"GEN {n} temperature=0 " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        toks, resume = [], None
        for line in p.stdout:
            if line.startswith("RESUME"):
                resume = int(line.split()[1])
            elif line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                break
        out.append((resume, toks))
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    tops = [re.sub(r"\s*\| logits [0-9a-f]{16}", "", t).strip()
            for t in re.findall(r"first-token top at position \d+:(.*)", Path(log).read_text(errors="replace"))]
    return out, tops


old = run("old", old_exe)
print("old:", old, flush=True)
new = run("new", new_exe)
print("new:", new, flush=True)
same = old == new
print(f"{'PASS' if same else 'FAIL'}: the new build with its switches off is the old one - the 44K read and its follow-up: "
      f"resume points, 32 greedy tokens each, the first token's top-8 to 5 decimals", flush=True)
