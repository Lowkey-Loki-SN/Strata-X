"""serve_eg_warm.py LOG_PREFIX EXE - DSV41_BACKLOG #83: do the cold engram fetches start the drive's slow mode?  One engine
at strata-dsv41.json's arguments and switches (EXTRA_ENV=K=V,... added), the engram tables left in the page cache as they are:

 1  ~84.4K tokens of this repo's docs read from 0 - their engram rows mostly cold
 2  the same text with another first token (nothing reused: the read starts at 0 again) - the same records streamed, its
    engram rows now in the page cache but for the first few positions
 Each read's slow-mode episodes (the drive guard's lines), its engram waits, its time and the drive's bytes"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, re, subprocess, sys
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, exe = sys.argv[1], sys.argv[2]
R = _REPO
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
git = lambda f: subprocess.run(["git", "-C", R, "show", f"645844e9:{f}"], capture_output=True, text=True, check=True).stdout
Q1 = tok.encode("\n\nIn one sentence: what does the text above do?<｜Assistant｜></think>", parse_special=True)
D = tok.encode("".join(git(f) for f in ("docs/DETAILS.md", "moex/docs/DSV41_PLAN.md", "moex/docs/DSV41_PORT.md", "docs/AI_SETUP.md")))
P1 = D[:84400] + Q1
P2 = tok.encode("Read this.\n\n") + D[:84400] + Q1
ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804"]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_ARENA") or k.startswith("STRATA_X_STAGE") or k.startswith("STRATA_POOL"):
        ENV.pop(k)
ENV.update(cfg["env"])
for kv in filter(None, os.environ.get("EXTRA_ENV", "").split(",")):
    k, v = kv.split("=", 1)
    ENV[k] = v
log = f"{prefix}.log"
err = open(log, "w")
p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=ENV)
for line in p.stdout:
    if line.startswith("READY"):
        break
for i, P in enumerate((P1, P2)):
    mark = len(Path(log).read_text(errors="replace"))
    p.stdin.write("GEN 1 temperature=0 " + ",".join(map(str, P)) + "\n")
    p.stdin.flush()
    for line in p.stdout:
        if line.startswith("DONE") or line.startswith("ERR"):
            break
    err.flush()
    s = Path(log).read_text(errors="replace")[mark:]
    g = lambda pat: (re.findall(pat, s) or ["-"])[-1]
    eps = [l for l in s.splitlines() if "the drive guard: the drive's slow mode" in l]
    print(f"  read {i + 1}: {g(r'serve: (\d+ prompt tokens [^,]*)')}; the drive: {g(r'the drive meanwhile: ([^;]*)')[:90]}; "
          f"{g(r'(\d+ tokens waited [0-9.]+ s for their engram rows)')}; the engram's waits {g(r'the engram rows ([0-9.]+ s)')}; "
          f"slow-mode episodes {len(eps)}", flush=True)
    for l in eps:
        print(f"      {l.split(': ', 1)[-1][:110]}", flush=True)   # without the engine's name
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=180)
err.close()
