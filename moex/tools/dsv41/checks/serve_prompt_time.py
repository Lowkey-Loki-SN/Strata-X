"""serve_prompt_time.py LOG_PREFIX EXE TOKENS ... - a prompt read's time at strata-dsv41.json as it stands (its arguments and
switches, 512K context), through the engine's serve protocol: for each TOKENS, a fresh engine reads that many tokens of
this repo's docs and sources (serve_long_context.py's text, no passphrase) from 0 and generates one token; the engram
tables are left in the page cache as they are, the SSD cache is the config's (60 GiB) - the read's time, its plan, the
drive's bytes, the slow-mode episodes met (a read with one is slower than the drive alone makes it)."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, re, subprocess, sys
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, exe, sizes = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
R = _REPO
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
files = subprocess.run(["git", "-C", R, "ls-files", "docs/*.md", "moex/docs/*.md", "src/x/*.cpp", "src/x/*.hpp", "src/x/*.cu",
                        "serve/*.py"], capture_output=True, text=True, check=True).stdout.split()
Q = tok.encode("\n\nIn one sentence: what does the text above do?<｜Assistant｜></think>", parse_special=True)
ids = []
for f in files:
    if len(ids) >= max(sizes):
        break
    ids += tok.encode(f"\n\n===== {f} =====\n\n" + (Path(R) / f).read_text(errors="replace"))
ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804"]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_ARENA") or k.startswith("STRATA_X_STAGE") or k.startswith("STRATA_POOL"):
        ENV.pop(k)
ENV.update(cfg["env"])
for n in sizes:
    P = ids[:n - len(Q)] + Q
    log = f"{prefix}-{n}.log"
    err = open(log, "w")
    p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=ENV)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    p.stdin.write("GEN 1 temperature=0 " + ",".join(map(str, P)) + "\n")
    p.stdin.flush()
    for line in p.stdout:
        if line.startswith("DONE") or line.startswith("ERR"):
            break
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    s = Path(log).read_text(errors="replace")
    g = lambda pat: (re.findall(pat, s) or ["-"])[-1]
    t = g(r"serve: \d+ prompt tokens \(0 reused, \d+ read\) in ([0-9.]+) s")
    print(f"  {len(P):,} tokens: read in {t} s ({len(P) / float(t):.0f} tok/s); {g(r'layer-major: ([^(]*)')}; the drive: "
          f"{g(r'the drive meanwhile: ([^;]*)')}; slow-mode episodes {s.count('the drive guard: the drive' + chr(39) + 's slow mode')}"
          if t != "-" else f"  {len(P):,} tokens: no DONE ({line.strip()[:120]})", flush=True)
