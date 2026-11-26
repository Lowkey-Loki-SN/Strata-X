"""serve_short_warm.py LOG_PREFIX EXE - a conversation's short reads at strata-dsv41.json as it stands (its arguments and
switches: the 60 GiB SSD cache, the adaptive tier, the route bias), through the serve protocol: a ~3,000-token first turn of
this repo's docs, then TURNS (4) follow-ups, each the conversation so far (its prefix reused) and ~SHORT_N (400) new tokens of
other docs and a question, DECODE_N (48) tokens generated a turn.  EXTRA_ENV=K=V,... is added to the config's switches;
FIRST_N (3000) the first turn's tokens; NO_BIAS=1 drops the route bias (the decode then routes the same experts whatever the
tiers hold: an exactness check, each turn's first-token logits compared across arms).  Prints each turn's read (tokens
reused / read, its time), its prompt-path line's parts (the records from the SSD / its cache, the lend, the routing's waits,
the refill), its decode and its first token's logits."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, re, subprocess, sys
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, exe = sys.argv[1], sys.argv[2]
TURNS = int(os.environ.get("TURNS", "4"))
SHORT = int(os.environ.get("SHORT_N", "400"))
N = int(os.environ.get("DECODE_N", "48"))
R = _REPO
cfg = json.loads(Path(os.environ.get("CONFIG") or f"{R}/strata-dsv41.json").read_text())   # CONFIG=FILE: another config
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
git = lambda f: subprocess.run(["git", "-C", R, "show", f"645844e9:{f}"], capture_output=True, text=True, check=True).stdout
sp = lambda s: tok.encode(s, parse_special=True)
docs = [tok.encode(git(f)) for f in ("docs/DETAILS.md", "docs/AI_SETUP.md", "moex/docs/DSV41_PORT.md", "docs/MULTI_GPU.md",
                                     "docs/AMD_HIP.md", "docs/MCP_SERVER.md")]
first = sp("<｜User｜>") + docs[0][:int(os.environ.get("FIRST_N", "3000"))] + sp("\n\nWhat does the text above describe?<｜Assistant｜></think>")
ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804"] + \
    os.environ.get("EXTRA_ARGS", "").split()   # EXTRA_ARGS: engine flags (e.g. DSpark: --draft-dspark builtin --spec 2)
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_ARENA") or k.startswith("STRATA_X_STAGE") or k.startswith("STRATA_POOL"):
        ENV.pop(k)
ENV.update(cfg["env"])
if os.environ.get("NO_BIAS") == "1":
    ENV.pop("STRATA_DSV41_ROUTE_BIAS", None)
for kv in filter(None, os.environ.get("EXTRA_ENV", "").split(",")):
    k, v = kv.split("=", 1)
    ENV[k] = v
log = f"{prefix}.log"
err = open(log, "w")
p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=ENV)
for line in p.stdout:
    if line.startswith("READY"):
        break
conv = first
for t in range(TURNS + 1):
    if t > 0:
        d, at = docs[1 + (t - 1) % (len(docs) - 1)], (t - 1) // (len(docs) - 1) * SHORT   # (a later turn: the doc's next part)
        conv = conv + sp("<｜User｜>") + d[at:at + SHORT] + sp("\n\nAnd this part: what does it add?<｜Assistant｜></think>")
    mark = len(Path(log).read_text(errors="replace"))
    p.stdin.write(f"GEN {N} temperature=0 " + ",".join(map(str, conv)) + "\n")
    p.stdin.flush()
    toks, done = [], ""
    for line in p.stdout:
        if line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith("DONE") or line.startswith("ERR"):
            done = line.strip()
            break
    err.flush()
    s = Path(log).read_text(errors="replace")[mark:]
    g = lambda pat: (re.findall(pat, s) or ["-"])[-1]
    pp = g(r"prompt path: (\d+ tokens.*)")
    parts = re.search(r"(\d+) read from the SSD, (\d+) from its cache.*?refill (\d+) from the cache / (\d+) read.*?routing ([0-9.]+) s, "
                      r"the engram rows ([0-9.]+) s.*?lending ([0-9.]+) s, for the routing ([0-9.]+) s.*?refilling ([0-9.]+) s", pp)
    head = g(r"serve: (\d+ prompt tokens \(\d+ reused, \d+ read\) in [0-9.]+ s, \d+ generated in [0-9.]+ s \([0-9.]+ T/s\))")
    print(f"  turn {t}: {head}; {g(r'prompt path: (\d+ tokens, chunk \d+ \(ring \d+, \d+ slots lent\) in [0-9.]+ s)')}", flush=True)
    if parts:
        a = parts.groups()
        print(f"      records {a[0]} from the SSD, {a[1]} from its cache; refill {a[2]} cached / {a[3]} read; waits: routing {a[4]} s, "
              f"engram {a[5]} s; no read in flight: lending {a[6]} s, routing {a[7]} s, refilling {a[8]} s", flush=True)
    print(f"      the drive: {g(r'the drive meanwhile: ([^;]*)')[:100]}; slow-mode episodes {s.count('the drive guard: the drive' + chr(39) + 's slow mode')}", flush=True)
    kept = g(r"prompt path: the SSD cache kept (.*)")
    print(f"      decode: {g(r'serve: decode: (.*)')[:150]}" + (f"; the cache kept {kept[:120]}" if kept != "-" else ""), flush=True)
    print(f"      first-token logits {g(r'first-token top at position \d+:[^|]*\| logits ([0-9a-f]{16})')}; tokens {','.join(map(str, toks[:12]))}", flush=True)
    conv = conv + toks
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=180)
err.close()
