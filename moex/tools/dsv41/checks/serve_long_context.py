"""serve_long_context.py LOG_PREFIX EXE [TOKENS] - past the 262,144 positions every earlier run stayed under: one read of TOKENS
(300,000 by default) at strata-dsv41.json's arguments and switches (512K context, the 60 GiB cache, layer-major), through the
engine's serve protocol - a passphrase in the text's first line, this repo's own docs and sources after it (no conversation
of the user's), and a question for the passphrase at the end - then 16 greedy tokens.

Pass: the read and the reply end without an error, the first token's logits are finite (its top-8 printed), and the reply
names the passphrase (recall over the whole read)."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, math, os, re, subprocess, sys, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, exe = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300000
R = _REPO
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
files = subprocess.run(["git", "-C", R, "ls-files", "docs/*.md", "moex/docs/*.md", "src/x/*.cpp", "src/x/*.hpp", "src/x/*.cu",
                        "serve/*.py"], capture_output=True, text=True, check=True).stdout.split()
head = "The passphrase for this whole text is: amber-walrus-7193. Remember it.\n\n"
Q = tok.encode("\n\nWhat is the passphrase given in the very first line of this text? Reply with the passphrase only."
               "<｜Assistant｜></think>", parse_special=True)
ids = tok.encode(head)
for f in files:
    if len(ids) >= N:
        break
    ids += tok.encode(f"\n\n===== {f} =====\n\n" + (Path(R) / f).read_text(errors="replace"))
if len(ids) < N:
    raise SystemExit(f"only {len(ids):,} tokens of text")
P = ids[:N - len(Q)] + Q
ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804"]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_ARENA"):
        ENV.pop(k)
ENV.update(cfg["env"])

log = f"{prefix}-engine.log"
err = open(log, "w")
p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=ENV)
for line in p.stdout:
    if line.startswith("READY"):
        break
t0 = time.monotonic()
p.stdin.write("GEN 16 temperature=0 " + ",".join(map(str, P)) + "\n")
p.stdin.flush()
out, done = [], None
for line in p.stdout:
    if line.startswith("T "):
        out.append(int(line.split()[1]))
    elif line.startswith("DONE") or line.startswith("ERR"):
        done = line.strip()
        break
wall = time.monotonic() - t0
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=300)
err.close()
s = Path(log).read_text(errors="replace")
reply = tok.decode(out)
top = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
vals = [float(x) for x in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", top[-1][1])] if top else []
errs = [l for l in s.splitlines() if any(w in l for w in ("ERR ", "failed", "error:", "CUDA error"))]   # serve_smoke.py's words
print(f"  {len(P):,} tokens read; DONE line: {done}; {wall:.0f} s for the read and 16 tokens", flush=True)
print(f"  the first token's top-8 at position {top[-1][0] if top else '?'}: {top[-1][1].strip()[:200] if top else '-'}", flush=True)
for l in re.findall(r"(?:layer-major: |serve: prompt path: |serve: decode: |the drive meanwhile: ).*", s)[-4:]:
    print(f"      {l[:260]}", flush=True)
print(f"  the reply: {reply!r}", flush=True)
ok = {
    "the read and the reply ended without an error": bool(done) and done.startswith("DONE") and not errs,
    "the first token's logits are finite": bool(vals) and all(math.isfinite(v) for v in vals),
    "the reply names the passphrase (recall across the read)": "amber-walrus-7193" in reply.replace(" ", ""),
}
for l in errs[:5]:
    print(f"      error line: {l[:200]}", flush=True)
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
