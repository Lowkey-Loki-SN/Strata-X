"""serve_lm_keep.py LOG_PREFIX EXE - DSV41_PLAN §28: layer-major windows of 4 at the server's 512K context.  A layer-major read
lent all but 128 of GPU 0's slots; STRATA_DSV41_LM_KEEP=32 lends all but 32 (the 96 more experts lent are copied into the SSD
cache first, STRATA_DSV41_PROMPT_KEEP, so the read streams them and the refill takes them from RAM).  Through the engine's
serve protocol, strata-dsv41.json's arguments and switches, --no-adaptive, the slots that fit, the 30 GiB test cache, the
engram tables left in the page cache; ~120K tokens of this repo's docs and sources (serve_long_context.py's text, no
passphrase) read from 0:

 A  the config: windows of 3
 B  STRATA_DSV41_LM_KEEP=32: windows of 4 over the same chunks - the first token's logits bit for bit with A's, the time and
    the drive's bytes against A's"""
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
Q = tok.encode("\n\nIn one sentence: what does the text above do?<｜Assistant｜></think>", parse_special=True)
files = subprocess.run(["git", "-C", R, "ls-files", "docs/*.md", "moex/docs/*.md", "src/x/*.cpp", "src/x/*.hpp", "src/x/*.cu",
                        "serve/*.py"], capture_output=True, text=True, check=True).stdout.split()
ids = []
for f in files:
    if len(ids) >= 120000:
        break
    ids += tok.encode(f"\n\n===== {f} =====\n\n" + (Path(R) / f).read_text(errors="replace"))
P84 = ids[:120000 - len(Q)] + Q   # (the name kept from serve_lm_windows.py: ~120K tokens here)
ARGS = ["--serve"] + [a for a in cfg["args"]] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804",
                                                 "--no-adaptive"]
ARGS[ARGS.index("--ssd-cache-gib") + 1] = "30"   # the test runs' cache (the user's OK), as serve_score_blocks.py
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_ARENA") or k.startswith("STRATA_X_STAGE"):
        ENV.pop(k)
ENV.update(cfg["env"])


def arm(tag, ids, extra=None):
    log = f"{prefix}-{tag}.log"
    err = open(log, "w")
    p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                         env=dict(ENV, **(extra or {})))
    for line in p.stdout:
        if line.startswith("READY"):
            break
    p.stdin.write("GEN 1 temperature=0 " + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    prompt_s = None
    for line in p.stdout:
        if line.startswith("DONE") or line.startswith("ERR"):
            f = line.split()
            prompt_s = float(f[3]) / 1000 if line.startswith("DONE") else line.strip()
            break
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    s = Path(log).read_text(errors="replace")
    t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
    lm = re.findall(r"layer-major: .*", s)
    drive = re.findall(r"the drive meanwhile: ([^;]*)", s)
    guard = [l for l in s.splitlines() if "the drive guard: the drive's slow mode" in l]
    r = {"hash": t[-1][2] if t else None, "prompt_s": prompt_s, "lm": lm[-1] if lm else "-", "drive": drive[-1] if drive else "-",
         "episodes": len(guard)}
    print(f"  {tag}: {len(ids):,} tokens, prompt {prompt_s} s, logits {r['hash']}; {r['lm'][:130]}; the drive: {r['drive'][:110]}; "
          f"slow-mode episodes {r['episodes']}", flush=True)
    return r


A = arm("a-config", P84)
B = arm("b-keep32", P84, {"STRATA_DSV41_LM_KEEP": "32"})
ok = {
    "A: windows of 3 (the config)": re.search(r"in windows of 3", A["lm"]) is not None,
    "B: windows of 4": re.search(r"in windows of 4", B["lm"]) is not None,
    "B: the first token's logits are A's bit for bit": A["hash"] is not None and A["hash"] == B["hash"],
}
if isinstance(A["prompt_s"], float) and isinstance(B["prompt_s"], float):
    print(f"  the ~120K read: {A['prompt_s']:.1f} s in windows of 3, {B['prompt_s']:.1f} s in windows of 4 "
          f"(x{A['prompt_s'] / B['prompt_s']:.2f}); slow-mode episodes {A['episodes']} / {B['episodes']} (a run with one is no "
          f"speed comparison)", flush=True)
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
