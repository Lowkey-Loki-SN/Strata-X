"""serve_cone_hang.py LOG_PREFIX EXE [EXE ...] - the audit of 2026-10-24: with STRATA_DSV41_REPLAY_CONE=1 (#108) a prompt call
with one read start whose last chunk leaves the chunk before it in the replay's tail (8,192 + 2,049-2,559 tokens) streamed
layers 21-39 for that chunk while the cone stopped its layer loop at the first layer that needs none of its rows (here 23):
the issuer held at a full ring, the chunk's last wait spun forever.  Through the engine's serve protocol, strata-dsv41.json's
arguments and switches, --no-adaptive, the 30 GiB test cache, for each EXE (the installed engine, then the fixed one):

 1  ~3,000 tokens of this repo's docs read from 0, one greedy token
 2  the same and ~10,400 tokens more - read from the live state at the first call's end, no checkpoint mark inside it
    (the next is 65,536): its 8,192 + 2,208-token plan's first chunk reaches 352 rows into the replay's 2,560
    - DONE, or a stall: no progress line for STALL_S seconds (the engine is then ended)"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, queue, re, subprocess, sys, threading, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, exes = sys.argv[1], sys.argv[2:]
STALL_S = float(os.environ.get("STALL_S", "150"))   # (a slow-mode episode holds a layer 60-79 s)
R = _REPO
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
git = lambda f: subprocess.run(["git", "-C", R, "show", f"645844e9:{f}"], capture_output=True, text=True, check=True).stdout
DOCS = tok.encode("".join(git(f) for f in ("docs/DETAILS.md", "moex/docs/DSV41_PLAN.md")))
P1 = DOCS[:3000]
P2 = P1 + DOCS[3000:13400]
ARGS = ["--serve"] + [a for a in cfg["args"]] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804",
                                                 "--no-adaptive"]
ARGS[ARGS.index("--ssd-cache-gib") + 1] = "30"   # the test runs' cache (the user's OK)
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_ARENA") or k.startswith("STRATA_X_STAGE"):
        ENV.pop(k)
ENV.update(cfg["env"])
assert ENV.get("STRATA_DSV41_REPLAY_CONE") == "1", "the config's cone is off: nothing to check"


def gen(p, lines, ids):
    """One GEN: -> ("DONE", seconds) or ("STALL", seconds since the last line) or ("ERR ...", s).  The engine's lines come
    through `lines` (a reader thread's queue: a select() on the pipe misses lines its buffered reader already holds)."""
    t0 = last = time.monotonic()
    p.stdin.write("GEN 1 temperature=0 " + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    while True:
        try:
            line = lines.get(timeout=5.0)
        except queue.Empty:
            if time.monotonic() - last > STALL_S:
                return "STALL", time.monotonic() - t0
            continue
        if line is None:
            return "EOF", time.monotonic() - t0
        last = time.monotonic()
        if line.startswith("DONE"):
            return "DONE", last - t0
        if line.startswith("ERR"):
            return line.strip(), last - t0


def reader(p, lines):
    for line in p.stdout:
        lines.put(line)
    lines.put(None)


results = {}
for i, exe in enumerate(exes):
    tag = f"{i}-{Path(exe).parent.name}"
    log = f"{prefix}-{tag}.log"
    err = open(log, "w")
    p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=ENV)
    lines = queue.Queue()
    threading.Thread(target=reader, args=(p, lines), daemon=True).start()
    while (line := lines.get()) is not None and not line.startswith("READY"):
        pass
    a = gen(p, lines, P1)
    b = gen(p, lines, P2) if a[0] == "DONE" else ("skipped", 0.0)
    if b[0] == "DONE":
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=180)
    else:
        p.kill()
        p.wait()
    err.close()
    s = Path(log).read_text(errors="replace")
    lm = re.findall(r"layer-major: .*", s)
    results[exe] = (a, b)
    print(f"  {tag}: call 1 ({len(P1):,} tokens) {a[0]} in {a[1]:.1f} s; call 2 (+{len(P2) - len(P1):,} tokens) {b[0]} after "
          f"{b[1]:.1f} s; {lm[-1][:110] if lm else '(no layer-major line)'}", flush=True)
fixed = results[exes[-1]]
ok = {"the last engine: both calls DONE": fixed[0][0] == "DONE" and fixed[1][0] == "DONE"}
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
