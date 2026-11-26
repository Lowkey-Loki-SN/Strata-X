"""serve_dm_alias.py LOG_PREFIX EXE - DSV41_PLAN §33 through the engine's serve protocol at strata-dsv41.json's arguments and
switches (512K context, the 60 GiB cache, a checkpoint every 65,536 tokens), --no-adaptive (every arm places the same experts
in VRAM, and the first token comes out of a decode window), the engram tables left in the page cache:

 A  the config: ~84.4K tokens of this repo's docs read from 0 (windows of 4), then a second request that shares the first
    65,536 tokens (it resumes at that checkpoint) and asks another question
 B  the same with STRATA_DSV41_DM_ALIAS=1 (the MoE's down rows over its gate+up rows, n floats apart: ~1 GB less a 9.4K
    chunk's loan) and STRATA_DSV41_LM_CHUNKS=8 (GPU 0's windows as large as its loan holds)
 C  B with STRATA_DSV41_RING_KEEP=1 (GPU 1's layer 20 left in its ring for the call's later chunks)
 Both requests' first-token logits bit for bit with A's in B and C; each read's plan, time, drive bytes, slow-mode episodes.
 ARMS=bc (default abc) runs only those; A_HASHES=<read>,<resume> stands in for an arm A run before (its two hashes);
 EXTRA_ENV=K=V,... adds to every arm's environment"""
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
Q2 = tok.encode("\n\nName the file the text above starts with.<｜Assistant｜></think>", parse_special=True)
D = tok.encode("".join(git(f) for f in ("docs/DETAILS.md", "moex/docs/DSV41_PLAN.md", "moex/docs/DSV41_PORT.md", "docs/AI_SETUP.md")))
P1 = D[:84400] + Q1
P2 = D[:65536] + Q2
ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--no-adaptive"]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_ARENA") or k.startswith("STRATA_X_STAGE") or k.startswith("STRATA_POOL"):
        ENV.pop(k)
ENV.update(cfg["env"])
for kv in filter(None, os.environ.get("EXTRA_ENV", "").split(",")):   # EXTRA_ENV=K=V,...: every arm's too (DSV41_PLAN §37)
    k, v = kv.split("=", 1)
    ENV[k] = v


def arm(tag, extra):
    log = f"{prefix}-{tag}.log"
    err = open(log, "w")
    p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                         env=dict(ENV, **extra))
    for line in p.stdout:
        if line.startswith("READY"):
            break
    res = []
    for P in (P1, P2):
        mark = len(Path(log).read_text(errors="replace"))
        p.stdin.write("GEN 1 temperature=0 " + ",".join(map(str, P)) + "\n")
        p.stdin.flush()
        resume, done = None, None
        for line in p.stdout:
            if line.startswith("RESUME"):
                resume = int(line.split()[1])
            elif line.startswith("DONE") or line.startswith("ERR"):
                done = line.strip()
                break
        err.flush()
        s = Path(log).read_text(errors="replace")[mark:]
        g = lambda pat: (re.findall(pat, s) or ["-"])[-1]
        t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
        rd = re.findall(r"serve: \d+ prompt tokens \((\d+) reused, (\d+) read\) in ([0-9.]+) s", s)
        res.append({"hash": t[-1][2] if t else None, "resume": resume, "done": done, "read": rd[-1] if rd else None,
                    "plan": g(r"layer-major: ([^\n]*)"), "drive": g(r"the drive meanwhile: ([^;]*)"),
                    "tails": g(r"the deferred tails: ([^\n]*)"), "streamed": g(r"(\d+ experts streamed: [^;]*)"),
                    "episodes": s.count("the drive guard: the drive's slow mode")})
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    for i, r in enumerate(res):
        print(f"  {tag} request {i + 1}: resume {r['resume']}, read {r['read']}, logits {r['hash']}; {r['plan'][:110]}; the drive: "
              f"{r['drive'][:90]}; {r['streamed'][:90]}; tails {r['tails'][:70]}; slow-mode episodes {r['episodes']}", flush=True)
    return res


ARMS = os.environ.get("ARMS", "abc")
none = [{"hash": None, "resume": None, "read": None, "plan": "-", "episodes": 0}, {"hash": None, "resume": 65536}]
if "a" in ARMS:
    A = arm("a-config", {})
else:   # an earlier run's arm A: its two hashes
    h = os.environ["A_HASHES"].split(",")
    A = [dict(none[0], hash=h[0], plan="in windows of 4 (an earlier run)"), dict(none[1], hash=h[1])]
B = arm("b-alias-windows", {"STRATA_DSV41_DM_ALIAS": "1", "STRATA_DSV41_LM_CHUNKS": "8"}) if "b" in ARMS else None
C = arm("c-ring-keep", {"STRATA_DSV41_DM_ALIAS": "1", "STRATA_DSV41_LM_CHUNKS": "8", "STRATA_DSV41_RING_KEEP": "1"}) if "c" in ARMS else None
ran = [X for X in (B, C) if X is not None]
ok = {
    "A: windows of 4 (the config)": "in windows of 4" in A[0]["plan"],
    "B / C: windows of more than 4": all(re.search(r"in windows of ([5-9])", X[0]["plan"]) is not None for X in ran),
    "both requests resumed where they should (0, 65,536)": all(x[0]["resume"] in (0, None) and x[1]["resume"] == 65536 for x in ran),
}
for tag, X in (("B", B), ("C", C)):
    if X is None:
        continue
    ok[f"{tag}: the read's first-token logits are A's bit for bit"] = A[0]["hash"] is not None and A[0]["hash"] == X[0]["hash"]
    ok[f"{tag}: the resume's first-token logits are A's bit for bit"] = A[1]["hash"] is not None and A[1]["hash"] == X[1]["hash"]
for tag, X in (("A", A), ("B", B), ("C", C)):
    if X is not None and X[0]["read"]:
        print(f"  {tag}: the read {float(X[0]['read'][2]):.1f} s; slow-mode episodes {X[0]['episodes']}", flush=True)
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
