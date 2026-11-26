"""serve_tail_batch.py LOG_PREFIX EXE - DSV41_PLAN §30: STRATA_DSV41_TAIL_BATCH=1 (the bounded replay's tails of a call run
together after its chunks, each replayed layer streamed once for all of them) through the engine's serve protocol at
strata-dsv41.json's arguments and switches (512K context, the 60 GiB cache, a checkpoint every 65,536 tokens), --no-adaptive
(both arms place the same experts in VRAM, and the first token comes out of a decode window), the engram tables left in the
page cache:

 A  the config: ~84.4K tokens of this repo's docs read from 0 (two tails: the cut at 65,536 and the end), then a second
    request that shares the first 65,536 tokens (it resumes at that checkpoint) and asks another question
 B  the same with STRATA_DSV41_TAIL_BATCH=1: both requests' first-token logits bit for bit with A's - the read's end state
    and the checkpoint's state, put together from the tails' run - and the read's time and drive bytes against A's"""
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
        t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
        rd = re.findall(r"serve: \d+ prompt tokens \((\d+) reused, (\d+) read\) in ([0-9.]+) s", s)
        drive = re.findall(r"the drive meanwhile: ([^;]*)", s)
        tails = re.findall(r"the deferred tails: [^\n]*", s)
        res.append({"hash": t[-1][2] if t else None, "resume": resume, "done": done, "read": rd[-1] if rd else None,
                    "drive": drive[-1] if drive else "-", "tails": tails[-1] if tails else "-",
                    "episodes": s.count("the drive guard: the drive's slow mode")})
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    for i, r in enumerate(res):
        print(f"  {tag} request {i + 1}: resume {r['resume']}, read {r['read']}, logits {r['hash']}; the drive: {r['drive'][:90]}; "
              f"{r['tails'][:110]}; slow-mode episodes {r['episodes']}", flush=True)
    return res


A = arm("a-config", {})
B = arm("b-tail-batch", {"STRATA_DSV41_TAIL_BATCH": "1"})
ok = {
    "B: the read's tails ran together": "pieces" in B[0]["tails"],
    "B: the read's first-token logits are A's bit for bit": A[0]["hash"] is not None and A[0]["hash"] == B[0]["hash"],
    "B: the second request resumed at the checkpoint (65,536)": B[1]["resume"] == 65536 and A[1]["resume"] == 65536,
    "B: ... and its first-token logits are A's bit for bit": A[1]["hash"] is not None and A[1]["hash"] == B[1]["hash"],
}
if A[0]["read"] and B[0]["read"]:
    a, b = float(A[0]["read"][2]), float(B[0]["read"][2])
    print(f"  the read: {a:.1f} s -> {b:.1f} s (x{a / b:.2f}); slow-mode episodes {A[0]['episodes']} / {B[0]['episodes']}", flush=True)
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
