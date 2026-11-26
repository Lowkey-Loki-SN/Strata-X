"""serve_cancel.py LOG - the audit's cancel-path fixes (DSV41_BACKLOG #34) through the engine's serve protocol (the dev build), each
with a reference.  The tier is frozen (--no-adaptive) so a re-read computes every expert where it did before: equal states give
bitwise-equal first-token distributions (STRATA_X_FIRST_TOP=8, 5 decimals in the log).

 (b) a short read stopped on its LAST window, then sent again: the engine must resume where the windows got (n - 1), and the
     first token's distribution must equal an uncancelled run's.  The read starts at an odd position (1001) and runs 89 rows,
     past the ratio-2 compressor ring's 16 rows.
 (a) a long prompt stopped between two prompt-path chunks (not at a checkpoint), then a short continuation that shares the
     tokens read: the engine must not resume at the chunk end (layers 21-39 are inexact there: bounded replay) but at the last
     checkpoint, and the continuation's distribution must match a fresh read of it."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, subprocess, sys, time
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from pathlib import Path
from serve import families
log = sys.argv[1]
S = os.environ.get("STRATA_CHECK_DIR", _os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks'))
P = json.load(open(f"{S}/long-44k-prompt.json"))["ids"]
cfg = json.loads(Path(_os.path.join(_REPO, 'strata-dsv41.json')).read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
M = _os.path.join(_MODELS, 'dsv41-1017')
cmd = [_os.path.join(_REPO, 'build-dsv41-dev/strata-x'), "--serve", "-m", f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf", "--pack",
       f"{M}/ud-iq3xxs", "--ssd-cache-gib", "60", "--max-context", "65536", "--threads", "20", "--short-read", "96", "--layer-split",
       "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--no-adaptive",
       "--expert-profile", _os.path.join(_DATA, 'dsv41-expert-profile-learned.txt')]
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_DSV41_HEADS_MMVQ="1",
           STRATA_DSV41_HC_SPLIT="1", STRATA_DSV41_PROMPT_KEEP="1", STRATA_X_FIRST_TOP="8")
err = open(log, "w")
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=env)
for line in p.stdout:
    if line.startswith("READY"):
        break


def tops():
    err.flush()
    return re.findall(r"first-token top at position (\d+):(.*)", Path(log).read_text())


def gen(ids, n=1, stop_after=None):
    """One request; stop_after(pp_lines) -> True sends STOP.  Returns (resume, pp reached list, DONE fields, tokens, its top)."""
    before = len(tops())
    p.stdin.write(f"GEN {n} temperature=0 " + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    resume, pp, out, done, stopped = None, [], [], None, False
    for line in p.stdout:
        if line.startswith("RESUME"):
            resume = int(line.split()[1])
        elif line.startswith("PP "):
            pp.append(int(line.split()[1]))
            if stop_after and not stopped and stop_after(pp):
                p.stdin.write("STOP\n")
                p.stdin.flush()
                stopped = True
        elif line.startswith("T "):
            out.append(int(line.split()[1]))
        elif line.startswith("DONE") or line.startswith("ERR"):
            done = line.split()
            break
    t = tops()
    return resume, pp, done, out, (t[before:] or [None])[-1]


def reset():   # an unrelated prompt read from 0 (the next request shares no prefix with live, and no checkpoint matches)
    gen(P[30000:30200])


results = []
# ---- (b)
n1, n2 = 1001, 1091
reset()
gen(P[:n1])
for attempt in range(3):
    r, pp, done, _, _ = gen(P[:n2], 1, stop_after=lambda pp: pp[-1] == n2 - 2)
    if done and done[5] == "cancel" and pp and pp[-1] == n2 - 1:
        break
    print(f"(b) attempt {attempt}: the stop did not land on the last window (PP {pp[-3:]}, DONE {done}); again", flush=True)
    reset(); gen(P[:n1])
rc, _, _, outc, topc = gen(P[:n2], 4)
reset(); gen(P[:n1]); _, _, _, oute, tope = gen(P[:n2], 4)
ok_b = rc == n2 - 1 and topc is not None and tope is not None and topc[1] == tope[1] and outc == oute
print(f"(b) stopped on the last window (PP up to {pp[-1]}, finish {done[5]}); resent: resumed at {rc} (want {n2 - 1}); first token "
      f"{'bitwise equal' if topc and tope and topc[1] == tope[1] else 'DIFFERENT'} to an uncancelled run; tokens {outc} vs {oute} "
      f"-> {'PASS' if ok_b else 'FAIL'}", flush=True)
print(f"    cancelled-then-resent top: {topc}\n    uncancelled top:           {tope}", flush=True)
# ---- (a)
reset()
r, pp, done, _, _ = gen(P, 1, stop_after=lambda pp: len(pp) == 1)   # stage 0 is on chunk 2 then: the stop lands at its end
k = pp[-1] if pp else 0
root = next(i for i in range(2048, len(P)) if P[i] in (128799, 128803, 128804))
cuts = {root, 16384, 32768}
S_ids = tok.encode("\n\nIn one sentence: what does the code above do?<｜Assistant｜></think>", parse_special=True)
B = P[:k] + S_ids
rb, ppb, doneb, outb, topb = gen(B, 4)
reset(); rr, _, _, outr, topr = gen(B, 4)
close = topb is not None and topr is not None and topb[1] == topr[1]
print(f"(a) stopped between chunks at {k} (PP {pp}; a cut: {k in cuts}; the cuts {sorted(cuts)}), finish {done[5] if done else '?'}; "
      f"the continuation ({len(B)} tokens): resumed at {rb} (a checkpoint, not {k}: {rb != k and rb in cuts | {0}}), first token "
      f"{'bitwise equal' if close else 'different'} to a fresh read; tokens {outb} vs {outr}", flush=True)
print(f"    continuation top: {topb}\n    fresh read top:   {topr}", flush=True)
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=120)
