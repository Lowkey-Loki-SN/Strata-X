"""serve_park.py LOG_PREFIX EXE - DSV41_PLAN §57: parked conversations through the serve protocol.

strata-dsv41.json's arguments and switches (the route bias off, --no-adaptive, the slots pinned per GPU by an outer
STRATA_X_STAGE_SLOTS, the 30 GiB test cache), greedy, the first token's logits by hash (STRATA_X_FIRST_TOP).  A is ~16K
tokens of this repo's plan and a question, B ~10K of its details and another; each follow-up (A2, B2) is its prompt, a fixed
reply and a new question - fixed, not the generated tokens, because only a request's first token is the same bits run to
run (a near-tie after it can go either way, serve_decode_ab.py) - so it resumes at its conversation's last-turn checkpoint.
  arm U  (--conversation-cache-mib 4096): A, A2                      the uninterrupted follow-up
  arm P  (--conversation-cache-mib 4096): A, B, A2, B2               A parked for B, put back for A2; then B the same way
  arm R  (--conversation-cache-mib 0):    A, B, A2                   today's engine: A2 reads A again
Pass: arm P's A2 resumes where arm U's does and its first token's logits are the same bits; B2 resumes at B's checkpoint;
arm R's A2 reads from the start (what parking saves).  Prints each request's resume, read time, and the park / restore
lines (their MiB and milliseconds)."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, re, subprocess, sys, time
from pathlib import Path

sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families  # noqa: E402

prefix, exe = sys.argv[1], sys.argv[2]
R = _REPO
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
sp = lambda s: tok.encode(s, parse_special=True)
git = lambda f: subprocess.run(["git", "-C", R, "show", f"645844e9:{f}"], capture_output=True, text=True, check=True).stdout
ask = lambda q: sp(f"\n\n{q}<｜Assistant｜></think>")
P_A = sp("<｜User｜>") + tok.encode(git("moex/docs/DSV41_PLAN.md"))[:16000] + ask("Summarise the plan above in one sentence.")
P_B = sp("<｜User｜>") + tok.encode(git("docs/DETAILS.md"))[:10000] + ask("What does the text above describe? One sentence.")
reply = lambda q: sp(f"It is a plan for running the model on this PC.<｜end▁of▁sentence｜><｜User｜>{q}<｜Assistant｜></think>")
A2 = P_A + reply("Which part of it reads the experts? One sentence.")
B2 = P_B + reply("Name one setting it describes. One sentence.")

args = list(cfg["args"])
i = args.index("--ssd-cache-gib")
args[i + 1] = "30"   # the test runs' cache
ARGS0 = ["--serve"] + args + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--image-pad",
                              "129264", "--no-adaptive", "--expert-slots", os.environ.get("PARK_SLOTS", "1700")]
ENV = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_DSV4", "STRATA_MMVQ"))}
ENV.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
ENV.update({k: v for k, v in cfg["env"].items() if k != "STRATA_DSV41_ROUTE_BIAS"})   # every arm routes the same experts


def run_arm(name, mib, seq):
    log = f"{prefix}-{name}.log"
    err = open(log, "w")
    p = subprocess.Popen([exe] + ARGS0 + ["--conversation-cache-mib", str(mib)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=err, text=True, bufsize=1, env=ENV)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    out = []
    for label, ids in seq:
        mark = len(Path(log).read_text(errors="replace"))
        t0 = time.time()
        p.stdin.write("GEN 8 temperature=0 " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        resume, toks, done = None, [], ""
        for line in p.stdout:
            if line.startswith("RESUME "):
                resume = int(line.split()[1])
            elif line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                done = line.strip()
                break
        err.flush()
        s = Path(log).read_text(errors="replace")[mark:]
        top = re.findall(r"\| logits ([0-9a-f]{16})", s)
        cache = re.findall(r"conversation cache: (.*)", s)
        out.append({"label": label, "n": len(ids), "resume": resume, "s": time.time() - t0, "hash": top[-1] if top else None,
                    "toks": toks, "cache": cache, "err": done if done.startswith("ERR") else ""})
        r = out[-1]
        print(f"  {name} {label}: {r['n']} tokens, resumed at {r['resume']}, {r['s']:.1f} s, logits {r['hash']} {r['err']}", flush=True)
        for c in cache:
            print(f"      conversation cache: {c[:200]}", flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    return {r["label"]: r for r in out}


U = run_arm("U", 4096, [("A", P_A), ("A2", A2)])
P = run_arm("P", 4096, [("A", P_A), ("B", P_B), ("A2", A2), ("B2", B2)])
Rr = run_arm("R", 0, [("A", P_A), ("B", P_B), ("A2", A2)])
ok_resume = P["A2"]["resume"] == U["A2"]["resume"] and (P["A2"]["resume"] or 0) > 15000
ok_bits = P["A2"]["hash"] is not None and P["A2"]["hash"] == U["A2"]["hash"]
ok_b2 = (P["B2"]["resume"] or 0) > 9000
ok_parked = any("parked" in c for c in P["B"]["cache"]) and any("restored" in c for c in P["A2"]["cache"])
saves = (Rr["A2"]["resume"] or 0) < 15000
for ok, what in ((ok_parked, "arm P: A parked when B came, put back for A2"),
                 (ok_resume, f"arm P's A2 resumes where arm U's does ({P['A2']['resume']} / {U['A2']['resume']})"),
                 (ok_bits, f"... and its first token's logits are the same bits ({P['A2']['hash']} / {U['A2']['hash']})"),
                 (ok_b2, f"arm P's B2 resumes at B's checkpoint ({P['B2']['resume']})"),
                 (saves, f"arm R (no parking): A2 reads from {Rr['A2']['resume']}, {Rr['A2']['s']:.1f} s against arm P's "
                         f"{P['A2']['s']:.1f} s")):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}", flush=True)
print("ALL PASS" if ok_parked and ok_resume and ok_bits and ok_b2 and saves else "SOME FAILED", flush=True)
