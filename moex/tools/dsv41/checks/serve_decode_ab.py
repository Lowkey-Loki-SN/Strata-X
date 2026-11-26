"""serve_decode_ab.py LOG_PREFIX EXE ARM ... - a decode A/B through the serve protocol at the server's context (512K, one tier:
--no-adaptive --expert-slots 1700), the config's switches with the route bias OFF (so every arm routes the same experts and
must give the same tokens), each ARM a fresh engine (K=V,K=V... environment overrides; "-" for none): a 2,000-token text (the
top of moex/docs/DSV41_PLAN.md at 645844e9) and a question, then N greedy tokens (DECODE_N, 256) - a warm-up pass and a
measured one (the same request again: the prompt reused, the SSD cache warmed by the first pass).  Prints each arm's two
passes' tok/s (the DONE line's decode time) and the engine's decode line, and checks every arm's tokens and first-token
logits equal."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, subprocess, sys
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, exe, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
N = int(os.environ.get("DECODE_N", "256"))
R = _REPO
M = _os.path.join(_MODELS, 'dsv41-1017')
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
text = subprocess.run(["git", "-C", R, "show", "645844e9:moex/docs/DSV41_PLAN.md"], capture_output=True, text=True, check=True).stdout
# DECODE_PROMPT_N=N (moex, DSV41_PLAN §56 C): N tokens of the text, not 2,000 - past 16,384 the index sources after layer 20
# select among 2,048 candidate blocks, which STRATA_DSV41_CAND_ONLY=1 changes how they score
# DECODE_QUESTION=TEXT (moex, DSV41_BACKLOG #125): another question after the text - e.g. one whose answer copies it, for
# the prompt lookup's drafts
Q = os.environ.get("DECODE_QUESTION", "Explain in detail how the plan above reads the experts from the drive.")
P = tok.encode(text)[:int(os.environ.get("DECODE_PROMPT_N", "2000"))] + tok.encode(f"\n\n{Q}<｜Assistant｜></think>",
                                                                                 parse_special=True)
# DECODE_CACHE=G: the SSD cache's GiB (30 by default, the test runs'; the server's is 60 - the RAM copy takes what is left)
ARGS = ["--serve", "-m", f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf", "--pack", os.environ.get("PACK") or f"{M}/ud-iq3xxs", "--ssd-cache-gib", os.environ.get("DECODE_CACHE", "30"),
        "--max-context", "524288", "--threads", "20", "--short-read", "96", "--layer-split", "20", "--eos-ids", "1",
        "--turn-tokens", "128799,128803,128804", "--no-adaptive", "--expert-slots", os.environ.get("DECODE_SLOTS", "1700"),
        "--expert-profile", _os.path.join(_DATA, 'dsv41-expert-profile-learned.txt')] + os.environ.get("EXTRA_ARGS", "").split()
# EXTRA_ARGS: engine flags added to every arm (a later flag wins: e.g. "--threads 19", DSV41_BACKLOG #69)
# DECODE_SLOTS=N: the VRAM tier's slots (1700 by default: what the server's 512K context leaves room for, both GPUs)
if os.environ.get("DECODE_ADAPTIVE") == "1":   # the VRAM tier's swaps on, as the server runs: its slots then follow the decode
    ARGS = [a for a in ARGS if a != "--no-adaptive"]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_MMVQ"):
        ENV.pop(k)
BIAS = os.environ.get("DECODE_BIAS") == "1"   # the config's route bias kept: the decode the server runs (its tokens then
ENV.update({k: v for k, v in cfg["env"].items() if BIAS or k != "STRATA_DSV41_ROUTE_BIAS"})   # differ run to run: no check)
PASSES = int(os.environ.get("DECODE_PASSES", "2"))

res = []
for i, a in enumerate(arms):
    extra = dict(kv.split("=", 1) for kv in a.split(",")) if a != "-" else {}
    arm_args = extra.pop("_ARGS", "").split()   # _ARGS=FLAGS (moex): engine flags of this arm only, after the others
    log = f"{prefix}-arm{i}.log"
    err = open(log, "w")
    p = subprocess.Popen([exe] + ARGS + arm_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True,
                         bufsize=1, env=dict(ENV, **extra))
    for line in p.stdout:
        if line.startswith("READY"):
            break
    passes = []
    for _ in range(PASSES):
        p.stdin.write(f"GEN {N} temperature=0 " + ",".join(map(str, P)) + "\n")
        p.stdin.flush()
        toks = []
        for line in p.stdout:
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                f = line.split()
                passes.append((toks, float(f[4]) / 1000 if line.startswith("DONE") and len(f) > 4 else None, line.strip()))
                break
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    s = Path(log).read_text(errors="replace")
    t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
    dec = re.findall(r"serve: decode: (.*)", s)
    r = {"arm": a, "hash": t[-1][2] if t else None, "tokens": [x[0] for x in passes], "s": [x[1] for x in passes], "dec": dec[-1] if dec else "-",
         "done": [x[2] for x in passes]}
    res.append(r)
    tps = [f"{len(tk) / sec:.2f}" if sec else "?" for tk, sec in zip(r["tokens"], r["s"])]
    print(f"  arm {i} ({a}): logits {r['hash']}; tok/s {' / '.join(tps)} (passes 1..{PASSES}); {r['done'][-1][:90]}", flush=True)
    for dl in re.findall(r"serve: decode: (.*)", s):
        print(f"      decode: {dl[:240]}", flush=True)
# the first token's logits come out of a decode window (the VRAM experts on the GPU, the rest in the CPU pool): equal in every
# arm.  The tokens after it: the CPU pool adds a window's experts in the order their records land (a run's timing, PLAN §14),
# so a near-tie can go either way between any two runs - they are printed, not checked
same = all(r["hash"] == res[0]["hash"] for r in res) and res[0]["hash"] is not None
print(f"  {'PASS' if same else 'FAIL'}  every arm's first-token logits are the same", flush=True)
print(f"  the greedy tokens {'agree' if all(r['tokens'] == res[0]['tokens'] for r in res) else 'part'} between the arms", flush=True)
for i, r in enumerate(res[1:], start=1):   # where each arm's tokens first differ from arm 0's, pass by pass
    firsts = []
    for a, b in zip(res[0]["tokens"], r["tokens"]):
        d = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
        firsts.append("same" if d is None and len(a) == len(b) else str(d if d is not None else min(len(a), len(b))))
    print(f"  arm {i} against arm 0: the first differing token, passes 1..{PASSES}: {' / '.join(firsts)}", flush=True)
