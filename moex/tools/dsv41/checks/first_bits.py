"""first_bits.py EXE LOG - one prompt's first-token logits hash (STRATA_X_FIRST_TOP) through the --serve protocol, as
serve_park.py sets up an engine (strata-dsv41.json's arguments and switches, the route bias off, --no-adaptive, slots
pinned by the outer STRATA_X_STAGE_SLOTS and --expert-slots, the 30 GiB test cache): two binaries' hashes equal = the
same bits on q3q2u.  The prompt: 4,000 tokens of the plan (git 645844e9) and a question, greedy, 4 tokens."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, re, subprocess, sys, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families  # noqa: E402
exe, log = sys.argv[1], sys.argv[2]
R = _REPO
# CONFIG (strata-dsv41.json): the config whose arguments and switches the engine runs with (moex: the native one since the
# UD-IQ3_XXS pack it named was deleted, 2026-11-08)
cfg = json.loads(Path(os.environ.get("CONFIG", f"{R}/strata-dsv41.json")).read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
sp = lambda s: tok.encode(s, parse_special=True)
# PROMPT_N (4000): the prompt's length in the docs' tokens (past 65,536 it crosses a checkpoint cut: the tails' run);
# SWITCHES: NAME=V,... added to the engine's environment (the switches under test)
N = int(os.environ.get("PROMPT_N", "4000"))
docs = ["moex/docs/DSV41_PLAN.md"] + (["docs/DETAILS.md", "docs/AI_SETUP.md", "moex/docs/DSV41_PORT.md"] if N > 4000 else [])
text = []
for f in docs:
    text += tok.encode(subprocess.run(["git", "-C", R, "show", "645844e9:" + f], capture_output=True, text=True, check=True).stdout)
if len(text) < N:   # (DSV41_PLAN §63: past the docs' ~90K tokens, the commit's other docs and sources, in path order)
    more = subprocess.run(["git", "-C", R, "ls-tree", "-r", "--name-only", "645844e9"], capture_output=True, text=True,
                          check=True).stdout.split()
    for f in [p for p in more if p.endswith((".md", ".cpp", ".hpp", ".cu", ".py")) and p not in docs]:
        if len(text) >= N:
            break
        text += tok.encode(subprocess.run(["git", "-C", R, "show", "645844e9:" + f], capture_output=True, text=True,
                                          check=True).stdout)
if len(text) < N:
    sys.exit("the docs hold %d tokens, fewer than PROMPT_N %d" % (len(text), N))
ids = sp("<｜User｜>") + text[:N] + sp("\n\nSummarise the plan above in one sentence.<｜Assistant｜></think>")
args = list(cfg["args"])
args[args.index("--ssd-cache-gib") + 1] = os.environ.get("CACHE_GIB", "30")   # (§63: CACHE_GIB=60, the server's)
cmd = [exe, "--serve"] + args + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--image-pad",
                                 "129264", "--no-adaptive", "--expert-slots", "1700"] + os.environ.get("EXTRA_ARGS", "").split()
# EXTRA_ARGS (DSV41_PLAN §64): engine flags after the others (a later flag wins), e.g. "--prefill-chunk 12288"
env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_DSV4", "STRATA_MMVQ"))}
env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
env.update({k: v for k, v in cfg["env"].items() if k != "STRATA_DSV41_ROUTE_BIAS"})
for kv in filter(None, os.environ.get("SWITCHES", "").split(",")):
    k, v = kv.split("=", 1)
    env[k] = v
err = open(log, "w")
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=env)
for line in p.stdout:
    if line.startswith("READY"):
        break
def gen(ids, tag):
    mark = len(Path(log).read_text(errors="replace"))
    t0 = time.time()
    p.stdin.write("GEN 4 temperature=0 " + ",".join(map(str, ids)) + "\n"); p.stdin.flush()
    toks, resume, done = [], None, ""
    for line in p.stdout:
        if line.startswith("RESUME "):
            resume = int(line.split()[1])
        elif line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith(("DONE", "ERR")):
            done = line.strip(); break
    err.flush()
    h = re.findall(r"\| logits ([0-9a-f]{16})", Path(log).read_text(errors="replace")[mark:])
    print(f"{exe} {tag}: {len(ids)} tokens, resumed at {resume}, {time.time() - t0:.1f} s, first-token logits "
          f"{h[-1] if h else None}, tokens {toks}, {done[:60]}", flush=True)

gen(ids, "read")
# RESUME_N: then the first RESUME_N tokens of the same text and another question - it resumes at the call's checkpoint cut
# below RESUME_N (65,536): the state the first read left there (a chunk end's copies, the tails' run)
RN = int(os.environ.get("RESUME_N", "0"))
if RN:
    gen(sp("<｜User｜>") + text[:RN] + sp("\n\nWhich parts of the text above read the experts?<｜Assistant｜></think>"), "resume")
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=180); err.close()
