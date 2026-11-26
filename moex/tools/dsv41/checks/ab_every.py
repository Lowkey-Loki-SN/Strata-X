"""ab_every.py (serve_arm_prod.py with ARM_ARGS: extra engine flags for the arm) LOG (the production binary) PROMPT N SEED - one arm of the cold-read A/B, with the long-generation loop test riding on it when N > 1:
a fresh engine (the dev build, the served pack (q3q2u, 212 GB), the production flags and the learned expert profile - read, never saved) reads
PROMPT (a JSON {"ids": [...]}) cold and generates N tokens at the release's sampling (temperature 1.0, top_k 0: the whole
vocabulary) with SEED.  The arm's switches come from the environment (STRATA_DSV41_EG_DIRECT, STRATA_DSV41_GATE_AHEAD,
STRATA_DSV41_ROUTE_BIAS); STRATA_X_FIRST_TOP=8 puts the first token's distribution in LOG, so the arms can be checked equal.
Prints the engine's prompt-path line, the first token's top-8, decode tok/s per 512 tokens, and the server's loop measure
(reasoning_repeat_coverage, #728: its guard stops at 0.25) on the reasoning and the answer, every 512 tokens and at the end."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, subprocess, sys, time
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from pathlib import Path
from serve import families
from serve.server import reasoning_repeat_coverage

log, prompt, n_new, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
S = os.environ.get("STRATA_CHECK_DIR", _os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks'))
ids = json.load(open(prompt))["ids"]
cfg = json.loads(Path(_os.path.join(_REPO, 'strata-dsv41.json')).read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
M = _os.path.join(_MODELS, 'dsv41-1017')
cmd = [os.environ.get("ARM_EXE", _os.path.join(_REPO, 'build-dsv41/strata-x')), "--serve", "-m", f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf", "--pack",
       f"{M}/ud-iq3xxs", "--ssd-cache-gib", "60", "--max-context", "262144", "--threads", "20", "--short-read", "96", "--layer-split",
       "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804",
       "--expert-profile", _os.path.join(_DATA, 'dsv41-expert-profile-learned.txt')] + os.environ.get("ARM_ARGS", "").split()
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_DSV41_HEADS_MMVQ="1",
           STRATA_DSV41_HC_SPLIT="1", STRATA_DSV41_PROMPT_KEEP="1", STRATA_X_FIRST_TOP="8")
print("arm env:", {k: v for k, v in env.items() if k.startswith(("STRATA_DSV41_EG", "STRATA_DSV41_GATE", "STRATA_DSV41_ROUTE"))},
      flush=True)
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "w"), text=True, bufsize=1, env=env)
for line in p.stdout:
    if line.startswith("READY"):
        break
think_end = tok.encode("</think>", parse_special=True)


def split(out):
    cut = next((i for i in range(len(out)) if out[i:i + len(think_end)] == think_end), len(out))
    return tok.decode(out[:cut]), tok.decode(out[cut + len(think_end):]) if cut < len(out) else "", cut


out, worst, done, marks = [], {"reasoning": (0.0, 0), "answer": (0.0, 0)}, "", []
t0 = time.time()
p.stdin.write(f"GEN {n_new} temperature=1.0 top_k=0 seed={seed} " + ",".join(map(str, ids)) + "\n")
p.stdin.flush()
for line in p.stdout:
    if line.startswith("T "):
        out.append(int(line.split()[1]))
        if len(out) == 1:
            marks.append((1, time.time()))
        if len(out) % 512 == 0:
            marks.append((len(out), time.time()))
            r, a, _ = split(out)
            for k, text in (("reasoning", r), ("answer", a)):
                c = reasoning_repeat_coverage(text)
                if c > worst[k][0]:
                    worst[k] = (c, len(out))
        continue
    if line.startswith("DONE") or line.startswith("ERR"):
        done = line.strip()
        break
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=180)
for line in open(log):
    if "prompt path:" in line or "first-token top" in line or "first sampled" in line:
        print(line.strip()[:700], flush=True)
print(done, flush=True)
if n_new > 1:
    rates = [f"{(b - a) / max(1e-9, tb - ta):.1f}" for (a, ta), (b, tb) in zip(marks, marks[1:])]
    print(f"decode tok/s per 512 (from the first token): {' '.join(rates)}", flush=True)
    r, a, cut = split(out)
    fin = {"reasoning": reasoning_repeat_coverage(r), "answer": reasoning_repeat_coverage(a)}
    ok = max(max(v[0] for v in worst.values()), max(fin.values())) < 0.25
    print(f"seed {seed}: {len(out)} tokens ({cut} of reasoning, {len(r.split())} words; answer {len(a.split())} words); repeat "
          f"coverage reasoning worst {worst['reasoning'][0]:.3f} at {worst['reasoning'][1]}, final {fin['reasoning']:.3f}; answer "
          f"worst {worst['answer'][0]:.3f} at {worst['answer'][1]}, final {fin['answer']:.3f} -> {'PASS' if ok else 'LOOP'}",
          flush=True)
    open(f"{S}/arm_out_s{seed}.json", "w").write(json.dumps(out))
