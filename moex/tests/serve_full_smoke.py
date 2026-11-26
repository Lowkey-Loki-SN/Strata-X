#!/usr/bin/env python3
"""A full-model serve smoke test with a long prompt: read speed from the PP lines, decode speed, reuse of the conversation,
a resume from a checkpoint after a divergence.  The GPUs must be free; loads the 111 GB model (about a minute)."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_protocol_test import Engine, X, DATA
from collections import Counter

toks = [int(x) for x in open(f"{DATA}/glm-prompt-20k.tok").read().strip().split(",")]
turn = Counter(toks[2000:]).most_common(1)[0][0]
t0 = time.time()
eng = Engine(os.path.join(X, "build/strata-x"), _os.path.join(_MODELS, 'GGUFs/glm53-flash/glm53-flash-q2k-fast-111.gguf'),
             ["--turn-tokens", str(turn), "--expert-profile", f"{DATA}/glm-expert-profile.txt"], base=("--ctx", "16384"))
print(f"loaded in {time.time() - t0:.0f} s; info {eng.info}", flush=True)
prompt = toks[:12000]
t = time.time()
r = eng.gen(prompt, 64)
dt = time.time() - t
done = r["done"]
print(f"request 1: {len(prompt)} prompt tokens, resume {r['resume']}, PP lines {len(r['pp'])}, first chunk {r['pp'][0] if r['pp'] else None}, "
      f"prompt {float(done[2])/1000:.1f} s ({len(prompt)/(float(done[2])/1000):.0f} T/s), decode {done[0]} tokens in {float(done[3])/1000:.1f} s "
      f"({int(done[0])/(float(done[3])/1000):.1f} T/s), hits {done[8]}/{done[9]}, wall {dt:.1f} s", flush=True)
conv = prompt + r["tokens"][:-1] + [r["tokens"][-1]] + toks[12000:12040]
t = time.time()
r2 = eng.gen(conv, 64)
d2 = r2["done"]
print(f"request 2 (the conversation continued): resume {r2['resume']} (held {len(prompt) + len(r['tokens']) - 1}), read {d2[13]} tokens in {float(d2[2])/1000:.2f} s, "
      f"decode {int(d2[0])/(float(d2[3])/1000):.1f} T/s", flush=True)
turns = [i for i, x in enumerate(prompt) if x == turn and i > 2000]
div = prompt[: turns[-3] + 3] + [x ^ 1 for x in prompt[turns[-3] + 3: turns[-3] + 20]] + toks[12000:12010] if len(turns) >= 3 else None
if div:
    t = time.time()
    r3 = eng.gen(div, 32)
    d3 = r3["done"]
    print(f"request 3 (diverges after a turn token): resume {r3['resume']}, read {d3[13]} tokens in {float(d3[2])/1000:.2f} s", flush=True)
eng.close()
