"""serve_kv_pack_1m.py LOG_PREFIX EXE - the 1M context of 2026-10-24 (strata-dsv41.json: --max-context 1048576 with
STRATA_DSV41_KV_PACK=1, DSV41_PLAN §48): the packed rows had been through the server only on short reads (3-5K positions).
Through the engine's serve protocol, strata-dsv41.json's arguments and switches as they are, --no-adaptive, the 30 GiB test
cache, the engram tables left in the page cache:

 A  ~84.5K tokens of this repo's docs (serve_lm_windows.py's text) read from 0 with the rows as F16 (KV_PACK=0), the slots
    that fit, then 16 greedy tokens
 B  the same with the packed rows, the slots A held on each GPU (STRATA_X_STAGE_SLOTS: the packed rows leave more room,
    and another VRAM set would part the first token's logits - the first token comes out of a decode window): its logits
    and its 16 tokens bit for bit with A's
 C  the packed rows, the slots that fit, no prompt: the attention state and the slots the server will hold"""
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
P84 = tok.encode("".join(git(f) for f in ("docs/DETAILS.md", "moex/docs/DSV41_PLAN.md", "moex/docs/DSV41_PORT.md",
                                          "docs/AI_SETUP.md")))[:84400] + Q
ARGS = ["--serve"] + [a for a in cfg["args"]] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804",
                                                 "--no-adaptive"]
ARGS[ARGS.index("--ssd-cache-gib") + 1] = "30"   # the test runs' cache (the user's OK)
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_ARENA") or k.startswith("STRATA_X_STAGE"):
        ENV.pop(k)
ENV.update(cfg["env"])


def arm(tag, ids, extra=None, n_gen=16):
    log = f"{prefix}-{tag}.log"
    err = open(log, "w")
    p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                         env=dict(ENV, **(extra or {})))
    for line in p.stdout:
        if line.startswith("READY"):
            break
    toks, prompt_s = [], None
    if ids:
        p.stdin.write(f"GEN {n_gen} temperature=0 " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        for line in p.stdout:
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                prompt_s = float(line.split()[3]) / 1000 if line.startswith("DONE") else line.strip()
                break
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    s = Path(log).read_text(errors="replace")
    t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
    state = re.findall(r"attention state ([0-9.]+) MiB for (\d+) positions", s)
    slots = [int(n) for n in re.findall(r"GPU \d: (\d+) expert slots", s)]
    lm = re.findall(r"layer-major: .*", s)
    r = {"hash": t[-1][2] if t else None, "prompt_s": prompt_s, "toks": toks, "state": state[-1] if state else None,
         "slots": slots, "lm": lm[-1] if lm else "-"}
    print(f"  {tag}: state {r['state']}, slots {slots}; {len(ids):,} tokens, prompt {prompt_s} s, logits {r['hash']}, "
          f"tokens {toks}; {r['lm'][:120]}", flush=True)
    return r


A = arm("a-f16", P84, {"STRATA_DSV41_KV_PACK": "0"})
cap = ",".join(map(str, A["slots"]))
B = arm("b-packed", P84, {"STRATA_DSV41_KV_PACK": "1", "STRATA_X_STAGE_SLOTS": cap})
C = arm("c-packed-uncapped", [], {"STRATA_DSV41_KV_PACK": "1"})
ok = {
    "A: 1,048,576 positions": A["state"] is not None and A["state"][1] == "1048576",
    "A and B: the read finished": isinstance(A["prompt_s"], float) and isinstance(B["prompt_s"], float),
    "B: the slots A held": B["slots"] == A["slots"],
    "B: the first token's logits are A's bit for bit": A["hash"] is not None and A["hash"] == B["hash"],
    "B: the 16 greedy tokens are A's": len(A["toks"]) == 16 and A["toks"] == B["toks"],
    "C: more slots than A": sum(C["slots"]) > sum(A["slots"]),
}
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
