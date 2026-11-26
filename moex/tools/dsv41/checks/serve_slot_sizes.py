"""serve_slot_sizes.py LOG_PREFIX EXE - DSV41_PLAN §24: STRATA_DSV41_SLOT_SIZES=1 (every VRAM slot its expert's size, not the
largest blob's) through the engine's serve protocol at strata-dsv41.json's arguments and switches, the route bias off and
--no-adaptive (every arm routes the same experts), a 2,000-token text (the top of moex/docs/DSV41_PLAN.md at 645844e9) and a
question, then 64 greedy tokens:

 A  uniform slots, at most 945 on GPU 0 and 755 on GPU 1 (STRATA_X_STAGE_SLOTS=945,755: the counts the uniform layout fits)
 B  sized slots, the same caps: the same experts in VRAM at other addresses - A's first-token logits and tokens, bit for bit
 C  sized slots, no cap: how many fit in the same VRAM"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, re, subprocess, sys, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, exe = sys.argv[1], sys.argv[2]
R = _REPO
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
text = subprocess.run(["git", "-C", R, "show", "645844e9:moex/docs/DSV41_PLAN.md"], capture_output=True, text=True, check=True).stdout
P = tok.encode(text)[:2000] + tok.encode("\n\nExplain in detail how the plan above reads the experts from the drive.<｜Assistant｜>"
                                         "</think>", parse_special=True)
ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--no-adaptive"]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_ARENA") or k.startswith("STRATA_X_STAGE"):
        ENV.pop(k)
ENV.update({k: v for k, v in cfg["env"].items() if k != "STRATA_DSV41_ROUTE_BIAS"})


def arm(tag, extra, args=()):
    log = f"{prefix}-{tag}.log"
    err = open(log, "w")
    p = subprocess.Popen([exe] + ARGS + list(args), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                         env=dict(ENV, **extra))
    for line in p.stdout:
        if line.startswith("READY"):
            break
    p.stdin.write("GEN 64 temperature=0 " + ",".join(map(str, P)) + "\n")
    p.stdin.flush()
    toks, done = [], None
    for line in p.stdout:
        if line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith("DONE") or line.startswith("ERR"):
            done = line.strip()
            break
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    s = Path(log).read_text(errors="replace")
    t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
    slots = re.findall(r"GPU (\d): (\d+) expert slots([^,]*)", s)
    dec = re.findall(r"serve: decode: (.*)", s)
    r = {"hash": t[-1][2] if t else None, "toks": toks, "done": done, "slots": [(g, n) for g, n, _ in slots],
         "dec": dec[-1] if dec else "-"}
    f = done.split() if done and done.startswith("DONE") else []
    tps = f"{len(toks) / (float(f[4]) / 1000):.2f} tok/s" if len(f) > 4 and float(f[4]) > 0 else "?"
    print(f"  {tag}: slots {r['slots']}; logits {r['hash']}; {len(toks)} tokens, {tps}; {r['dec'][:200]}", flush=True)
    return r


A = arm("a-uniform", {"STRATA_X_STAGE_SLOTS": "945,755"}, ["--expert-slots", "1700"])
B = arm("b-sized-capped", {"STRATA_DSV41_SLOT_SIZES": "1", "STRATA_X_STAGE_SLOTS": "945,755"}, ["--expert-slots", "1700"])
C = arm("c-sized", {"STRATA_DSV41_SLOT_SIZES": "1"})
ok = {
    "A and B hold the same counts (945 + 755)": A["slots"] == B["slots"] == [("0", "945"), ("1", "755")],
    "B: the first token's logits are A's bit for bit": A["hash"] is not None and A["hash"] == B["hash"],
    "B: the 64 greedy tokens are A's": len(A["toks"]) == 64 and A["toks"] == B["toks"],
    "C: more slots than the uniform layout fits": sum(int(n) for _, n in C["slots"]) > 945 + 755,
}
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
