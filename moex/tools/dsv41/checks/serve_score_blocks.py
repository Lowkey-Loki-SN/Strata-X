"""serve_score_blocks.py LOG_PREFIX TODAY_EXE NEW_EXE - DSV41_PLAN §19 (DSV41_BACKLOG #84): the indexer scored in blocks of 64 rows
(kDsv41ScoreRows) at the server's context (--max-context 524288), through the engine's serve protocol, one tier
(--no-adaptive --expert-slots 1700: what 512K leaves room for), the config's switches, the engram tables left in the page
cache (the arms compare the record stream, not cold engram reads).

 A  today's build, 16,398 tokens (moex/docs/DSV41_PLAN.md at 645844e9 and a question) read from 0: the first token's logits
 B  the new one, the same: bit for bit with A
 C  the new one, ~84.5K tokens of this repo's docs (serve_cold_start.py's text), the chunk loop (STRATA_DSV41_LAYER_MAJOR=0)
 D  the new one, the same with layer-major: its windows (the 9.4K-token chunks did not fit a window of 2 before), its time
    against C's, and its logits bit for bit with C's"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, subprocess, sys, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, today_exe, new_exe = sys.argv[1], sys.argv[2], sys.argv[3]
R = _REPO
M = _os.path.join(_MODELS, 'dsv41-1017')
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
git = lambda f: subprocess.run(["git", "-C", R, "show", f"645844e9:{f}"], capture_output=True, text=True, check=True).stdout
Q = tok.encode("\n\nIn one sentence: what does the text above do?<｜Assistant｜></think>", parse_special=True)
P16 = tok.encode(git("moex/docs/DSV41_PLAN.md"))[:16384] + Q
P84 = tok.encode("".join(git(f) for f in ("docs/DETAILS.md", "moex/docs/DSV41_PLAN.md", "moex/docs/DSV41_PORT.md",
                                          "docs/AI_SETUP.md")))[:84400] + Q
ARGS = ["--serve", "-m", f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf", "--pack", f"{M}/ud-iq3xxs", "--ssd-cache-gib", "30",
        "--max-context", "524288", "--threads", "20", "--short-read", "96", "--prompt-cache-every", "65536",
        "--prompt-cache-deep", "8", "--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804",
        "--no-adaptive", "--expert-slots", "1700", "--expert-profile", _os.path.join(_DATA, 'dsv41-expert-profile-learned.txt')]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4"):
        ENV.pop(k)
ENV.update(cfg["env"])


def arm(tag, exe, ids, extra=None):
    log = f"{prefix}-{tag}.log"
    err = open(log, "w")
    p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                         env=dict(ENV, **(extra or {})))
    for line in p.stdout:
        if line.startswith("READY"):
            break
    p.stdin.write("GEN 1 temperature=0 " + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    prompt_s = None
    for line in p.stdout:
        if line.startswith("DONE") or line.startswith("ERR"):
            f = line.split()
            prompt_s = float(f[3]) / 1000 if line.startswith("DONE") else line.strip()
            break
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    s = Path(log).read_text(errors="replace")
    t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
    lm = re.findall(r"layer-major: .*", s)
    eg = re.findall(r"waited ([0-9.]+) s for their engram rows", s)
    drive = re.findall(r"the drive meanwhile: ([^;]*)", s)
    guard = [l for l in s.splitlines() if "the drive guard: the drive's slow mode" in l]
    r = {"hash": t[-1][2] if t else None, "prompt_s": prompt_s, "lm": lm[-1] if lm else "-", "drive": drive[-1] if drive else "-",
         "episodes": len(guard)}
    print(f"  {tag}: {len(ids):,} tokens, prompt {prompt_s} s, logits {r['hash']}; {r['lm'][:120]}; the drive: {r['drive'][:110]}; "
          f"slow-mode episodes {r['episodes']}", flush=True)
    return r


A = arm("a-today-16k", today_exe, P16)
B = arm("b-new-16k", new_exe, P16)
C = arm("c-new-84k-chunks", new_exe, P84, {"STRATA_DSV41_LAYER_MAJOR": "0"})
D = arm("d-new-84k-lm", new_exe, P84)
ok = {
    "B: the indexer in blocks is today's bit for bit (16K at 512K context)": A["hash"] is not None and A["hash"] == B["hash"],
    "D: the 84K read runs layer-major (windows of 2 or more)": re.search(r"in windows of [2-9]", D["lm"]) is not None,
    "D: layer-major's 84K read is the chunk loop's bit for bit": C["hash"] is not None and C["hash"] == D["hash"],
}
if isinstance(C["prompt_s"], float) and isinstance(D["prompt_s"], float):
    print(f"  the 84K read: {C['prompt_s']:.1f} s in the chunk loop, {D['prompt_s']:.1f} s layer-major (x{C['prompt_s'] / D['prompt_s']:.2f}); "
          f"slow-mode episodes {C['episodes']} / {D['episodes']} (a run with one is no speed comparison)", flush=True)
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
