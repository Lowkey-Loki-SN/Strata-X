"""serve_guard_decode.py LOG_PREFIX TODAY_EXE NEW_EXE - DSV41_PLAN §18: the decode at the drive guard's floor, bit for bit.  The
route bias (STRATA_DSV41_ROUTE_BIAS, decode only) picks among the experts resident at the time, and the RAM copy is sized
by MemAvailable at each start (46.35 and 50.44 GiB in serve_guard.py's arms A and B), so with it on two starts decode
differently; here it is off, and the decode reads the same records whatever their tier.  serve_guard.py's text and 32
greedy tokens: today's build, then the new one with the config's §18 switches and the guard on counters this script writes
- a healthy drive's until the read's last PP line, the slow mode's from there: the decode at the floor (one record reader,
128 KB a call, the engram rows' pages one at a time).  The first token's logits hash and the 32 tokens must be the same."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, subprocess, sys, threading, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, today_exe, new_exe = sys.argv[1], sys.argv[2], sys.argv[3]
M = _os.path.join(_MODELS, 'dsv41-1017')
cfg = json.loads(Path(_os.path.join(_REPO, 'strata-dsv41.json')).read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
text = subprocess.run(["git", "-C", _REPO, "show", "645844e9:moex/docs/DSV41_PLAN.md"], capture_output=True,
                      text=True, check=True).stdout
P = tok.encode(text)[:16384] + tok.encode("\n\nIn one sentence: what does the plan above do?<｜Assistant｜></think>",
                                          parse_special=True)
ARGS = ["--serve", "-m", f"{M}/dsv41-1017-dense-q8_0-dsp-eg.gguf", "--pack", f"{M}/ud-iq3xxs", "--ssd-cache-gib", "30",
        "--max-context", "65536", "--threads", "20", "--short-read", "96", "--layer-split", "20", "--eos-ids", "1",
        "--turn-tokens", "128799,128803,128804", "--no-adaptive", "--expert-slots", "1800",
        "--expert-profile", _os.path.join(_DATA, 'dsv41-expert-profile-learned.txt')]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4"):
        ENV.pop(k)
ENV.update({k: v for k, v in cfg["env"].items() if k not in ("STRATA_DSV41_EG_READER", "STRATA_DSV41_ROUTE_BIAS")})
ENV.update(STRATA_DSV41_LAYER_MAJOR="1", STRATA_DSV41_BOUNCE="96,32")


def run(tag, exe, extra, cpath=None):
    log = f"{prefix}-{tag}.log"
    err = open(log, "w")
    state = {"slow": False, "reads": 0, "ms": 0.0, "stop": False}

    def counters():   # the guard's counters: 8,000 reads a second at 0.2 ms, or the slow mode's 1,000 at 1,000 ms
        while not state["stop"]:
            n = 100 if state["slow"] else 800
            state["reads"] += n
            state["ms"] += n * (1000 if state["slow"] else 0.2)
            Path(cpath + ".tmp").write_text(f"{state['reads']} 0 0 {int(state['ms'])} 0 0 0 0 0 0 0\n")
            os.replace(cpath + ".tmp", cpath)
            time.sleep(0.1)

    if cpath:
        threading.Thread(target=counters, daemon=True).start()
        time.sleep(0.3)
    p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                         env=dict(ENV, **extra))
    for line in p.stdout:
        if line.startswith("READY"):
            break
    p.stdin.write("GEN 32 temperature=0 " + ",".join(map(str, P)) + "\n")
    p.stdin.flush()
    toks, t_slow = [], None
    for line in p.stdout:
        if line.startswith("PP ") and int(line.split()[1]) >= len(P) - 1 and cpath and not state["slow"]:
            state["slow"], t_slow = True, time.monotonic()   # the read is done: the decode at the floor
        elif line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith("DONE") or line.startswith("ERR"):
            break
    t_end = time.monotonic()
    state["stop"] = True
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
    s = Path(log).read_text(errors="replace")
    t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
    floor = [l for l in s.splitlines() if "the drive guard: the drive's slow mode" in l]
    print(f"  {tag}: logits {t[-1][2] if t else None}, 32 tokens {toks}" +
          (f"; the decode at the floor: {len(floor)} engagement(s), {t_end - t_slow:.1f} s from the read's end to DONE" if t_slow else ""),
          flush=True)
    return (t[-1][2] if t else None), toks


print("=== the route bias off; the switches: " + " ".join(f"{k[7:]}={v}" for k, v in sorted(ENV.items()) if k.startswith("STRATA_DSV4")),
      flush=True)
a = run("e-today", today_exe, {})
b = run("e-floor", new_exe, {"STRATA_DSV41_DRIVE_GUARD": "1", "STRATA_DSV41_DRIVE_GUARD_STAT": f"{prefix}-e-counters.txt",
                             "STRATA_DSV41_EG_WINDOW": "1024", "STRATA_DSV41_EG_SCALES_PRELOAD": "1"}, f"{prefix}-e-counters.txt")
print(f"  {'PASS' if a == b and a[0] else 'FAIL'}  the decode at the guard's floor is today's bit for bit (the logits hash and 32 "
      f"greedy tokens)", flush=True)
