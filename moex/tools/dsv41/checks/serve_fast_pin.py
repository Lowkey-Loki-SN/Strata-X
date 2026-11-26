"""serve_fast_pin.py LOG_PREFIX OLD_EXE NEW_EXE - DSV41_PLAN §21: STRATA_DSV41_FAST_PIN=1 (the SSD cache's blocks, its bounce slots
and the RAM copy faulted in by 8 threads before cudaHostRegister, not cudaHostAlloc'ed / registered untouched) through the
engine's serve protocol with strata-dsv41.json's arguments (the 60 GiB cache, 512K context) and switches, one tier
(--no-adaptive --expert-slots 1700, so both arms place the same experts):

 A  OLD_EXE as the config runs it: the time to READY, its phases, the first token's logits of a 2,000-token text (the top of
    moex/docs/DSV41_PLAN.md at 645844e9) and a question
 B  NEW_EXE with STRATA_DSV41_FAST_PIN=1: the same, the logits bit for bit with A's, READY sooner"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, re, subprocess, sys, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families

prefix, old_exe, new_exe = sys.argv[1], sys.argv[2], sys.argv[3]
R = _REPO
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
text = subprocess.run(["git", "-C", R, "show", "645844e9:moex/docs/DSV41_PLAN.md"], capture_output=True, text=True, check=True).stdout
P = tok.encode(text)[:2000] + tok.encode("\n\nIn one sentence: what does the text above do?<｜Assistant｜></think>", parse_special=True)
ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--no-adaptive",
                                    "--expert-slots", "1700"]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
for k in list(ENV):
    if k.startswith("STRATA_DSV4") or k.startswith("STRATA_ARENA"):
        ENV.pop(k)
ENV.update(cfg["env"])


def arm(tag, exe, extra):
    log = f"{prefix}-{tag}.log"
    err = open(log, "w")
    t0 = time.monotonic()
    p = subprocess.Popen([exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
                         env=dict(ENV, **extra))
    for line in p.stdout:
        if line.startswith("READY"):
            break
    ready = time.monotonic() - t0
    p.stdin.write("GEN 1 temperature=0 " + ",".join(map(str, P)) + "\n")
    p.stdin.flush()
    for line in p.stdout:
        if line.startswith("DONE") or line.startswith("ERR"):
            break
    tq = time.monotonic()
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    quit_s = time.monotonic() - tq
    err.close()
    s = Path(log).read_text(errors="replace")
    t = re.findall(r"first-token top at position (\d+):([^|\n]*)\| logits ([0-9a-f]{16})", s)
    g = lambda pat: (re.findall(pat, s) or ["?"])[-1]
    r = {"hash": t[-1][2] if t else None, "ready": ready, "cache": g(r"slots = ([0-9.]+ GiB pinned in [0-9.]+ s[^,]*)"),
         "ram": g(r"RAM copy ([0-9.]+ GiB reserved in [0-9.]+ s)"), "load": g(r"experts loaded in ([0-9.]+ s)"),
         "dense": g(r"dense tensors, ([0-9.]+ GiB in [0-9.]+ s)"), "quit": quit_s}
    print(f"  {tag}: READY {ready:.1f} s after the start (the SSD cache {r['cache']}; the RAM copy {r['ram']}; the load pass "
          f"{r['load']}; dense {r['dense']}); logits {r['hash']}; the engine ended {quit_s:.1f} s after QUIT", flush=True)
    return r


A = arm("a-old", old_exe, {})
B = arm("b-new-fastpin", new_exe, {"STRATA_DSV41_FAST_PIN": "1"})
ok = {
    "B: the first token's logits are A's bit for bit": A["hash"] is not None and A["hash"] == B["hash"],
    "B: the cache's blocks were faulted in and registered (STRATA_DSV41_FAST_PIN)": "faulted in and registered" in B["cache"],
    "B: READY sooner than A": B["ready"] < A["ready"],
}
print(f"  READY {A['ready']:.1f} -> {B['ready']:.1f} s ({A['ready'] - B['ready']:+.1f} s)", flush=True)
for k, v in ok.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(ok.values()) else "SOME FAILED", flush=True)
