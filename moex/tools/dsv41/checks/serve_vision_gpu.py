"""serve_vision_gpu.py LOG_PREFIX EXE - DSV41_PLAN §54: the image encoder in the engine through the serve protocol, with
strata-dsv41.json's arguments and switches (the route bias off for exact routing, --no-adaptive, the 30 GiB test cache) plus
--vision-weights, two engine starts (each in a systemd-run --user scope):

  A (the default: the encoder's workspace in GPU 1's free VRAM - it keeps ~2 GiB free for the desktop)
    - the "7319" picture's rows from the CPU encoder (as the server runs it: fp32, fused, the warm-up) and from VENC, each read
      behind the same question, greedy: the answer and the first token's top 8 (STRATA_X_FIRST_TOP);
    - a 1920x1080 picture's VENC time; then a text request.
  B (STRATA_DSV41_VENC_LEND=1: GPU 0's top slots lent to the encoder and refilled after)
    - VENC of the "7319" picture: its rows file must be A's bytes; read behind the question; then a text request;
    - the log: no failed check of the refilled slots.
  C (STRATA_X_STAGE_SLOTS=900,800: GPU 0 left room, so the workspace comes from its free VRAM - with the server's slots it
    has ~704 MiB free after the load and less after a request, and the encoder lends slots, as A does)
    - VENC of the "7319" picture: "free", and A's bytes.
ARMS=AB (etc.) runs some of them; a later arm compares with the rows an earlier run left.

PASS: the VENC answers "7319", B's and C's rows A's bytes, C's workspace the free VRAM, the text requests done, no slot-check
failure."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

R = Path(_REPO)
sys.path[:0] = [str(R), str(R / "tools")]
from serve import families  # noqa: E402
from serve import vision_dsv41_plan as plan  # noqa: E402

prefix, exe = sys.argv[1], sys.argv[2]
cfg = json.loads((R / "strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
v = cfg["vision"]
mcfg = json.loads(Path(v["model"]).read_text())
max_n = plan.max_tokens_of(mcfg, int(v.get("max_tokens") or 0))
work = Path(f"{prefix}.d")
work.mkdir(parents=True, exist_ok=True)
number = Path(_os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks/v43-number.png')).read_bytes()
b = io.BytesIO()
Image.fromarray(np.random.default_rng(542).integers(0, 256, size=(1080, 1920, 3)).astype(np.uint8), "RGB").save(b, "PNG")
big = b.getvalue()
for name, data in (("number", number), ("big", big)):
    a, vh, vw, lh, lw = plan.patches_of(data, mcfg, max_n)
    plan.write_patches(work / f"{name}.svp", a, vh, vw, lh, lw)
manifest = work / "vision.svm"
plan.write_manifest(manifest, v["mmproj"], mcfg)

# the CPU encoder's rows of the "7319" picture (the server's settings)
venv = {k: val for k, val in os.environ.items() if not k.startswith("STRATA_")}
venv.update({k: val for k, val in cfg["env"].items() if k.startswith("STRATA_DSV41_VISION")})
venv.pop("STRATA_DSV41_VISION_EARLY", None)
(work / "number.png").write_bytes(number)
p = subprocess.Popen([v["python"], v["exe"], "--mmproj", v["mmproj"], "--model", v["model"], "--threads", str(v.get("threads", 20))],
                     cwd=str(work), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=venv)
for line in p.stdout:
    if line.startswith("READY"):
        break
t = time.time()
p.stdin.write("ENC number.png number.cpu.sve\n")
p.stdin.flush()
ok = p.stdout.readline().strip()
cpu_s = time.time() - t
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=60)
if not ok.startswith("OK"):
    sys.exit(f"the CPU encoder: {ok}")
n_rows = int(ok.split()[1])

ARGS = ["--serve"] + list(cfg["args"]) + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804",
                                          "--image-pad", "129264", "--no-adaptive", "--vision-weights", str(manifest)]
ARGS[ARGS.index("--ssd-cache-gib") + 1] = "30"   # the test runs' cache (the user's OK)
base = {k: val for k, val in os.environ.items() if not k.startswith(("STRATA_DSV4", "STRATA_ARENA", "STRATA_X_STAGE", "STRATA_POOL"))}
base.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
base.update(cfg["env"])
base.pop("STRATA_DSV41_ROUTE_BIAS", None)
sp = lambda x: tok.encode(x, parse_special=True)  # noqa: E731
QUESTION = "\n\nWhat number is shown in the picture? Answer with the number only.<｜Assistant｜></think>"


def start(tag, extra):
    log = f"{prefix}-{tag}.log"
    err = open(log, "w")
    eng = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=err, text=True, bufsize=1, env=dict(base, **extra))
    for line in eng.stdout:
        if line.startswith("READY"):
            break
    return eng, err, log


def call(eng, line, until=("DONE", "ERR", "VOK")):
    eng.stdin.write(line + "\n")
    eng.stdin.flush()
    toks, last = [], ""
    for out in eng.stdout:
        if out.startswith("T "):
            toks.append(int(out.split()[1]))
        elif out.startswith(until):
            last = out.strip()
            break
    return toks, last


def ask(eng, err, log, sve, n):
    mark = len(Path(log).read_text(errors="replace"))
    ids = sp("<｜User｜>") + [129264] * n + sp(QUESTION)
    toks, last = call(eng, f"GENI 8 temperature=0 {sve} " + ",".join(map(str, ids)))
    err.flush()
    top = (re.findall(r"first-token top at position \d+: ([^|]*)", Path(log).read_text(errors="replace")[mark:]) or ["-"])[-1]
    return tok.decode(toks).strip(), top.strip()[:110], last


def text(eng):
    toks, last = call(eng, "GEN 16 temperature=0 " + ",".join(map(str, sp("<｜User｜>Say hello in one word.<｜Assistant｜></think>"))))
    return last.startswith("DONE"), tok.decode(toks).strip()


ok_all = True
results = {}
arms = (("A", {}), ("B", {"STRATA_DSV41_VENC_LEND": "1"}), ("C", {"STRATA_X_STAGE_SLOTS": "900,800"}))
for tag, extra in (a for a in arms if a[0] in os.environ.get("ARMS", "ABC")):
    eng, err, log = start(tag, extra)
    if tag == "A":
        ans, top, _ = ask(eng, err, log, str((work / "number.cpu.sve").resolve()), n_rows)
        print(f"  A, the CPU encoder's rows ({n_rows}, {cpu_s:.2f} s): the answer {ans!r}; top {top}", flush=True)
    out = work / f"number.{tag}.sve"
    _, vok = call(eng, f"VENC 1 {(work / 'number.svp').resolve()} {out.resolve()}")
    print(f"  {tag}, VENC of the \"7319\" picture: {vok}", flush=True)
    ok_all &= vok.startswith("VOK")
    if tag == "A":   # (two GPUs: GPU 1 keeps ~2 GiB free for the desktop - the encoder's first choice)
        ok_all &= vok.endswith(" free=gpu1")
    if tag == "C":
        ok_all &= " free=gpu" in vok
        results[tag] = out.read_bytes() if vok.startswith("VOK") else None
    elif vok.startswith("VOK"):
        n = int(vok.split()[1])
        ans, top, _ = ask(eng, err, log, str(out.resolve()), n)
        print(f"  {tag}, its rows read: the answer {ans!r}; top {top}", flush=True)
        ok_all &= ans.startswith("7319")
        results[tag] = out.read_bytes()
    if tag == "A":
        _, vok = call(eng, f"VENC 1 {(work / 'big.svp').resolve()} {(work / 'big.sve').resolve()}")
        print(f"  A, VENC of a 1920x1080 picture: {vok}", flush=True)
        ok_all &= vok.startswith("VOK")
    if tag != "C":
        done, said = text(eng)
        print(f"  {tag}, a text request after: {'done' if done else 'FAILED'} ({said!r})", flush=True)
        ok_all &= done
    eng.stdin.write("QUIT\n")
    eng.stdin.flush()
    eng.wait(timeout=180)
    err.close()
    bad = re.findall(r"refilled slots' check: .*", Path(log).read_text(errors="replace"))
    if bad:
        print(f"  {tag}: {bad[0][:160]}", flush=True)
        ok_all = False
if "A" not in results and (work / "number.A.sve").exists():
    results["A"] = (work / "number.A.sve").read_bytes()   # an earlier run's
for tag in ("B", "C"):
    if tag in results:
        same = results.get("A") is not None and results.get("A") == results[tag]
        print(f"  {tag}'s rows A's bytes: {'yes' if same else 'NO'}", flush=True)
        ok_all &= same
print("ALL PASS" if ok_all else "SOME FAILED", flush=True)
sys.exit(0 if ok_all else 1)
