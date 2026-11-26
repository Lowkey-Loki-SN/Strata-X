"""serve_pic_encoder.py LOG_PREFIX EXE [PICTURE] - the image encoder's settings against each other on the model's answer: the
picture (ref-runs/engine-checks/v43-number.png, "7319") encoded by serve/vision_dsv41.py as the release computes it (bf16, the
math path's attention) and as STRATA_DSV41_VISION_DTYPE=fp32 STRATA_DSV41_VISION_ATTN=fused computes it, then one engine
(strata-dsv41.json's arguments and switches through the serve protocol, the route bias off, one tier) reading each one's rows
behind the same question, greedy: each way's encode time, its first token's top 8 and its answer."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families  # noqa: E402

prefix, exe = sys.argv[1], sys.argv[2]
pic = Path(sys.argv[3] if len(sys.argv) > 3 else _os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks/v43-number.png'))
R = _REPO
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
work = Path(f"{prefix}.d")
work.mkdir(parents=True, exist_ok=True)
(work / "pic.png").write_bytes(pic.read_bytes())
base_env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_DSV4", "STRATA_ARENA", "STRATA_X_STAGE", "STRATA_POOL"))}
base_env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1")
base_env.update(cfg["env"])
WAYS = (("bf16 math", {"STRATA_DSV41_VISION_DTYPE": "bf16", "STRATA_DSV41_VISION_ATTN": "math"}),
        ("fp32 fused", {"STRATA_DSV41_VISION_DTYPE": "fp32", "STRATA_DSV41_VISION_ATTN": "fused"}))
v = cfg["vision"]
rows = {}
for name, extra in WAYS:
    env = dict(base_env, **extra)
    p = subprocess.Popen([v["python"], v["exe"], "--mmproj", v["mmproj"], "--model", v["model"], "--threads", str(v.get("threads", 20))],
                         cwd=str(work), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    out = name.replace(" ", "_") + ".sve"
    t = time.time()
    p.stdin.write(f"ENC pic.png {out}\n")
    p.stdin.flush()
    ok = p.stdout.readline().strip()
    s = time.time() - t
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=60)
    if not ok.startswith("OK"):
        sys.exit(f"{name}: {ok}")
    rows[name] = (int(ok.split()[1]), str((work / out).resolve()), s)
ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--image-pad", "129264",
                                    "--no-adaptive", "--expert-slots", "1700"]
env = dict(base_env, STRATA_X_FIRST_TOP="8")
env.pop("STRATA_DSV41_ROUTE_BIAS", None)
log = f"{prefix}.log"
err = open(log, "w")
eng = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                       text=True, bufsize=1, env=env)
for line in eng.stdout:
    if line.startswith("READY"):
        break
sp = lambda x: tok.encode(x, parse_special=True)
for name, _ in WAYS:
    n, sve, s = rows[name]
    ids = sp("<｜User｜>") + [129264] * n + sp("\n\nWhat number is shown in the picture? Answer with the number only.<｜Assistant｜></think>")
    mark = len(Path(log).read_text(errors="replace"))
    eng.stdin.write(f"GENI 8 temperature=0 {sve} " + ",".join(map(str, ids)) + "\n")
    eng.stdin.flush()
    toks = []
    for line in eng.stdout:
        if line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith("DONE") or line.startswith("ERR"):
            break
    err.flush()
    top = (re.findall(r"first-token top at position \d+: ([^|]*)", Path(log).read_text(errors="replace")[mark:]) or ["-"])[-1]
    print(f"{name}: {n} rows encoded in {s:.2f} s; the answer {tok.decode(toks)!r}; first-token top {top.strip()[:120]}", flush=True)
eng.stdin.write("QUIT\n")
eng.stdin.flush()
eng.wait(timeout=180)
err.close()
