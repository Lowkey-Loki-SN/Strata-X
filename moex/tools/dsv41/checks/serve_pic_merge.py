"""serve_pic_merge.py LOG_PREFIX EXE - DSV41_BACKLOG #113: a picture read in its own prompt call against the same picture read in
one call with the text around it (STRATA_DSV41_PIC_MERGE=1).  The pictures encoded once by serve/vision_dsv41.py as configured
(ref-runs/engine-checks/v43-number.png, "7319", and a generated WxH screenshot, W, H: 1920, 1080), then per arm a fresh engine
(strata-dsv41.json's arguments and switches through the serve protocol): (1) the number picture behind a user turn's start and
a question, greedy - its answer and first token's top 8; (2) PREFIX_N (1500) tokens of this repo's docs, cached; (3) the
screenshot's rows and a question behind that prefix, DECODE_N (16) tokens - the read's time and its prompt-path lines' records
(from the SSD / its cache, the windows' reads)."""
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

prefix_log, exe = sys.argv[1], sys.argv[2]
R = _REPO
W, H = int(os.environ.get("W", "1920")), int(os.environ.get("H", "1080"))
PREFIX = int(os.environ.get("PREFIX_N", "1500"))
N = int(os.environ.get("DECODE_N", "16"))
ARMS = os.environ.get("ARMS", "off,on").split(",")
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
git = lambda f: subprocess.run(["git", "-C", R, "show", f"645844e9:{f}"], capture_output=True, text=True, check=True).stdout
sp = lambda s: tok.encode(s, parse_special=True)
docs = [tok.encode(git(f)) for f in ("docs/DETAILS.md",)]
ENV = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_DSV4", "STRATA_ARENA", "STRATA_X_STAGE", "STRATA_POOL"))}
ENV.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1")
ENV.update(cfg["env"])
work = Path(f"{prefix_log}.d")
work.mkdir(parents=True, exist_ok=True)

# ---- the pictures, encoded once (kept in the work directory for a later run)
if not (work / "shot.png").exists():
    import numpy as np  # noqa: E402
    from PIL import Image  # noqa: E402
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:H, 0:W]
    img = np.stack([(x * 255 // (W - 1)), (y * 255 // (H - 1)), ((x + y) % 256)], -1).astype(np.float32)
    for _ in range(60):
        x0, y0 = rng.integers(0, W - 8), rng.integers(0, H - 8)
        img[y0:y0 + rng.integers(4, H // 4), x0:x0 + rng.integers(4, W // 4)] = rng.integers(0, 256, 3)
    Image.fromarray(np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)).save(work / "shot.png")
    (work / "num.png").write_bytes(Path(_os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks/v43-number.png')).read_bytes())
rows = {}
v = cfg["vision"]
venc = None
for name in ("num", "shot"):
    meta = work / f"{name}.rows"
    if (work / f"{name}.sve").exists() and meta.exists():
        rows[name] = int(meta.read_text())
        continue
    if venc is None:
        venc = subprocess.Popen([v["python"], v["exe"], "--mmproj", v["mmproj"], "--model", v["model"], "--threads", str(v.get("threads", 20))],
                                cwd=str(work), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=ENV)
        for line in venc.stdout:
            if line.startswith("READY"):
                break
    t = time.time()
    venc.stdin.write(f"ENC {name}.png {name}.sve\n")
    venc.stdin.flush()
    ok = venc.stdout.readline().strip()
    if not ok.startswith("OK"):
        sys.exit(f"the encoder: {ok}")
    rows[name] = int(ok.split()[1])
    meta.write_text(str(rows[name]))
    print(f"{name}: {rows[name]} rows encoded in {time.time() - t:.1f} s", flush=True)
if venc is not None:
    venc.stdin.write("QUIT\n")
    venc.stdin.flush()
    venc.wait(timeout=60)

ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--image-pad", "129264"]
for arm in ARMS:
    env = dict(ENV, STRATA_X_QUIET="0", STRATA_X_FIRST_TOP="8")
    if arm == "on":
        env["STRATA_DSV41_PIC_MERGE"] = "1"
    log = f"{prefix_log}.{arm}.log"
    err = open(log, "w")
    p = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                         text=True, bufsize=1, env=env)
    for line in p.stdout:
        if line.startswith("READY"):
            break

    def gen(ids, emb=None):
        mark = len(Path(log).read_text(errors="replace"))
        t0 = time.time()
        p.stdin.write((f"GENI {N} temperature=0 {emb} " if emb else f"GEN {N} temperature=0 ") + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        toks = []
        for line in p.stdout:
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                break
        err.flush()
        s = Path(log).read_text(errors="replace")[mark:]
        g = lambda pat: (re.findall(pat, s) or ["-"])[-1]
        recs = [tuple(map(int, m)) for m in re.findall(r"(\d+) read from the SSD, (\d+) from its cache", s)]
        top = g(r"first-token top at position \d+: ([^|]*)").strip()[:110]
        return time.time() - t0, toks, g(r"serve: (\d+ prompt tokens \(\d+ reused, \d+ read\) in [0-9.]+ s)"), recs, top, \
            g(r"the drive meanwhile: ([^;]*)")[:90]

    s1, t1, h1, r1, top1, _ = gen(sp("<｜User｜>") + [129264] * rows["num"] +
                                  sp("\n\nWhat number is shown in the picture? Answer with the number only.<｜Assistant｜></think>"),
                                  str((work / "num.sve").resolve()))
    print(f"[{arm}] (1) the number picture: {h1}; the answer {tok.decode(t1[:8])!r}; first-token top {top1}", flush=True)
    pre = sp("<｜User｜>") + docs[0][:PREFIX]
    gen(pre + sp("\n\nIn one line: what is this?<｜Assistant｜></think>"))
    s3, t3, h3, r3, _, drv = gen(pre + sp("\n\n") + [129264] * rows["shot"] + sp("\n\nWhat does the part above show?<｜Assistant｜></think>"),
                                 str((work / "shot.sve").resolve()))
    print(f"[{arm}] (3) the screenshot's {rows['shot']} rows behind the prefix: {h3}; reply in {s3:.1f} s; prompt calls' records "
          f"(SSD, cache) {r3}: {sum(a for a, _ in r3)} + {sum(b for _, b in r3)}; the drive {drv}", flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=180)
    err.close()
