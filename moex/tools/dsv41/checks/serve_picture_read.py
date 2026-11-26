"""serve_picture_read.py LOG_PREFIX EXE - where a picture's time goes (DSV41_BACKLOG: the image read): the encoder
(serve/vision_dsv41.py with the config's vision settings and switches) on a generated WxH screenshot (W, H: 1920, 1080), then
the engine (strata-dsv41.json's arguments and switches, through the serve protocol) reading, behind the same cached prefix
of PREFIX_N (1500) tokens of this repo's docs, (a) the picture's rows and a question, (b) as many text tokens of other docs
and the same question - each a fresh read after the prefix, DECODE_N (16) tokens generated.  Prints the encode time, each
read's time and its prompt-path line's parts (records from the SSD / its cache, the drive's bytes)."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import io
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
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
git = lambda f: subprocess.run(["git", "-C", R, "show", f"645844e9:{f}"], capture_output=True, text=True, check=True).stdout
sp = lambda s: tok.encode(s, parse_special=True)
docs = [tok.encode(git(f)) for f in ("docs/DETAILS.md", "docs/AI_SETUP.md", "moex/docs/DSV41_PORT.md")]
ENV = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_DSV4", "STRATA_ARENA", "STRATA_X_STAGE", "STRATA_POOL"))}
ENV.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1")
ENV.update(cfg["env"])
for kv in filter(None, os.environ.get("EXTRA_ENV", "").split(",")):
    k, v = kv.split("=", 1)
    ENV[k] = v
work = Path(f"{prefix_log}.d")
work.mkdir(parents=True, exist_ok=True)

# ---- the picture, encoded by the server's encoder as configured
import numpy as np  # noqa: E402  (the Strata venv's numpy, for the generated picture)
from PIL import Image  # noqa: E402
rng = np.random.default_rng(7)
y, x = np.mgrid[0:H, 0:W]
img = np.stack([(x * 255 // (W - 1)), (y * 255 // (H - 1)), ((x + y) % 256)], -1).astype(np.float32)
for _ in range(60):
    x0, y0 = rng.integers(0, W - 8), rng.integers(0, H - 8)
    img[y0:y0 + rng.integers(4, H // 4), x0:x0 + rng.integers(4, W // 4)] = rng.integers(0, 256, 3)
Image.fromarray(np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)).save(work / "shot.png")
v = cfg["vision"]
venc = subprocess.Popen([v["python"], v["exe"], "--mmproj", v["mmproj"], "--model", v["model"], "--threads", str(v.get("threads", 20))],
                        cwd=str(work), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=ENV)
for line in venc.stdout:
    if line.startswith("READY"):
        break
t = time.time()
venc.stdin.write("ENC shot.png shot.sve\n")
venc.stdin.flush()
ok = venc.stdout.readline().strip()
enc_s = time.time() - t
venc.stdin.write("QUIT\n")
venc.stdin.flush()
venc.wait(timeout=60)
if not ok.startswith("OK"):
    sys.exit(f"the encoder: {ok}")
rows = int(ok.split()[1])
print(f"picture {W}x{H}: {rows} rows encoded in {enc_s:.1f} s (STRATA_DSV41_VISION_DTYPE={ENV.get('STRATA_DSV41_VISION_DTYPE', 'bf16')}, "
      f"STRATA_DSV41_VISION_ATTN={ENV.get('STRATA_DSV41_VISION_ATTN', 'math')})", flush=True)

# ---- the engine: the prefix once, then the picture's read and the text's read behind it
ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--image-pad", "129264"]
log = f"{prefix_log}.log"
err = open(log, "w")
p = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                     text=True, bufsize=1, env=dict(ENV, STRATA_X_QUIET="0"))
for line in p.stdout:
    if line.startswith("READY"):
        break
pre = sp("<｜User｜>") + docs[0][:PREFIX]
question = sp("\n\nWhat does the part above show?<｜Assistant｜></think>")


def gen(ids, emb=None):
    mark = len(Path(log).read_text(errors="replace"))
    t0 = time.time()
    p.stdin.write((f"GENI {N} temperature=0 {emb} " if emb else f"GEN {N} temperature=0 ") + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    first = None
    for line in p.stdout:
        if line.startswith("T ") and first is None:
            first = time.time() - t0
        elif line.startswith("DONE") or line.startswith("ERR"):
            break
    err.flush()
    s = Path(log).read_text(errors="replace")[mark:]
    g = lambda pat: (re.findall(pat, s) or ["-"])[-1]
    return first, g(r"serve: (\d+ prompt tokens \(\d+ reused, \d+ read\) in [0-9.]+ s)"), g(r"prompt path: (\d+ tokens.*)")[:400]


gen(pre + sp("\n\nIn one line: what is this?<｜Assistant｜></think>"))   # (the prefix cached)
pic = pre + sp("\n\n") + [129264] * rows + question
first, head, path = gen(pic, str((work / "shot.sve").resolve()))
print(f"  (a) the picture's {rows} rows: first token {first:.1f} s; {head}\n      {path}", flush=True)
text = pre + sp("\n\n") + docs[1][:rows] + question
first, head, path = gen(text)
print(f"  (b) {rows} text tokens: first token {first:.1f} s; {head}\n      {path}", flush=True)
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=180)
err.close()
