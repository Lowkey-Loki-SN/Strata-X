"""vision_gpu_parity.py WORK_DIR [PARITY_EXE] - DSV41_PLAN §54: the image encoder on the GPU (build-dsv41-dev/dsv41_vision_parity:
the engine's encoder alone, in a workspace of its own) against the CPU encoder (serve/vision_dsv41.py with the server's settings
from strata-dsv41.json: fp32, the fused attention, the warm-up pass) on five pictures, both measured against a float64 encode
(vision_fp64_ref.py): two fp32 encoders that sum in different orders each err ~1e-6 to 1e-5 against float64, so their distance
from each other is no yardstick.  PASS: each picture encoded twice to the same bytes, its GPU error against float64 within 1.5x
the CPU encoder's (how the rounding falls varies with the picture: 1.37x on the tall one), at most the CPU's on average
(geometric mean of the ratios), and a negative control that fails: block 16's MLP output weights read from block 17's (a
manifest line pointed elsewhere).  The CPU encoder and the float64 one run on the vision block's interpreter (llama.cpp's venv);
the GPU tool takes ~1 GB of GPU 0 - run it with the GPUs free, in a systemd-run --user scope."""
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
from pathlib import Path

import numpy as np
from PIL import Image

R = Path(_REPO)
sys.path[:0] = [str(R)]
from serve import vision_dsv41_plan as plan  # noqa: E402

work = Path(sys.argv[1])
exe = sys.argv[2] if len(sys.argv) > 2 else str(R / "build-dsv41-dev/dsv41_vision_parity")
work.mkdir(parents=True, exist_ok=True)
cfg = json.loads((R / "strata-dsv41.json").read_text())
v = cfg["vision"]
mcfg = json.loads(Path(v["model"]).read_text())
max_n = plan.max_tokens_of(mcfg, int(v.get("max_tokens") or 0))


def png(a):
    b = io.BytesIO()
    Image.fromarray(a.astype(np.uint8), "RGB").save(b, "PNG")
    return b.getvalue()


rng = np.random.default_rng(541)
pics = {"7319": Path(_os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks/v43-number.png')).read_bytes()}
yy, xx = np.mgrid[0:720, 0:1280]
pics["640x480-noise"] = png(rng.integers(0, 256, size=(480, 640, 3)))
pics["1280x720-gradient"] = png(np.stack([xx * 255 // 1279, yy * 255 // 719, (xx + yy) % 256], axis=-1))
pics["1920x1080-noise"] = png(rng.integers(0, 256, size=(1080, 1920, 3)))
pics["500x1600-noise"] = png(rng.integers(0, 256, size=(1600, 500, 3)))

# the CPU encoder's rows, as the server runs it (its env from the config: fp32, fused, the warm-up)
env = {k: val for k, val in os.environ.items() if not k.startswith("STRATA_")}
env.update({k: val for k, val in cfg["env"].items() if k.startswith("STRATA_DSV41_VISION")})
env.pop("STRATA_DSV41_VISION_EARLY", None)
p = subprocess.Popen([v["python"], v["exe"],   # as the server starts it (its plan module beside it)
                      "--mmproj", v["mmproj"], "--model", v["model"], "--threads", str(v.get("threads", 20))],
                     cwd=str(work), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
for line in p.stdout:
    if line.startswith("READY"):
        break
for name, data in pics.items():
    (work / f"{name}.png").write_bytes(data)
    p.stdin.write(f"ENC {name}.png {name}.cpu.sve\n")
    p.stdin.flush()
    ok = p.stdout.readline().strip()
    if not ok.startswith("OK"):
        sys.exit(f"the CPU encoder on {name}: {ok}")
    a, vh, vw, lh, lw = plan.patches_of(data, mcfg, max_n)
    plan.write_patches(work / f"{name}.svp", a, vh, vw, lh, lw)
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=60)

manifest = work / "vision.svm"
plan.write_manifest(manifest, v["mmproj"], mcfg)
genv = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0")
bad = 0
for name in pics:   # the GPU's span (tol 1: the verdict is float64's, below), twice to the same bytes
    r = subprocess.run([exe, str(manifest), str(work / f"{name}.svp"), str(work / f"{name}.cpu.sve"), "1", str(work / f"{name}.gpu.sve")],
                       capture_output=True, text=True, env=genv)
    same = r.returncode == 0 and "the same bytes" in r.stdout
    bad += not same
    print(f"  {name}: {(r.stdout or r.stderr).strip().replace(' (tol 1)', ' against the CPU encoder').replace(' - PASS', '')}", flush=True)
f64 = subprocess.run([v["python"], "-I", str(R / "moex/tools/dsv41/checks/vision_fp64_ref.py"), str(work), *pics], capture_output=True,
                     text=True)
ratios = []
for line in f64.stdout.splitlines():
    m = re.search(r"(\S+): \d+ rows - against float64: the CPU encoder ([\d.e+-]+), the GPU encoder ([\d.e+-]+)", line)
    if not m:
        continue
    c, g = float(m.group(2)), float(m.group(3))
    ratios.append(g / c)
    ok = g <= 1.5 * c
    bad += not ok
    print(f"  {m.group(1)} against float64: the CPU encoder {c:.3g}, the GPU encoder {g:.3g} ({g / c:.2f}x) - {'PASS' if ok else 'FAIL'}",
          flush=True)
if len(ratios) != len(pics):
    bad += 1
    print(f"  the float64 encode: {(f64.stderr or f64.stdout).strip()[-400:]}", flush=True)
else:
    gm = float(np.exp(np.mean(np.log(ratios))))
    bad += gm > 1.0
    print(f"  on average (geometric mean) the GPU's error is {gm:.2f}x the CPU encoder's - {'PASS' if gm <= 1.0 else 'FAIL'}", flush=True)

# the negative control: block 16's mlp.w2 read where block 17's is
lines = manifest.read_text().splitlines()
w17 = next(x for x in lines if " vision.blocks.17.mlp.w2.weight " in x).split(" ")
neg = [(" ".join(x.split(" ")[:2] + w17[2:]) if " vision.blocks.16.mlp.w2.weight " in x else x) for x in lines]
(work / "vision-neg.svm").write_text("\n".join(neg) + "\n")
r = subprocess.run([exe, str(work / "vision-neg.svm"), str(work / "640x480-noise.svp"), str(work / "640x480-noise.cpu.sve")],
                   capture_output=True, text=True, env=genv)
neg_ok = r.returncode == 1
print(f"  negative control (block 16's mlp.w2 from block 17's): {(r.stdout or r.stderr).strip()} - "
      f"{'FAILS, as it must' if neg_ok else 'DID NOT FAIL'}", flush=True)
bad += not neg_ok
print("ALL PASS" if bad == 0 else f"{bad} FAILED", flush=True)
sys.exit(1 if bad else 0)
