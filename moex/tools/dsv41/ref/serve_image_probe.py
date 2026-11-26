"""moex/tools/dsv41/ref/serve_image_probe.py OUT_DIR LOG [--pack DIR] [--bin PATH] - the engine's side of ref_image.py: the prompt
(OUT_DIR/ids.npy, the picture's span OUT_DIR/image.sve) through Strata's engine protocol (GENI: the span read causally, routed
with exp_probs_b_vl, no engram), one greedy request per prefix from just past the span to the whole prompt (each reuses the
previous one's state: one token read), with STRATA_X_FIRST_TOP=8 - the engine's log-probabilities at each of those positions
land in LOG for `ref_image.py compare`."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import argparse
import json
import os
import pathlib
import subprocess

import numpy as np

D = pathlib.Path(_os.path.join(_MODELS, 'dsv41-1017'))
X = pathlib.Path(__file__).resolve().parents[4]
ap = argparse.ArgumentParser()
ap.add_argument("out", type=pathlib.Path)
ap.add_argument("log", type=pathlib.Path)
ap.add_argument("--pack", type=pathlib.Path, default=D / "native")
ap.add_argument("--bin", type=pathlib.Path, default=X / "build-dsv41-dev/strata-x")
ap.add_argument("--ssd-cache-gib", default="60")   # a test beside a CPU reference: 30 (the server keeps 60)
a = ap.parse_args()
span = json.loads((a.out / "span.json").read_text())
ids = np.load(a.out / "ids.npy").tolist()
sve = str((a.out / "image.sve").resolve())
cmd = [str(a.bin), "--serve", "-m", str(D / "dsv41-1017-dense-q8_0-dsp-eg.gguf"), "--pack", str(a.pack), "--ssd-cache-gib", a.ssd_cache_gib,
       "--max-context", "8192", "--threads", "20", "--short-read", "96", "--layer-split", "20", "--eos-ids", "1",
       "--turn-tokens", "128799,128803,128804", "--image-pad", "129264", "--no-adaptive"]
env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_DSV41_HEADS_MMVQ="1",
           STRATA_DSV41_HC_SPLIT="1", STRATA_X_FIRST_TOP="8")
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(a.log, "w"), text=True, bufsize=1, env=env)
for line in p.stdout:
    if line.startswith("READY"):
        break
end = span["start"] + span["rows"]
for L in range(end + 1, len(ids) + 1):
    p.stdin.write(f"GENI 1 temperature=0 {sve} " + ",".join(map(str, ids[:L])) + "\n")
    p.stdin.flush()
    for line in p.stdout:
        if line.startswith("DONE") or line.startswith("ERR"):
            if line.startswith("ERR"):
                print(f"prefix {L}: {line.strip()}", flush=True)
            break
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=120)
print(f"{len(ids) - end} prefixes from {end + 1} to {len(ids)} tokens -> {a.log}")
