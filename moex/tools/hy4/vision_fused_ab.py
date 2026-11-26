#!/usr/bin/env python3
"""moex/tools/hy4/vision_fused_ab.py - Hy4-flash's GPU image encoder (hy4_vision_parity, src/x/hy4_vision.cu) through both
of its attention paths on the same patches - STRATA_HY4_VENC_FUSED=1 (k_attn_fused, the default) and =0 (the SGEMMs and
k_softmax) - each against one float64 encode of the CPU encoder (serve/vision_hy4.py), with vision_gpu_parity.py's
functions: the rows' relative L2, the worst row, whether a repeat is bit for bit the same, the encode's time.

    vision_fused_ab.py --mmproj MMPROJ.gguf --bin build-dsv41-dev/hy4_vision_parity [--device 1] PICTURE...

HY4_PORT.md, the GPU encoder's fused attention (2026-11-19): 448 x 320 rel 9.34e-7 / 9.93e-7, 1920 x 1088 2.38e-6 / 2.46e-6
(worst rows 2.98e-5 / 3.47e-5), 1,973 / 3,212 ms.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "serve"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision_hy4 as V        # noqa: E402
import vision_hy4_plan as P   # noqa: E402
from vision_gpu_parity import read_sve, rel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("pictures", nargs="+")
    a = ap.parse_args()
    enc64 = V.Encoder(Path(a.mmproj), torch.float64)
    with tempfile.TemporaryDirectory() as td:
        for pic in a.pictures:
            data = Path(pic).read_bytes()
            x = V.load_image(data, 0, None)
            patches, gh, gw = P.patches_of(data, None)
            svh = os.path.join(td, "p.svh")
            P.write_patches(svh, patches, gh, gw)
            out = {}
            for arm in ("1", "0", "1"):   # the fused path twice: a repeat must be the same bytes
                sve = os.path.join(td, "p%s.sve" % arm)
                env = dict(os.environ, STRATA_HY4_VENC_FUSED=arm)
                r = subprocess.run([a.bin, a.mmproj, svh, sve, str(a.device)], capture_output=True, text=True, env=env)
                if r.returncode:
                    print(f"{pic}: STRATA_HY4_VENC_FUSED={arm}: the GPU encoder failed: {r.stderr.strip()[-400:]}")
                    return 1
                line = [ln for ln in r.stderr.splitlines() if ln.startswith("encode 1")]
                ms = line[0].split(" in ")[1].split(" (")[0] if line and " in " in line[0] else "?"
                out.setdefault(arm, []).append((read_sve(sve)[0], ms))
            t0 = time.time()
            f64 = enc64.rows(x.double())[0].numpy()
            print(f"{Path(pic).name} {gw * 16}x{gh * 16} ({gh * gw} patches), float64 in {time.time() - t0:.0f} s", flush=True)
            for arm, runs in out.items():
                g = runs[0][0]
                worst = max(float(np.linalg.norm(g[i] - f64[i]) / np.linalg.norm(f64[i])) for i in range(len(f64)))
                same = all(np.array_equal(g.view(np.uint32), rr[0].view(np.uint32)) for rr in runs[1:])
                print(f"  {'fused ' if arm == '1' else 'SGEMMs'}: rel {rel(g, f64):.2e}, worst row {worst:.2e}; repeat bit for bit: "
                      f"{same}; {' | '.join(rr[1] for rr in runs)}", flush=True)
            f, s = out["1"][0][0], out["0"][0][0]
            print(f"  fused vs SGEMMs: rel {rel(f, s):.2e}, max |d| {np.abs(f - s).max():.2e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
