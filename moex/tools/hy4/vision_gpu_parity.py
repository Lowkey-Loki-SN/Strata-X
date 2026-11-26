#!/usr/bin/env python3
"""moex/tools/hy4/vision_gpu_parity.py - Hy4-flash's image encoder on the GPU (hy4_vision_parity, strata/x/hy4_vision.hpp)
against the CPU encoder (serve/vision_hy4.py) on the same pictures, both fp32, with a float64 encode of the CPU encoder as
the reference for both.  The patches are the server's (serve/vision_hy4_plan.py patches_of: no torch), checked bit for bit
against the CPU encoder's own input first.

    vision_gpu_parity.py --mmproj MMPROJ.gguf --bin build-dsv41-dev/hy4_vision_parity [--device 1] [--no-f64] PICTURE...

Per picture: the GPU's and the CPU's largest |difference| and relative L2 against the float64 rows (and against each other),
the GPU's time.  Exit 1 when a GPU row is further from the float64 rows than 1e-4 relative (the CPU encoder's own error is
~1e-6), or when the patches differ.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "serve"))
import vision_hy4 as V        # noqa: E402
import vision_hy4_plan as P   # noqa: E402


def read_sve(path):
    raw = Path(path).read_bytes()
    magic, n, nx, ny, w = struct.unpack_from("<5i", raw, 0)
    assert magic == 0x31455653
    return np.frombuffer(raw, np.float32, n * w, 20).reshape(n, w), nx, ny


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--no-f64", action="store_true", help="skip the float64 reference (the GPU against the CPU only)")
    ap.add_argument("pictures", nargs="+")
    a = ap.parse_args()
    lim = P.processor_limits(None)
    enc32 = V.Encoder(Path(a.mmproj), torch.float32)
    enc64 = None if a.no_f64 else V.Encoder(Path(a.mmproj), torch.float64)
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for pic in a.pictures:
            data = Path(pic).read_bytes()
            x = V.load_image(data, 0, lim)
            patches, gh, gw = P.patches_of(data, lim)
            ref_in = x.numpy().reshape(3, gh, 16, gw, 16).transpose(1, 3, 0, 2, 4).reshape(gh * gw, 768)
            same_in = np.array_equal(ref_in.view(np.uint32), patches.view(np.uint32))
            svh, sve = os.path.join(td, "p.svh"), os.path.join(td, "p.sve")
            P.write_patches(svh, patches, gh, gw)
            t0 = time.time()
            r = subprocess.run([a.bin, a.mmproj, svh, sve, str(a.device)], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{pic}: the GPU encoder refused: {r.stderr.strip()}")
                bad += 1
                continue
            gpu, nx, ny = read_sve(sve)
            enc_line = [ln for ln in r.stderr.splitlines() if ln.startswith("encode 1")]
            t1 = time.time()
            cpu, cnx, cny = enc32.rows(x)
            cpu = cpu.numpy()
            t_cpu = time.time() - t1
            line = (f"{Path(pic).name} {gw * 16}x{gh * 16}: patches same as the CPU's: {same_in}; grid {nx}x{ny} (CPU {cnx}x{cny}); "
                    f"GPU vs CPU rel {rel(gpu, cpu):.2e} max |d| {np.abs(gpu - cpu).max():.2e}; CPU {t_cpu:.1f} s; "
                    f"GPU {enc_line[0].split(' in ')[1].split(' (')[0] if enc_line else '?'}")
            ok = same_in and (nx, ny) == (cnx, cny)
            if enc64 is not None:
                f64, _, _ = enc64.rows(x.double())
                f64 = f64.numpy()
                g_rel, c_rel = rel(gpu, f64), rel(cpu, f64)
                worst = max(float(np.linalg.norm(gpu[i] - f64[i]) / np.linalg.norm(f64[i])) for i in range(len(f64)))
                line += f"; vs float64: GPU rel {g_rel:.2e} (worst row {worst:.2e}), CPU rel {c_rel:.2e}"
                ok = ok and worst < 1e-4
            print(line, flush=True)
            bad += 0 if ok else 1
    print("ALL PASS" if bad == 0 else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
