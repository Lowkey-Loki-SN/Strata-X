#!/usr/bin/env python3
"""compare_layers.py DUMP REF_DIR [--pos P] [--tokens IDS.npy] - one position, layer by layer: the engine's recorded output
streams of every layer (a `strata-x -o DUMP --match '^(l_last-[0-9]+|logits)$' --probe-from P` run: l_last-<il>, [4, 5120])
against the release's model.py residual after the same layer (moex/tools/dsv41/ref/ref_run.py: hNN.npy [S, 4, 5120]), and
the logits (logits.npy [S, vocab]).  Prints each layer's relative L2 error (a layer where it jumps is where the two part) and
the distributions' KLD / top tokens at P.  No GPU."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_selection import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("ref")
    ap.add_argument("--pos", type=int, required=True)
    a = ap.parse_args()
    rec = load(a.dump)[a.pos]
    ref = Path(a.ref)
    prev = None
    for il in range(40):
        name = f"l_last-{il}"
        if name not in rec or not (ref / f"h{il:02d}.npy").exists():
            continue
        e = rec[name].astype(np.float64).reshape(4, -1)
        r = np.load(ref / f"h{il:02d}.npy", mmap_mode="r")[a.pos].astype(np.float64).reshape(4, -1)
        rel = np.linalg.norm(e - r) / max(1e-30, np.linalg.norm(r))
        per = [np.linalg.norm(e[k] - r[k]) / max(1e-30, np.linalg.norm(r[k])) for k in range(4)]
        jump = "" if prev is None or rel < 2 * prev + 0.01 else "   <-- jumps"
        print(f"layer {il:2d}: relative L2 {rel:.3e} (streams {' '.join(f'{x:.2e}' for x in per)}){jump}")
        prev = rel
    if "logits" in rec and (ref / "logits.npy").exists():
        le = rec["logits"].astype(np.float64)
        lr = np.load(ref / "logits.npy", mmap_mode="r")[a.pos].astype(np.float64)
        pe = np.exp(le - le.max()); pe /= pe.sum()
        pr = np.exp(lr - lr.max()); pr /= pr.sum()
        kld = float(np.sum(pr * (np.log(pr + 1e-30) - np.log(pe + 1e-30))))
        print(f"logits at {a.pos}: KLD(ref || engine) {kld:.4f}; top-5 ref {np.argsort(-lr)[:5].tolist()} "
              f"engine {np.argsort(-le)[:5].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
