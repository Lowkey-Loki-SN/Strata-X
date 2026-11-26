#!/usr/bin/env python3
"""check_selection.py DUMP [--keep 2048] [--bsize 8] [--topk 512] - DeepSeek-V4.1's two-level top-k checked deep in the context
(the audit of 2026-10-19, DSV41_BACKLOG #36b: the candidate filter only cuts past keep x bsize = 16,384 compressed rows, and no
check had reached that far).  The engine's own inputs of each recorded decode row - a `strata-x -o DUMP --match ... --probe-from
N` run: idx_q, idx_w (the indexer's query heads and weights), idx_kc (its source's index keys, every visible row), idx_score (its
scores, after the candidate mask on a narrowed layer), idx_cand (the candidate bits), idx_sel (the selected rows) - go through
model.py's Indexer.forward and select_candidate_blocks in numpy (float64), and the engine's three outputs are compared:

  scores      sum over heads of relu(q_h . k_j) * w_h for every visible row j (the engine's tensor-core sum vs float64);
  candidates  the source layer's `keep` best blocks of `bsize` rows by their best score, the newest block pinned in, from the
              ENGINE's scores (so a near-tie in the scores cannot move a block) - and from float64 scores, as a second count;
  selection   the `topk` best of the (masked) scores; rows that differ are put against the k-th score - a near-tie differs by a
              rounding, a wrong row by more.

No GPU, no model: numpy and the dump (moex/ref/recorder.cpp: index.tsv + dump.bin)."""
import argparse
import sys
from pathlib import Path

import numpy as np


def load(dump):
    idx = Path(dump) / "index.tsv"
    raw = np.memmap(Path(dump) / "dump.bin", dtype=np.uint8, mode="r")
    recs = {}
    for line in idx.read_text().splitlines():
        step, name, typ, n0, n1, n2, n3, off = line.split("\t")
        ne = [int(n0), int(n1), int(n2), int(n3)]
        cnt = ne[0] * ne[1] * ne[2] * ne[3]
        dt = np.float32 if typ == "f32" else np.int32
        a = np.frombuffer(raw, dtype=dt, count=cnt, offset=int(off)).reshape([d for d in ne[::-1] if d > 1] or [1])
        recs.setdefault(int(step), {})[name] = a
    return recs


def candidates(score, keep, bsize):
    """model.py select_candidate_blocks for one decode row (every row visible): a bool per block."""
    n = score.shape[0]
    nblk = (n + bsize - 1) // bsize
    pad = np.full(nblk * bsize, -np.inf)
    pad[:n] = score
    best = pad.reshape(nblk, bsize).max(axis=1)
    best[(n - 1) // bsize] = np.inf                    # the newest block, pinned in
    if nblk <= keep:
        return np.ones(nblk, bool)
    order = np.lexsort((np.arange(nblk), -best))       # descending, ties to the lower block (as the engine; torch's topk is unspecified)
    out = np.zeros(nblk, bool)
    out[order[:keep]] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--keep", type=int, default=2048, help="candidate_topk_blocks (the GGUF's deepseek41.candidate.*)")
    ap.add_argument("--bsize", type=int, default=8, help="candidate_block_size")
    ap.add_argument("--topk", type=int, default=512, help="index_topk")
    ap.add_argument("--source", type=int, default=20, help="candidate_source_layer")
    a = ap.parse_args()
    recs = load(a.dump)
    bad = 0
    for step in sorted(recs):
        r = recs[step]
        layers = sorted(int(k.split("-")[1]) for k in r if k.startswith("idx_sel-"))
        cand_src = None
        for il in layers:
            q, w, kc = r[f"idx_q-{il}"].astype(np.float64), r[f"idx_w-{il}"].astype(np.float64), r[f"idx_kc-{il}"].astype(np.float64)
            se = r[f"idx_score-{il}"].astype(np.float64)
            sel = r[f"idx_sel-{il}"]
            n_vis = kc.shape[0]
            ref = (np.maximum(kc @ q.T, 0.0) * w[None, :]).sum(axis=1)           # [n_vis]
            line = f"step {step} layer {il}: {n_vis} visible rows ({(n_vis + a.bsize - 1) // a.bsize} blocks)"
            narrowed = il > a.source and cand_src is not None
            fin = np.isfinite(se)
            if il == a.source:
                rel = np.abs(se - ref).max() / max(1e-30, np.abs(ref).max())
                bits = np.unpackbits(r[f"idx_cand-{il}"].view(np.uint8), bitorder="little")[: (n_vis + a.bsize - 1) // a.bsize].astype(bool)
                c_eng = candidates(se, a.keep, a.bsize)
                c_ref = candidates(ref, a.keep, a.bsize)
                ok_c = np.array_equal(bits, c_eng)
                line += (f"; scores max rel err {rel:.2e}; candidates {int(bits.sum())} blocks, from its scores "
                         f"{'equal' if ok_c else 'DIFFER (%d blocks)' % int((bits != c_eng).sum())}, from float64 scores "
                         f"{int((bits != c_ref).sum())} blocks differ")
                bad += not ok_c or rel > 1e-3
                cand_src = bits
                mask = np.ones(n_vis, bool)
            elif narrowed:
                mask = np.repeat(cand_src, a.bsize)[:n_vis]
                ok_m = np.array_equal(fin, mask)
                rel = np.abs(se[mask] - ref[mask]).max() / max(1e-30, np.abs(ref[mask]).max())
                line += (f"; masked to layer {a.source}'s candidates: {'equal' if ok_m else 'DIFFER (%d rows)' % int((fin != mask).sum())}"
                         f" ({int(mask.sum())} rows); scores max rel err {rel:.2e}")
                bad += not ok_m or rel > 1e-3
            else:
                mask = np.ones(n_vis, bool)
                rel = np.abs(se - ref).max() / max(1e-30, np.abs(ref).max())
                line += f"; scores max rel err {rel:.2e}"
                bad += rel > 1e-3
            k = min(a.topk, n_vis)
            m_eng = np.where(mask, se, -np.inf)
            want = set(np.argsort(-m_eng, kind="stable")[:k].tolist())
            got = set(int(x) for x in sel if 0 <= x < n_vis)
            kth = np.sort(m_eng[np.isfinite(m_eng)])[::-1][k - 1] if np.isfinite(m_eng).sum() >= k else -np.inf
            diff = want ^ got
            gaps = [abs(m_eng[j] - kth) / max(1e-30, abs(kth)) for j in diff]
            ties = all(g < 1e-5 for g in gaps)
            line += (f"; selection {len(got)} rows, {len(want & got)} of {k} as numpy's from the engine's scores"
                     + ("" if not diff else f" ({len(diff)} differ at the {k}-th score, gaps up to {max(gaps):.1e}: "
                                            f"{'near-ties' if ties else 'NOT near-ties'})"))
            m_ref = np.where(mask, ref, -np.inf)
            want2 = set(np.argsort(-m_ref, kind="stable")[:k].tolist())
            line += f"; {len(want2 & got)} of {k} as float64's"
            bad += bool(diff) and not ties
            print(line, flush=True)
    print(f"check_selection: {'PASS' if bad == 0 else 'FAIL (%d)' % bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
