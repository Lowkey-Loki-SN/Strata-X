"""moex/tools/ernie55/kld_compare.py - KLD bases of the same token sequence compared row by row (llama-perplexity's format:
the reference model's - ref/imatrix.py, ref/kld_base.py - or an engine's, strata-x --kld F --kld-save OUT): per scored
position the KL divergence of each pair (the first named the reference), the same top token, the probability of the
sequence's next token and of chosen ids under each base, by region and in bins of positions.

    python3 -I moex/tools/ernie55/kld_compare.py --base NAME=FILE --base NAME=FILE ... [--regions A-B:name,...]
        [--ids 20,8] [--bins 2048] [--out per_position.npz]

Rows are decoded from the format's uint16 log-probabilities (a row's floor is its maximum minus 16 nats, so a probability
below ~1e-7 of the top reads as that floor) and renormalized. numpy only; a base of 24,791 rows of the 201,216-entry
vocabulary is 10 GB, read in blocks (--block rows at a time).
"""
import argparse, struct, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--base', action='append', required=True)
ap.add_argument('--regions', default='')
ap.add_argument('--ids', default='20')
ap.add_argument('--out', default='')
ap.add_argument('--block', type=int, default=512)
ap.add_argument('--bins', type=int, default=0)
a = ap.parse_args()
names, files = zip(*[b.split('=', 1) for b in a.base])
hdr = None
maps = []
for f in files:
    with open(f, 'rb') as fh:
        assert fh.read(8) == b'_logits_'
        n_ctx, V, n_chunk = struct.unpack('<3i', fh.read(12))
        toks = np.frombuffer(fh.read(4 * n_ctx * n_chunk), dtype='<i4')
    if hdr is None:
        hdr = (n_ctx, V, n_chunk, toks)
    else:
        assert (n_ctx, V, n_chunk) == hdr[:3] and np.array_equal(toks, hdr[3]), f'{f}: another sequence'
    nv = 2 * ((V + 1) // 2) + 4
    first = n_ctx // 2
    n_tok = n_ctx - 1 - first
    off = 20 + 4 * n_ctx * n_chunk
    maps.append(np.memmap(f, dtype='<u2', mode='r', offset=off, shape=(n_tok, nv)))
n_ctx, V, _, toks = hdr
first = n_ctx // 2
n_tok = n_ctx - 1 - first
ids = [int(x) for x in a.ids.split(',') if x]
K = len(names)
pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
kld = np.zeros((len(pairs), n_tok), np.float32)
same_top = np.zeros((len(pairs), n_tok), bool)
p_next = np.zeros((K, n_tok), np.float32)
p_ids = np.zeros((K, len(ids), n_tok), np.float32)

def logp(rows):
    head = rows[:, :4].copy().view('<f4')          # [n, 2]: scale, min log-prob
    q = rows[:, 4:4 + V].astype(np.float32)
    return q * head[:, 0:1] + head[:, 1:2]

for s in range(0, n_tok, a.block):
    e = min(s + a.block, n_tok)
    lps = [logp(np.asarray(m[s:e])) for m in maps]
    nxt = toks[first + 1 + s:first + 1 + e]
    ps = []
    for k, lp in enumerate(lps):
        lse = np.log(np.exp(lp - lp.max(1, keepdims=True)).sum(1)) + lp.max(1)   # renormalize (the floor at max-16)
        lp -= lse[:, None]
        ps.append(np.exp(lp))
        p_next[k, s:e] = ps[k][np.arange(e - s), nxt]
        for t, i in enumerate(ids):
            p_ids[k, t, s:e] = ps[k][:, i]
    for pi, (i, j) in enumerate(pairs):
        kld[pi, s:e] = (ps[i] * (lps[i] - lps[j])).sum(1)      # KL(i || j): i the reference
        same_top[pi, s:e] = lps[i].argmax(1) == lps[j].argmax(1)
    print(f'  rows {e}/{n_tok}', file=sys.stderr, flush=True)

regions = [(first, n_ctx - 1, 'all scored')]
for r in a.regions.split(','):
    if r:
        span, nm = r.split(':', 1)
        x, y = span.split('-')
        regions.append((int(x), int(y), nm))
for x, y, nm in regions:
    lo, hi = max(x, first) - first, min(y, n_ctx - 1) - first
    if hi <= lo:
        continue
    print(f'== positions {lo + first}-{hi + first - 1} ({nm}, {hi - lo} rows)')
    for pi, (i, j) in enumerate(pairs):
        v = kld[pi, lo:hi]
        print(f'  KL({names[i]} || {names[j]}): mean {v.mean():.5f} median {np.median(v):.5f} p99 {np.quantile(v, 0.99):.4f} '
              f'max {v.max():.3f} | same top {100 * same_top[pi, lo:hi].mean():.2f}%')
    for k in range(K):
        pn = p_next[k, lo:hi]
        msg = f'  {names[k]:8s}: next token mean logp {np.log(np.maximum(pn, 1e-30)).mean():.4f}, mean p {pn.mean():.4f}'
        for t, i in enumerate(ids):
            pv = p_ids[k, t, lo:hi]
            msg += f' | p(id {i}) mean {pv.mean():.2e} max {pv.max():.3f} (>0.01 at {(pv > 0.01).sum()} rows)'
        print(msg)
if a.bins:
    print(f'== by depth, bins of {a.bins} positions: mean KL per pair | same top')
    for s0 in range(first, n_ctx - 1, a.bins):
        lo, hi = s0 - first, min(s0 + a.bins, n_ctx - 1) - first
        cells = ' | '.join(f'{kld[pi, lo:hi].mean():.4f} {100 * same_top[pi, lo:hi].mean():5.1f}%' for pi in range(len(pairs)))
        print(f'  {s0:6d}-{s0 + hi - lo - 1:6d}: {cells}')
if a.out:
    np.savez_compressed(a.out, first=first, names=np.array(names), pairs=np.array(pairs), kld=kld, same_top=same_top,
                        p_next=p_next, p_ids=p_ids, ids=np.array(ids), toks=toks)
