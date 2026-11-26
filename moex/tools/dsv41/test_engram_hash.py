"""moex/tools/dsv41/test_engram_hash.py - engram_tables.py + engram_hash.hpp against the release's own engram.py.

    python test_engram_hash.py --model-dir DIR --reference DIR_WITH_engram.py --exe engram_hash_dump [--tables F]

Checks, with no GPU and no weights:
  1. the tables (token map, compressed vocab, pad id, multipliers, primes, offsets) equal the reference module's
     buffers;
  2. the C++ row numbers equal NgramHashState's for real text, random ids, image (dead) spans, sequence starts, and a
     prefill followed by one-token decode steps (the reference's cache carries the look-back across the split).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engram_tables  # noqa: E402


def load_reference(ref_dir: pathlib.Path):
    spec = importlib.util.spec_from_file_location("dsv41_ref_engram", ref_dir / "engram.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=pathlib.Path)
    ap.add_argument("--reference", required=True, type=pathlib.Path)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--tables", type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    ref = load_reference(a.reference)
    tc = json.loads((a.model_dir / "config.json").read_text())
    tc = tc.get("text_config", tc)
    args = types.SimpleNamespace(
        engram_layer_ids=tc["engram_layer_ids"], engram_max_ngram_size=tc["engram_max_ngram_size"],
        engram_n_heads=tc["engram_n_heads"], engram_vocab_size=tc["engram_vocab_size"],
        engram_num_embeddings=tc["engram_num_embeddings"], engram_head_dim=tc["engram_head_dim"],
        engram_pad_id=engram_tables.pad_token(tc), engram_compressed_vocab_size=tc["engram_compressed_vocab_size"],
        max_batch_size=1, max_seq_len=8192)
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_dir))
    layout = ref.EngramLayout.from_args(args)
    state = ref.NgramHashState(args, layout, tokenizer)

    # 1. tables
    tables = engram_tables.build(a.model_dir)
    fails = []
    def same(name, mine, theirs):
        if not np.array_equal(np.asarray(mine, dtype=np.int64), np.asarray(theirs, dtype=np.int64)):
            fails.append(name)
        print(f"  {name:12s} {'ok' if name not in fails else 'DIFFERENT'}")
    print("tables vs the reference module:")
    same("token_map", tables["token_map"], state.token_map.numpy())
    same("pad_id", [tables["pad_id"]], [state.pad_id])
    same("multipliers", tables["multipliers"], state.multipliers.numpy())
    same("primes", tables["primes"], state.primes.numpy())
    same("offsets", tables["offsets"], state.offsets.numpy())
    print(f"  compressed vocab {tables['n_compressed']} (config {tc['engram_compressed_vocab_size']})")

    tables_path = a.tables
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="dsv41-engram-"))
    if tables_path is None:
        tables_path = tmp / "engram_tables.bin"
        engram_tables.write(tables, tables_path)

    # 2. row numbers
    rng = np.random.default_rng(a.seed)
    text = (pathlib.Path(__file__).read_text() + (HERE.parent.parent / "docs" / "DSV41_PLAN.md").read_text())
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    seqs = []  # (tokens, dead, prefill)
    seqs.append((text_ids[:3000], [False] * 3000, 2900))                      # prose/code, 100 decode steps
    seqs.append((text_ids[3000:3005], [False] * 5, 1))                        # short, decode from position 1
    r = rng.integers(0, len(tokenizer), size=2000).tolist()                   # every id, incl. specials
    seqs.append((r, [False] * 2000, 1500))
    img = text_ids[:1200]
    dead = [False] * 1200
    for s0, s1 in ((10, 40), (500, 501), (1100, 1180)):                       # image spans, one near the split
        dead[s0:s1] = [True] * (s1 - s0)
    seqs.append((img, dead, 1150))
    seqs.append(([engram_tables.pad_token(tc)] * 6 + text_ids[:50], [False] * 56, 0))  # pad tokens; all decode
    seqs.append((text_ids[:64], [True] + [False] * 63, 32))                   # dead first token

    inp = tmp / "in.bin"
    with open(inp, "wb") as f:
        f.write(struct.pack("<I", len(seqs)))
        for toks, dd, pre in seqs:
            f.write(struct.pack("<II", len(toks), pre))
            f.write(np.asarray(toks, dtype="<i4").tobytes())
            f.write(np.asarray(dd, dtype=np.uint8).tobytes())
    out = tmp / "out.bin"
    subprocess.run([a.exe, str(tables_path), str(inp), str(out)], check=True)
    got = np.fromfile(out, dtype="<i8")

    L, C = len(tc["engram_layer_ids"]), (tc["engram_max_ngram_size"] - 1) * tc["engram_n_heads"]
    k, total = 0, 0
    print("row numbers vs NgramHashState (prefill, then one token per step):")
    for si, (toks, dd, pre) in enumerate(seqs):
        ids = torch.tensor([toks])
        mask = ~torch.tensor([dd])
        parts = []
        if pre:
            parts.append(state(ids[:, :pre], 0, mask[:, :pre]))
        for p in range(pre, len(toks)):
            parts.append(state(ids[:, p:p + 1], p, mask[:, p:p + 1]))
        want = torch.cat(parts, dim=1).reshape(-1).numpy()
        n = len(toks) * L * C
        mine = got[k:k + n]
        k += n
        bad = int((mine != want).sum())
        total += bad
        print(f"  seq {si}: {len(toks):5d} tokens, prefill {pre:5d}: {n:7d} rows, {bad} different")
        if bad:
            fails.append(f"seq {si}")
    if k != got.size:
        fails.append("output size")
    print("PASS" if not fails else f"FAIL: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    raise SystemExit(main())
