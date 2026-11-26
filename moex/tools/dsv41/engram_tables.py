"""moex/tools/dsv41/engram_tables.py - the fixed inputs of DeepSeek-V4.1's engram hash, built once per model.

The engine hashes n-grams of compressed token ids (src: the release's inference/engram.py). Two of its inputs are
expensive to reproduce in C++ and never change for a model, so they are computed here and stored in the pack:
  - the compressed token map (HF tokenizers' NFKC/NFD/accent/lowercase/whitespace normalizer over every token id);
  - the hash multipliers (numpy PCG64 per layer) and the per-(layer, n-gram size, head) primes and row offsets.

    python engram_tables.py --model-dir DIR_WITH_config.json_AND_tokenizer.json --out engram_tables.bin

File layout (little-endian):
  char[8] "DSV41ENG", u32 version (1),
  u32 n_vocab, u32 n_compressed, i32 pad_id (compressed), u32 n_layers, u32 max_ngram, u32 n_heads, u32 head_dim,
  u32 layer_ids[n_layers], u64 num_embeddings[n_layers],
  i64 multipliers[n_layers][max_ngram],
  i64 primes[n_layers][max_ngram - 1][n_heads],
  i64 offsets[n_layers][(max_ngram - 1) * n_heads],
  i32 token_map[n_vocab]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct

import numpy as np

MAGIC = b"DSV41ENG"


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def pad_token(tc: dict) -> int:
    """The token that fills n-gram slots with no history: `engram_pad_id` in the release's inference/config.json,
    `engram_pad_token_id` in its HF config.json."""
    for key in ("engram_pad_id", "engram_pad_token_id"):
        if key in tc:
            return int(tc[key])
    raise SystemExit("config has no engram pad token id")


def compressed_token_map(model_dir: pathlib.Path) -> tuple[list[int], int]:
    """The release's build_compressed_token_map, over the tokenizer the release's generate.py loads."""
    from tokenizers import Regex, normalizers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def build(model_dir: pathlib.Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    layer_ids = list(tc["engram_layer_ids"])
    max_ngram, n_heads = tc["engram_max_ngram_size"], tc["engram_n_heads"]

    token_map, n_compressed = compressed_token_map(model_dir)
    if n_compressed != tc["engram_compressed_vocab_size"]:
        raise SystemExit(f"compressed vocab {n_compressed} != config {tc['engram_compressed_vocab_size']}: "
                         "the tokenizer or the normalizer differs from training, every hash would be wrong")

    # primes: drawn in order over all layers, never reused (EngramLayout.from_args)
    seen: set[int] = set()
    primes = []
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram - 1):
            sizes, cur = [], tc["engram_vocab_size"] - 1
            for _ in range(n_heads):
                cur += 1
                while not is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    offsets = []
    for li, per_ngram in enumerate(primes):
        flat = [p for sizes in per_ngram for p in sizes]
        if sum(flat) != tc["engram_num_embeddings"][li]:
            raise SystemExit(f"layer {layer_ids[li]}: primes sum to {sum(flat)}, table has "
                             f"{tc['engram_num_embeddings'][li]} rows")
        offsets.append(list(np.cumsum([0, *flat[:-1]]).astype(np.int64)))

    # multipliers (compute_hash_multipliers): odd, bounded so id * m cannot overflow int64
    bound = max(1, (np.iinfo(np.int64).max // n_compressed) // 2)
    multipliers = []
    for layer_id in layer_ids:
        values = np.random.default_rng(10007 * layer_id).integers(low=0, high=bound, size=(max_ngram,),
                                                                  dtype=np.int64)
        multipliers.append([int(v) * 2 + 1 for v in values])

    return dict(n_vocab=len(token_map), n_compressed=n_compressed, pad_id=token_map[pad_token(tc)],
                layer_ids=layer_ids, num_embeddings=list(tc["engram_num_embeddings"]), max_ngram=max_ngram,
                n_heads=n_heads, head_dim=tc["engram_head_dim"], multipliers=multipliers, primes=primes,
                offsets=offsets, token_map=token_map)


def write(t: dict, path: pathlib.Path) -> None:
    L = len(t["layer_ids"])
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", 1))
        f.write(struct.pack("<IIiIIII", t["n_vocab"], t["n_compressed"], t["pad_id"], L, t["max_ngram"],
                            t["n_heads"], t["head_dim"]))
        f.write(struct.pack(f"<{L}I", *t["layer_ids"]))
        f.write(struct.pack(f"<{L}Q", *t["num_embeddings"]))
        f.write(np.asarray(t["multipliers"], dtype="<i8").tobytes())
        f.write(np.asarray(t["primes"], dtype="<i8").tobytes())
        f.write(np.asarray(t["offsets"], dtype="<i8").tobytes())
        f.write(np.asarray(t["token_map"], dtype="<i4").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    t = build(a.model_dir)
    write(t, a.out)
    print(f"{a.out}: vocab {t['n_vocab']} -> {t['n_compressed']} compressed ids, pad {t['pad_id']}, "
          f"layers {t['layer_ids']}, rows {t['num_embeddings']}")


if __name__ == "__main__":
    main()
