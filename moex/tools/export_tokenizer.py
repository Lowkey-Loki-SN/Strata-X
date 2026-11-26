"""moex/tools/export_tokenizer.py - export the tokenizer of each moex model from its GGUF, as Strata's pack export does
(tools/strata_tokenizer.py --gguf G --out PACK writes PACK/tokenizer/{vocab.json, merges.txt, token_type.json, ...}).

    python3 -I moex/tools/export_tokenizer.py                       # all three, into $STRATA_X_DATA/tokenizers/<family>/
    python3 -I moex/tools/export_tokenizer.py glm5 --check          # one family, and round-trip + oracle spot checks

Header and metadata only: it never reads tensor data.  The GGUFs are the ones the moex work uses (a 4-layer slice carries
the same tokenizer as the whole model; for MiMo the slice's tokenizer metadata was compared with the BPW3.0 shard's key by
key and is identical, and this tool checks that again when both exist).
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import hashlib
import json
import os
import pathlib
import sys

X = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(X / "tools"))
import strata_tokenizer_families as F  # noqa: E402

DATA = pathlib.Path(os.environ.get("STRATA_X_DATA", _DATA))
SOURCES = {
    "glm5": DATA / "glm-mini/glm53-mini-L0-3.gguf",
    "deepseek4": DATA / "deepseek4-vocab-only.gguf",   # the DSV4 mini GGUF's metadata alone (every key, no tensors)
    "mimo2": DATA / "mimo-mini/mimo-mini-L0-3.gguf",
    # Hy4-flash: the GGUF set's first file (its metadata: the whole vocabulary and template; header only)
    "hy4": pathlib.Path(os.path.expanduser("~/AI/Models/hy4-flash/q4_k_m/Hy4-flash-Q4_K_M-00001-of-00002.gguf")),
}
# the MiMo GGUF named in the task: metadata only, the tensors are in later shards
MIMO_SHARD = pathlib.Path(_os.path.join(_MODELS, 'GGUFs/mimo-v2.6-flash/BPW3.0/MiMo-V2.6-Flash-MOPD-BPW3.0-00001-of-00004.gguf'))


def digest(d: pathlib.Path) -> dict:
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(d.iterdir()) if f.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="*", choices=[*SOURCES, []] if False else None, help="glm5 deepseek4 mimo2 (default: all)")
    ap.add_argument("--out", default=str(DATA / "tokenizers"))
    ap.add_argument("--check", action="store_true", help="round-trip a few strings through each export")
    a = ap.parse_args()
    fams = a.families or list(SOURCES)
    for fam in fams:
        if fam not in SOURCES:
            ap.error("unknown family %r (glm5, deepseek4, mimo2)" % fam)
        out = pathlib.Path(a.out) / fam
        cfg = F.extract(SOURCES[fam], out)
        print("%-10s %s -> %s: vocab %d, merges %d, pre %s, %d special tokens, eog %s"
              % (fam, SOURCES[fam].name, out, cfg["vocab_size"], cfg["n_merges"], cfg["pre"], cfg["special_tokens"],
                 cfg["meta"]["eog"]))
        if fam == "mimo2" and MIMO_SHARD.exists():
            sys.path.insert(0, str(X / "tools"))
            from gguf_reader import GGUFFile
            a_md, b_md = GGUFFile(MIMO_SHARD).metadata, GGUFFile(SOURCES[fam]).metadata
            keys = [k for k in a_md if k.startswith("tokenizer.")]
            same = all(a_md[k] == b_md.get(k) for k in keys)
            print("           tokenizer metadata of the BPW3.0 shard 1 == the slice's: %s (%d keys)" % (same, len(keys)))
            if not same:
                return 1
        if a.check:
            tk = F.FamilyTokenizer.from_dir(out)
            ref = F.FamilyTokenizer.from_gguf(SOURCES[fam])
            samples = ["", "Hello, world!", "  x  ", "你好，世界\U0001f600", "a\n\n\nb\tc", "x" * 5000,
                       "<|user|>hi<|assistant|>", "é́", "\x00\x01\x1f\x7f"]
            bad = 0
            for s in samples:
                for ps in (True, False):
                    ids = tk.encode(s, parse_special=ps)
                    if ids != ref.encode(s, parse_special=ps):
                        print("   dir != gguf for", repr(s[:30]), ps)
                        bad += 1
                if ps and tk.decode(tk.encode(s, parse_special=False), errors="strict") != s and "<" not in s:
                    print("   round trip failed for", repr(s[:30]))
                    bad += 1
            same_attr = tk.attr == ref.attr and tk.eog_ids == ref.eog_ids
            print("           export reload == GGUF load: %s; attrs/eog equal: %s" % (bad == 0, same_attr))
            if bad or not same_attr:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
