"""moex/tools/bench_tokenizer.py - speed of the pure-Python tokenizers on a 100 KB text, and the heap path for long words.

    python3 -I moex/tools/bench_tokenizer.py [glm5 deepseek4 mimo2 qwen]

For each tokenizer: tokens per second on 100 KB of mixed text (English prose, source code, Chinese; one fresh instance, so
the word cache starts empty) and again with the cache warm, on a worst-case text with no repeated words, and the time of a
5,000-character single word (above 64 symbols the BPE takes strata_tokenizer's heap path: `_bpe_heap`).  Qwen's reference
tokenizer is included for comparison.
"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
import json
import pathlib
import random
import sys
import time

X = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(X / "tools"))
sys.path.insert(0, str(X / "moex" / "tools"))
import strata_tokenizer as ST            # noqa: E402
import strata_tokenizer_families as F    # noqa: E402

DATA = pathlib.Path(_os.path.join(_DATA, 'tokenizers'))


def text_100k(seed=3) -> str:
    rng = random.Random(seed)
    parts = []
    for p in sorted(X.glob("README*.md")) + sorted((X / "docs").glob("*.md"))[:10]:
        parts.append(p.read_text(encoding="utf-8"))
    for p in sorted((X / "serve").glob("*.py"))[:6]:
        parts.append(p.read_text(encoding="utf-8"))
    big = "\n".join(parts)
    out, n = [], 0
    while n < 100_000:
        a = rng.randrange(0, len(big) - 3000)
        chunk = big[a:a + rng.randint(500, 3000)]
        out.append(chunk)
        n += len(chunk.encode("utf-8"))
    return "".join(out).encode("utf-8")[:100_000].decode("utf-8", "ignore")


def novel_text(seed=5, nbytes=100_000) -> str:
    """No repeated word at all: random 'words' of random letters, so the cache never helps."""
    rng = random.Random(seed)
    alpha = "abcdefghijklmnopqrstuvwxyz"
    words, n = [], 0
    while n < nbytes:
        w = "".join(rng.choice(alpha) for _ in range(rng.randint(3, 11)))
        words.append(w)
        n += len(w) + 1
    return " ".join(words)


def load(name):
    if name == "qwen":
        d = DATA / "qwen3.8-flash-next"
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        return ST.Tokenizer(tokens, (d / "merges.txt").read_text(encoding="utf-8").split("\n"),
                            json.loads((d / "token_type.json").read_text()))
    return F.FamilyTokenizer.from_dir(DATA / name)


def main():
    names = sys.argv[1:] or ["qwen", "glm5", "deepseek4", "mimo2"]
    text, novel = text_100k(), novel_text()
    print("text: %d characters, %d bytes (prose, source code, Chinese)" % (len(text), len(text.encode("utf-8"))))
    for name in names:
        tok = load(name)
        t = time.perf_counter()
        ids = tok.encode(text, parse_special=True)
        cold = time.perf_counter() - t
        t = time.perf_counter()
        ids2 = tok.encode(text, parse_special=True)
        warm = time.perf_counter() - t
        assert ids == ids2
        tok2 = load(name)
        t = time.perf_counter()
        nids = tok2.encode(novel, parse_special=True)
        nov = time.perf_counter() - t
        word = "".join(random.Random(9).choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5000))
        calls = []
        orig = tok._bpe_heap
        tok._bpe_heap = lambda w, orig=orig: (calls.append(len(w)), orig(w))[1]
        t = time.perf_counter()
        wids = tok.encode(word)
        tw = time.perf_counter() - t
        print("%-10s %7d tokens  cold %.2f s = %6.0f tok/s | warm %.2f s = %7.0f tok/s | no repeated words: %d tokens %.2f s = %6.0f tok/s | "
              "5000-char word: %d tokens, %.1f ms, heap path used: %s"
              % (name, len(ids), cold, len(ids) / cold, warm, len(ids) / warm, len(nids), nov, len(nids) / nov, len(wids), tw * 1000, bool(calls)))


if __name__ == "__main__":
    main()
