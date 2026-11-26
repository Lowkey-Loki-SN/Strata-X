"""moex/tools/verify_tokenizer.py - the family tokenizers against llama.cpp's own tokenizer (the oracle).

    python3 -I moex/tools/verify_tokenizer.py glm5 [deepseek4 mimo2] [--sweep] [--scale 1.0] [--seed 1] [--report DIR]

Per family it checks, in this order, and exits non-zero on any difference:
  vocab    the loader's decisions: bos/eos/eot/pad/nl and the add_* flags, the end-of-generation set, the attrs of every
           token, and every token's text and bytes as llama_token_to_piece writes it (special=true)
  corpus   tools/tokenizer_corpus.py's strings (tens of thousands) with all four flag combinations of llama_tokenize
           (add_special x parse_special); each disagreement is shrunk to a small string that still disagrees
  sweep    (--sweep) every Unicode code point in four contexts, with parse_special on and off
  pieces   decode(encode(s)) of the corpus against llama_detokenize / the joined pieces
The oracle is moex/build/tok_oracle (vocabulary only: no weights, no GPU); build it with moex/tools/tok_oracle/build.sh.
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")

import argparse
import collections
import json
import os
import pathlib
import sys
import time

X = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(X / "tools"))
sys.path.insert(0, str(X / "moex" / "tools"))
sys.path.insert(0, str(X / "moex" / "tools" / "tok_oracle"))
import strata_tokenizer_families as F  # noqa: E402
import tokenizer_corpus as C           # noqa: E402
from oracle import Oracle              # noqa: E402

DATA = pathlib.Path(os.environ.get("STRATA_X_DATA", _DATA))
# deepseek4: the DSV4 mini GGUF's metadata alone (every key, no tensors: the oracle loads vocab only); DeepSeek-V4.1's
# tokenizer.json has the same vocabulary and merges (checked 2026-10-19)
GGUF = {"glm5": DATA / "glm-mini/glm53-mini-L0-3.gguf", "deepseek4": DATA / "deepseek4-vocab-only.gguf",
        "mimo2": DATA / "mimo-mini/mimo-mini-L0-3.gguf"}


def show(s: str, n: int = 90) -> str:
    r = ascii(s)
    return r if len(r) <= n else r[:n - 3] + "..."


class Pair:
    """The tokenizer under test and the oracle, with the oracle restarted if a string kills it (it uses std::regex for
    some pre-tokenizers, which can overflow its stack on very long inputs)."""

    def __init__(self, fam: str, tokdir: pathlib.Path | None = None):
        self.fam = fam
        self.gguf = GGUF[fam]
        self.tok = F.FamilyTokenizer.from_dir(tokdir or (DATA / "tokenizers" / fam))
        self.oracle = Oracle(str(self.gguf))
        self.crashes = 0

    def ref(self, s: str, add_special: bool, parse_special: bool):
        try:
            return self.oracle.tokenize(s, add_special, parse_special)
        except (EOFError, BrokenPipeError, OSError):
            self.crashes += 1
            self.oracle = Oracle(str(self.gguf))
            return None

    def ref_many(self, texts, add_special, parse_special):
        try:
            return self.oracle.tokenize_many(texts, add_special, parse_special)
        except (EOFError, BrokenPipeError, OSError):
            self.crashes += 1
            self.oracle = Oracle(str(self.gguf))
            return [self.ref(t, add_special, parse_special) for t in texts]

    def mine(self, s, add_special, parse_special):
        return self.tok.encode(s, parse_special=parse_special, add_special=add_special)

    def differs(self, s, add_special, parse_special) -> bool:
        r = self.ref(s, add_special, parse_special)
        return r is not None and r != self.mine(s, add_special, parse_special)


def shrink(pair: Pair, s: str, flags, budget: int = 300) -> str:
    """The smallest string (by chunk removal) that still disagrees with the oracle."""
    chunk = max(1, len(s) // 2)
    while chunk >= 1 and budget > 0 and len(s) > 1:
        i, changed = 0, False
        while i < len(s) and budget > 0:
            cand = s[:i] + s[i + chunk:]
            budget -= 1
            if cand and pair.differs(cand, *flags):
                s, changed = cand, True
            else:
                i += chunk
        if chunk == 1 and not changed:
            break
        chunk = chunk if changed and chunk == 1 else max(1, chunk // 2)
    return s


FLAGS = [(False, True), (False, False), (True, True), (True, False)]       # (add_special, parse_special)


def check_vocab(pair: Pair, report: dict) -> int:
    tok, o = pair.tok, pair.oracle
    bad = 0
    info = o.info()
    mine = {"n_vocab": len(tok.tokens), "bos": tok.role_ids.get("bos"), "eos": tok.role_ids.get("eos"), "eot": tok.role_ids.get("eot"),
            "pad": tok.role_ids.get("pad"), "add_bos": int(tok.add_bos), "add_eos": int(tok.add_eos), "eog": tok.eog_ids,
            "fim_pre": tok.role_ids.get("fim_pre"), "fim_suf": tok.role_ids.get("fim_suf"), "fim_mid": tok.role_ids.get("fim_mid"),
            "fim_pad": tok.role_ids.get("fim_pad"), "fim_rep": tok.role_ids.get("fim_rep"), "fim_sep": tok.role_ids.get("fim_sep")}
    diffs = {}
    for k, v in mine.items():
        ref = info[k]
        if k in ("bos", "eos", "eot", "pad", "fim_pre", "fim_suf", "fim_mid", "fim_pad", "fim_rep", "fim_sep"):
            v = -1 if v is None else v
        if v != ref:
            diffs[k] = (v, ref)
    # llama.cpp's default bos for a gpt2 vocab is token 11 when the GGUF gives none; add_bos is false then, so it never matters
    diffs.pop("bos", None) if info["add_bos"] == 0 and tok.role_ids.get("bos") is None else None
    print("  vocab: info %s" % ("equal" if not diffs else "DIFFERENT %s" % diffs))
    bad += len(diffs)
    report["info"] = {"mine": mine, "oracle": info, "diffs": {k: list(v) for k, v in diffs.items()}}
    attrs = o.attrs()
    attr_diff = [(i, tok.attr[i], a) for (i, a, e, t) in attrs if tok.attr[i] != a]
    eog_diff = [i for (i, a, e, t) in attrs if e != (i in set(tok.eog_ids))]
    text_diff = [i for (i, a, e, t) in attrs if t != tok.tokens[i].encode("utf-8")]
    print("  vocab: %d tokens: attr differences %d, eog differences %d, text differences %d"
          % (len(attrs), len(attr_diff), len(eog_diff), len(text_diff)))
    bad += len(attr_diff) + len(eog_diff) + len(text_diff)
    report["attrs"] = {"attr_diff": attr_diff[:50], "eog_diff": eog_diff[:50], "text_diff": text_diff[:50]}
    piece_diff = []
    for i in range(len(tok.tokens)):
        if tok.token_bytes(i) != o.piece(i, True):
            piece_diff.append(i)
    print("  vocab: token bytes (llama_token_to_piece, special=true) differ for %d of %d tokens" % (len(piece_diff), len(tok.tokens)))
    bad += len(piece_diff)
    report["pieces"] = {"diff": piece_diff[:50]}
    return bad


def check_corpus(pair: Pair, corpus: dict[str, list[str]], report: dict, only=None) -> int:
    mism = []
    t0 = time.time()
    total = 0
    per_cat = collections.OrderedDict()
    for cat, items in corpus.items():
        if only and cat not in only:
            continue
        bad_cat = 0
        for flags in FLAGS:
            refs = pair.ref_many(items, *flags)
            for i, (s, r) in enumerate(zip(items, refs)):
                total += 1
                if r is None:                         # the oracle died on it: nothing to compare with
                    mism.append({"category": cat, "index": i, "flags": flags, "text": show(s, 200), "mine": None, "oracle": "CRASH",
                                 "length": len(s)})
                    continue
                m = pair.mine(s, *flags)
                if m != r:
                    bad_cat += 1
                    k = next((j for j in range(min(len(m), len(r))) if m[j] != r[j]), min(len(m), len(r)))
                    mism.append({"category": cat, "index": i, "flags": flags, "text": show(s, 200), "length": len(s),
                                 "first_diff_at_token": k, "mine": m[max(0, k - 2):k + 6], "oracle": r[max(0, k - 2):k + 6],
                                 "_s": s})
        per_cat[cat] = (len(items), bad_cat)
    # shrink each distinct mismatch (a string that disagrees under several flag combinations is shrunk once per combination)
    seen = set()
    for m in mism:
        s = m.pop("_s", None)
        if s is None or (m["category"], m["text"]) in seen and len(seen) > 400:
            continue
        seen.add((m["category"], m["text"]))
        m["shrunk"] = ascii(shrink(pair, s, m["flags"])) if len(s) > 1 else ascii(s)
    print("  corpus: %d categories, %d strings x 4 flag combinations = %d comparisons in %.1f s; mismatches: %d; oracle crashes: %d"
          % (len(per_cat), sum(n for n, _ in per_cat.values()), total, time.time() - t0, len([m for m in mism if m['oracle'] != 'CRASH']),
             pair.crashes))
    for cat, (n, b) in per_cat.items():
        print("    %-18s %6d strings  %5d mismatches" % (cat, n, b))
    report["corpus"] = {"per_category": per_cat, "mismatches": mism, "comparisons": total, "oracle_crashes": pair.crashes}
    return len([m for m in mism if m["oracle"] != "CRASH"])


def check_sweep(pair: Pair, report: dict, step: int = 1) -> int:
    contexts = [("a%sb", "a{}b"), ("%s", "{}"), (" %sa", " {}a"), ("%s\n1", "{}\n1")]
    mism = []
    t0 = time.time()
    cps = [c for c in range(0, 0x110000, step) if not 0xD800 <= c <= 0xDFFF]
    n = 0
    for pat, _ in contexts:
        for parse in (True, False):
            for lo in range(0, len(cps), 20000):
                chunk = cps[lo:lo + 20000]
                texts = [pat % chr(c) for c in chunk]
                refs = pair.ref_many(texts, False, parse)
                for c, s, r in zip(chunk, texts, refs):
                    n += 1
                    if r is None:
                        mism.append({"cp": "U+%04X" % c, "context": pat, "parse_special": parse, "oracle": "CRASH"})
                        continue
                    m = pair.mine(s, False, parse)
                    if m != r:
                        mism.append({"cp": "U+%04X" % c, "context": pat, "parse_special": parse, "mine": m, "oracle": r})
    print("  sweep: %d code points x %d contexts x 2 = %d comparisons in %.0f s; mismatches: %d" % (len(cps), len(contexts), n,
                                                                                                time.time() - t0, len(mism)))
    for m in mism[:12]:
        print("    ", m)
    report["sweep"] = {"comparisons": n, "mismatches": mism}
    return len(mism)


def check_pieces(pair: Pair, corpus: dict[str, list[str]], report: dict, limit: int = 4000) -> int:
    """decode() of what the oracle tokenized against llama.cpp's own detokenize (and the piece-by-piece join)."""
    tok, o = pair.tok, pair.oracle
    bad = 0
    items = [s for cat in ("prose_en", "scripts", "emoji", "code_py", "json", "special_real", "fuzz") for s in corpus.get(cat, [])[:600]][:limit]
    diffs = []
    for s in items:
        ids = o.tokenize(s, False, True)
        mine = tok.decode(ids)
        ref_join = b"".join(o.piece(i, True) for i in ids).decode("utf-8", "replace")
        if mine != ref_join:
            bad += 1
            diffs.append((ascii(s)[:80], ascii(mine)[:80], ascii(ref_join)[:80]))
    print("  pieces: decode(oracle ids) vs joined llama_token_to_piece: %d of %d differ" % (bad, len(items)))
    report["pieces_decode"] = {"checked": len(items), "diffs": diffs[:30]}
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="*", default=list(GGUF))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-step", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated corpus categories")
    ap.add_argument("--report", default=str(X / "moex" / "out"))
    a = ap.parse_args()
    failed = 0
    for fam in a.families:
        if fam not in GGUF:
            ap.error("unknown family %r" % fam)
        print("== %s (%s)" % (fam, GGUF[fam].name))
        pair = Pair(fam)
        report = {"family": fam, "gguf": str(GGUF[fam]), "seed": a.seed, "scale": a.scale}
        bad = check_vocab(pair, report)
        corpus = C.build(pair.tok, seed=a.seed, scale=a.scale)
        bad += check_corpus(pair, corpus, report, set(filter(None, a.only.split(","))))
        bad += check_pieces(pair, corpus, report)
        if a.sweep:
            bad += check_sweep(pair, report, a.sweep_step)
        out = pathlib.Path(a.report)
        out.mkdir(parents=True, exist_ok=True)
        (out / ("tokenizer-%s.json" % fam)).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        print("  => %s: %d differences" % (fam, bad))
        failed += bad
        pair.oracle.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
