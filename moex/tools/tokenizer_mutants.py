"""moex/tools/tokenizer_mutants.py - does the corpus see a wrong tokenizer?  "0 mismatches" only means something if the same
comparison fails for plausible mistakes.  Each mutant below is one such mistake (a Unicode library's classes instead of
llama.cpp's table, Python's \\s, a missing ignore_merges, one regex stage dropped, ...).  It is applied to a family's
tokenizer and the corpus is run against the oracle; every mutant must disagree somewhere.

    python3 -I moex/tools/tokenizer_mutants.py [glm5] [deepseek4] [mimo2] [--scale 0.3]
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import sys

X = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(X / "tools"))
sys.path.insert(0, str(X / "moex" / "tools"))
sys.path.insert(0, str(X / "moex" / "tools" / "tok_oracle"))
import regex                           # noqa: E402
import strata_tokenizer_families as F  # noqa: E402
import strata_tokenizer as ST          # noqa: E402
import tokenizer_corpus as C           # noqa: E402
import verify_tokenizer as V           # noqa: E402


def native_pre(exprs, unicode_classes=True):
    """A pre-tokenizer written the way a Unicode library would do it: the `regex` package on the raw text, \\p{..} and \\s as
    Python knows them (this is what tools/strata_tokenizer.py does for Qwen)."""
    class P:
        stages = [(regex.compile(e), False) for e in exprs]

        def split(self, text):
            spans = [(0, len(text))]
            for rx, _ in self.stages:
                nxt = []
                for a, b in spans:
                    pos = a
                    for m in rx.finditer(text, a, b):
                        s, e = m.span()
                        if s > pos:
                            nxt.append((pos, s))
                        nxt.append((s, e))
                        pos = e
                    if pos < b:
                        nxt.append((pos, b))
                spans = nxt
            return [text[a:b] for a, b in spans]
    return P()


def mutate(tok: F.FamilyTokenizer, name: str):
    """(the mutant tokenizer, whether the mutation changes nothing for this family)."""
    t = copy.copy(tok)
    t._cache = {}
    t._noop = False
    exprs = tok.family_pre["exprs"]

    def rewritten(new):
        new = tuple(new)
        t._noop = new == tuple(exprs)
        return F.PreTokenizer(new)
    if name == "regex_package_classes":
        t.pretok = native_pre(exprs)
    elif name == "python_whitespace":
        t.pretok = F.PreTokenizer(exprs)
        old = F._SPACE_BODY
        # \s as str.isspace has it: adds U+001C..U+001F to the whitespace of the collapsed text
        F._SPACE_BODY = old + "\\x1c-\\x1f"
        try:
            t.pretok = F.PreTokenizer(exprs)
        finally:
            F._SPACE_BODY = old
    elif name == "no_ignore_merges":
        t.ignore_merges = False
    elif name == "force_ignore_merges":
        t.ignore_merges = True
    elif name == "digits_single":
        t.pretok = rewritten(e.replace("\\p{N}{1,3}", "\\p{N}") for e in exprs)
    elif name == "digits_triple":
        t.pretok = rewritten(e.replace("|\\p{N}|", "|\\p{N}{1,3}|") for e in exprs)
    elif name == "drop_first_stage":
        t.pretok = F.PreTokenizer(exprs[1:])
    elif name == "drop_middle_stage":
        t.pretok = rewritten(exprs[:1] + exprs[2:]) if len(exprs) == 3 else rewritten(exprs)
    elif name == "drop_last_stage":
        t.pretok = F.PreTokenizer(exprs[:-1])
    elif name == "marks_in_words":      # qwen35's \p{M} letters
        t.pretok = rewritten(e.replace("\\p{L}+", "[\\p{L}\\p{M}]+").replace("[^\\s\\p{L}\\p{N}]", "[^\\s\\p{L}\\p{M}\\p{N}]") for e in exprs)
    elif name == "lowercase_contractions_only":
        t.pretok = F.PreTokenizer(tuple(e.replace("[sS]", "s").replace("[tT]", "t").replace("[rR][eE]", "re").replace("[vV][eE]", "ve")
                                        .replace("[mM]", "m").replace("[lL][lL]", "ll").replace("[dD]", "d") for e in exprs))
    elif name == "control_always_special":
        t._always_re = t._special_re
    elif name == "user_defined_needs_parse":
        t._always_re = None
    elif name == "no_loader_control_override":
        base = [F.TYPE_TO_ATTR.get(x, 0) for x in tok.token_types]
        specials = [i for i, a in enumerate(base) if a & F.SPECIAL_ATTRS]
        t.special_tokens = {tok.tokens[i]: i for i in sorted(specials, key=lambda i: -len(tok.tokens[i]))}
        always = [tok.tokens[i] for i in specials if not base[i] & (F.ATTR_CONTROL | F.ATTR_UNKNOWN)]
        t._always_re = tok._alt(always)
        t._special_re = tok._alt(list(t.special_tokens))
    elif name == "shortest_special_first":
        t._special_re = regex.compile("|".join(regex.escape(s) for s in sorted(tok.special_tokens, key=len)))
    elif name == "no_marks_collapse":    # non-ASCII kept as is: the classes of an ASCII-only table
        t.pretok = F.PreTokenizer(exprs)
        F_collapse = F.collapse
        F.collapse = lambda s: s
        t._restore = lambda: setattr(F, "collapse", F_collapse)
    else:
        raise KeyError(name)
    return t, t._noop


MUTANTS = ["regex_package_classes", "python_whitespace", "no_ignore_merges", "force_ignore_merges", "digits_single", "digits_triple",
           "drop_first_stage", "drop_middle_stage", "drop_last_stage", "marks_in_words", "lowercase_contractions_only", "control_always_special",
           "user_defined_needs_parse", "no_loader_control_override", "no_marks_collapse"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="*", default=["glm5", "deepseek4", "mimo2"])
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    survivors = 0
    for fam in a.families:
        pair = V.Pair(fam)
        corpus = C.build(pair.tok, seed=1, scale=a.scale)
        items = [(cat, s) for cat, ss in corpus.items() for s in ss]
        base_ok = pair.tok
        refs = {}
        for flags in V.FLAGS:
            refs[flags] = pair.ref_many([s for _, s in items], *flags)
        print("== %s: %d strings x 4 flag combinations" % (fam, len(items)))
        for name in (a.only.split(",") if a.only else MUTANTS):
            if name in ("force_ignore_merges", "no_ignore_merges") and False:
                continue
            tok, noop = mutate(base_ok, name)
            bad, first = 0, None
            for flags in V.FLAGS:
                for (cat, s), r in zip(items, refs[flags]):
                    if r is None:
                        continue
                    try:
                        m = tok.encode(s, parse_special=flags[1], add_special=flags[0])
                    except Exception as e:      # a mutant may break outright
                        m = ("error", repr(e))
                    if m != r:
                        bad += 1
                        first = first or (cat, V.show(s, 40))
            if hasattr(tok, "_restore"):
                tok._restore()
            same_as_base = noop or (name == "no_ignore_merges" and not base_ok.ignore_merges) or (
                name == "force_ignore_merges" and base_ok.ignore_merges)
            note = "  (a no-op for this family: it already is so)" if same_as_base else ""
            if bad == 0 and not same_as_base:
                survivors += 1
            print("   %-30s %6d disagreements%s%s" % (name, bad, ("  first: %s %s" % first) if first else "", note))
        pair.oracle.close()
    print("mutants the corpus did not detect: %d" % survivors)
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
