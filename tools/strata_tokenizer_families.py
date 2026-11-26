"""tools/strata_tokenizer_families.py - the tokenizers of the model families beyond Qwen: GLM-5 (pre `glm5`),
DeepSeek-V4 (pre `joyai-llm`) and MiMo-V2 (pre `qwen2`).

WHY THIS IS A SECOND MODULE.  tools/strata_tokenizer.py is the reference tokenizer of the Qwen3.8-Flash-Next pack and
its behaviour is pinned by the live install, so it is left exactly as it is.  This module imports it, reuses its
byte-level BPE (`_bpe`, with the heap path for long words, `BYTE_TO_UNICODE`) and replaces only what differs per
family: which characters end a word (the pre-tokenizer), which strings are special tokens, whether a word that is
already a token skips the merges, and how a token turns back into bytes.  The oracle for all of it is llama.cpp's own
tokenizer (src/llama-vocab.cpp and src/unicode.cpp of the glm5next-mtp tree); moex/tools/tok_oracle runs it behind a
pipe and moex/tools/verify_tokenizer.py compares this module with it on tens of thousands of strings.

WHAT llama.cpp DOES, AND WHERE IT IS NOT WHAT A UNICODE LIBRARY DOES.
  * Classes.  \\p{L}, \\p{N}, \\p{M}, \\p{P}, \\p{S} and "whitespace" are not looked up in a Unicode library: llama.cpp
    reads a generated table (unicode-data.cpp).  tools/strata_unicode_tables.py is a copy of it.  The families here
    use that table, so a word ends where llama.cpp ends it even for code points the `regex` package classifies
    differently (Unicode versions, U+001C..U+001F, private use).
  * The "collapsed text".  For a pre-tokenizer written as a regex (joyai-llm), llama.cpp replaces every non-ASCII code
    point by ONE class byte and rewrites the regex's \\p{..} classes to match those bytes (unicode_regex_split).  This
    module does exactly that: it collapses the text, rewrites the pattern the same way and runs the `regex` package on
    the collapsed string, one character per code point, so offsets carry over.  The patterns for glm5 and qwen2 are
    handled by hand-written loops in llama.cpp (unicode_regex_split_custom_*); they are the same regex over the same
    classes, and the oracle comparison is what shows that.
  * Several regexes run in sequence (joyai-llm has three): each one splits the pieces the previous one left, and
    anything it does not match stays a piece of its own.
  * `ignore_merges` (glm5): a word that is already a vocabulary entry is one token without looking at the merges.
  * Special tokens are found by scanning the raw text for each of them, longest first (tokenizer_st_partition), after
    llama.cpp's loader has OR-ed CONTROL into tokens whose text is on its EOG/FIM lists even when the GGUF types
    them NORMAL (MiMo's `</s>`).  One alternation regex finds the same tokens as long as no two special tokens
    overlap, which `special_overlaps` checks per vocabulary.
  * Output: token_to_piece writes a CONTROL / USER_DEFINED / UNKNOWN token's text as it is and byte-decodes only
    NORMAL tokens, so DeepSeek's `<｜begin▁of▁sentence｜>` (characters outside the byte alphabet) decodes.

The Qwen tokenizer is not touched by any of this; `strata_tokenizer.Tokenizer` stays the default of the server.
"""
from __future__ import annotations

import bisect
import json
import pathlib
import sys

import regex

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import strata_tokenizer as _base            # noqa: E402
import strata_unicode_tables as _ut         # noqa: E402
from strata_tokenizer import BYTE_TO_UNICODE, UNICODE_TO_BYTE, Tokenizer  # noqa: E402

# ------------------------------------------------------------------------------------------------ llama.cpp's attrs
# GGUF token_type -> llama_token_attr bits (src/llama-vocab.h)
ATTR_UNKNOWN, ATTR_UNUSED, ATTR_NORMAL, ATTR_CONTROL, ATTR_USER_DEFINED, ATTR_BYTE = 1, 2, 4, 8, 16, 32
ATTR_LSTRIP, ATTR_RSTRIP = 1 << 7, 1 << 8
TYPE_TO_ATTR = {0: 0, 1: ATTR_NORMAL, 2: ATTR_UNKNOWN, 3: ATTR_CONTROL, 4: ATTR_USER_DEFINED, 5: ATTR_UNUSED,
                6: ATTR_BYTE}
SPECIAL_ATTRS = ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN      # what tokenizer_st_partition scans for

# The texts llama.cpp's loader looks for (llama_vocab::impl::load, "auto-detect special tokens by text"): a token with
# one of them gets CONTROL OR-ed into its attrs, and the EOG ones end generation.  Copied from the tree the oracle is
# built from; moex/tools/verify_tokenizer.py compares the result with `llama_vocab_get_attr` / `llama_vocab_is_eog`.
EOT_TEXTS = ("<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>", "<EOT>",
             "_<EOT>", "[EOT]", "<｜end▁of▁sentence｜>", "<end_of_utterance>")
EOM_TEXTS = ("<|eom_id|>",)
FIM_TEXTS = {
    "pre": ("<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<｜fim▁begin｜>", "<PRE>", "▁<PRE>",
            "<|code_prefix|>", "<|prefix|>"),
    "suf": ("<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<｜fim▁hole｜>", "<SUF>", "▁<SUF>",
            "<|code_suffix|>", "<|suffix|>"),
    "mid": ("<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<｜fim▁end｜>", "<MID>", "▁<MID>",
            "<|code_middle|>", "<|middle|>"),
    "pad": ("<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"),
    "rep": ("<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>"),
    "sep": ("<|file_sep|>",),
}
EOG_TEXTS = ("<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>", "<end_of_turn>",
             "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
             "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>", "<｜end▁of▁sentence｜>",
             "[e~[")
GGUF_ID_KEYS = {"bos": "tokenizer.ggml.bos_token_id", "eos": "tokenizer.ggml.eos_token_id",
                "eot": "tokenizer.ggml.eot_token_id", "eom": "tokenizer.ggml.eom_token_id",
                "unk": "tokenizer.ggml.unknown_token_id", "sep": "tokenizer.ggml.seperator_token_id",
                "pad": "tokenizer.ggml.padding_token_id", "mask": "tokenizer.ggml.mask_token_id",
                "fim_pre": "tokenizer.ggml.fim_pre_token_id", "fim_suf": "tokenizer.ggml.fim_suf_token_id",
                "fim_mid": "tokenizer.ggml.fim_mid_token_id", "fim_pad": "tokenizer.ggml.fim_pad_token_id",
                "fim_rep": "tokenizer.ggml.fim_rep_token_id", "fim_sep": "tokenizer.ggml.fim_sep_token_id"}


def derive_vocab_meta(tokens: list[str], types: list[int] | None, ids: dict[str, int | None]) -> dict:
    """llama.cpp's loader, for the parts that decide how text is tokenized and when generation ends: the final attrs of
    every token (the GGUF type plus the CONTROL the loader adds), the EOG set, and the special ids.

    `ids` holds the GGUF's own `tokenizer.ggml.*_token_id` values by role (bos eos eot eom pad sep fim_pre ...; None when
    the key is absent).  Returns {"attr": [..], "eog": [..], "ids": {role: id | None}, "ambiguous": [..]}.  `ambiguous`
    names a role whose auto-detection found several tokens: llama.cpp takes the first in the iteration order of an
    unordered_map, which cannot be reproduced, so nothing is guessed (it is empty for the three families here).
    """
    n = len(tokens)
    attr = [TYPE_TO_ATTR.get(t, 0) for t in types] if types else [ATTR_NORMAL] * n
    by_text: dict[str, int] = {t: i for i, t in enumerate(tokens)}
    roles = dict(ids)
    ambiguous = []

    def control(i):
        attr[i] |= ATTR_CONTROL

    # find EOT / EOM / FIM tokens by their text when the GGUF names none
    def detect(role: str, texts) -> None:
        if roles.get(role) is not None:
            return
        found = [by_text[t] for t in texts if t in by_text]
        if found:
            if len(found) > 1:
                ambiguous.append(role)
            roles[role] = found[0]
            control(found[0])

    detect("eot", EOT_TEXTS)
    detect("eom", EOM_TEXTS)
    for role, texts in FIM_TEXTS.items():
        detect("fim_" + role, texts)
    # control tokens with "unused" in their text are UNUSED as well
    for i in range(n):
        if attr[i] & ATTR_CONTROL and "unused" in tokens[i]:
            attr[i] |= ATTR_UNUSED
    # tokens that end generation
    eog: set[int] = set()
    for role in ("fim_pad", "fim_rep", "fim_sep"):
        if roles.get(role) is not None:
            eog.add(roles[role])
    for t in EOG_TEXTS:
        if t in by_text:
            eog.add(by_text[t])
            control(by_text[t])
    for t in ("<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"):      # gpt-oss: always rendered
        if t in by_text:
            attr[by_text[t]] = ATTR_USER_DEFINED
    for role in ("eos", "eot", "eom"):
        if roles.get(role) is not None:
            eog.add(roles[role])
    texts_of = {tokens[i]: i for i in eog}
    if {"<|return|>", "<|call|>", "<|end|>"} <= texts_of.keys() or {"<|calls|>", "<|flush|>", "<|end|>"} <= texts_of.keys():
        eog.discard(texts_of["<|end|>"])                       # o200k_harmony / solar-open
        attr[texts_of["<|end|>"]] = ATTR_USER_DEFINED
    if "<|tool_response>" in texts_of and "</s>" in texts_of:
        eog.discard(texts_of["</s>"])                          # gemma4 / paddleocr
        attr[texts_of["</s>"]] = ATTR_NORMAL
    return {"attr": attr, "eog": sorted(eog), "ids": roles, "ambiguous": ambiguous}


# ------------------------------------------------------------------------------------------------ llama.cpp's classes
FLAG_NUMBER, FLAG_LETTER, FLAG_MARK, FLAG_PUNCT, FLAG_SYMBOL = 0x02, 0x04, 0x10, 0x20, 0x40
_RANGE_STARTS = [s for s, _ in _ut.RANGES]
_RANGE_FLAGS = [f for _, f in _ut.RANGES]
WHITESPACE = frozenset(_ut.WHITESPACE)

# class marks of the collapsed text: one character per non-ASCII code point (llama.cpp uses the bytes 0xD1..0xD5, 0x0B
# for whitespace and 0xD0 for the rest; any distinct characters do, ASCII keeps its own)
M_LETTER, M_NUMBER, M_MARK, M_SPACE, M_PUNCT, M_SYMBOL, M_OTHER = "\x80", "\x81", "\x82", "\x83", "\x84", "\x85", "\x86"


def cpt_flags(cp: int) -> int:
    """llama.cpp's category flags of a code point (unicode_cpt_flags_from_cpt)."""
    if cp >= 0x110000:
        return 1
    return _RANGE_FLAGS[bisect.bisect_right(_RANGE_STARTS, cp) - 1]


def cpt_mark(cp: int) -> str:
    """The class mark of a code point in the collapsed text (unicode_regex_split's loop)."""
    if cp < 128:
        return chr(cp)
    if cp in WHITESPACE:
        return M_SPACE
    cat = cpt_flags(cp) & 0xFF
    return {FLAG_NUMBER: M_NUMBER, FLAG_LETTER: M_LETTER, FLAG_PUNCT: M_PUNCT, FLAG_MARK: M_MARK,
            FLAG_SYMBOL: M_SYMBOL}.get(cat, M_OTHER)


class _Collapser(dict):
    """str.translate table that fills itself: ASCII maps to itself, any other code point to its class mark."""

    def __init__(self):
        super().__init__({i: chr(i) for i in range(128)})

    def __missing__(self, cp):
        v = self[cp] = cpt_mark(cp)
        return v


_COLLAPSE = _Collapser()


def collapse(text: str) -> str:
    """The text with every non-ASCII code point replaced by its class mark (one character each, so offsets agree)."""
    return text if text.isascii() else text.translate(_COLLAPSE)


def _ascii_members(flag: int) -> str:
    """The ASCII characters of a category, as a regex class body (from the table; llama.cpp hard-codes the same lists)."""
    out = []
    for cp in range(128):
        if (cpt_flags(cp) & 0xFF) == flag and cp not in WHITESPACE:
            out.append(cp)
    parts, i = [], 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1] == out[j] + 1:
            j += 1
        a, b = out[i], out[j]
        esc = lambda c: "\\" + chr(c) if chr(c) in "\\]^-[" else chr(c)       # noqa: E731
        parts.append(esc(a) if a == b else esc(a) + "-" + esc(b))
        i = j + 1
    return "".join(parts)


# llama.cpp's k_ucat_enum / k_ucat_map: the class a \p{..} stands for in the collapsed regex
_CATEGORY = {"\\p{N}": (M_NUMBER, FLAG_NUMBER), "\\p{L}": (M_LETTER, FLAG_LETTER), "\\p{P}": (M_PUNCT, FLAG_PUNCT),
             "\\p{M}": (M_MARK, FLAG_MARK), "\\p{S}": (M_SYMBOL, FLAG_SYMBOL), "\\p{Lu}": (M_LETTER, FLAG_LETTER),
             "\\p{Ll}": (M_LETTER, FLAG_LETTER), "\\p{Lt}": (M_LETTER, FLAG_LETTER), "\\p{Lm}": (M_LETTER, FLAG_LETTER),
             "\\p{Lo}": (M_LETTER, FLAG_LETTER)}
_SPACE_BODY = "\\t-\\r " + M_SPACE            # \s of the collapsed text: the C-locale space characters + the mark


def collapse_pattern(expr: str) -> tuple[str, bool]:
    """A llama.cpp regex_expr -> (the pattern for the collapsed text, whether it uses categories).  The same rewrite as
    unicode_regex_split: \\p{..} becomes a class of the category's mark and ASCII members (inside [...] the body is
    inserted), `\\s` is the C-locale whitespace plus the mark of non-ASCII whitespace, other text is kept.  A pattern with
    no category (joyai-llm's CJK ranges) is returned as it is, for the raw text."""
    if not any(c in expr for c in _CATEGORY):
        return expr, False
    out, inside, i, n = [], False, 0, len(expr)
    while i < n:
        c = expr[i]
        if c == "[" and (i == 0 or expr[i - 1] != "\\"):
            out.append("[")
            inside = True
        elif inside and c == "]" and expr[i - 1] != "\\":
            out.append("]")
            inside = False
        elif c == "\\" and expr.startswith("\\p{", i):
            close = expr.find("}", i + 3)
            pat = expr[i:close + 1] if close != -1 and close <= i + 10 else None
            if pat in _CATEGORY:
                mark, flag = _CATEGORY[pat]
                body = mark + _ascii_members(flag)
                out.append(body if inside else "[" + body + "]")
                i = close
            else:
                out.append(c)
        elif c == "\\" and i + 1 < n and expr[i + 1] in "sS":
            if expr[i + 1] == "s":
                out.append(_SPACE_BODY if inside else "[" + _SPACE_BODY + "]")
            else:
                if inside:
                    raise ValueError("\\S inside a class is not used by any family")
                out.append("[^" + _SPACE_BODY + "]")
            i += 1
        elif c == "\\" and i + 1 < n:                   # an escape that is not ours: both characters, untouched
            out.append(c + expr[i + 1])
            i += 1
        else:
            out.append(c)
        i += 1
    return "".join(out), True


# ------------------------------------------------------------------------------------------------ the families
class HfSplitPreTokenizer:
    """moex (ERNIE-5.5): HF tokenizers' pre-tokenizer Sequence of Digits then Isolated Splits (see ERNIE55_SPLITS): each
    stage cuts every piece the previous one left - its matches and the gaps between them pieces of their own - with the
    pattern matched on the piece alone."""

    def __init__(self, splits: tuple[str, ...]):
        self.exprs = splits
        self.stages = [regex.compile(r"\p{N}+")] + [regex.compile(p) for p in splits]

    def split(self, text: str) -> list[str]:
        pieces = [text]
        for rx in self.stages:
            out = []
            for p in pieces:
                pos = 0
                for m in rx.finditer(p):
                    a, b = m.span()
                    if a == b:
                        continue
                    if a > pos:
                        out.append(p[pos:a])
                    out.append(p[a:b])
                    pos = b
                if pos < len(p):
                    out.append(p[pos:])
            pieces = out
        return pieces


class PreTokenizer:
    """llama.cpp's regex_exprs of a pre-type, run in sequence on the (collapsed) text; see the module's note."""

    def __init__(self, exprs: tuple[str, ...]):
        self.exprs = exprs
        self.stages = []
        for e in exprs:
            pat, uses_classes = collapse_pattern(e)
            self.stages.append((regex.compile(pat), uses_classes))
        self._single = len(self.stages) == 1 and self.stages[0][1]

    def split(self, text: str) -> list[str]:
        """The words of `text`, in order, covering all of it."""
        if self._single:
            rx = self.stages[0][0]
            src = collapse(text)
            if src is text:                                  # ASCII: the matches are the words
                words = rx.findall(text)
                if sum(map(len, words)) == len(text):
                    return words
            else:
                found = rx.findall(src)
                if sum(map(len, found)) == len(src):         # every character matched: slice the original
                    out, pos = [], 0
                    for w in found:
                        end = pos + len(w)
                        out.append(text[pos:end])
                        pos = end
                    return out
        n = len(text)
        spans = [(0, n)]
        collapsed = None
        for rx, uses_classes in self.stages:
            if uses_classes:
                if collapsed is None:
                    collapsed = collapse(text)
                src = collapsed
            else:
                src = text
            nxt = []
            for a, b in spans:
                pos = a
                for m in rx.finditer(src, a, b):
                    s, e = m.span()
                    if s > pos:
                        nxt.append((pos, s))                 # a gap is a piece of its own
                    nxt.append((s, e))
                    pos = e
                if pos < b:
                    nxt.append((pos, b))
            spans = nxt
        return [text[a:b] for a, b in spans]


_LLAMA3_LIKE = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|%s| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

# pre-type name (GGUF `tokenizer.ggml.pre`) -> how llama.cpp tokenizes it.  `exprs` are copied from llama-vocab.cpp:
# glm5 shares CHATGLM4's regex (`\p{N}{1,3}` digit groups) and sets ignore_merges; qwen2 is MiMo's (one digit at a time);
# joyai-llm is DeepSeek-V4's (digit groups, a CJK run, and one big alternation, in sequence).
FAMILY_PRE = {
    "glm5": dict(llama="CHATGLM4", exprs=(_LLAMA3_LIKE % "\\p{N}{1,3}",), ignore_merges=True),
    "glm4": dict(llama="CHATGLM4", exprs=(_LLAMA3_LIKE % "\\p{N}{1,3}",), ignore_merges=True),
    "qwen2": dict(llama="QWEN2", exprs=(_LLAMA3_LIKE % "\\p{N}",), ignore_merges=False),
    "joyai-llm": dict(llama="JOYAI_LLM", ignore_merges=False, exprs=(
        "\\p{N}{1,3}",
        "[一-龥぀-ゟ゠-ヿ]+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
        "| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")),
}
# moex: Hy4-flash's `hy_v4` takes joyai-llm's regexes exactly (llama-vocab.cpp: LLAMA_VOCAB_PRE_TYPE_HY_V4 shares their case)
FAMILY_PRE["hy_v4"] = dict(llama="HY_V4", ignore_merges=False, exprs=FAMILY_PRE["joyai-llm"]["exprs"])
# The first stage of joyai-llm in llama.cpp's source reads `[一-龥぀-ゟ゠-ヿ]+`: U+4E00-U+9FA5, U+3040-U+309F, U+30A0-U+30FF.

# moex (ERNIE-5.5-Flash, pre `ernie55`): llama.cpp has no ERNIE-5.5, so its oracle is the release's own tokenizer.json (HF
# tokenizers), whose pre-tokenizer is a Sequence: Digits (individual_digits false: each run of numeric characters, Rust's
# char::is_numeric = \p{N}, a piece of its own), then five Splits with the Isolated behaviour - digit runs cut into right-
# aligned groups of 3 (and of 510 first), a CJK run, the word alternation - then ByteLevel without its regex.  HF matches
# each Split on each piece alone with onig (Ruby syntax), so \A, \z and \G anchor at the piece; the patterns below are the
# release's in the `regex` package's spelling: onig's \z is \Z, and its `X{3}+` (one or more groups of 3, not possessive)
# is `(?:X{3})+`.  The classes are the `regex` package's Unicode ones, as onig's are Unicode (U+200A is a space to both).
# serve/test_ernie55.py and moex/tools/ernie55/verify_tokenizer.py compare the whole tokenizer with HF tokenizers.
ERNIE55_SPLITS = (
    r"\p{Nd}{1,510}(?=(?>\p{Nd}{510})*(?:\P{Nd}|$))|\G\p{Nd}{510}",
    r"\A\p{Nd}{1,2}(?=(?:\p{Nd}{3})+\Z)",
    r"\A\p{Nd}{3}|\G\p{Nd}{3}",
    "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|"
    r"\s+(?!\S)|\s+",
)
FAMILY_PRE["ernie55"] = dict(llama="(none: the release's HF tokenizers Sequence)", ignore_merges=False, exprs=(),
                             hf_splits=ERNIE55_SPLITS)


def special_overlaps(texts: list[str]) -> list[tuple[str, str]]:
    """Pairs of special tokens (X, Y) where a proper suffix of X is a proper prefix of Y: a text can hold them
    overlapping, which is where llama.cpp's token-by-token scan and one alternation regex could disagree."""
    prefixes: dict[str, list[str]] = {}
    for y in texts:
        for k in range(1, len(y)):
            prefixes.setdefault(y[:k], []).append(y)
    out = []
    for x in texts:
        for k in range(1, len(x)):
            for y in prefixes.get(x[len(x) - k:], ()):
                if x != y and len(y) > k:
                    out.append((x, y))
    return sorted(set(out))


class FamilyTokenizer(Tokenizer):
    """The byte-level BPE of a family beyond Qwen: Strata's `_bpe` (heap path included), llama.cpp's pre-tokenizer
    classes, special-token handling and byte output.  `encode` and `decode` have the base class's signature."""

    CACHE_MAX = 1 << 18

    def __init__(self, tokens, merges, token_types=None, pre="glm5", special_ids=None, *, meta: dict | None = None,
                 neutralize_skip=("/nothink",)):
        if pre not in FAMILY_PRE:
            raise ValueError("no pre-tokenizer for %r; known: %s" % (pre, ", ".join(sorted(FAMILY_PRE))))
        super().__init__(tokens, merges, token_types, pre=pre, special_ids=special_ids)
        spec = FAMILY_PRE[pre]
        self.family_pre = spec
        self.ignore_merges = spec["ignore_merges"]
        self.pretok = HfSplitPreTokenizer(spec["hf_splits"]) if "hf_splits" in spec else PreTokenizer(spec["exprs"])
        # llama.cpp's bpe_ranks.emplace keeps the FIRST rank of a repeated pair (the base class keeps the last)
        ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            a, b = m.split(" ", 1)
            ranks.setdefault((a, b), i)
        self.n_duplicate_merges = len(merges) - len(ranks)
        self.ranks = ranks
        # attrs: the GGUF types plus the loader's overrides (derived, or given by the exporter in `meta`)
        ids_by_role = {role: (special_ids or {}).get(key) for role, key in GGUF_ID_KEYS.items()}
        derived = derive_vocab_meta(tokens, token_types, ids_by_role) if meta is None else meta
        self.attr = derived["attr"]
        self.eog_ids = sorted(derived["eog"])
        self.role_ids = derived["ids"]
        if any(a & (ATTR_LSTRIP | ATTR_RSTRIP) for a in self.attr):
            raise ValueError("token attributes LSTRIP/RSTRIP are not supported by the families")
        # special tokens, longest text first (llama.cpp sorts the cache by byte length)
        specials = [i for i, a in enumerate(self.attr) if a & SPECIAL_ATTRS]
        specials.sort(key=lambda i: (-len(tokens[i].encode("utf-8")), i))
        self.special_tokens = {tokens[i]: i for i in specials}
        always = [tokens[i] for i in specials if not self.attr[i] & (ATTR_CONTROL | ATTR_UNKNOWN)]
        self._always_re = self._alt(always)
        self._special_re = self._alt([tokens[i] for i in specials])
        self.special_overlaps = special_overlaps([tokens[i] for i in specials])
        # what the server's hygiene must turn into plain text inside a message: every special token the tokenizer would
        # match in text, but not a user-typed switch (`/nothink`)
        self.control_tokens = [t for t in self.special_tokens if t not in neutralize_skip]
        self.add_bos = bool((special_ids or {}).get("tokenizer.ggml.add_bos_token", False))
        self.add_eos = bool((special_ids or {}).get("tokenizer.ggml.add_eos_token", False))
        self._cache: dict[str, tuple[int, ...]] = {}
        self._byte_table = {i: BYTE_TO_UNICODE[i] for i in range(256)}

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_gguf(cls, path) -> "FamilyTokenizer":
        sys.path.insert(0, str(_HERE))
        from gguf_reader import GGUFFile
        md = GGUFFile(pathlib.Path(path)).metadata
        return cls.from_metadata(md)

    @classmethod
    def from_metadata(cls, md: dict) -> "FamilyTokenizer":
        need = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
        missing = [k for k in need if k not in md]
        if missing:
            raise ValueError("GGUF is missing %s; it does not carry a tokenizer" % ", ".join(missing))
        if md.get("tokenizer.ggml.model") != "gpt2":
            raise ValueError("expected a byte-level BPE tokenizer (model 'gpt2'), got %r" % md.get("tokenizer.ggml.model"))
        extra = {k: md[k] for k in md if k.startswith("tokenizer.ggml.") and (k.endswith("_token_id") or k.startswith(
            "tokenizer.ggml.add_"))}
        return cls(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"]),
                   list(md.get("tokenizer.ggml.token_type") or []) or None, md.get("tokenizer.ggml.pre", ""), extra)

    @classmethod
    def from_dir(cls, path) -> "FamilyTokenizer":
        """Read an exported `tokenizer/` directory (vocab.json, merges.txt, token_type.json, tokenizer.json)."""
        p = pathlib.Path(path)
        vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (p / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((p / "token_type.json").read_text())
        cfg = json.loads((p / "tokenizer.json").read_text(encoding="utf-8"))
        meta = cfg.get("meta")
        if meta is not None:                         # the exporter keeps only where the attrs differ from the GGUF types
            attr = [TYPE_TO_ATTR.get(t, 0) for t in types]
            for k, v in meta["attr_overrides"].items():
                attr[int(k)] = v
            meta = {"attr": attr, "eog": meta["eog"], "ids": meta["ids"]}
        return cls(tokens, merges, types, cfg["pre"], cfg.get("special_ids") or {}, meta=meta)

    # -------------------------------------------------------------- the algorithm
    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        cache, ids, bpe, ignore = self._cache, self.ids, self._bpe, self.ignore_merges
        table = self._byte_table
        for piece in self.pretok.split(text):
            hit = cache.get(piece)
            if hit is None:
                mapped = piece.encode("utf-8", "surrogatepass").decode("latin-1").translate(table)
                if ignore and mapped in ids:
                    hit = (ids[mapped],)
                else:
                    hit = tuple(ids[t] for t in bpe(mapped))
                if len(cache) >= self.CACHE_MAX:
                    cache.clear()
                cache[piece] = hit
            out.extend(hit)
        return out

    def encode(self, text: str, parse_special: bool = False, plain=(), add_special: bool = False) -> list[int]:
        """Tokenize `text` as llama_tokenize does (parse_special: CONTROL tokens too; USER_DEFINED ones always).
        `plain`: spans of `text` that are ordinary text even where they spell a special token (#537)."""
        out: list[int] = []
        if add_special and self.add_bos and self.role_ids.get("bos") is not None:
            out.append(self.role_ids["bos"])
        if text:
            out += self._encode_matching(text, self._special_re if parse_special else self._always_re, plain)
        if add_special and self.add_eos and self.role_ids.get("eos") is not None:
            out.append(self.role_ids["eos"])
        return out

    # -------------------------------------------------------------- bytes out
    def token_bytes(self, i: int) -> bytes:
        """llama_token_to_piece: CONTROL / USER_DEFINED / UNKNOWN text as it is, NORMAL tokens through the byte alphabet,
        everything else (UNUSED, BYTE) nothing."""
        cache = self.__dict__.setdefault("_bytes_cache", {})
        b = cache.get(i)
        if b is None:
            if i < 0 or i >= len(self.tokens):
                raise IndexError("token id %d is outside the vocabulary (%d)" % (i, len(self.tokens)))
            a = self.attr[i]
            if a & (ATTR_CONTROL | ATTR_UNKNOWN | ATTR_USER_DEFINED):
                b = self.tokens[i].encode("utf-8")
            elif a & ATTR_NORMAL:
                raw = bytearray()
                for ch in self.tokens[i]:
                    v = UNICODE_TO_BYTE.get(ch)
                    if v is None:
                        raw += ("[UNK_BYTE_0x%s%s]" % (ch.encode("utf-8").hex(), self.tokens[i])).encode("utf-8")
                        break
                    raw.append(v)
                else:
                    pass
                b = bytes(raw)
            else:
                b = b""
            cache[i] = b
        return b


def load(path_or_gguf) -> FamilyTokenizer:
    """A FamilyTokenizer from a GGUF file or an exported tokenizer directory."""
    p = pathlib.Path(path_or_gguf)
    return FamilyTokenizer.from_dir(p) if p.is_dir() else FamilyTokenizer.from_gguf(p)


# ------------------------------------------------------------------------------------------------ the exported directory
def extract(gguf_path, out_dir) -> dict:
    """Write a family's tokenizer into `out_dir` (flat: vocab.json, merges.txt, token_type.json, tokenizer.json and the
    model's chat_template.jinja), as tools/strata_tokenizer.py does for the Qwen pack's `tokenizer/` directory, so the
    engine and the server never open the weight shards for it.  tokenizer.json ships what is not recomputed by a reader:
    the pre-tokenizer's regexes as llama.cpp spells them, `ignore_merges`, the special ids, and the loader's decisions
    (the tokens whose attrs differ from the GGUF types, the end-of-generation set, the special ids by role)."""
    sys.path.insert(0, str(_HERE))
    from gguf_reader import GGUFFile
    md = GGUFFile(pathlib.Path(gguf_path)).metadata
    tk = FamilyTokenizer.from_metadata(md)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    types = list(md.get("tokenizer.ggml.token_type") or []) or None
    default = [TYPE_TO_ATTR.get(t, 0) for t in types] if types else [ATTR_NORMAL] * len(tk.tokens)
    overrides = {str(i): a for i, (a, d) in enumerate(zip(tk.attr, default)) if a != d}
    cfg = {
        "model": "gpt2",
        "pre": tk.pre,
        "llama_pre_type": tk.family_pre["llama"],
        "vocab_size": len(tk.tokens),
        "n_merges": len(tk.ranks),
        "special_ids": tk.special_ids,
        "add_bos_token": tk.add_bos,
        "add_eos_token": tk.add_eos,
        "ignore_merges": tk.ignore_merges,
        "pre_patterns": list(tk.family_pre["exprs"]),
        "pre_pattern_source": "llama.cpp src/llama-vocab.cpp llm_tokenizer_bpe (LLAMA_VOCAB_PRE_TYPE_%s); classes: "
                              "src/unicode-data.cpp (tools/strata_unicode_tables.py)" % tk.family_pre["llama"],
        "meta": {"attr_overrides": overrides, "eog": tk.eog_ids, "ids": tk.role_ids},
        "special_tokens": len(tk.special_tokens),
        "special_overlaps": len(tk.special_overlaps),
    }
    (out / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join("%s %s" % k for k, _ in sorted(tk.ranks.items(), key=lambda kv: kv[1])),
                                    encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(tk.token_types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    tpl = md.get("tokenizer.chat_template")
    if tpl:
        (out / "chat_template.jinja").write_text(tpl, encoding="utf-8", newline="\n")
    return cfg


def extract_hf(tokenizer_json, out_dir, pre: str, roles: dict[str, int], eog: list[int]) -> dict:
    """moex (ERNIE-5.5): a family's tokenizer directory from the release's HF tokenizer.json (its GGUF carries no vocabulary):
    the BPE vocabulary and merges as they are; the token types from its added_tokens (special: CONTROL, matched in a
    template's text only; the others: USER_DEFINED, matched in any text - HF's added-token matching); `roles` the special
    ids by role, `eog` the ids that end generation (the release's generation_config.json)."""
    cfg_hf = json.loads(pathlib.Path(tokenizer_json).read_text(encoding="utf-8"))
    model = cfg_hf["model"]
    if model.get("type") != "BPE" or model.get("ignore_merges"):
        raise ValueError("expected a BPE tokenizer.json without ignore_merges")
    vocab = model["vocab"]
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    if any(t is None for t in tokens):
        raise ValueError("tokenizer.json's vocabulary has holes")
    types = [1] * len(tokens)
    for a in cfg_hf.get("added_tokens", []):
        if a["id"] >= len(tokens) or tokens[a["id"]] != a["content"]:
            raise ValueError("added token %r is not the vocabulary's entry %d" % (a["content"], a["id"]))
        if a.get("lstrip") or a.get("rstrip") or a.get("single_word"):
            raise ValueError("added token %r strips or is a single word: not supported" % a["content"])
        types[a["id"]] = 3 if a.get("special") else 4
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    special_ids = {GGUF_ID_KEYS[r]: i for r, i in roles.items() if r in GGUF_ID_KEYS}
    special_ids["tokenizer.ggml.add_bos_token"] = False
    special_ids["tokenizer.ggml.add_eos_token"] = False
    tk = FamilyTokenizer(tokens, merges, types, pre, special_ids,
                         meta={"attr": [TYPE_TO_ATTR.get(t, 0) for t in types], "eog": sorted(eog),
                               "ids": {r: roles.get(r) for r in GGUF_ID_KEYS}})
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model": "gpt2",
        "pre": pre,
        "llama_pre_type": FAMILY_PRE[pre]["llama"],
        "vocab_size": len(tokens),
        "n_merges": len(tk.ranks),
        "special_ids": special_ids,
        "add_bos_token": False,
        "add_eos_token": False,
        "ignore_merges": False,
        "pre_patterns": list(FAMILY_PRE[pre].get("hf_splits") or FAMILY_PRE[pre]["exprs"]),
        "pre_pattern_source": "the release's tokenizer.json pre_tokenizer (HF tokenizers' Sequence), "
                              "strata_tokenizer_families.ERNIE55_SPLITS",
        "meta": {"attr_overrides": {}, "eog": sorted(eog), "ids": tk.role_ids},
        "special_tokens": len(tk.special_tokens),
        "special_overlaps": len(tk.special_overlaps),
        "source": pathlib.Path(tokenizer_json).name,   # the file's name only: an exported directory carries no local path
    }
    (out / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join("%s %s" % k for k, _ in sorted(tk.ranks.items(), key=lambda kv: kv[1])),
                                    encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    return cfg


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="export a family's tokenizer from its GGUF, or from a release's HF "
                                             "tokenizer.json when its GGUF carries no vocabulary (ERNIE-5.5)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", help="the model's (first) GGUF file")
    src.add_argument("--hf-json", help="the release's tokenizer.json (with --pre, --roles and --eog)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pre", help="--hf-json: the family's pre-tokenizer (FAMILY_PRE), e.g. ernie55")
    ap.add_argument("--roles", default="", help="--hf-json: special ids by role, e.g. bos=2,eos=8,unk=0,pad=0")
    ap.add_argument("--eog", default="", help="--hf-json: the ids that end generation, e.g. 2,8")
    a = ap.parse_args()
    if a.hf_json:
        if not a.pre or a.pre not in FAMILY_PRE:
            ap.error("--hf-json needs --pre, one of " + ", ".join(sorted(FAMILY_PRE)))
        roles = {k: int(v) for k, v in (x.split("=") for x in a.roles.split(",") if x)}
        cfg = extract_hf(a.hf_json, a.out, a.pre, roles, [int(x) for x in a.eog.split(",") if x])
    else:
        cfg = extract(a.gguf, a.out)
    print("tokenizer: vocab %d, merges %d, pre %s, %d special tokens, eog %s"
          % (cfg["vocab_size"], cfg["n_merges"], cfg["pre"], cfg["special_tokens"], cfg["meta"]["eog"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
