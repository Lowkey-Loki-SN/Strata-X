"""moex/tools/tokenizer_corpus.py - the corpus the family tokenizers are checked against the oracle with.

    from tokenizer_corpus import build
    corpus = build(tokenizer, seed=1)       # {category: [str, ...]}

A tokenizer disagrees with its oracle at edges, so the corpus is made of edges: every category below exists because a
different tokenizer rule (or a different Unicode table) could split it differently.  Most categories are generated from
building blocks with a seeded generator, so a run is reproducible; the real-text categories sample files on this
machine (documents in seven languages, source in a dozen languages) at seeded offsets.  `tokenizer` is a
FamilyTokenizer: the special-token categories are built from ITS special tokens.

Categories
  prose_en, prose_multi     real prose: English; the README translations (de es fr ja pt-BR zh-CN)
  contractions              apostrophes in every position, both cases, curly quotes
  code_<lang>               real source (cpp h c cu py ts svelte js sh yml cmake glsl metal kt swift html css toml)
                            and short hand-written samples of the languages not on disk
  json, jsonl, yaml_toml    compact / pretty / nested / escaped / large numbers
  markdown_tables           pipes, alignment rows, CJK and emoji cells
  scripts                   Chinese, Japanese, Korean, Arabic, Hebrew, Thai, Devanagari, Tamil, Bengali, Cyrillic, Greek,
                            Georgian, Armenian, Ethiopic, Khmer, Burmese, Vietnamese (NFC and NFD)
  emoji                     single, ZWJ sequences, flags, skin tones, keycaps, variation selectors, tag sequences
  marks                     combining marks after every kind of base, Zalgo, Indic matras
  whitespace                runs of every whitespace code point and mixtures, at the start, the end and between words
  tabs_newlines             indentation, CR/LF/CRLF mixtures, blank-line runs
  digits                    runs of 1..40, separators, every kind of non-ASCII digit and number
  urls_paths                URLs, e-mail, paths, IPv4/IPv6, UUIDs, hashes, base64
  special_plain             each special token's text typed as plain text, broken, doubled, spaced, in context
  special_real              real special tokens between text (what a rendered prompt contains)
  long_words                single words of 64..20000 characters of every kind (the heap path, ignore_merges)
  fuzz                      random strings over a weighted alphabet of every class, with special-token fragments
  boundaries                the first and last code point of every range of llama.cpp's table, in contexts
  controls                  every C0/C1 control, DEL, noncharacters, private use, format characters
  templates                 (added by the caller) rendered chat prompts
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)

import os
import pathlib
import random
import sys
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
X = HERE.parents[1]
sys.path.insert(0, str(X / "tools"))
import strata_unicode_tables as UT  # noqa: E402

ROOTS = [pathlib.Path(_os.path.join(_AI, 'llama.cpp-glm5next-mtp')), X / "serve", X / "tools", X / "docs", X / "src",
         pathlib.Path("/usr/share/dictionaries-common"), pathlib.Path("/usr/share/console-setup"), pathlib.Path("/usr/sbin")]
LANG_EXT = {"cpp": (".cpp", ".cc"), "h": (".h", ".hpp"), "c": (".c",), "cu": (".cu", ".cuh"), "py": (".py",),
            "ts": (".ts",), "svelte": (".svelte",), "js": (".js",), "sh": (".sh",), "yml": (".yml", ".yaml"),
            "cmake": (".cmake", "CMakeLists.txt"), "glsl": (".glsl", ".comp"), "metal": (".metal",), "kt": (".kt",),
            "swift": (".swift",), "html": (".html",), "css": (".css",), "toml": (".toml",), "md": (".md",),
            "json": (".json",), "m": (".m",), "pl": (".pl",), "rb": (".rb",)}


def _files(ext: tuple[str, ...], limit: int = 400) -> list[pathlib.Path]:
    out = []
    for root in ROOTS:
        if not root.exists():
            continue
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if d not in (".git", "node_modules", "build", "__pycache__", "third_party", ".venv"))
            for f in sorted(fn):
                if f.endswith(ext) and not f.endswith(".min.js"):
                    out.append(pathlib.Path(dp) / f)
                    if len(out) >= limit:
                        return out
    return out


def _chunks(path: pathlib.Path, rng: random.Random, n: int, lo: int = 40, hi: int = 1800) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    if len(text) < 20:
        return []
    out = []
    for _ in range(n):
        a = rng.randrange(0, max(1, len(text) - lo))
        nl = text.find("\n", a)
        a = nl + 1 if 0 <= nl < a + 200 and rng.random() < 0.7 else a      # mostly start at a line, sometimes mid-token
        b = min(len(text), a + rng.randint(lo, hi))
        if rng.random() < 0.5:
            nl = text.rfind("\n", a, b)
            b = nl + 1 if nl > a else b
        if b > a:
            out.append(text[a:b])
    return out


def _sample_files(lang: str, rng: random.Random, files: int, per_file: int, **kw) -> list[str]:
    paths = _files(LANG_EXT[lang])
    rng.shuffle(paths)
    out: list[str] = []
    for p in paths[:files]:
        out += _chunks(p, rng, per_file, **kw)
    return out


# ------------------------------------------------------------------------------------------------ hand-made blocks
CONTRACTION_WORDS = ["don", "I", "we", "they", "she", "he", "it", "you", "that", "what", "DON", "WE", "It", "O", "rock",
                     "n", "y", "can", "won", "isn", "aren", "wouldn", "he'd", "the", "a", ""]
CONTRACTIONS = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D", "'Ll", "'rE", "'vE",
                "'", "''", "'''", "'x", "'ss", "'tt", "'rex", "'lle", "'\u00e9", "\u2019s", "\u2019t", "\u2018s", "`s", "\u00b4s",
                "'s'", "'S'", "' s", "'\u017f", "'\u212a", "'\u0131",
                "'SS", "'TX", "'REX", "'VEX", "'MX", "'LLX", "'DX", "'sS", "'Re", "'Sx", "'Tt", "'Rex", "'Vex", "'Mx", "'Llx", "'Dx",
                "'SSS", "'RE_", "'VE1", "'LL-", "'D.", "'Ss", "'rEx", "'vEx", "'lLx"]
SCRIPTS = {
    "zh": ["\u4f60\u597d\uff0c\u4e16\u754c\uff01", "\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u6211\u4eec\u53bb\u516c\u56ed\u73a9\u5427\u3002",
           "\u673a\u5668\u5b66\u4e60\u548c\u6df1\u5ea6\u5b66\u4e60\u7684\u533a\u522b\u662f\u4ec0\u4e48\uff1f", "\u7e41\u9ad4\u4e2d\u6587\u6e2c\u8a66\uff1a\u9019\u662f\u4e00\u500b\u6e2c\u8a66\u3002",
           "\u300c\u5f15\u7528\u300d\u3010\u6807\u9898\u3011\u300a\u4e66\u540d\u300b\u2014\u2014\u7701\u7565\u53f7\u2026\u2026"],
    "ja": ["\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059\u3002", "\u30ab\u30bf\u30ab\u30ca\u3068\u3072\u3089\u304c\u306a\u3068\u6f22\u5b57\u304c\u6df7\u3056\u3063\u3066\u3044\u307e\u3059\u3002",
           "\u3053\u3093\u306b\u3061\u306f\u3001\u4e16\u754c\uff01\u30c6\u30b9\u30c8123abc", "\u30fc\u30fc\u30fc\u30fb\u30fb\u30fb\u3002\u3002\u3002", "\uff7a\uff9d\uff86\uff81\uff8a"],
    "ko": ["\uc548\ub155\ud558\uc138\uc694 \uc138\uacc4", "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8 \ud14c\uc2a4\ud2b8\uc785\ub2c8\ub2e4.", "\u1112\u1161\u11ab\u1100\u116e\u11a8\u110b\u1165 (NFD jamo)"],
    "ar": ["\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "\u0627\u0644\u0644\u063a\u0629 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0645\u0639 \u0627\u0644\u062a\u0634\u0643\u064a\u0644: \u0643\u064e\u062a\u064e\u0628\u064e",
           "\u0627\u0644\u0631\u0642\u0645 \u0661\u0662\u0663\u0664 \u0648 \u06f1\u06f2\u06f3", "\u200f\u0645\u0646 \u0627\u0644\u064a\u0645\u064a\u0646 \u0625\u0644\u0649 \u0627\u0644\u064a\u0633\u0627\u0631\u200e"],
    "he": ["\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd", "\u05e2\u05b4\u05d1\u05b0\u05e8\u05b4\u05d9\u05ea \u05e2\u05dd \u05e0\u05d9\u05e7\u05d5\u05d3"],
    "th": ["\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e0a\u0e32\u0e27\u0e42\u0e25\u0e01", "\u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22\u0e44\u0e21\u0e48\u0e21\u0e35\u0e0a\u0e48\u0e2d\u0e07\u0e27\u0e48\u0e32\u0e07\u0e23\u0e30\u0e2b\u0e27\u0e48\u0e32\u0e07\u0e04\u0e33"],
    "hi": ["\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e", "\u0915\u094d\u0937\u0924\u094d\u0930\u093f\u092f \u0915\u094d\u0937\u092e\u093e \u0924\u094d\u0930\u093f\u0936\u0942\u0932 \u0967\u0968\u0969"],
    "ta": ["\u0bb5\u0ba3\u0b95\u0bcd\u0b95\u0bae\u0bcd \u0b89\u0bb2\u0b95\u0bae\u0bcd", "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd \u0bae\u0bca\u0bb4\u0bbf"],
    "bn": ["\u09b9\u09cd\u09af\u09be\u09b2\u09cb \u0993\u09af\u09bc\u09be\u09b0\u09cd\u09b2\u09cd\u09a1", "\u09ac\u09be\u0982\u09b2\u09be \u09ad\u09be\u09b7\u09be"],
    "ru": ["\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440!", "\u0420\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442 \u0441 \u00ab\u043a\u0430\u0432\u044b\u0447\u043a\u0430\u043c\u0438\u00bb \u2014 \u0438 \u0442\u0438\u0440\u0435."],
    "el": ["\u0393\u03b5\u03b9\u03b1 \u03c3\u03bf\u03c5 \u039a\u03cc\u03c3\u03bc\u03b5", "\u03ac\u03ad\u03ae\u03af\u03cc\u03cd\u03ce \u03b1\u0301"],
    "ka": ["\u10d2\u10d0\u10db\u10d0\u10e0\u10ef\u10dd\u10d1\u10d0 \u10db\u10e1\u10dd\u10e4\u10da\u10d8\u10dd"], "hy": ["\u0532\u0561\u0580\u0587 \u0531\u0577\u056d\u0561\u0580\u0570"],
    "am": ["\u1230\u120b\u121d \u12d3\u1208\u121d"], "km": ["\u179f\u17bd\u179f\u17d2\u178f\u17b8\u1796\u17b7\u1797\u1796\u179b\u17c4\u1780"],
    "my": ["\u1019\u1002\u1004\u103a\u1039\u1002\u101c\u102c\u1015\u102b \u1000\u103c\u100a\u103a\u103d\u1031\u1037"],
    "vi": ["Xin ch\u00e0o th\u1ebf gi\u1edbi", "Xin cha\u0300o the\u0301\u0309 gio\u031b\u0301i (NFD)"],
    "mix": ["Hello \u4e16\u754c 123 \u3053\u3093\u306b\u3061\u306f! \u0645\u0631\u062d\u0628\u0627 \U0001f600", "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00fcber \u00e5ngstr\u00f6m \u00f1",
            "Stra\u00dfe \u1e9e \u0130stanbul \u0131 \u017f \u212a K"],
}
EMOJI = ["\U0001f600", "\U0001f680", "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466", "\U0001f1fa\U0001f1f8\U0001f1ef\U0001f1f5",
         "\U0001f44d\U0001f3fd", "1\ufe0f\u20e3", "\u2764\ufe0f", "\u2764", "\u263a\ufe0e", "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
         "\U0001f9d1\u200d\U0001f4bb", "\u00a9\ufe0f", "\u2705\u274c\u26a0\ufe0f", "\U0001f525\U0001f525\U0001f525", "\U0001f4a9", "\U0001f9e0",
         "\u2728", "\u2b50\ufe0f"]
MARKS = ["e\u0301", "e\u0301\u0302\u0303\u0304\u0305", "a\u0308b\u0308c\u0308", "\u0915\u094d\u0937", "\u0928\u093f", "\u0937\u094d\u0928",
         "z\u0351\u036b\u0343\u036a\u0302\u036b\u033d\u034f\u0334\u0319\u0324\u031e\u0349\u035a\u032f\u031e\u0320\u034d", "\u0e01\u0e34\u0e49", "\u1000\u103c\u102c", "x\u200dy", "\u0301", "\u0301a",
         "1\u0301", " \u0301", "\u0301\u0301\u0301", "!\u0301", "\n\u0301", "\u4e2d\u0301", "\u3099", "\ud55c\u3099", "\u20dd", "a\u20ddb"]
WS_CHARS = [" ", "\t", "\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85", "\xa0", "\u1680", "\u2000", "\u2001",
            "\u2002", "\u2003", "\u2004", "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200a", "\u200b", "\u200c", "\u200d",
            "\u2028", "\u2029", "\u202f", "\u205f", "\u2060", "\u3000", "\ufeff", "\u180e", "\u061c", "\u200e", "\u200f"]
NUMBERS = ["\u00b2", "\u00b3", "\u00b9", "\u00bc", "\u00bd", "\u00be", "\u2160", "\u2167", "\u2170", "\u2465", "\u2474", "\u2488",
           "\uff11\uff12\uff13", "\u0663\u0664\u0665", "\u06f3\u06f4", "\u0967\u0968\u0969", "\u09e7\u09e8", "\u0e51\u0e52\u0e53", "\u16ee",
           "\u3007", "\u4e00\u4e8c\u4e09", "\u5341", "\u2154", "\u215b", "\U0001d7d8\U0001d7d9", "\U0001f10b", "\u0f33", "\u2189"]
LONG_KINDS = ["abcdefghij", "x", "Aa", "0123456789", "\u4e2d\u6587", "\u3053\u3093\u306b\u3061\u306f", "\u0645\u0631\u062d\u0628\u0627", "Zm9v", "deadbeef",
              "!@#$%^&*", "_", "-", "a1", "\U0001f600", "\u00e9", "{\"k\":[1,2,3]}", "function(){return 1}", "AAAA", "\u0915\u094d\u0937", "a\u0301"]


def prose_pieces(rng: random.Random) -> list[str]:
    """English prose from the repository's own documents (and the translations in their own language)."""
    out: list[str] = []
    for p in sorted(X.glob("README*.md")) + sorted((X / "docs").glob("*.md"))[:14]:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        paras = [t for t in text.split("\n\n") if t.strip()]
        rng.shuffle(paras)
        out += paras[:18]
    return out


def multi_pieces(rng: random.Random) -> list[str]:
    out: list[str] = []
    for p in sorted(X.glob("README.*.md")):
        text = p.read_text(encoding="utf-8")
        paras = [t for t in text.split("\n\n") if t.strip()]
        rng.shuffle(paras)
        out += paras[:30]
    return out


def json_samples(rng: random.Random) -> list[str]:
    import json
    out = []
    objs = [{"a": 1, "b": [1, 2, 3], "c": {"d": None, "e": True}}, {"name": "\u4f60\u597d", "emoji": "\U0001f600", "n": 1.5e-10},
            [1, 22, 333, 4444, 55555, 666666], {"s": "line1\nline2\ttab \"q\" \\ back"}, {"k" * 40: "v" * 90}, {"nested": [[[[[]]]]]},
            {"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
                                                          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}]
    for o in objs:
        out += [json.dumps(o), json.dumps(o, indent=2), json.dumps(o, ensure_ascii=False, separators=(",", ":")),
                json.dumps(o, ensure_ascii=True)]
    out += ['{"a":1}\n{"b":2}\n{"c":3}\n', '{"big": 123456789012345678901234567890, "f": -0.000001234, "e": 1E+30}',
            '{"unicode": "\\u00e9\\ud83d\\ude00", "esc": "\\/\\b\\f\\n\\r\\t"}', "[" + ",".join(str(i) for i in range(300)) + "]"]
    for p in _files((".json",))[:20]:
        out += _chunks(p, rng, 2, 40, 1500)
    return out


def markdown_tables(rng: random.Random) -> list[str]:
    rows = ["| Name | Value | Notes |\n|------|:-----:|------:|\n| a | 1 | x |\n| b | 22 | y y |\n",
            "|a|b|\n|-|-|\n|1|2|\n", "| \u4e2d\u6587 | \u65e5\u672c\u8a9e | \U0001f600 |\n| --- | --- | --- |\n| \u6d4b\u8bd5 | \u30c6\u30b9\u30c8 | \u2705 |\n",
            "| Tool | Tokens/s |\n|---|---|\n| `x` | **70.5** |\n| _y_ | 1,234.5 |\n", "|  |  |\n|--|--|\n|  |  |\n",
            "Header\n------\n\n* item one\n* item two\n  * nested\n\n1. first\n2. second\n\n> quote\n> more\n\n```py\nprint(1)\n```\n",
            "| a | b |\n|---|---|\n" + "\n".join("| %d | %s |" % (i, "x" * (i % 17)) for i in range(40))]
    for p in sorted(X.glob("README*.md")) + sorted((X / "docs").glob("*.md")):
        t = p.read_text(encoding="utf-8")
        lines = t.split("\n")
        for i, ln in enumerate(lines):
            if ln.startswith("|") and ln.count("|") >= 3 and i + 2 < len(lines):
                rows.append("\n".join(lines[max(0, i - 1):i + 6]) + "\n")
                if len(rows) > 90:
                    return rows
    return rows


def whitespace_blocks(rng: random.Random) -> list[str]:
    out = []
    for c in WS_CHARS:
        for n in (1, 2, 3, 5, 8, 17, 64, 129):
            run = c * n
            out += [run, "a" + run, run + "a", "a" + run + "b", run + "\n", "\n" + run, "x" + run + "\n", "1" + run + "2", "." + run + "."]
    for n in (200, 255, 256, 257, 511, 512, 513, 1000, 1023, 1024, 1025, 2048, 4097, 5000):
        for c in (" ", "\t", "\n"):
            out += [c * n, "a" + c * n + "b", c * n + "x", "x" + c * n]
        out.append(" " * n + "\n" + " " * n)
        out.append("\n".join(" " * (j % 9) for j in range(40)))
    for _ in range(150):
        n = rng.randint(1, 30)
        out.append("".join(rng.choice(WS_CHARS) for _ in range(n)))
        out.append("x" + "".join(rng.choice(" \t\n\r") for _ in range(n)) + "y")
        out.append("".join(rng.choice(WS_CHARS[:5] + ["a", "1", ".", "'"]) for _ in range(n + 5)))
    return out


def tabs_newlines(rng: random.Random) -> list[str]:
    out = ["\t\t\n\n\t", "\r\n\r\n", "\r\r\n\n\r", "a\r\nb\r\nc", "a\n\n\n\nb", "\n" * 80, "\r\n" * 60, "  \n  \n  \n", "\t\n\t\n",
           "def f():\n\tif x:\n\t\treturn 1\n\treturn 2\n", "def f():\n    if x:\n        return 1\n    return 2\n\n\n",
           "line\n  \nline\n \t \nline\n", "x\n\n  y\n\n\n    z", "\n\n\n  indented after blank lines", "end with newline\n",
           "end with two\n\n", "trailing spaces   \nnext   \n", "mixed\t \t\n \t \nrows", "\u2028x\u2029y\x85z", "a\x0bb\x0cc"]
    for _ in range(120):
        n = rng.randint(2, 24)
        out.append("".join(rng.choice(["\n", "\n", "\r\n", "\t", " ", " ", "  ", "a", "bc", "\r"]) for _ in range(n)))
    return out


def digit_blocks(rng: random.Random) -> list[str]:
    out = []
    for n in range(1, 41):
        out.append("".join(rng.choice("0123456789") for _ in range(n)))
        out.append("x" + "7" * n + "y")
    out += ["1,234,567.89", "-0.001", "3.14159265358979", "0xDEADBEEF", "1e-10", "+1 (555) 123-4567", "2026-10-07T12:34:56Z", "12:34:56.789",
            "1/2 3/4", "1\u00b2+2\u00b2=5", "No. 5", "#1", "100%", "$1,000", "\u20ac1.000,00", "1_000_000", "007", "00000000", "9" * 100,
            "a1b22c333d4444e55555"]
    for s in NUMBERS:
        out += [s, "a" + s + "b", s + s, "1" + s + "2", " " + s, s + " ", s + "x", "x" + s]
    for _ in range(100):
        out.append("".join(rng.choice("0123456789\u00b2\u0663\uff11,.- ax") for _ in range(rng.randint(2, 30))))
    return out


def url_blocks(rng: random.Random) -> list[str]:
    import base64
    import hashlib
    import uuid
    out = ["https://example.com/path/to/page?query=1&x=%20y#frag", "http://localhost:8080/v1/chat/completions", "ftp://user:pw@host/dir/",
           "mailto:someone@example.com", "someone.name+tag@sub.example.co.uk", "/usr/local/bin/python3.12", "C:\\Users\\Name\\Documents\\file.txt",
           "~/AI/Strata-X/moex/tools/x.py", "../../a/b/../c", "192.168.0.1", "::1", "2001:0db8:85a3:0000:0000:8a2e:0370:7334", "[::]:8080",
           "git@github.com:user/repo.git", "www.example.com", "https://\u4f8b\u3048.jp/\u30d1\u30b9?q=\u691c\u7d22", "file:///tmp/x%20y.txt",
           "https://example.com/" + "a" * 200, "data:image/png;base64," + base64.b64encode(os.urandom(60)).decode()]
    r2 = random.Random(7)
    for i in range(30):
        out.append(str(uuid.UUID(int=r2.getrandbits(128))))
        out.append(hashlib.sha256(str(i).encode()).hexdigest())
        out.append(base64.b64encode(bytes(r2.getrandbits(8) for _ in range(r2.randint(4, 300)))).decode())
        out.append("commit " + hashlib.sha1(str(i).encode()).hexdigest()[:r2.randint(7, 40)])
    return out


def contraction_blocks() -> list[str]:
    out = []
    for w in CONTRACTION_WORDS:
        for c in CONTRACTIONS:
            out.append(w + c)
    out += ["don't", "Don't STOP", "I'm sure they'll say we've seen it, she'd know", "rock'n'roll", "'tis the season", "a 'quoted' word",
            "'Hello'", "''double''", "it's' x", "x'", "'", " '", "' ", "'\n", "\n'", "'s", " 's", "'S", "1's", ".'s", "\u4e2d's", "e\u0301's",
            "'re", "'ve", "'ll", "'d", "'m", "'t", "I'M", "WE'VE", "THEY'LL", "SHE'D", "CAN'T", "wouldn't've", "y'all'd've"]
    return out


def special_blocks(tokenizer, rng: random.Random) -> tuple[list[str], list[str]]:
    """(plain, real): the tokenizer's special tokens typed in every way a user could, and genuinely used."""
    specials = list(tokenizer.special_tokens)
    plain: list[str] = []
    real: list[str] = []
    for t in specials:
        plain += [t, "a" + t + "b", " " + t, t + " ", "\n" + t + "\n", t + t, t[:-1], t[1:], t[:len(t) // 2], t[len(t) // 2:],
                  '{"x": "' + t + '"}']
    for t in rng.sample(specials, min(len(specials), 150)):
        k = rng.randrange(1, max(2, len(t) - 1))
        plain += [t[:k] + " " + t[k:], t[:k] + t, t + t[k:], t.upper(), t.lower(), t.replace("|", "\uff5c"), "<" + t, t + ">",
                  t[:k] + t[k:] + t[:k] + t[k:]]
    for _ in range(300):
        a, b = rng.choice(specials), rng.choice(specials)
        k = rng.randrange(1, max(2, len(a)))
        plain += [a + b, a + " " + b, a[:k] + b, a + "x" + b, a[k:] + b[:k], a + "\n" + b, a[:-1] + b[1:]]
    # fragments of the usual brackets
    for f in ["<|", "|>", "<|user", "|user|>", "< |user|>", "<|user| >", "<||>", "<|<|user|>|>", "<<|user|>>", "<|\uff5c", "\uff5c>", "<\uff5c",
              "<think", "think>", "</think", "</ think>", "<tool_call", "<tool_call >", "<arg_key>", "<tool_call><arg_key>", "[gMASK", "gMASK]",
              "[[gMASK]]", "/nothink", "/ nothink", "/nothinkk", "x/nothink", "/nothink/nothink", "\u2581", "<unk>", "<s>", "</s>", "<pad>",
              "<|endoftext|", "<|endoftext|>", "<|end_of_text|>", "<|EOT|>", "<|eot_id|>", "<end_of_turn>", "<\uff5cend\u2581of\u2581sentence\uff5c>",
              "<\uff5cDSML\uff5ctool_calls>", "\uff5cDSML\uff5c", "<dsml:", "</dsml:", "<tool_response>", "</tool_response>", "<|im_start|>assistant\n",
              "<|im_end|>", "<|fim_prefix|>", "<|file_sep|>", "<|repo_name|>"]:
        plain += [f, "a" + f, f + "b", " " + f + " ", f * 3]
    # real use: what a rendered prompt contains
    sp = [t for t in specials if len(t) < 40]
    for _ in range(300):
        parts = []
        for _ in range(rng.randint(1, 5)):
            parts.append(rng.choice(sp))
            parts.append(rng.choice(["", "\n", " ", "Hello", "Hello, world!", "\u4f60\u597d", "1234", " the", "{\"a\": 1}", "  ", "\n\n"]))
        real.append("".join(parts))
    real += [t for t in specials]
    return plain, real


def long_word_blocks(rng: random.Random) -> list[str]:
    out = []
    for unit in LONG_KINDS:
        for n in (63, 64, 65, 66, 100, 257, 1000, 4097):
            s = unit * (n // max(1, len(unit)) + 1)
            out += [s[:n], " " + s[:n], s[:n] + " x", "x" + s[:n]]
    for n in (10000, 20000):
        for unit in ("abcdefghij", "\u4e2d\u6587", "0123456789", "x", "\u3053\u3093\u306b\u3061\u306f", "AAAA", "a1", "\U0001f600"):
            out.append((unit * n)[:n])
    # mixed letter/digit/punct long run without whitespace
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.\u00e9\u4e2d"
    for n in (70, 300, 2000, 9000):
        out.append("".join(rng.choice(alphabet) for _ in range(n)))
    return out


def fuzz_blocks(tokenizer, rng: random.Random, n: int) -> list[str]:
    classes = [
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"), list("0123456789"), list(" \t\n\r"), list("'\"`"),
        list("!@#$%^&*()-_=+[]{};:,.<>/?\\|~"), ["\u00e9", "\u00fc", "\u00df", "\u0130", "\u0131", "\u017f", "\u212a", "\u03a9", "\u0436", "\u05d0"],
        ["\u4e2d", "\u6587", "\u65e5", "\u672c", "\u3042", "\u30a2", "\ud55c", "\u1100", "\u1161"], ["\u0301", "\u0308", "\u200d", "\ufe0f", "\u093f", "\u0e31"],
        ["\u00a0", "\u2003", "\u3000", "\u200b", "\x85", "\u2028", "\x1c", "\x7f", "\x00", "\x01"],
        ["\U0001f600", "\U0001f1fa", "\U0001f3fd", "\u2764", "\u2728", "\U0001f9e0"], ["\u00b2", "\u00bd", "\u0663", "\uff11", "\u2167", "\u2460"],
        ["\u2018", "\u2019", "\u201c", "\u201d", "\u2014", "\u2026", "\u00ab", "\u00bb", "\u3001", "\u3002", "\uff0c", "\uff01"], ["\u20ac", "\u00a3", "\u00a5", "\u00b0", "\u00d7", "\u2211", "\u221e"],
        ["\ue000", "\uf8ff", "\ufffd", "\ufffe", "\uffff", "\U000f0e01", "\U0010ffff", "\U000e0001"],
    ]
    frags = [t for t in tokenizer.special_tokens if len(t) <= 24]
    out = []
    for _ in range(n):
        weights = [rng.random() ** 2 for _ in classes]
        length = rng.choice([1, 2, 3, 4, 6, 8, 12, 20, 40, 80, 160])
        parts = []
        for _ in range(length):
            if frags and rng.random() < 0.03:
                t = rng.choice(frags)
                parts.append(t if rng.random() < 0.6 else t[:rng.randrange(1, len(t))])
            else:
                parts.append(rng.choice(rng.choices(classes, weights)[0]))
        out.append("".join(parts))
    return out


def boundary_blocks(rng: random.Random) -> list[str]:
    """The first and last code point of every range of llama.cpp's class table, in the contexts that matter."""
    out = []
    edge = []
    r = UT.RANGES
    for i in range(len(r) - 1):
        a, b = r[i][0], r[i + 1][0] - 1
        for cp in {a, b, (a + b) // 2}:
            if 0xD800 <= cp <= 0xDFFF:
                continue
            edge.append(chr(cp))
    for ch in edge:
        out += [ch, "a" + ch + "b", ch + "\n", " " + ch, ch + "1", "'" + ch]
    return out


def control_blocks() -> list[str]:
    out = []
    for cp in list(range(0, 0x20)) + list(range(0x7f, 0xa0)):
        c = chr(cp)
        out += [c, "a" + c + "b", " " + c, c + " ", c + c + c, "1" + c + "2", c + "\n"]
    for cp in [0xfdd0, 0xfdef, 0xfffe, 0xffff, 0x1fffe, 0x10ffff, 0xe000, 0xf8ff, 0xf0000, 0x100000, 0xe0001, 0xe007f, 0xad, 0x200b, 0x2060,
               0xfeff, 0x061c, 0x2066, 0x2069, 0x180e, 0xfff9, 0xfffc, 0xfffd]:
        c = chr(cp)
        out += [c, "a" + c + "b", " " + c, c + c, "\u4e2d" + c + "\u6587"]
    return out


def code_hand() -> list[str]:
    return [
        'fn main() {\n    let v: Vec<i32> = (0..10).map(|x| x * 2).collect();\n    println!("{:?}", v);\n}\n',
        'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("hello, \u4e16\u754c")\n}\n',
        'public class A {\n  public static void main(String[] args) {\n    System.out.println("x" + 1);\n  }\n}\n',
        'SELECT a.id, COUNT(*) AS n FROM t a LEFT JOIN u b ON a.id = b.id WHERE a.x > 10 GROUP BY a.id HAVING n > 1 ORDER BY n DESC;',
        '#!/bin/bash\nset -euo pipefail\nfor f in *.txt; do\n  echo "$f" | tr a-z A-Z\ndone\n',
        '<?php\n$a = ["x" => 1];\necho $a["x"] . "\\n";\n?>', 'def f(x: int) -> str:\n    return f"{x!r:>10}"\n',
        'const f = async (x) => { await g(x); return `${x}-${x * 2}`; };\n', 'x <- c(1,2,3); y = x^2; plot(x, y, type="l")',
        'SELECT * FROM `t` WHERE `a` = \'it\'\'s\';', 'interface A<T extends object = {}> { [k: string]: T | undefined; }\n',
        '\\begin{equation}\n  \\int_0^\\infty e^{-x^2}\\,dx = \\frac{\\sqrt{\\pi}}{2}\n\\end{equation}\n', 'a ? b : c ?? d?.e ?? f!',
        'std::vector<std::pair<int, std::string>> v{{1, "a"}, {2, "b"}};\nauto it = std::find_if(v.begin(), v.end(), [](auto &p) { return p.first == 2; });\n',
        '    # comment with trailing spaces   \n\tx = 1\t# tab comment\n', "echo $((1+2)) && ls -la | grep '^d'", 'if (a<b && c>=d || !e) { x++; y--; z += 1 << 3; }',
        '/* block\n * comment\n */\n// line comment\n#define MAX(a,b) ((a)>(b)?(a):(b))\n', '<div class="a b" id=\'x\' data-k="v">text &amp; more</div>',
        'a{color:#fff;margin:0 auto;font:12px/1.5 "Helvetica Neue",Arial}', 'key: value\nlist:\n  - a\n  - b: 1\n    c: [1, 2]\n', '[section]\nname = "x"\nn = 1_000\n',
        'let s = "emoji \\u{1F600} \\x41 \\n";\nlet t = r#"raw "quote""#;\n', 'λx. x x', 'forall a b. (a -> b) -> [a] -> [b]', '0b1010_1010 0o755 0xFF_FF 1e3f 1.5d',
    ]


# ------------------------------------------------------------------------------------------------ the corpus
def build(tokenizer, seed: int = 1, scale: float = 1.0) -> dict[str, list[str]]:
    """The corpus by category.  `scale` multiplies the sampled (not the hand-made) categories."""
    rng = random.Random(seed)
    cats: dict[str, list[str]] = {}

    def keep(name: str, items):
        seen, out = set(), []
        for s in items:
            if s not in seen:
                seen.add(s)
                out.append(s)
        cats[name] = out

    pieces = prose_pieces(rng)
    keep("prose_en", [p[:rng.choice([60, 200, 600, 2000, 6000])] for p in pieces] +
         ["The quick brown fox jumps over the lazy dog.", "Hello, world!", "  Leading and trailing  ", "Mr. Smith's 3 dogs aren't here.",
          "To be, or not to be: that is the question.", "ALL CAPS SENTENCE WITH NUMBERS 123 AND SYMBOLS #$%", "e.g. i.e. etc.", "A\u00a0B", "word" * 50])
    keep("prose_multi", [p[:rng.choice([80, 300, 1200, 4000])] for p in multi_pieces(rng)])
    keep("contractions", contraction_blocks())
    for lang in ("cpp", "h", "c", "cu", "py", "ts", "svelte", "js", "sh", "yml", "cmake", "glsl", "metal", "kt", "swift", "html", "css", "toml", "m", "pl", "rb"):
        files = max(2, int(10 * scale))
        keep("code_" + lang, _sample_files(lang, rng, files, max(2, int(6 * scale))))
    keep("code_hand", code_hand())
    keep("json", json_samples(rng))
    keep("markdown_tables", markdown_tables(rng))
    scripts = []
    for lang, items in SCRIPTS.items():
        for s in items:
            scripts += [s, s * 3, " " + s + " ", s + "\n" + s]
    keep("scripts", scripts)
    keep("emoji", EMOJI + [e + e for e in EMOJI] + ["a" + e + "b" for e in EMOJI] + [" " + e for e in EMOJI] + ["".join(EMOJI)])
    marks = []
    for m in MARKS:
        marks += [m, m + m, " " + m, "a " + m, m + "\n", "'" + m, "1" + m]
    keep("marks", marks)
    keep("whitespace", whitespace_blocks(rng))
    keep("tabs_newlines", tabs_newlines(rng))
    keep("digits", digit_blocks(rng))
    keep("urls_paths", url_blocks(rng))
    plain, real = special_blocks(tokenizer, rng)
    keep("special_plain", plain)
    keep("special_real", real)
    keep("long_words", long_word_blocks(rng))
    keep("fuzz", fuzz_blocks(tokenizer, rng, int(12000 * scale)))
    keep("boundaries", boundary_blocks(rng))
    keep("controls", control_blocks())
    return cats
