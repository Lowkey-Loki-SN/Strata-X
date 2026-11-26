"""moex/tools/ernie55/verify_tokenizer.py - the server's ERNIE-5.5 tokenizer (tools/strata_tokenizer_families.py, pre ernie55,
the directory extract_hf writes) against the release's own (HF tokenizers on tokenizer.json), id for id.

Two interpreters: HF tokenizers lives in the vLLM venv, the server's `regex` in its own - and the comparison must use the
server's `regex` (its Unicode tables decide the pre-tokenizer's classes).

    ~/AI/vLLM/.venv/bin/python -I verify_tokenizer.py hf OUT.jsonl FILE...      the release's ids per paragraph
    ~/AI/Strata/.venv/bin/python -I verify_tokenizer.py check OUT.jsonl         the server's, compared

Each paragraph (the files split at blank lines) is encoded whole, the added tokens matched in it as HF matches them (the
family tokenizer's parse_special=True), and decoded back; a list of strings with every control and added token, digit
runs, CJK, odd spaces and mixed scripts is checked too.
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import json
import pathlib
import sys

TJ = _os.path.join(_MODELS, 'HF/ERNIE-5.5-Flash-Preview-FP8/tokenizer.json')
DIR = _os.path.join(_DATA, 'tokenizers/ernie55')

EXTRA = [
    "<|ERNIE_MASK|><|ernie:im_start|>system\n<ernie:global_setting>\nreasoning_effort=high\n</ernie:global_setting><|ernie:im_end|>\n",
    "<|ernie:im_start|>user\nWhat is 17 * 23?<|ernie:im_end|>\n<|ernie:im_start|>assistant\n<ernie:think>",
    "<ernie:tool_call>\n<invoke name=\"get_weather\"><parameter name=\"city\">Paris</parameter></invoke>\n</ernie:tool_call>",
    "<|ernie:im_start|>tool\n<ernie:tool_output>sunny</ernie:tool_output><|ernie:im_end|>",
    "1 12 123 1234 12345 123456 1234567 12345678901234567890 3.14159 1,000,000 ½ ² ٣٤٥ १२३",
    "价格是12345元，今天是2026年11月25日。こんにちは、カタカナ。한국어 텍스트.",
    "tab\tsep  two  spaces nbsp thin hair⁠joiner　ideographic end ",
    "emoji 😀👍🏽 and zwj 👨‍👩‍👧 mixed Ελληνικά русский العربية עברית",
    "code: for (int i = 0; i < n; ++i) { x[i] += y[i] * 2.0f; } // comment\n\n\n  indented",
    "<|ERNIE_PAD|><ERNIE_mask:1208></ernie:think>plain text with a fake <|ernie:im_end tag",
    "\n\n\n", " ", "", "a", "€", "\u0000\u0001\u001f",
]


def paragraphs(files):
    out = []
    for f in files:
        t = pathlib.Path(f).read_text(encoding="utf-8", errors="replace")
        out += [p for p in t.split("\n\n") if p]
    return out + EXTRA


def main() -> int:
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "hf":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(TJ)
        with open(path, "w", encoding="utf-8") as f:
            for p in paragraphs(sys.argv[3:]):
                ids = tok.encode(p, add_special_tokens=False).ids
                f.write(json.dumps({"text": p, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)},
                                   ensure_ascii=False) + "\n")
        return 0
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))
    import strata_tokenizer_families as F
    tk = F.FamilyTokenizer.from_dir(DIR)
    n = bad = bad_dec = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        got = tk.encode(r["text"], parse_special=True)
        n += 1
        if got != r["ids"]:
            bad += 1
            if bad <= 8:
                k = next((i for i, (a, b) in enumerate(zip(got, r["ids"])) if a != b), min(len(got), len(r["ids"])))
                print(f"ids differ at {k}: {got[max(0, k - 3):k + 4]} vs {r['ids'][max(0, k - 3):k + 4]} in {r['text'][:80]!r}")
        dec = tk.decode(r["ids"])
        if isinstance(dec, bytes):
            dec = dec.decode("utf-8", "replace")
        if dec != r["decoded"]:
            bad_dec += 1
            if bad_dec <= 4:
                print(f"decode differs: {dec[:60]!r} vs {r['decoded'][:60]!r}")
    print(f"{n} texts: {bad} encode differently, {bad_dec} decode differently")
    return 1 if bad or bad_dec else 0


if __name__ == "__main__":
    sys.exit(main())
