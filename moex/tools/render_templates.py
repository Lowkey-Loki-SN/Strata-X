"""moex/tools/render_templates.py - render every fixture conversation with every family's chat template and write the prompts out.

    python3 -I moex/tools/render_templates.py [glm5 deepseek4 mimo2] [--show CASE] [--variant NAME]

The rendering is the server's own: openai_to_messages, the hygiene marks, the family template in the front end's Jinja
environment, then the family tokenizer.  Files: moex/out/templates/<family>/<case>.<variant>.txt (the prompt) and
<case>.<variant>.ids.txt (its tokens, special ones shown in place).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

X = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(X / "moex" / "tests"))
import family_fixtures as FX  # noqa: E402


def annotate(tok, ids) -> str:
    parts = []
    for i in ids:
        text = tok.tokens[i]
        if tok.attr[i] & (tok.attr[i] & 0x18):      # CONTROL or USER_DEFINED: show as [text]
            parts.append("[" + text + "]")
        else:
            parts.append(tok.token_bytes(i).decode("utf-8", "replace"))
    return "|".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="*", default=["glm5", "deepseek4", "mimo2"])
    ap.add_argument("--show", default="")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--out", default=str(X / "moex" / "out" / "templates"))
    a = ap.parse_args()
    for fam in a.families:
        svc, spec = FX.service_for(fam)
        out = pathlib.Path(a.out) / fam
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for case in FX.CASES:
            for variant in FX.REQUEST_VARIANTS:
                try:
                    prompt, ids, thinking = FX.render(svc, case, variant)
                except Exception as e:                      # a template that refuses a request is a result too
                    prompt, ids, thinking = "ERROR: %s: %s" % (type(e).__name__, e), [], None
                (out / ("%s.%s.txt" % (case, variant))).write_text(prompt, encoding="utf-8")
                (out / ("%s.%s.ids.txt" % (case, variant))).write_text(annotate(svc.tok, ids) + "\n", encoding="utf-8")
                n += 1
                if a.show == case and variant == a.variant:
                    print("== %s / %s / %s (thinking=%s, %d tokens)\n%s\n" % (fam, case, variant, thinking, len(ids), prompt))
        print("%s: %d prompts -> %s" % (fam, n, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
