"""moex: serve/prompt_memo.py gives the ids of one full encode, turn after turn (DeepSeek-V4.1's tokenizer and template).

Agent-shaped conversations built from repo files (no session content): growing turns, an edit in the middle, a
compaction-like rewrite, control-token and <think> literals inside messages (plain spans), two conversations taking
turns.  Every encode is compared with tok.encode of the whole prompt, and the reuse is checked to happen.

    python -m unittest serve.test_prompt_memo
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")

import json
import random
import sys
import unittest
from pathlib import Path

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X / "tools"))
sys.path.insert(0, str(X))
from serve import families  # noqa: E402
from serve.frontend import mark_think_literals, unmark_think_literals  # noqa: E402
from serve.prompt_memo import LiteralMemo, PromptMemo, common_prefix  # noqa: E402
from serve.server import Service  # noqa: E402

SPEC = families.get("deepseek41")
TOK_DIR = Path(_os.path.join(_DATA, 'tokenizers/deepseek4'))


def corpus():
    out = []
    for p in sorted((X / "serve").glob("*.py"))[:40] + sorted((X / "moex" / "docs").glob("*.md"))[:6]:
        t = p.read_text(encoding="utf-8")
        for i in range(0, min(len(t), 60000), 6000):
            out.append(t[i:i + 6000])
    return out


@unittest.skipUnless(TOK_DIR.is_dir(), "the DeepSeek-V4.1 tokenizer is not on this machine")
class PromptMemoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = SPEC.load_tokenizer(TOK_DIR)
        cls.svc = Service(None, cls.tok, SPEC.load_template(), model_name=SPEC.model_name, family=SPEC)
        cls.texts = corpus()
        # literals a message may hold: control tokens and the think tags (they become plain spans)
        cls.literals = [t for t in cls.tok.control_tokens[:40]] + ["<think>", "</think>"]

    def conversation(self, rng, turns, salt=""):
        msgs = [{"role": "system", "content": "You are a coding agent." + salt},
                {"role": "user", "content": "Audit the prompt path." + salt}]
        for k in range(turns):
            body = self.texts[rng.randrange(len(self.texts))]
            if rng.random() < 0.25:                 # a literal inside a tool result or a reasoning
                lit = rng.choice(self.literals)
                at = rng.randrange(len(body) + 1)
                body = body[:at] + lit + body[at:]
            call = {"id": f"c{k}", "type": "function",
                    "function": {"name": "read", "arguments": json.dumps({"path": f"f{k}.py"})}}
            msgs.append({"role": "assistant", "content": "", "reasoning_content": self.texts[(k * 7) % len(self.texts)][:300],
                         "tool_calls": [call]})
            msgs.append({"role": "tool", "tool_call_id": f"c{k}", "content": body})
        return msgs

    def check(self, msgs, expect_reuse=None):
        kw = {"reasoning_effort": "max"}
        ids = self.svc.encode_prompt(msgs, None, kw)
        marked, mtools, changed = mark_think_literals(msgs, None, self.svc.literals, **self.svc.literal_keep)
        prompt = self.svc.render_prompt(marked, mtools, kw)
        if changed:
            prompt, plain = unmark_think_literals(prompt, self.svc.literals)
            full = self.tok.encode(prompt, parse_special=True, plain=plain)
        else:
            full = self.tok.encode(prompt, parse_special=True)
        self.assertEqual(ids, full)
        if expect_reuse is not None:
            self.assertGreaterEqual(self.svc.prompt_memo.reused, expect_reuse)
        return ids

    def test_turns_edits_and_two_conversations(self):
        rng = random.Random(7)
        self.svc.prompt_memo.entries = []
        a = self.conversation(rng, 6)
        first = self.check(a)
        for _ in range(6):                          # growing turns: most of the prompt comes from the last request
            a = a + self.conversation(rng, 1)[2:]
            self.check(a, expect_reuse=len(first) // 2)
        b = self.conversation(random.Random(99), 5, salt=" (b)")
        self.check(b)                               # another conversation in between
        a = a + self.conversation(rng, 1)[2:]
        self.check(a, expect_reuse=len(first) // 2)  # the first one still remembered (keep=2)
        a2 = [dict(m) for m in a]                   # an edit in the middle: reuse up to it, exact after it
        mid = len(a2) // 2
        a2[mid] = dict(a2[mid], content=str(a2[mid].get("content")) + " edited <think>")
        self.check(a2)
        c = [a[0], {"role": "user", "content": "Summary so far: ..."}] + a[-4:]   # a compaction-like rewrite
        self.check(c)

    def test_random_cut_points_against_full_encode(self):
        # the memo alone, on texts that differ at random places (inside text, at and around special tokens)
        rng = random.Random(3)
        specials = list(self.tok.special_tokens)[:60]
        memo = PromptMemo(self.tok, keep=2, mode="1")
        base = "".join(self.texts[rng.randrange(len(self.texts))][:800] + rng.choice(specials) for _ in range(40))
        for trial in range(60):
            at = rng.randrange(len(base))
            new = base[:at] + rng.choice(["", "x", rng.choice(specials), "\n\n", "é→"]) + base[at:]
            spans = []
            if rng.random() < 0.5:                  # a plain span over a special token's text somewhere
                for m in memo.pat.finditer(new):
                    if rng.random() < 0.1:
                        spans.append((m.start(), m.end()))
            self.assertEqual(memo.encode(new, spans), self.tok.encode(new, parse_special=True, plain=spans), trial)
            if trial % 3 == 0:
                base = new

    def test_literal_memo_same_results(self):
        rng = random.Random(5)
        msgs = self.conversation(rng, 8)
        memo = LiteralMemo(self.svc.literals)
        for _ in range(2):
            memo.begin()
            got = mark_think_literals(msgs, None, self.svc.literals, **self.svc.literal_keep, memo=memo)
            want = mark_think_literals(msgs, None, self.svc.literals, **self.svc.literal_keep)
            self.assertEqual(got[0], want[0])
            self.assertEqual(got[2], want[2])

    def test_common_prefix(self):
        for a, b in [("", ""), ("abc", "abd"), ("abc", "abc"), ("abc", "abcd"), ("x" * 1000 + "a", "x" * 1000 + "b")]:
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            self.assertEqual(common_prefix(a, b), n)


if __name__ == "__main__":
    unittest.main()
