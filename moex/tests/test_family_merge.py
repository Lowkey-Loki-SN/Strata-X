#!/usr/bin/env python3
"""moex/tests/test_family_merge.py - the family path after the merge of upstream 0.1.43.1 (a0116b27).

    ~/AI/Strata/.venv/bin/python moex/tests/test_family_merge.py

  1. #2026: Qwen's assistant messages keep their `</tool_call>` as the token (a client replaying the model's raw call text);
     a family marks every one of its call tags in every message, so a `</tool_call>` inside a past call's argument stays
     text: quoting it in an argument adds no `</tool_call>` token to the rendered history.
  2. "replicas" (upstream's data-parallel engines, each on its own cards) is refused for a family config: these models
     need every card for one engine.
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

X = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(X))
sys.path.insert(0, str(X / "tools"))

from serve import families  # noqa: E402
from serve.frontend import openai_to_messages  # noqa: E402
from serve.server import MockEngine, Service  # noqa: E402

WRITE = {"type": "function", "function": {
    "name": "write_file", "description": "Write a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]}}}


def request(argument: str) -> dict:
    return {"model": "m", "tools": [WRITE], "messages": [
        {"role": "user", "content": "Write the parser test."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "write_file", "arguments": json.dumps({"path": "t.py", "content": argument})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "written"},
        {"role": "user", "content": "Thanks."}]}


class AssistantCallTags(unittest.TestCase):
    def test_family_keeps_a_quoted_call_end_as_text(self):
        for key in ("glm5", "mimo2"):          # the families whose call syntax closes with </tool_call>
            with self.subTest(family=key):
                spec = families.get(key)
                tok = spec.load_tokenizer(spec.tokenizer_path())
                svc = Service(MockEngine(tok, "hi", max_context=8192), tok, spec.load_template(spec.tokenizer_path()),
                              model_name=spec.model_name, family=spec)
                end = tok.encode("</tool_call>", parse_special=True)
                self.assertEqual(len(end), 1, "</tool_call> is one control token")
                quoted = svc.encode_prompt(*openai_to_messages(request("assert p.feed('</tool_call>') == []")))
                plain = svc.encode_prompt(*openai_to_messages(request("x = 1")))
                # the template's own </tool_call> tokens (GLM's tool instructions write an example call too) and nothing more
                self.assertGreaterEqual(plain.count(end[0]), 1)
                self.assertEqual(quoted.count(end[0]), plain.count(end[0]), "a quoted </tool_call> in an argument is text")


class ReasoningClose(unittest.TestCase):
    """#1053 (opt-in reasoning_close_retry): the text that closes a reply's thinking is the template's - Qwen's has a blank
    line after </think>, a family's answer follows it directly (as its reasoning_wrap_up)."""

    def test_each_family_closes_as_its_template(self):
        for key in ("glm5", "mimo2", "deepseek4"):
            with self.subTest(family=key):
                spec = families.get(key)
                tok = spec.load_tokenizer(spec.tokenizer_path())
                svc = Service(MockEngine(tok, "hi", max_context=4096), tok, spec.load_template(spec.tokenizer_path()),
                              model_name=spec.model_name, family=spec)
                self.assertEqual(svc.reasoning_close(), "\n</think>")
                self.assertTrue(spec.reasoning_wrap_up.endswith("</think>"))


class Replicas(unittest.TestCase):
    def test_refused_for_a_family(self):
        for key in ("glm5", "mimo2", "deepseek4"):
            spec = families.get(key)
            for value in (2, [{"gpus": [0]}, {"gpus": [1]}]):
                with self.subTest(family=key, replicas=value), self.assertRaises(ValueError):
                    families.check_config({"replicas": value}, spec)
            for value in (None, 1, False):
                families.check_config({"replicas": value} if value is not None else {}, spec)   # accepted


if __name__ == "__main__":
    unittest.main(verbosity=2)
