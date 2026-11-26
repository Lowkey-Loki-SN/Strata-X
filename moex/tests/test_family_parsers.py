#!/usr/bin/env python3
"""moex/tests/test_family_parsers.py - serve/family_parsers.py.

    ~/AI/Strata/.venv/bin/python moex/tests/test_family_parsers.py       (needs jinja2: the front end's)

  1. the machine, with Qwen's syntax, gives OutputParser's events - on a set of model outputs, whole, one character at a
     time and in random pieces, with and without streamed tool calls, thinking and not;
  2. each family's own template is the other half of the contract: a past assistant turn rendered by the template (reasoning,
     text, calls of every argument type) is parsed back into exactly what was rendered, in every chunking;
  3. the cases a syntax has of its own: prose that names the tag, a call quoted in a code fence, a call inside the
     reasoning, output that stops in the middle, DeepSeek's several invokes in one block, MiMo's own `<think>`;
  4. the forced call's opening text parses as the start of that call.
"""
import json
import os
import random
import sys
import unittest

X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, X)
sys.path.insert(0, os.path.join(X, "tools"))
sys.path.insert(0, os.path.join(X, "moex", "tests"))

from serve import families  # noqa: E402
from serve import family_parsers as FP  # noqa: E402
from serve.frontend import OutputParser  # noqa: E402

TOOLS = [
    {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}}}},
    {"name": "add", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"},
                                                                    "tags": {"type": "array"}}}},
    {"name": "write_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}, "append": {"type": "boolean"}, "meta": {"type": "object"}}}},
    {"name": "foo"},
]


def norm(events):
    """Events -> a comparable list: adjacent reasoning/content/args pieces joined, call ids numbered by first sight."""
    out, ids = [], {}
    for e in events:
        cid = ids.setdefault(e.call.id, len(ids)) if e.call is not None else None
        if e.kind in ("reasoning", "content"):
            if not e.text:
                continue
            if out and out[-1][0] == e.kind:
                out[-1] = (e.kind, out[-1][1] + e.text)
            else:
                out.append((e.kind, e.text))
        elif e.kind == "tool_args":
            if out and out[-1][0] == "tool_args" and out[-1][2] == cid:
                out[-1] = ("tool_args", out[-1][1] + e.text, cid)
            else:
                out.append(("tool_args", e.text, cid))
        elif e.kind == "tool_start":
            out.append(("tool_start", e.call.name, cid))
        else:
            out.append(("tool_call", e.call.name, json.dumps(e.call.arguments, sort_keys=True, ensure_ascii=False), cid))
    return out


def pieces(text, mode, seed=0):
    if mode == "whole":
        return [text]
    if mode == "char":
        return list(text)
    rng = random.Random(seed)
    out, i = [], 0
    while i < len(text):
        n = rng.randint(1, 9)
        out.append(text[i:i + n])
        i += n
    return out


def run(parser, text, mode, seed=0, reason=None):
    evs = []
    for p in pieces(text, mode, seed):
        evs += parser.feed(p)
    evs += parser.finish(reason)
    return evs


MODES = [("whole", 0), ("char", 0), ("random", 1), ("random", 2), ("random", 3)]

# ------------------------------------------------------------------------------------------------ 1. Qwen's syntax
QWEN_CASES = {
    "plain": ("Let me think.\n</think>\n\nHello there.", True),
    "no_think": ("Hello there.\nSecond line.", False),
    "call": ("Sure.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=unit>\ncelsius\n"
             "</parameter>\n</function>\n</tool_call>", False),
    "types": ("<tool_call>\n<function=add>\n<parameter=a>\n-7\n</parameter>\n<parameter=b>\n0\n</parameter>\n<parameter=tags>\n"
              "[\"x\", \"y z\"]\n</parameter>\n</function>\n</tool_call>", False),
    "write": ("<tool_call>\n<function=write_file>\n<parameter=path>\na/b.txt\n</parameter>\n<parameter=content>\nline1\n"
              "line2 \"q\" <tag> 你好\n\n</parameter>\n<parameter=append>\nfalse\n</parameter>\n<parameter=meta>\n{\"n\": 1}\n"
              "</parameter>\n</function>\n</tool_call>", False),
    "two_calls": ("a\n\n<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n"
                  "</tool_call>\n\n<tool_call>\n<function=add>\n<parameter=a>\n3\n</parameter>\n<parameter=b>\n4\n</parameter>\n"
                  "</function>\n</tool_call>\n\nafter", False),
    "prose_mention": ("I'll use a <tool_call> block to do that, then <tool_call>.\n", False),
    "fenced": ("Example:\n```\n<tool_call>\n<function=foo>\n</function>\n</tool_call>\n```\ndone", False),
    "inline_quote": ("write `<tool_call>` to call.\n", False),
    "in_reasoning_rescued": ("I will call.\n<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n"
                             "</parameter>\n</function>\n</tool_call>\n</think>", True),
    "in_reasoning_refused": ("I will call.\n<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>\n"
                             "but more thinking</think>answer", True),
    "in_reasoning_midline": ("so I write <tool_call> in the text</think>ok", True),
    "end_tag_in_value": ("<tool_call>\n<function=write_file>\n<parameter=content>\ndoc: </parameter> then x\n</parameter>\n"
                         "<parameter=path>\np\n</parameter>\n</function>\n</tool_call>", False),
    "cut_in_value": ("<tool_call>\n<function=write_file>\n<parameter=content>\nabc", False),
    "cut_in_raw": ("<tool_call>\n<function=add>\n<parameter=a>\n12", False),
    "closer_missing": ("<tool_call>\n<function=foo>\n</function>\n", False),
    "cut_in_head": ("text <tool_call>\n<function=fo", False),
    "unknown_function": ("<tool_call>\n<function=nope>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>", False),
}


class QwenEquivalence(unittest.TestCase):
    # Cases where upstream's OutputParser changed after the port (prose that names the tag, a call cut part-way, a
    # missing closer, an inline quote): the "qwen" syntax is only the reference, and each family's own rules are tested
    # with its parser (serve/test_*.py).
    DIVERGED = {"closer_missing", "cut_in_head", "cut_in_raw", "cut_in_value", "inline_quote", "prose_mention"}

    def test_same_events_as_output_parser(self):
        for name, (text, thinking) in QWEN_CASES.items():
            if name in self.DIVERGED:
                continue
            for stream in (True, False):
                for mode, seed in MODES:
                    for reason in (None, "length"):
                        with self.subTest(case=name, stream=stream, mode=mode, seed=seed, reason=reason):
                            ref = OutputParser(thinking=thinking, tools=TOOLS, stream_tools=stream)
                            got = FP.FamilyParser(FP.SYNTAXES["qwen"], thinking=thinking, tools=TOOLS, stream_tools=stream)
                            a, b = norm(run(ref, text, mode, seed, reason)), norm(run(got, text, mode, seed, reason))
                            self.assertEqual(a, b)
                            self.assertEqual((ref.rescued, ref.refused), (got.rescued, got.refused))

    def test_without_tools_the_tag_is_text_in_reasoning(self):
        text = "think <tool_call> more</think>ok"
        ref = OutputParser(thinking=True, tools=None, stream_tools=True)
        got = FP.FamilyParser(FP.SYNTAXES["qwen"], thinking=True, tools=None, stream_tools=True)
        self.assertEqual(norm(run(ref, text, "char")), norm(run(got, text, "char")))


# ------------------------------------------------------------------------------------------------ 2. the templates
class TemplateRoundTrip(unittest.TestCase):
    """What each family's template writes for a past turn, the parser reads back."""

    @classmethod
    def setUpClass(cls):
        cls.T = {}
        for key in ("glm5", "deepseek4", "mimo2"):
            spec = families.get(key)
            cls.T[key] = (spec, spec.load_template(spec.tokenizer_path()))

    CALLS = [
        {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}},
        {"name": "add", "arguments": {"a": -7, "b": 0, "tags": ["x", "y z"]}},
        {"name": "write_file", "arguments": {"path": "a/b.txt", "content": "line1\nline2 \"q\" <tag> 你好 \U0001f600\n",
                                             "append": False, "meta": {"n": 1, "xs": [1, 2.5, None, "s"]}}},
        {"name": "write_file", "arguments": {"path": "x", "content": "\nstarts and ends with a newline\n", "append": True}},
        {"name": "write_file", "arguments": {"path": "doc", "content": "end tags in a value: </arg_value> </parameter> "
                                            "</｜DSML｜parameter> </function> </tool_call>\n</｜DSML｜invoke>"}},
        {"name": "get_weather", "arguments": {"city": "123", "unit": "true"}},       # strings that look like JSON
        {"name": "foo", "arguments": {}},
    ]

    def assistant_text(self, key, reasoning, content, calls):
        spec, tpl = self.T[key]
        user = [{"role": "user", "content": "go"}]
        tools = [{"type": "function", "function": t} for t in TOOLS]
        asst = {"role": "assistant", "content": content, "reasoning_content": reasoning,
                "tool_calls": [{"id": f"c{i}", "type": "function", "function": c} for i, c in enumerate(calls)] or None}
        if asst["tool_calls"] is None:
            del asst["tool_calls"]
        kw = {"enable_thinking": True}
        prefix = tpl.render(user, tools=tools, add_generation_prompt=True, **kw)
        full = tpl.render(user + [asst], tools=tools, add_generation_prompt=False, **kw)
        self.assertTrue(full.startswith(prefix), (key, prefix[-60:], full[len(prefix) - 60:len(prefix) + 40]))
        text = full[len(prefix):]
        for end in ("<|im_end|>", "<｜end▁of▁sentence｜>"):
            if text.endswith(end):
                text = text[:-len(end)]
        return text

    def check(self, key, reasoning, content, calls, thinking_first):
        text = self.assistant_text(key, reasoning, content, calls)
        for stream in (True, False):
            for mode, seed in MODES:
                with self.subTest(family=key, stream=stream, mode=mode, seed=seed, text=text[:70]):
                    p = FP.make_parser(families.get(key).parser_name, True, TOOLS, stream_tools=stream)
                    evs = run(p, text, mode, seed)
                    got_r = "".join(e.text for e in evs if e.kind == "reasoning")
                    got_c = "".join(e.text for e in evs if e.kind == "content")
                    got_calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual(got_r, reasoning)
                    self.assertEqual(got_c, content or "")
                    self.assertEqual([(c.name, c.arguments) for c in got_calls], [(c["name"], c["arguments"]) for c in calls])
                    if stream:    # the streamed arguments, joined, are the call's arguments
                        starts_ = [e.call for e in evs if e.kind == "tool_start"]
                        self.assertEqual([c.id for c in starts_], [c.id for c in got_calls])
                        for c in got_calls:
                            joined = "".join(e.text for e in evs if e.kind == "tool_args" and e.call.id == c.id)
                            self.assertEqual(json.loads(joined), c.arguments)

    def test_reasoning_text_and_calls(self):
        for key in ("glm5", "deepseek4", "mimo2"):
            self.check(key, "I should call the tools.\nSecond line of thought.", "Let me check.", self.CALLS, True)

    def test_each_call_alone(self):
        for key in ("glm5", "deepseek4", "mimo2"):
            for c in self.CALLS:
                self.check(key, "think", "", [c], True)

    def test_text_only(self):
        for key in ("glm5", "deepseek4", "mimo2"):
            self.check(key, "Simple arithmetic: 2+2=4.", "4", [], True)

    def test_parallel_calls(self):
        for key in ("glm5", "deepseek4", "mimo2"):
            self.check(key, "r", "go", self.CALLS[:3], True)


# ------------------------------------------------------------------------------------------------ 3. each syntax's own cases
def events(key, text, thinking=True, reason=None, stream=True, mode="char", tools=TOOLS):
    p = FP.make_parser(key, thinking, tools, stream_tools=stream)
    return run(p, text, mode, 0, reason), p


class Glm(unittest.TestCase):
    def test_newlines_between_tags(self):
        text = "go</think>\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>unit</arg_key>\n" \
               "<arg_value>celsius</arg_value>\n</tool_call>"
        evs, _ = events("glm5", text)
        calls = [e.call for e in evs if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Paris", "unit": "celsius"})])

    def test_no_arguments(self):
        evs, _ = events("glm5", "</think><tool_call>foo</tool_call>")
        calls = [e.call for e in evs if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls], [("foo", {})])
        self.assertEqual("".join(e.text for e in evs if e.kind == "tool_args"), "{}")

    def test_prose_naming_the_tag(self):
        evs, _ = events("glm5", "</think>Use the <tool_call> tag for that.")
        self.assertEqual(norm(evs), [("content", "Use the <tool_call> tag for that.")])

    def test_quoted_in_fence(self):
        text = "</think>Example:\n```\n<tool_call>foo<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>\n```\n"
        evs, _ = events("glm5", text)
        self.assertFalse([e for e in evs if e.kind in ("tool_start", "tool_call")])

    def test_rescued_from_reasoning(self):
        text = "need the weather\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call></think>"
        evs, p = events("glm5", text)
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], [("get_weather", {"city": "Rome"})])
        self.assertEqual(p.rescued, 1)

    def test_cut_call_is_not_closed(self):
        evs, _ = events("glm5", "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Par")
        self.assertTrue([e for e in evs if e.kind == "tool_start"])
        self.assertFalse([e for e in evs if e.kind == "tool_call"])


def norm_kinds(evs):
    return [e for e in evs if e.kind in ("content", "reasoning") and e.text]


class Deepseek(unittest.TestCase):
    D = "｜DSML｜"

    def block(self, invokes):
        d = self.D
        inner = "".join(f'<{d}invoke name="{n}">\n' + "".join(
            f'<{d}parameter name="{k}" string="{"true" if isinstance(v, str) else "false"}">{v if isinstance(v, str) else json.dumps(v)}'
            f'</{d}parameter>\n' for k, v in a.items()) + f"</{d}invoke>\n" for n, a in invokes)
        return f"\n\n<{d}tool_calls>\n{inner}</{d}tool_calls>"

    def test_two_invokes(self):
        text = "go" + "</think>" + "text" + self.block([("add", {"a": 1, "b": 2}), ("get_weather", {"city": "Rome"})])
        for mode in ("whole", "char"):
            evs, _ = events("deepseek4", text, mode=mode)
            self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                             [("add", {"a": 1, "b": 2}), ("get_weather", {"city": "Rome"})])
            self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "text")

    def test_call_after_reasoning_without_end_think(self):
        # the model may leave the thinking by writing the block (llama.cpp: thinking_end_tags = </think>, the block)
        text = "need weather" + self.block([("get_weather", {"city": "Rome"})])
        evs, p = events("deepseek4", text, reason=None)
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"], [("get_weather", {"city": "Rome"})])
        self.assertEqual(p.rescued, 1)

    def test_string_attribute_wins_over_the_schema(self):
        d = self.D
        text = f'</think><{d}tool_calls><{d}invoke name="add"><{d}parameter name="a" string="true">5</{d}parameter></{d}invoke></{d}tool_calls>'
        evs, _ = events("deepseek4", text)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"a": "5"}])

    def test_not_thinking(self):
        text = "plain answer" + self.block([("foo", {})])
        evs, _ = events("deepseek4", text, thinking=False)
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "plain answer")
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_call"], ["foo"])


class Mimo(unittest.TestCase):
    def test_own_think(self):
        evs, _ = events("mimo2", "<think>pondering</think>Answer.")
        self.assertEqual(norm(evs), [("reasoning", "pondering"), ("content", "Answer.")])

    def test_no_think_at_all(self):
        evs, _ = events("mimo2", "Just an answer.")
        self.assertEqual(norm(evs), [("content", "Just an answer.")])

    def test_empty_think(self):
        evs, _ = events("mimo2", "<think></think>Answer.")
        self.assertEqual(norm(evs), [("content", "Answer.")])

    def test_thinking_off_prompt_has_the_empty_think(self):
        evs, _ = events("mimo2", "Answer.", thinking=False)
        self.assertEqual(norm(evs), [("content", "Answer.")])

    def test_exact_values(self):
        text = "<think></think><tool_call><function=write_file><parameter=content>\nx\n</parameter><parameter=path>p</parameter></function></tool_call>"
        evs, _ = events("mimo2", text)
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"content": "\nx\n", "path": "p"}])

    def test_leading_space_before_think(self):
        evs, _ = events("mimo2", "\n<think>a</think>b")
        self.assertEqual(norm(evs), [("reasoning", "a"), ("content", "b")])


class QuotedThinkEnd(unittest.TestCase):
    """#537 for the families: a `</think>` the reasoning quotes stays reasoning.  The families' templates put the answer right
    after the real </think> (no line break, unlike Qwen's), so only a tag the same mark encloses - opened on its line - is a
    quote; a reasoning that ends on a quote or a backtick still ends there."""
    PRE = {"glm5": "", "deepseek4": "", "mimo2": "<think>"}

    def check(self, body, reasoning, content):
        for key, pre in self.PRE.items():
            for mode, seed in MODES:
                with self.subTest(family=key, mode=mode, seed=seed):
                    p = FP.make_parser(key, True, TOOLS, stream_tools=True)
                    evs = run(p, pre + body, mode, seed)
                    want = ([("reasoning", reasoning)] if reasoning else []) + ([("content", content)] if content else [])
                    self.assertEqual(norm(evs), want)

    def test_backticks_enclose_it(self):
        self.check("the parser ends at `</think>` and keeps going</think>Answer.",
                   "the parser ends at `</think>` and keeps going", "Answer.")

    def test_double_quotes_enclose_it(self):
        self.check('the tag "</think>" closes it</think>Hi.', 'the tag "</think>" closes it', "Hi.")

    def test_apostrophes_never_enclose_it(self):
        # the audit of 2026-10-24 (serve/test_deepseek41.py): an apostrophe cannot tell a quote from "I'll", so it never
        # quotes the tag - the first </think> ends the reasoning and the rest is the answer's text
        self.check("I'll write '</think>' there</think>Done.", "I'll write '", "' there</think>Done.")

    def test_reasoning_ending_on_a_closing_quote(self):
        # the quote before the tag closes "hi" (two on the line): the real end, and the answer opens with a quote
        self.check('I will say "hi"</think>"Hello" back.', 'I will say "hi"', '"Hello" back.')

    def test_reasoning_ending_on_a_backtick(self):
        self.check("it uses `x`</think>The answer.", "it uses `x`", "The answer.")

    def test_reasoning_ending_on_a_quote_answer_without_one(self):
        self.check('so "yes"</think>Yes.', 'so "yes"', "Yes.")

    def test_a_fence_closed_right_before_the_tag(self):
        # DSV41_BACKLOG #56: the "```" that closes a code block right before the tag is a fence line, not an open quote
        # (three backticks, an odd count): an answer that starts with a fence of its own is the answer - it stayed reasoning
        body, reasoning, content = "Plan:\n```py\nx=1\n```</think>```py\nprint(1)\n```", "Plan:\n```py\nx=1\n```", \
            "```py\nprint(1)\n```"
        self.check(body, reasoning, content)
        self.check("Code:\n  ```\n  a\n  ```</think>`b` is it.", "Code:\n  ```\n  a\n  ```", "`b` is it.")
        for key, pre in self.PRE.items():
            text = pre + body
            for cut in range(1, len(text)):
                with self.subTest(family=key, cut=cut):
                    p = FP.make_parser(key, True, TOOLS, stream_tools=True)
                    evs = p.feed(text[:cut]) + p.feed(text[cut:]) + p.finish()
                    self.assertEqual(norm(evs), [("reasoning", reasoning), ("content", content)])

    def test_output_ends_on_a_held_tag(self):
        for key, pre in self.PRE.items():
            with self.subTest(family=key):
                p = FP.make_parser(key, True, TOOLS, stream_tools=True)
                evs = run(p, pre + "thinking about `", "char")
                p2 = FP.make_parser(key, True, TOOLS, stream_tools=True)
                evs2 = run(p2, pre + "thinking about `</think>", "char")
                self.assertEqual(norm(evs2), norm(evs))      # the held tag ends the thinking, adds no text


class LongLine(unittest.TestCase):
    """upstream 2a91a144: one long line streamed in small pieces is kept as parts (no re-copy per piece); the events are
    the same in every chunking."""

    def test_long_line_same_events(self):
        line = "x" * 50000 + " `code` " + "y" * 50000
        for key, pre in QuotedThinkEnd.PRE.items():
            with self.subTest(family=key):
                whole = run(FP.make_parser(key, True, TOOLS), pre + line + "</think>" + line, "whole")
                small = run(FP.make_parser(key, True, TOOLS), pre + line + "</think>" + line, "random", 7)
                self.assertEqual(norm(small), norm(whole))
                self.assertEqual(norm(whole), [("reasoning", line), ("content", line)])


# ------------------------------------------------------------------------------------------------ 4. forced calls
class Forced(unittest.TestCase):
    def test_openings_parse_as_the_start_of_that_call(self):
        for key in ("glm5", "deepseek4", "mimo2"):
            spec = families.get(key)
            for choice, want in (({"type": "function", "function": {"name": "get_weather"}}, "get_weather"), ("required", None)):
                with self.subTest(family=key, choice=choice):
                    force = spec.forced_call(choice, TOOLS)
                    self.assertTrue(force)
                    p = FP.make_parser(key, False, TOOLS, stream_tools=True)
                    evs = p.feed(force)
                    if want:
                        if key != "glm5":     # GLM's name has no end tag: it is known when the next tag (or a space) arrives
                            self.assertEqual([e.call.name for e in evs if e.kind == "tool_start"], [want])
                        # the model goes on with the arguments and the call closes
                        rest = {"glm5": '<arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>',
                                "deepseek4": '<｜DSML｜parameter name="city" string="true">Rome</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
                                "mimo2": '<parameter=city>Rome</parameter></function></tool_call>'}[key]
                        evs += p.feed(rest) + p.finish()
                        self.assertEqual([e.call.name for e in evs if e.kind == "tool_start"], [want])
                        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"city": "Rome"}])

    def test_auto_and_none_force_nothing(self):
        for key in ("glm5", "deepseek4", "mimo2"):
            spec = families.get(key)
            self.assertIsNone(spec.forced_call("auto", TOOLS))
            self.assertIsNone(spec.forced_call(None, TOOLS))

    def test_one_tool_required_names_it(self):
        for key in ("glm5", "deepseek4", "mimo2"):
            spec = families.get(key)
            self.assertEqual(spec.forced_call("required", [TOOLS[0]]), spec.forced_call("required", [TOOLS[0]]))
            self.assertIn("get_weather", spec.forced_call("required", [TOOLS[0]]))


if __name__ == "__main__":
    unittest.main(verbosity=1)
