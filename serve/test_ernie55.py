#!/usr/bin/env python3
"""serve/test_ernie55.py - ERNIE-5.5-Flash's template, parser and tokenizer (family "ernie55") against the release's own
encoder, decoder and tokenizer.

    ~/AI/Strata/.venv/bin/python serve/test_ernie55.py      the template, the parser's cases, the tokenizer's ids
    ~/AI/vLLM/.venv/bin/python -I serve/test_ernie55.py     ... also the parser against decoding_ernie.py (which
                                                                     needs jsonschema) and the tokenizer against HF tokenizers

The references are the release's encoding_ernie.py, decoding_ernie.py and tokenizer.json, read from the model's folder
($ERNIE55_MODEL, by default ~/AI/Models/HF/ERNIE-5.5-Flash-Preview-FP8; they are not copied into the repo); a test
that needs one is skipped without it.  No GPU, no network, no server.

  1. The template: conversations rendered by serve/family_templates/ernie55.jinja through the server's own renderer
     (FamilyChatTemplate, Strata's request kwargs mapped by Ernie55.thinking_kwargs) and by render_chat() are the same
     string - hand cases and random conversations, with and without the generation prompt, thinking on and off, every
     effort, tools or none.
  2. The parser: what the encoder writes for an assistant turn, fed whole, a character at a time, in random pieces and
     cut in two at every position, gives what parse_completion gives (reasoning, answer, calls with their typed arguments).
  3. The tokenizer: the exported directory (tools/strata_tokenizer_families.py extract_hf) gives HF's ids for rendered
     prompts and for the protocol's markers; the family's entry (eos, vocabulary, effort mapping).
  4. The server's request path (ServerPath): OpenAI and Anthropic request bodies through frontend.openai_to_messages /
     anthropic_to_messages with the family's Dialect, then the template, give the encoder's string - an agent's loop with
     reasoning on every assistant turn and tool results, a turn of reasoning alone, a late system message, a past turn's
     reasoning under vLLM's / OpenRouter's `reasoning` (read as its reasoning_content), Anthropic thinking blocks.
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import importlib.util
import json
import os
import random
import re
import sys
import unittest
from pathlib import Path

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X))
sys.path.insert(0, str(X / "tools"))

from serve import families  # noqa: E402
from serve import family_parsers as FP  # noqa: E402

MODEL = Path(os.environ.get("ERNIE55_MODEL", _os.path.join(_MODELS, 'HF/ERNIE-5.5-Flash-Preview-FP8')))
TOKDIR = Path(os.environ.get("STRATA_X_DATA", _DATA)) / "tokenizers" / "ernie55"


def _load(name: str, src: str, fix_relative: bool = False):
    path = MODEL / src
    if not path.is_file():
        return None
    code = path.read_text(encoding="utf-8")
    if fix_relative:   # decoding_ernie.py imports its sibling relatively: the same module, imported by name
        code = code.replace("from .encoding_ernie import", "from ernie55_release_encoding import")
    module = type(sys)(name)
    module.__file__ = str(path)
    sys.modules[name] = module   # (its dataclasses look themselves up there)
    exec(compile(code, str(path), "exec"), module.__dict__)
    return module


enc = _load("ernie55_release_encoding", "encoding_ernie.py")
try:
    import jsonschema  # noqa: F401
    dec = _load("ernie55_release_decoding", "decoding_ernie.py", fix_relative=True) if enc else None
except ImportError:
    dec = None
try:
    from tokenizers import Tokenizer as HfTokenizer
    hf = HfTokenizer.from_file(str(MODEL / "tokenizer.json")) if (MODEL / "tokenizer.json").is_file() else None
except ImportError:
    hf = None

needs_encoder = unittest.skipIf(enc is None, f"the release's encoding_ernie.py is not in {MODEL} (set ERNIE55_MODEL)")
needs_decoder = unittest.skipIf(dec is None, "decoding_ernie.py needs the release's folder and jsonschema (the vLLM venv)")
needs_hf = unittest.skipIf(hf is None, "HF tokenizers and the release's tokenizer.json (the vLLM venv)")
needs_tokdir = unittest.skipIf(not (TOKDIR / "vocab.json").is_file(), f"no exported tokenizer in {TOKDIR}")

SPEC = families.get("ernie55")

TOOLS = [
    {"name": "get_weather", "description": "The weather in a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                                     "unit": {"type": "string", "enum": ["c", "f"]}},
                    "required": ["city"]}},
    {"name": "add", "description": "Add two numbers",
     "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"},
                                                     "tags": {"type": "array", "items": {"type": "string"}}},
                    "required": ["a", "b"]}},
    {"name": "write_file", "description": "Write a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                                     "overwrite": {"type": "boolean"}, "meta": {"type": "object"}},
                    "required": ["path", "content"]}},
]
WRAPPED = [{"type": "function", "function": t} for t in TOOLS]


def call(name, args):
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def U(text):
    return {"role": "user", "content": text}


TURNS = {
    "text": {"reasoning_content": "Simple arithmetic: 17 * 23 = 391.", "content": "391"},
    "multiline": {"reasoning_content": "Line one.\nLine two with `code` and \"quotes\".\n\n- item",
                  "content": "Answer:\n\n```python\nprint('hi')\n```\nDone."},
    "one_call": {"reasoning_content": "Need the weather.", "content": "",
                 "tool_calls": [call("get_weather", {"city": "Paris", "days": 3})]},
    "typed_calls": {"reasoning_content": "write and add", "content": "Doing both.",
                    "tool_calls": [call("write_file", {"path": "a.txt", "content": "line 1\nline 2 <b>", "overwrite": True,
                                                       "meta": {"k": [1, 2]}}),
                                   call("add", {"a": -7, "b": 0.5, "tags": ["x", "y z"]})]},
    "numeric_string": {"reasoning_content": "r", "content": "check",
                       "tool_calls": [call("get_weather", {"city": "123", "unit": "c"})]},
    "unicode": {"reasoning_content": "思考中…", "content": "你好 \U0001f600 café"},
    "empty_reasoning": {"reasoning_content": "", "content": "Plain."},
    "three_calls": {"reasoning_content": "many", "content": "Three.",
                    "tool_calls": [call("get_weather", {"city": "Paris"}), call("get_weather", {"city": "Rome"}),
                                   call("add", {"a": 2, "b": 3})]},
}


def strata_kwargs(thinking: bool, effort: str | None) -> dict:
    kw = {} if thinking else {"enable_thinking": False}
    if effort is not None:
        kw["reasoning_effort"] = effort
    return kw


def release_kwargs(thinking: bool, effort: str | None) -> dict:
    kw = {} if thinking else {"enable_thinking": False}
    if thinking and effort is not None:
        kw["reasoning_effort"] = SPEC.EFFORTS[effort]
    return kw


# ------------------------------------------------------------------------------------------------ the template
@needs_encoder
class TemplateMatchesEncoder(unittest.TestCase):
    def setUp(self):
        self.tpl = SPEC.load_template()

    def same(self, messages, tools=None, gen=True, thinking=True, effort=None):
        want = enc.encode_messages(messages, WRAPPED if tools else None, add_generation_prompt=gen,
                                   **release_kwargs(thinking, effort))
        got = self.tpl.render(messages, tools=TOOLS if tools else None, add_generation_prompt=gen,
                              **strata_kwargs(thinking, effort))
        self.assertEqual(got, want)

    def test_hand_cases(self):
        tc = [call("get_weather", {"days": 3, "city": "Paris"})]
        cases = [
            [U("What is 17 * 23?")],
            [{"role": "system", "content": "Be brief."}, U("Hi")],
            [U("Hi"), {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"}, U("Weather in Paris?")],
            [U("Weather?"), {"role": "assistant", "content": "", "tool_calls": tc}, {"role": "tool", "content": "sunny"},
             {"role": "tool", "content": "warm"}],
            [U("Weather?"), {"role": "assistant", "content": "", "tool_calls": tc}, {"role": "tool", "content": "sunny"},
             {"role": "assistant", "content": "It is sunny.", "reasoning_content": "from the tool"}, U("Thanks")],
            [{"role": "user", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]},
             {"role": "assistant", "content": "<ernie:think>inline</ernie:think>The answer."},
             {"role": "system", "content": "late system"}, U("more")],
            [U("a\n\n"), {"role": "assistant", "content": "\nb\n"}, U("\n")],
            [{"role": "system", "content": [{"type": "text", "text": "sys as parts"}]}, U("q")],
            [U("x"), {"role": "assistant", "content": "y", "reasoning_content": "   "}, U("z")],
            [U("中文问题？"), {"role": "assistant", "content": "回答。", "reasoning_content": "思考"}, U("再问")],
        ]
        for i, msgs in enumerate(cases):
            for tools in (False, True):
                for thinking in (True, False):
                    for gen in (True, False):
                        for effort in (None, "low", "medium", "high", "xhigh", "max"):
                            if not thinking and effort is not None:
                                continue
                            with self.subTest(case=i, tools=tools, thinking=thinking, gen=gen, effort=effort):
                                self.same(msgs, tools, gen, thinking, effort)

    def test_random_conversations(self):
        rng = random.Random(55)
        words = ["alpha", "beta", "the sea", "17 * 23", "naïve café", "中文", "x\ny", "  spaced  ", "<b>tag</b>", "1,234.5"]
        for k in range(300):
            msgs = []
            if rng.random() < 0.4:
                msgs.append({"role": "system", "content": " ".join(rng.choices(words, k=3))})
            for _ in range(rng.randint(1, 4)):
                msgs.append(U(" ".join(rng.choices(words, k=rng.randint(1, 5)))))
                r = rng.random()
                if r < 0.35:
                    m = {"role": "assistant", "content": " ".join(rng.choices(words, k=2))}
                    if rng.random() < 0.6:
                        m["reasoning_content"] = " ".join(rng.choices(words, k=3))
                    msgs.append(m)
                elif r < 0.6:
                    calls = [call(rng.choice(["get_weather", "add"]), {"city": rng.choice(words), "days": rng.randint(0, 9)})
                             for _ in range(rng.randint(1, 3))]
                    msgs.append({"role": "assistant", "content": rng.choice(["", "Calling."]), "tool_calls": calls,
                                 "reasoning_content": rng.choice(["", "need data"])})
                    for _ in calls:
                        msgs.append({"role": "tool", "content": rng.choice(words)})
            thinking = rng.random() < 0.7
            effort = rng.choice([None, "low", "medium", "high", "xhigh"]) if thinking else None
            with self.subTest(k=k):
                self.same(msgs, tools=rng.random() < 0.5, gen=rng.random() < 0.8, thinking=thinking, effort=effort)


# ------------------------------------------------------------------------------------------------ the parser
def events(text, thinking, pieces, stream=True):
    p = FP.make_parser("ernie55", thinking, TOOLS, stream_tools=stream)
    out = []
    for piece in pieces:
        out += p.feed(piece)
    return out + p.finish()


def summary(evs):
    calls = [e.call for e in evs if e.kind == "tool_call"]
    return ("".join(e.text for e in evs if e.kind == "reasoning"), "".join(e.text for e in evs if e.kind == "content"),
            [(c.name, c.arguments) for c in calls])


def chunkings(text):
    yield "whole", [text]
    yield "char", list(text)
    for seed in (1, 2, 3):
        rng, out, i = random.Random(seed), [], 0
        while i < len(text):
            n = rng.randint(1, 9)
            out.append(text[i:i + n])
            i += n
        yield f"random{seed}", out
    for cut in range(1, len(text)):
        yield f"cut{cut}", [text[:cut], text[cut:]]


def completion_of(turn, thinking):
    """the text the model writes for `turn` after the generation prompt (without the end token the engine stops on), as
    the encoder renders the turn"""
    kw = {} if thinking else {"enable_thinking": False}
    prefix = enc.encode_messages([U("go")], WRAPPED, add_generation_prompt=True, **kw)
    full = enc.encode_messages([U("go"), dict(turn, role="assistant")], WRAPPED, add_generation_prompt=False, **kw)
    assert full.startswith(prefix), (prefix[-40:], full[:len(prefix) + 40])
    text = full[len(prefix):]
    assert text.endswith(enc.END)
    return text[:-len(enc.END)], prefix


@needs_encoder
class ServerPath(unittest.TestCase):
    """The server's path for a request (serve/server.py: the family's Dialect, frontend's conversion, the renderer's turn
    filter) against the encoder on the conversation as the model should read it."""

    def setUp(self):
        from serve import frontend
        self.fe = frontend
        self.tpl = SPEC.load_template()
        self.dialect = frontend.Dialect(**SPEC.dialect) if SPEC.dialect else frontend.DEFAULT_DIALECT

    @staticmethod
    def as_release(msgs):
        """The messages as the model should read them: a client's `reasoning` is the turn's reasoning_content."""
        out = []
        for m in msgs:
            m = dict(m)
            if "reasoning" in m and "reasoning_content" not in m:
                m["reasoning_content"] = m.pop("reasoning")
            out.append(m)
        return out

    def openai(self, msgs, tools=None):
        req = {"model": "ernie-5.5-flash", "messages": msgs}
        if tools:
            req["tools"] = WRAPPED
        m2, t2, _ = self.fe.openai_to_messages(req, self.dialect)
        return self.tpl.render(m2, tools=t2, add_generation_prompt=True)

    def test_openai_conversations(self):
        rc = [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]
        rc2 = [{"id": "call_2", "type": "function", "function": {"name": "add", "arguments": json.dumps({"a": 1, "b": 2})}}]
        sysm = {"role": "system", "content": "You are an agent."}
        cases = {
            "an agent's loop, reasoning on every turn": [
                sysm, U("Weather in Paris, then add 1 and 2."),
                {"role": "assistant", "content": "", "reasoning_content": "weather first", "tool_calls": rc},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                {"role": "assistant", "content": "Sunny. Now the sum.", "reasoning_content": "then add", "tool_calls": rc2},
                {"role": "tool", "tool_call_id": "call_2", "content": "3"},
                {"role": "assistant", "content": "Sunny, and 3.", "reasoning_content": "done"}, U("Thanks.")],
            "a turn of reasoning alone": [sysm, U("Go."), {"role": "assistant", "content": "", "reasoning_content": "ran out"},
                                          U("Continue.")],
            "a late system message": [sysm, U("Hi."), {"role": "assistant", "content": "Hello.", "reasoning_content": "greet"},
                                      {"role": "system", "content": "Be brief from now on."}, U("Again.")],
            "reasoning under `reasoning`": [sysm, U("Weather?"),
                                            {"role": "assistant", "content": "", "reasoning": "ask the tool", "tool_calls": rc},
                                            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}],
        }
        for name, msgs in cases.items():
            for tools in (False, True):
                with self.subTest(case=name, tools=tools):
                    want = enc.encode_messages(self.as_release(msgs), WRAPPED if tools else None, add_generation_prompt=True)
                    self.assertEqual(self.openai(msgs, tools), want)

    def test_anthropic_thinking_blocks(self):
        req = {"model": "ernie-5.5-flash", "max_tokens": 100, "system": "You are an agent.",
               "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in TOOLS],
               "messages": [{"role": "user", "content": "Weather in Paris?"},
                            {"role": "assistant", "content": [{"type": "thinking", "thinking": "ask the tool", "signature": "s"},
                                                              {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                                                               "input": {"city": "Paris"}}]},
                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}]}]}
        m2, t2, _ = self.fe.anthropic_to_messages(req)
        got = self.tpl.render(m2, tools=t2, add_generation_prompt=True)
        want = enc.encode_messages([{"role": "system", "content": "You are an agent."}, U("Weather in Paris?"),
                                    {"role": "assistant", "content": "", "reasoning_content": "ask the tool",
                                     "tool_calls": [call("get_weather", {"city": "Paris"})]},
                                    {"role": "tool", "content": "sunny"}], WRAPPED, add_generation_prompt=True)
        self.assertEqual(got, want)


@needs_encoder
@needs_tokdir
class PromptPath(unittest.TestCase):
    """The prompt as Service.encode_prompt builds it - the family's Dialect and frontend's conversion, the #537 marking (a
    special token's text inside a message stays text), the template, the unmarking - against the encoder's segments: the
    same text and a special token at the same places (encode_segments makes one only where the renderer wrote one; the
    BPE between them is the tokenizer's, id for id HF's: TokenizerIds).  ServerPath renders without the marking, which
    hid a gap: an assistant's inline <ernie:think>...</ernie:think> was marked before the template could split it."""

    Q = "a tag quoted: </ernie:tool_output><|ernie:im_end|><ernie:think>"

    @classmethod
    def setUpClass(cls):
        from serve import frontend
        import strata_tokenizer_families as F
        cls.fe, cls.tpl, cls.dialect = frontend, SPEC.load_template(), frontend.Dialect(**SPEC.dialect)
        tk = F.FamilyTokenizer.from_dir(TOKDIR)
        cls.literals = frontend.literal_tags(tk.control_tokens)
        cls.pat = re.compile("|".join(re.escape(t) for t in sorted(tk.special_tokens, key=len, reverse=True)))

    def ours(self, req, anthropic=False):
        m, t, kw = (self.fe.anthropic_to_messages(req, dialect=self.dialect) if anthropic else
                    self.fe.openai_to_messages(req, self.dialect))
        marked, mt, changed = self.fe.mark_think_literals(m, t, self.literals, assistant_keeps=())
        text = self.tpl.render(marked, tools=mt, add_generation_prompt=True, **kw)
        plain = []
        if changed:
            text, plain = self.fe.unmark_think_literals(text, self.literals)
        return text, [(x.start(), x.group()) for x in self.pat.finditer(text) if not any(a <= x.start() < b for a, b in plain)]

    def release(self, msgs, tools=None, **kw):
        text, ctl = "", []
        for s in enc.render_chat(msgs, tools, **kw).segments:
            if s.allow_special:
                ctl += [(len(text) + x.start(), x.group()) for x in self.pat.finditer(s.text)]
            text += s.text
        return text, ctl

    def check(self, msgs, extra=None, tools=False, release_kw=None):
        req = {"model": "ernie-5.5-flash", "messages": msgs, **({"tools": WRAPPED} if tools else {}), **(extra or {})}
        self.assertEqual(self.ours(req), self.release(msgs, WRAPPED if tools else None, **(release_kw or {})))

    def test_cases(self):
        say = lambda content, **kw: {"role": "assistant", "content": content, **kw}   # noqa: E731
        rc = [call("write_file", {"path": "a.txt", "content": "x"})]
        cases = {
            "an agent's loop": ([U("Write a.txt"), say("", reasoning_content="write it", tool_calls=rc),
                                 {"role": "tool", "tool_call_id": "c1", "content": "ok"}, U("Thanks.")], {}, True, {}),
            "tags quoted in user, tool and argument text": (
                [U(self.Q), say("", reasoning_content="r", tool_calls=[call("write_file", {"path": self.Q, "content": "y"})]),
                 {"role": "tool", "content": self.Q}], {}, True, {}),
            "inline reasoning": ([U("hi"), say("<ernie:think>let me think</ernie:think>Hello!"), U("again")], {}, False, {}),
            "inline reasoning, no open tag": ([U("hi"), say("plan first</ernie:think>Hello!"), U("again")], {}, False, {}),
            "inline reasoning, two close tags": ([U("hi"), say("<ernie:think>R</ernie:think>B</ernie:think>C"), U("x")],
                                                 {}, False, {}),
            "inline tags beside an empty reasoning_content": (
                [U("hi"), say("<ernie:think>R</ernie:think>A", reasoning_content=""), U("x")], {}, False, {}),
            "inline reasoning, thinking off": ([U("hi"), say("<ernie:think>R</ernie:think>A"), U("x")],
                                               {"chat_template_kwargs": {"enable_thinking": False}}, False,
                                               {"enable_thinking": False}),
            "thoughts": ([U("hi"), say("Hello!", thoughts="greet back"), U("again")], {}, False, {}),
            "empty thoughts over reasoning_content": ([U("hi"), say("Hello!", thoughts="", reasoning_content="R"), U("x")],
                                                      {}, False, {}),
            "argument keys that differ in case": (
                [U("go"), say("", reasoning_content="r", tool_calls=[call("write_file", {"content": "y", "Path": "x",
                                                                                         "aB": 2, "a_b": 1})]),
                 {"role": "tool", "content": "ok"}], {}, True, {}),
            "parallel_tool_calls false": ([U("Weather?")], {"parallel_tool_calls": False}, True,
                                          {"options": {"parallel_tool_calls": False}}),
            "parallel_tool_calls true": ([U("Weather?")], {"parallel_tool_calls": True}, True, {}),
        }
        for name, (msgs, extra, tools, release_kw) in cases.items():
            with self.subTest(case=name):
                self.check(msgs, extra, tools, release_kw)

    def test_anthropic(self):
        """disable_parallel_tool_use as parallel_tool_calls false, and inline reasoning in a turn without a thinking block."""
        req = {"model": "ernie-5.5-flash", "max_tokens": 100,
               "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in TOOLS],
               "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
               "messages": [U("hi"), {"role": "assistant", "content": "<ernie:think>R</ernie:think>Hello!"},
                            {"role": "assistant", "content": [{"type": "text", "text": "plan</ernie:think>Done."}]}, U("x")]}
        want = self.release([U("hi"), {"role": "assistant", "content": "<ernie:think>R</ernie:think>Hello!"},
                             {"role": "assistant", "content": "plan</ernie:think>Done."}, U("x")], WRAPPED,
                            options={"parallel_tool_calls": False})
        self.assertEqual(self.ours(req, anthropic=True), want)

    def test_responses(self):
        from serve import responses
        req = {"parallel_tool_calls": False, "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "<ernie:think>R</ernie:think>A"}]}]}
        msgs = responses.input_messages(req, self.dialect)
        self.assertEqual((msgs[-1]["reasoning_content"], msgs[-1]["content"]), ("R", "A"))
        self.assertIs(responses.template_kwargs(req, {}, self.dialect).get("parallel_tool_calls"), False)
        self.assertNotIn("parallel_tool_calls", responses.template_kwargs({}, {}, self.dialect))


class OneCall(unittest.TestCase):
    def test_first_call_only(self):
        from serve.frontend import Event, FirstCallOnly, ToolCall, one_call_only
        a, b = ToolCall("write_file", {"path": "a"}), ToolCall("write_file", {"path": "b"})
        evs = [Event("reasoning", "r"), Event("tool_start", call=a), Event("tool_args", "{", call=a), Event("tool_call", call=a),
               Event("tool_start", call=b), Event("tool_args", "{", call=b), Event("tool_call", call=b), Event("content", "x")]
        f = FirstCallOnly()
        got = f(evs[:5]) + f(evs[5:])                    # the second call's start in the first chunk, the rest in the next
        self.assertEqual([(e.kind, e.call.arguments if e.call else None) for e in got],
                         [("reasoning", None), ("tool_start", {"path": "a"}), ("tool_args", {"path": "a"}),
                          ("tool_call", {"path": "a"}), ("content", None)])
        self.assertEqual(f.dropped, 1)
        for req, want in (({"parallel_tool_calls": False}, True), ({"parallel_tool_calls": True}, False), ({}, False),
                          ({"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}, True),
                          ({"tool_choice": "auto"}, False)):
            self.assertIs(one_call_only(req), want, req)

    def test_other_families_unchanged(self):
        """The new switches are ERNIE's alone: the default dialect reads no `thoughts`, splits nothing and passes no option."""
        from serve import frontend
        req = {"messages": [U("hi"), {"role": "assistant", "content": "a</ernie:think>b", "thoughts": "t"}],
               "tools": WRAPPED, "parallel_tool_calls": False}
        m, _, kw = frontend.openai_to_messages(req)
        self.assertEqual(m[-1], {"role": "assistant", "content": "a</ernie:think>b"})
        self.assertNotIn("parallel_tool_calls", kw)


class Sampling(unittest.TestCase):
    """The keys an ERNIE request reaches the engine with (StrataEngine.sampling_keys with the family's full_vocab): the
    release's T 1.0 / top_p 0.95 with no top-k cut - never the engine's default top_k 20."""

    def keys(self, **s):
        from serve.server import StrataEngine
        return StrataEngine.sampling_keys(s, full_vocab=SPEC.full_vocab).split()

    def test_no_cut(self):
        self.assertTrue(SPEC.full_vocab)
        self.assertEqual(self.keys(temperature=1.0, top_p=0.95), ["temperature=1.0", "top_p=0.95", "top_k=0"])   # none named
        self.assertIn("top_k=0", self.keys(temperature=1.0, top_p=0.95, top_k=-1))   # a client's "off"
        self.assertIn("top_k=0", self.keys(temperature=1.0, top_p=0.95, top_k=0))
        self.assertIn("top_k=64", self.keys(temperature=1.0, top_p=0.95, top_k=64))  # the run config's (the widest list)
        self.assertIn("top_k=20", self.keys(temperature=1.0, top_p=0.95, top_k=20))  # a client's own cut wins


class ParserCheck(unittest.TestCase):
    def check_text(self, text, thinking, want):
        for name, pieces in chunkings(text):
            for stream in (True, False):
                with self.subTest(chunking=name, stream=stream, text=text[:60]):
                    evs = events(text, thinking, pieces, stream)
                    self.assertEqual(summary(evs), want)
                    if stream:
                        calls = [e.call for e in evs if e.kind == "tool_call"]
                        self.assertEqual([e.call.id for e in evs if e.kind == "tool_start"], [c.id for c in calls])
                        for c in calls:
                            joined = "".join(e.text for e in evs if e.kind == "tool_args" and e.call.id == c.id)
                            self.assertEqual(json.loads(joined), c.arguments)


@needs_decoder
class ParserMatchesDecoder(ParserCheck):
    """The encoder's assistant turn -> decoding_ernie.parse_completion, and -> the family parser."""

    def test_turns(self):
        for name, turn in TURNS.items():
            for thinking in (True, False):
                with self.subTest(turn=name, thinking=thinking):
                    text, _ = completion_of(turn, thinking)
                    kw = {} if thinking else {"enable_thinking": False}
                    rendered = enc.render_chat([U("go")], WRAPPED, add_generation_prompt=True, **kw)
                    ref = dec.parse_completion(text + enc.END, rendered=rendered, tools=WRAPPED)
                    want = (ref["reasoning_content"], ref["content"],
                            [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in ref["tool_calls"]])
                    self.check_text(text, thinking, want)


@needs_encoder
class ParserCases(ParserCheck):
    def test_turns_read_back(self):
        """What the encoder writes for a turn parses back into the turn (reasoning dropped when thinking is off)."""
        for name, turn in TURNS.items():
            for thinking in (True, False):
                with self.subTest(turn=name, thinking=thinking):
                    text, _ = completion_of(turn, thinking)
                    reasoning = turn.get("reasoning_content", "") if thinking and turn.get("reasoning_content", "").strip() else ""
                    want = (reasoning, turn.get("content", ""),
                            [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in turn.get("tool_calls", [])])
                    self.check_text(text, thinking, want)

    def test_call_without_tools_is_text(self):
        text = 'r</ernie:think>x<ernie:tool_call>\n<invoke name="f"><parameter name="a">1</parameter></invoke>\n</ernie:tool_call>'
        p = FP.make_parser("ernie55", True, None, stream_tools=False)
        evs = p.feed(text) + p.finish()
        self.assertEqual([e for e in evs if e.kind == "tool_call"], [])

    def test_forced_call(self):
        self.assertEqual(FP.SYNTAXES["ernie55"].forced("get_weather"), '<ernie:tool_call>\n<invoke name="get_weather">')


# ------------------------------------------------------------------------------------------------ the tokenizer, the entry
class Entry(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(SPEC.arch, "ernie55")
        self.assertEqual(SPEC.n_vocab, 201216)
        self.assertEqual(set(SPEC.eos_ids), {8, 2})
        self.assertTrue(SPEC.thinking_default)
        self.assertEqual(SPEC.default_sampling, {"temperature": 1.0, "top_p": 0.95})

    def test_effort_mapping(self):
        self.assertEqual(SPEC.thinking_kwargs({"enable_thinking": False}), {"enable_thinking": False})
        self.assertEqual(SPEC.thinking_kwargs({}), {})
        for strata, release in (("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "high"), ("max", "high")):
            self.assertEqual(SPEC.thinking_kwargs({"reasoning_effort": strata}), {"reasoning_effort": release})
        with self.assertRaises(ValueError):
            SPEC.thinking_kwargs({"reasoning_effort": "huge"})


@needs_tokdir
class TokenizerIds(unittest.TestCase):
    def setUp(self):
        import strata_tokenizer_families as F
        self.tk = F.FamilyTokenizer.from_dir(TOKDIR)

    def test_markers(self):
        for text, i in (("<|ERNIE_MASK|>", 2), ("<|ernie:im_start|>", 7), ("<|ernie:im_end|>", 8), ("<ernie:think>", 19),
                        ("</ernie:think>", 20)):
            self.assertEqual(self.tk.encode(text, parse_special=True), [i])
        self.assertEqual(sorted(self.tk.eog_ids), [2, 8])

    @needs_hf
    @needs_encoder
    def test_prompts_as_hf(self):
        tpl = SPEC.load_template()
        for turn in TURNS.values():
            for thinking in (True, False):
                prompt = tpl.render([U("go"), dict(turn, role="assistant"), U("again")], tools=TOOLS,
                                    **strata_kwargs(thinking, None))
                self.assertEqual(self.tk.encode(prompt, parse_special=True), hf.encode(prompt, add_special_tokens=False).ids)


if __name__ == "__main__":
    unittest.main(verbosity=1)
