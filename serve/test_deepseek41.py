#!/usr/bin/env python3
"""serve/test_deepseek41.py - DeepSeek-V4.1's template and parser (family "deepseek41") against the release's encoder.

    ~/AI/Strata/.venv/bin/python serve/test_deepseek41.py          (jinja2, and regex for the tokenizer part)
    python -m pytest -q serve/test_deepseek41.py

The reference is the release's encoding/encoding.py, imported from the model's folder ($DSV41_ENCODING, by default
~/AI/Models/dsv41-1017/reference/encoding/encoding.py); the tests that need it are skipped without it.  No GPU,
no network, no server.

  1. Templates: conversations rendered by serve/family_templates/deepseek41.jinja through the server's own renderer
     (FamilyChatTemplate, with Strata's request kwargs) and by encode_messages() are the same string, byte for byte.  The
     server's input is mapped onto the encoder's as the template's header says: the tools go on the first system
     message (or on an empty one put first), thinking and effort as Deepseek41.thinking_kwargs maps them, and a completed
     empty assistant turn is left out on both sides (the server's renderer drops it before any template, #843).
     add_generation_prompt false is the encoder's text without the assistant prompt it writes after a final user turn;
     true after a final assistant turn is the encoder's text and a new assistant prompt.
  2. The request path: OpenAI request bodies through frontend.openai_to_messages, then the template (and the tokens,
     when the tokenizer directory is there).
  3. The parser: what the encoder writes for an assistant turn, fed whole, a character at a time, in random pieces and
     cut in two at every position, gives what parse_message_from_completion_text gives; what the template writes for a
     past turn reads back as that turn.
  4. The rest: the effort mapping, the forced call, V4's unspaced DSML, the /props caps.
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import copy
import importlib.util
import json
import os
import random
import sys
import unittest
from pathlib import Path

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X))
sys.path.insert(0, str(X / "tools"))

from serve import families  # noqa: E402
from serve import family_parsers as FP  # noqa: E402
from serve.family_template import wrap_tools  # noqa: E402
from serve.frontend import (Dialect, TemplateRequestError, _has_image, _text_of, anthropic_to_messages,  # noqa: E402
                            openai_to_messages)

ENCODING = Path(os.environ.get("DSV41_ENCODING", _os.path.join(_MODELS, 'dsv41-1017/reference/encoding/encoding.py')))


def load_encoder():
    """The release's encoder, imported from its own path (it is not copied into the repo)."""
    if not ENCODING.is_file():
        return None
    spec = importlib.util.spec_from_file_location("dsv41_reference_encoding", ENCODING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


enc = load_encoder()
needs_encoder = unittest.skipIf(enc is None, f"the release's encoder is not at {ENCODING} (set DSV41_ENCODING)")

SPEC = families.get("deepseek41")
TPL = SPEC.load_template()
DIALECT = Dialect(**SPEC.dialect)              # what the server hands the converters for this family (Service.dialect)
BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
EFFORT = "<｜System｜>Reasoning Effort: {} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
D = "｜DSML｜"

# ------------------------------------------------------------------------------------------------ conversations
# Tools as the server hands them to the template (flat: FamilyChatTemplate wraps them)
WEATHER = {"name": "get_weather", "description": "Get the current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                           "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                          "required": ["city"]}}
ADD = {"name": "add", "description": "Add two integers.",
       "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"},
                                                       "tags": {"type": "array", "items": {"type": "string"}}},
                      "required": ["a", "b"]}}
WRITE = {"name": "write_file", "description": "Write a file.",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                                         "append": {"type": "boolean"}, "meta": {"type": "object"}},
                        "required": ["path", "content"]}}
LOOKUP = {"name": "lookup", "description": "Look up a value",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}}


def call(name, args, cid=None, **extra):
    """A past call as frontend.py builds it ({"function": {"name", "arguments": a mapping}}); `cid` adds a client's id."""
    c = {"function": {"name": name, "arguments": args}, **extra}
    if cid:
        c["id"] = cid
    return c


U = lambda t, **kw: {"role": "user", "content": t, **kw}                                    # noqa: E731
A = lambda t, **kw: {"role": "assistant", "content": t, **kw}                               # noqa: E731
S = lambda t, **kw: {"role": "system", "content": t, **kw}                                  # noqa: E731
T = lambda t, cid=None: {"role": "tool", "content": t, **({"tool_call_id": cid} if cid else {})}   # noqa: E731
LR = lambda t: {"role": "latest_reminder", "content": t}                                    # noqa: E731

ARGS = {"path": "a/b.txt", "content": "line1\nline2 \"q\" <tag> 你好 \U0001f600\n", "append": False,
        "meta": {"n": 1, "xs": [1, 2.5, None, "s"], "deep": {"t": True}}}

CASES = {
    # the reference test_encoding.py's text-only conversations
    "question": {"messages": [U("question")]},
    "leading_system": {"messages": [S("You are a helpful assistant."), U("hello")]},
    "mid_system": {"messages": [S("sys"), U("q1"), A("a1", reasoning_content="r1"), S("mid sys")]},
    "tool_instructions": {"messages": [S("system"), U("question")], "tools": [LOOKUP]},
    "drop_thinking_without_tools": {"messages": [
        U("q1"), A("a1", reasoning_content="r1"), U("q2"),
        A(None, reasoning_content="r2", tool_calls=[call("lookup", "{}", "c1", type="function")]), T("value", "c1")]},
    "drop_thinking_next_turn": {"messages": [
        U("q1"), A("a1", reasoning_content="r1"), U("q2"),
        A(None, reasoning_content="r2", tool_calls=[call("lookup", "{}", "c1", type="function")]), T("value", "c1"),
        U("q3")]},
    "task_query": {"messages": [U("classify me", task="query")]},
    # plain chat
    "single_user": {"messages": [U("Hello")]},
    "empty_system": {"messages": [S(""), U("Hi")]},
    "multi_turn": {"messages": [S("Be brief."), U("Hi"), A("Hello! How can I help?"), U("Tell me a joke."),
                                A("Why did the chicken cross the road?"), U("Why?")]},
    "multi_turn_reasoning": {"messages": [S("Be brief."), U("Hi"), A("Hello!", reasoning_content="The user greets me."),
                                          U("2+2?"), A("4", reasoning_content="Simple arithmetic: 2+2=4."), U("And 3+3?")]},
    "consecutive_users": {"messages": [U("part one"), U("part two"), A("ok"), U("again"), U("and again")]},
    "content_parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Part one. "},
                                                                {"type": "text", "text": "Part two."}]}]},
    "image_placeholder": {"messages": [{"role": "user", "content": [{"type": "text", "text": "inspect"},
                                                                    {"type": "image", "source": "/unused/image.png"}]}]},
    "unicode": {"messages": [S("你是助手。"), U("こんにちは \U0001f600 مرحبا café"), A("你好"), U("再见")]},
    "whitespace": {"messages": [U("  leading and trailing  \n\n"), A("  spaced answer \n"), U("\tTabbed\n")]},
    "special_text": {"messages": [U("<｜User｜> </think> <think> <｜DSML｜ calls> <｜end▁of▁sentence｜> <tool_result>"),
                                  A("ok"), U("again")]},
    "assistant_first": {"messages": [A("I start."), U("ok")]},
    "system_then_assistant": {"messages": [S("sys"), A("hello", reasoning_content="r"), U("q")]},
    "empty_assistant_mid": {"messages": [U("Hi"), A(""), U("Hello?"), A("", reasoning_content="r"), U("Bye")]},
    # the conversation ends on an assistant turn
    "assistant_last": {"messages": [U("Hi"), A("Hello.", reasoning_content="greet")]},
    "empty_assistant_last": {"messages": [U("Hi"), A("")]},
    "assistant_last_open": {"messages": [U("Hi"), A("Hel", wo_eos=True)]},
    # tools
    "tools_defs": {"messages": [S("You can use tools."), U("What's the weather in Paris?")], "tools": [WEATHER, ADD]},
    "tools_no_system": {"messages": [U("Hi")], "tools": [ADD]},
    "tools_empty_system": {"messages": [S(""), U("Hi")], "tools": [ADD]},
    "tool_roundtrip": {"messages": [
        S("You can use tools."), U("What's the weather in Paris?"),
        A("Let me check.", reasoning_content="I should call get_weather.",
          tool_calls=[call("get_weather", {"city": "Paris", "unit": "celsius"})]),
        T("{\"temp\": 18, \"sky\": \"cloudy\"}")], "tools": [WEATHER, ADD]},
    "tool_roundtrip_answered": {"messages": [
        U("What's the weather in Paris?"),
        A(None, reasoning_content="Call the tool.", tool_calls=[call("get_weather", {"city": "Paris"})]),
        T("18 C, cloudy"), A("It is 18 C and cloudy in Paris.", reasoning_content="The tool said 18 C."),
        U("And in Rome?")], "tools": [WEATHER, ADD]},
    "parallel_calls": {"messages": [
        U("Weather in Paris and Rome, and 2+3?"),
        A(None, tool_calls=[call("get_weather", {"city": "Paris"}), call("get_weather", {"city": "Rome"}),
                            call("add", {"a": 2, "b": 3})]),
        T("Paris: 18 C"), T("Rome: 25 C"), T("5")], "tools": [WEATHER, ADD]},
    "multi_step_tools": {"messages": [
        U("Add 1 and 2, then add the result to 10."),
        A("First step.", reasoning_content="one", tool_calls=[call("add", {"a": 1, "b": 2})]), T("3"),
        A("Second step.", reasoning_content="two", tool_calls=[call("add", {"a": 3, "b": 10})]), T("13")],
        "tools": [ADD]},
    "args_types": {"messages": [
        U("Write it."),
        A(None, tool_calls=[call("write_file", ARGS), call("add", {"a": -7, "b": 0, "tags": ["x", "y z"]}),
                            call("add", {})]),
        T("ok"), T("-7"), T("0")], "tools": [WRITE, ADD]},
    "tool_then_user": {"messages": [
        U("Weather?"), A("", tool_calls=[call("get_weather", {"city": "Oslo"})]), T("cold"), U("Thanks. And Rome?")],
        "tools": [WEATHER]},
    "tool_first": {"messages": [T("an orphan result"), U("q")]},
    "tool_results_out_of_order": {"messages": [
        U("three things"),
        A(None, tool_calls=[call("add", {"a": 1, "b": 1}, "c1"), call("add", {"a": 2, "b": 2}, "c2"),
                            call("add", {"a": 3, "b": 3}, "c3")]),
        T("6", "c3"), T("2", "c1"), T("4", "c2"), U("sum them")], "tools": [ADD]},
    "tool_results_unknown_ids": {"messages": [
        U("go"), A(None, tool_calls=[call("add", {"a": 1, "b": 1}, "c1"), call("add", {"a": 2, "b": 2}, "c2")]),
        T("4", "c2"), T("?", "zz"), T("2", "c1")], "tools": [ADD]},
    "tool_result_parts": {"messages": [
        U("look"), A(None, tool_calls=[call("add", {"a": 1, "b": 1})]),
        {"role": "tool", "content": [{"type": "text", "text": "result"}, {"type": "image", "source": "x.png"},
                                     {"type": "text", "text": "after"}]}], "tools": [ADD]},
    # system messages after the start (a direct caller: the server's request path turns them into user turns)
    "mid_system_then_user": {"messages": [S("Start."), U("Hi"), A("Hello"), S("Hook context: Linux."), U("ls?")]},
    "mid_system_no_first_system": {"messages": [U("q1"), A("a1", reasoning_content="r1"), S("late")]},
    "mid_system_after_tools": {"messages": [
        U("go"), A(None, tool_calls=[call("add", {"a": 1, "b": 2})]), T("3"), S("Be terse from now on.")],
        "tools": [ADD]},
    "mid_system_with_its_tools": {"messages": [
        U("q"), A("a", reasoning_content="r"), S("More tools.", tools=wrap_tools([ADD]))]},
    # what a direct caller can give beyond the server
    "response_format": {"messages": [
        S("Answer in JSON.", response_format={"type": "json_schema", "json_schema": {
            "name": "answer", "schema": {"type": "object", "properties": {"x": {"type": "integer"}}}}}),
        U("q")]},
    "response_format_and_tools": {"messages": [S("", response_format={"type": "json_object"}), U("q")],
                                  "tools": [ADD]},
    "latest_reminder": {"messages": [S("sys"), U("q"), LR("Today is 2026-10-18.")]},
    "latest_reminder_answered": {"messages": [U("q"), LR("date"), A("answer", reasoning_content="r"), U("q2")]},
    "latest_reminder_after_assistant": {"messages": [U("q"), A("a"), LR("note"), U("q2")]},
    "task_action": {"messages": [U("hi", task="action")]},
    "task_title": {"messages": [U("hi"), A("hello", reasoning_content="r", task="title")]},
    "task_title_answered": {"messages": [U("hi"), A("hello", task="title"), A("Greeting", reasoning_content="x")]},
    "task_mid": {"messages": [U("search this", task="query"), A("q1 q2"), U("now answer")]},
    "task_user_joins": {"messages": [U("a"), U("b", task="query")]},
    "task_user_stays": {"messages": [U("a", task="query"), U("b")]},
    "args_as_text": {"messages": [
        U("go"),
        A(None, tool_calls=[call("lookup", '{"query":"value","limit":2}'), call("lookup", json.dumps('{"query": "x"}')),
                            call("lookup", "{not json"), call("lookup", "[1, 2]"), call("lookup", ""),
                            call("lookup", None), call("lookup", {})]),
        T("r")], "tools": [LOOKUP]},
}

# the namespaces of the reference's tests: on the tool, in the function, a name or {"name", "description"}, qualified names
NS_TOOLS = [
    {"type": "function", "namespace": {"name": "search", "description": "Search tools."}, "function": dict(LOOKUP)},
    {"type": "function", "function": dict(LOOKUP, name="fetch", namespace="web")},
    {"type": "function", "function": dict(LOOKUP, name="files::read")},
    {"type": "function", "namespace": "files", "function": dict(LOOKUP, name="write", description=None)},
    {"type": "function", "function": dict(LOOKUP, namespace=None)},
    {"type": "function", "namespace": {"name": "plain"}, "function": {"name": "nodesc", "parameters": {}}},
]
CASES["namespaces"] = {"messages": [
    S("system"), U("question"),
    A("summary", reasoning_content="r", tool_calls=[
        call("lookup", {"query": "value"}, namespace="search"),
        {"function": {"name": "fetch", "namespace": "web", "arguments": {"query": "v"}}},
        call("files::read", {"query": "q"}),
        call("lookup", {"query": "dict"}, namespace={"name": "search"}),
        call("lookup", {"query": "plain"})]),
    T("ok")], "tools": NS_TOOLS}
CASES["same_named_tools"] = {"messages": [
    S("system"), U("question"),
    A("summary", tool_calls=[call("lookup", '{"query":"value"}', type="function"),
                             call("lookup", '{"query":"value"}', type="function", namespace="search"),
                             call("lookup", '{"query":"value"}', type="function", namespace="files")])],
    "tools": [{"type": "function", "function": dict(LOOKUP)},
              {"type": "function", "namespace": {"name": "search"}, "function": dict(LOOKUP)},
              {"type": "function", "namespace": {"name": "files"}, "function": dict(LOOKUP)}]}

# Strata's request kwargs (frontend.effort_kwargs' dialect; integers only from a direct caller)
VARIANTS = {
    "chat": {},
    "think": {"enable_thinking": True},
    "low": {"reasoning_effort": "low"},
    "medium": {"reasoning_effort": "medium"},
    "xhigh": {"reasoning_effort": "xhigh"},
    "int42": {"reasoning_effort": 42},
    "off": {"enable_thinking": False, "reasoning_effort": "low"},
}


def keeps(messages):
    """The server's renderer leaves out a completed empty assistant turn (#843, frontend.ChatTemplate.render) before any
    template sees it: the encoder gets the same list.  A turn of reasoning alone is no empty turn for this family
    (FamilySpec.reasoning_turns, the audit of 2026-10-19): the encoder renders it, and so does the server."""
    return [m for i, m in enumerate(messages) if i == len(messages) - 1 or not (
        m.get("role") == "assistant" and not _text_of(m.get("content")).strip() and not _has_image(m.get("content"))
        and not m.get("tool_calls") and not m.get("reasoning_content"))]


def reference_input(messages, tools):
    """The server's (messages, tools) as the encoder takes them: the tools on the first system message, or on an empty
    one put first."""
    msgs = [dict(m) for m in keeps(messages)]
    if tools:
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["tools"] = wrap_tools(tools)
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": wrap_tools(tools)})
    return msgs


def mode_of(kwargs):
    return ("thinking" if SPEC.thinking_on(kwargs) else "chat"), SPEC.thinking_kwargs(kwargs).get("reasoning_effort")


def render(messages, tools=None, gen=True, **kwargs):
    return TPL.render(copy.deepcopy(messages), tools=copy.deepcopy(tools), add_generation_prompt=gen, **kwargs)


def expected(messages, tools=None, gen=True, **kwargs):
    """encode_messages() of the same conversation, and what add_generation_prompt changes (the module's header)."""
    mode, effort = mode_of(kwargs)
    inp = reference_input(copy.deepcopy(messages), copy.deepcopy(tools))
    ref = enc.encode_messages(copy.deepcopy(inp), thinking_mode=mode, reasoning_effort=effort)
    merged = enc.merge_tool_messages(inp)
    last = merged[-1] if merged else {}
    header = "<｜Assistant｜>" + ("<think>" if mode == "thinking" else "</think>")
    if last.get("task") is not None:
        return ref
    if last.get("role") == "assistant":
        return ref + header if gen and not last.get("wo_eos") else ref
    if not gen and last.get("role") in ("user", "system") and ref.endswith(header):
        return ref[:-len(header)]
    return ref


# ------------------------------------------------------------------------------------------------ 1. the template
@needs_encoder
class TemplateMatchesEncoder(unittest.TestCase):
    def test_every_case_every_variant(self):
        for name, case in CASES.items():
            for variant, kwargs in VARIANTS.items():
                for gen in (True, False):
                    with self.subTest(case=name, variant=variant, add_generation_prompt=gen):
                        self.assertEqual(render(case["messages"], case.get("tools"), gen, **kwargs),
                                         expected(case["messages"], case.get("tools"), gen, **kwargs))

    def test_effort_as_the_encoder_takes_it(self):
        # the template takes the encoder's own values too (a direct caller): None, the three aliases, 1 and 100
        for effort in (None, "low", "high", "max", 1, 100):
            with self.subTest(effort=effort):
                got = TPL.template.render(messages=[S("s"), U("q")], tools=None, add_generation_prompt=True,
                                          thinking=True, reasoning_effort=effort, bos_token=BOS)
                self.assertEqual(got, enc.encode_messages([S("s"), U("q")], thinking_mode="thinking",
                                                          reasoning_effort=effort))

    def test_random_conversations(self):
        """Conversations put together at random from the cases' messages (tools or not, every variant)."""
        rng = random.Random(41)
        pool = [m for c in CASES.values() for m in c["messages"] if not m.get("task") and m.get("role") != "system"]
        for n in range(300):
            msgs = [S("sys")] if rng.random() < 0.5 else []
            msgs += [copy.deepcopy(rng.choice(pool)) for _ in range(rng.randint(1, 7))]
            if rng.random() < 0.3:
                msgs.insert(rng.randint(1, len(msgs)), S("a system message later on"))
            tools = rng.choice([None, [ADD], [WEATHER, ADD, WRITE, LOOKUP]])
            variant = rng.choice(list(VARIANTS))
            gen = rng.random() < 0.8
            with self.subTest(n=n, variant=variant, gen=gen):
                self.assertEqual(render(msgs, tools, gen, **VARIANTS[variant]),
                                 expected(msgs, tools, gen, **VARIANTS[variant]))


class TemplateText(unittest.TestCase):
    """The reference test_encoding.py's literal expectations (no encoder needed)."""

    def test_effort_budgets(self):
        for kwargs, budget in (({"enable_thinking": True}, 75), ({"reasoning_effort": "low"}, 50),
                               ({"reasoning_effort": "medium"}, 75), ({"reasoning_effort": "xhigh"}, 100),
                               ({"reasoning_effort": "high"}, 75), ({"reasoning_effort": "max"}, 100),
                               ({"reasoning_effort": 1}, 1), ({"reasoning_effort": 42}, 42),
                               ({"reasoning_effort": 100}, 100)):
            with self.subTest(**kwargs):
                self.assertEqual(render([U("question")], **kwargs),
                                 BOS + EFFORT.format(budget) + "<｜User｜>question<｜Assistant｜><think>")

    def test_chat_mode_has_no_effort_or_system_token(self):
        for kwargs in ({}, {"enable_thinking": False}, {"enable_thinking": False, "reasoning_effort": "xhigh"}):
            with self.subTest(**kwargs):
                self.assertEqual(render([U("hello")], **kwargs), BOS + "<｜User｜>hello<｜Assistant｜></think>")

    def test_leading_system_message_uses_system_token(self):
        self.assertEqual(render([S("You are a helpful assistant."), U("hello")]),
                         BOS + "<｜System｜>You are a helpful assistant.<｜User｜>hello<｜Assistant｜></think>")

    def test_mid_conversation_system_message(self):
        self.assertEqual(render(CASES["mid_system"]["messages"], reasoning_effort=88),
                         BOS + EFFORT.format(88) + "sys<｜User｜>q1<｜Assistant｜></think>a1" + EOS +
                         "<｜System｜>mid sys<｜Assistant｜><think>")

    def test_tool_instructions_use_spaced_dsml_tags(self):
        prompt = render([S("system"), U("question")], [LOOKUP])
        self.assertIn(f'<{D} calls>\n<{D} invoke name="$TOOL_NAME">\n<{D} parameter name="$PARAMETER_NAME" '
                      f'string="true|false">$PARAMETER_VALUE</{D} parameter>\n...\n</{D} invoke>', prompt)
        for v4 in (f"<{D}tool_calls>", f"<{D}invoke", f"<{D}parameter"):
            self.assertNotIn(v4, prompt)
        self.assertIn("### Available Tool Schemas\n\n" + json.dumps(LOOKUP, ensure_ascii=False) +
                      "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool "
                      "calls.\n<｜User｜>question<｜Assistant｜></think>", prompt)
        self.assertTrue(prompt.startswith(BOS + "<｜System｜>system\n\n## Tools\n\n"))

    def test_tools_without_a_system_message(self):
        # the encoder reads tools from a system message: an empty one is put first
        self.assertTrue(render([U("Hi")], [ADD]).startswith(BOS + "<｜System｜>\n\n## Tools\n\n"))

    def test_assistant_call_turn(self):
        # V41_TOOL_CALL_OUTPUT of the reference's tests, after the prompt's <think>
        out = ('  reason  </think>summary\n\n'
               f'<{D} calls>\n<{D} invoke name="lookup">\n<{D} parameter name="query" string="true">value</{D} parameter>\n'
               f'<{D} parameter name="limit" string="false">2</{D} parameter>\n</{D} invoke>\n</{D} calls>' + EOS)
        msgs = [U("question"), A("summary", reasoning_content="  reason  ",
                                 tool_calls=[call("lookup", '{"query":"value","limit":2}', type="function")])]
        self.assertEqual(render(msgs, gen=False, enable_thinking=True),
                         BOS + EFFORT.format(75) + "<｜User｜>question<｜Assistant｜><think>" + out)

    def test_drop_thinking_without_tools(self):
        prompt = render(CASES["drop_thinking_without_tools"]["messages"], enable_thinking=True)
        self.assertIn("<｜User｜>q1<｜Assistant｜></think>a1" + EOS, prompt)
        self.assertNotIn("r1", prompt)
        self.assertIn("<think>r2</think>", prompt)
        self.assertTrue(prompt.endswith("<tool_result>value</tool_result><｜Assistant｜><think>"))
        nxt = render(CASES["drop_thinking_next_turn"]["messages"], enable_thinking=True)
        self.assertNotIn("r2", nxt)
        self.assertTrue(nxt.endswith("q3<｜Assistant｜><think>"))

    def test_tools_keep_every_turns_reasoning(self):
        prompt = render(CASES["tool_roundtrip_answered"]["messages"], [WEATHER], enable_thinking=True)
        self.assertIn("<｜Assistant｜><think>Call the tool.</think>", prompt)
        self.assertIn("<think>The tool said 18 C.</think>It is 18 C", prompt)

    def test_tool_results_follow_the_call_order(self):
        prompt = render(CASES["tool_results_out_of_order"]["messages"], [ADD])
        self.assertIn("<｜User｜><tool_result>2</tool_result>\n\n<tool_result>4</tool_result>\n\n"
                      "<tool_result>6</tool_result>\n\nsum them<｜Assistant｜></think>", prompt)

    def test_task_tokens(self):
        prompt = render([U("classify me", task="query")])
        self.assertTrue(prompt.endswith("classify me<｜query｜>"))
        self.assertNotIn("<｜Assistant｜>", prompt)
        self.assertTrue(render([U("hi", task="action")], enable_thinking=True).endswith(
            "<｜User｜>hi<｜Assistant｜><think><｜action｜>"))
        self.assertTrue(render([U("hi"), A("hello", task="title")]).endswith("hello" + EOS + "<｜title｜>"))
        with self.assertRaises(TemplateRequestError):
            render([U("hi", task="summarize")])

    def test_namespaced_tool_schemas(self):
        for tool, schema in (
                ({"type": "function", "namespace": "search", "function": dict(LOOKUP)}, dict(LOOKUP, name="search::lookup")),
                ({"type": "function", "function": dict(LOOKUP, namespace="search")}, dict(LOOKUP, name="search::lookup")),
                ({"type": "function", "namespace": {"name": "search", "description": "Search tools."},
                  "function": dict(LOOKUP)}, dict(LOOKUP, name="search::lookup", description="Search tools.\nLook up a value"))):
            with self.subTest(tool=tool):
                prompt = render([S("system"), U("q")], [tool])
                self.assertIn(json.dumps(schema, ensure_ascii=False), prompt)
                self.assertNotIn('"namespace":', prompt)

    def test_unknown_role_and_effort_are_refused(self):
        with self.assertRaises(TemplateRequestError):
            render([{"role": "function", "content": "x"}, U("q")])
        for effort in (0, 101, True, 1.5, "medium"):      # the template's own check (the spec maps first)
            with self.subTest(effort=effort), self.assertRaises(TemplateRequestError):
                TPL.template.render(messages=[U("q")], add_generation_prompt=True, thinking=True,
                                    reasoning_effort=effort, bos_token=BOS)


class Spec(unittest.TestCase):
    def test_thinking_and_effort_mapping(self):
        for kwargs, want in (({}, {"thinking": False}),
                             ({"enable_thinking": True}, {"thinking": True}),
                             ({"enable_thinking": False, "reasoning_effort": "low"}, {"thinking": False}),
                             ({"reasoning_effort": "low"}, {"thinking": True, "reasoning_effort": "low"}),
                             ({"reasoning_effort": "medium"}, {"thinking": True, "reasoning_effort": "high"}),
                             ({"reasoning_effort": "xhigh"}, {"thinking": True, "reasoning_effort": "max"}),
                             ({"reasoning_effort": "high"}, {"thinking": True, "reasoning_effort": "high"}),
                             ({"reasoning_effort": "max"}, {"thinking": True, "reasoning_effort": "max"}),
                             ({"reasoning_effort": 63}, {"thinking": True, "reasoning_effort": 63})):
            with self.subTest(**kwargs):
                self.assertEqual(SPEC.thinking_kwargs(kwargs), want)
                self.assertEqual(SPEC.thinking_on(kwargs), want["thinking"])
        for bad in (0, 101, True, 2.5, "extreme", [1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SPEC.thinking_kwargs({"reasoning_effort": bad})

    def test_registry_entry(self):
        self.assertIs(families.get("deepseek41"), SPEC)
        self.assertEqual((SPEC.title, SPEC.model_name, SPEC.arch, SPEC.parser_name, SPEC.tokenizer_dir, SPEC.pre),
                         ("DeepSeek-V4.1-Flash", "deepseek-v4.1-flash", "deepseek41", "deepseek41", "deepseek4", "joyai-llm"))
        self.assertEqual((SPEC.eos_ids, SPEC.eog_ids, SPEC.bos_id, SPEC.n_vocab), ((1,), (1,), 0, 129280))
        self.assertEqual(SPEC.turn_token_ids, (128799, 128803, 128804))
        # moex vision (2026-10-19): the release's encoder writes <｜deepseek_image｜> for a picture; the server repeats it once
        # per row of the span (serve/vision_dsv41.py's records carry no lead pads) and the config's "vision" is taken
        self.assertTrue(SPEC.vision_ready)
        self.assertEqual(SPEC.image_tokens, (None, "<\uff5cdeepseek_image\uff5c>", 129264))
        self.assertEqual(SPEC.image_lead_pads, 0)
        self.assertEqual(families.get("deepseek4").image_lead_pads, 3)      # V4's mtmd records keep their lead pads
        families.check_config({"family": "deepseek41", "vision": {"exe": "x"}}, SPEC)   # no ValueError

    def test_the_packaged_template_not_the_v4_export(self):
        # the tokenizer directory is V4's and may hold V4's exported chat_template.jinja: V4.1 never takes it
        packaged = (X / "serve" / "family_templates" / "deepseek41.jinja").read_text(encoding="utf-8")
        self.assertEqual(SPEC.load_template(SPEC.tokenizer_path()).source, packaged)
        self.assertEqual(TPL.source, packaged)

    def test_props_caps(self):
        # what moex/tests/test_family_props.py reads through /props, without the HTTP server
        self.assertEqual(TPL.caps, {"supports_tools": True, "supports_tool_calls": True, "supports_system_role": True,
                                    "supports_parallel_tool_calls": True, "supports_preserve_reasoning": False})


# ------------------------------------------------------------------------------------------------ 2. the request path
REQUESTS = {
    "plain": {"messages": [{"role": "user", "content": "Hi"}]},
    "developer": {"messages": [{"role": "developer", "content": "Developer rules."}, {"role": "user", "content": "Hi"}]},
    "late_system": {"messages": [{"role": "system", "content": "Start."}, {"role": "user", "content": "Hi"},
                                 {"role": "assistant", "content": "Hello"},
                                 {"role": "system", "content": "Hook context."}, {"role": "user", "content": "ls?"}]},
    "parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Part one. "},
                                                        {"type": "text", "text": "Part two."}]}]},
    "tools": {"tools": [{"type": "function", "function": WEATHER}], "reasoning_effort": "high", "messages": [
        {"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": None, "reasoning_content": "Call it.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "18 C"}]},
    "medium": {"reasoning_effort": "medium", "messages": [{"role": "user", "content": "q"}]},
    "low": {"reasoning_effort": "low", "messages": [{"role": "user", "content": "q"}]},
    "none": {"reasoning_effort": "none", "messages": [{"role": "user", "content": "q"}]},
    "kwargs_on": {"chat_template_kwargs": {"enable_thinking": True}, "messages": [{"role": "user", "content": "q"}]},
    "kwargs_off": {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "high",
                   "messages": [{"role": "user", "content": "q"}]},
}


@needs_encoder
class RequestPath(unittest.TestCase):
    def test_requests_render_as_the_encoder(self):
        for name, req in REQUESTS.items():
            with self.subTest(request=name):
                messages, tools, kwargs = openai_to_messages(copy.deepcopy(req))
                self.assertEqual(TPL.render(messages, tools=tools, **kwargs), expected(messages, tools, True, **kwargs))

    def test_request_effort_levels(self):
        def prompt(name, dialect=DIALECT, **extra):
            messages, tools, kwargs = openai_to_messages(dict(copy.deepcopy(REQUESTS[name]), **extra), dialect)
            return TPL.render(messages, tools=tools, **kwargs)
        self.assertIn(EFFORT.format(75), prompt("tools"))            # a client's "high" is the release's high (75)
        self.assertIn(EFFORT.format(100), prompt("tools", reasoning_effort="max"))
        self.assertIn(EFFORT.format(100), prompt("tools", reasoning_effort="xhigh"))
        self.assertIn(EFFORT.format(75), prompt("medium"))
        self.assertIn(EFFORT.format(50), prompt("low"))
        self.assertIn(EFFORT.format(75), prompt("kwargs_on"))
        for name in ("plain", "none", "kwargs_off"):
            self.assertTrue(prompt(name).endswith("<｜Assistant｜></think>"), name)
            self.assertNotIn("Reasoning Effort", prompt(name))
        # a system message after the start stays one (<｜System｜>, the encoder's mid-conversation system message)
        self.assertIn(EOS + "<｜System｜>Hook context.<｜User｜>ls?<｜Assistant｜></think>", prompt("late_system"))
        # the default dialect (every other family) is what it was: high is xhigh (the template's max), a late system
        # message a user turn the next user message joins
        self.assertIn(EFFORT.format(100), prompt("tools", dialect=Dialect()))
        self.assertIn(EOS + "<｜User｜>Hook context.\n\nls?<｜Assistant｜></think>", prompt("late_system", dialect=Dialect()))


# The serving cases of the audit of 2026-10-19 (DSV41_BACKLOG #35), as a client sends them.  Each is checked end to end: the
# server's path (frontend.openai_to_messages with this family's Dialect, then the template) against the release's encoder
# fed the same body as it is - the client's messages (tools on the first system message), its effort word as the release
# takes it (medium is the release's default, high; xhigh and maximum are max).
CALLS2 = [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}},
          {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}]
EDGE = {
    "results_out_of_order": {"tools": [{"type": "function", "function": WEATHER}], "reasoning_effort": "high", "messages": [
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {"role": "assistant", "content": "", "reasoning_content": "Both.", "tool_calls": CALLS2},
        {"role": "tool", "tool_call_id": "c2", "content": "Rome 21 C"}, {"role": "tool", "tool_call_id": "c1", "content": "Paris 18 C"},
        {"role": "user", "content": "And now?"}]},
    "late_system": REQUESTS["late_system"],
    "late_developer_thinking": {"reasoning_effort": "max", "messages": [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello", "reasoning_content": "greet"},
        {"role": "developer", "content": "Be brief."}]},
    "user_parts": REQUESTS["parts"],
    "system_parts": {"messages": [{"role": "system", "content": [{"type": "text", "text": "Rule one."},
                                                                 {"type": "text", "text": "Rule two."}]},
                                  {"role": "user", "content": "Hi"}]},
    "assistant_parts": {"messages": [{"role": "user", "content": "Hi"},
                                     {"role": "assistant", "content": [{"type": "text", "text": "A."}, {"type": "text", "text": "B."}]},
                                     {"role": "user", "content": "More"}]},
    "tool_parts": {"tools": [{"type": "function", "function": WEATHER}], "messages": [
        {"role": "user", "content": "Paris?"}, {"role": "assistant", "content": "", "tool_calls": CALLS2[:1]},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "18 C"}, {"type": "text", "text": "cloudy"}]}]},
    "reasoning_only_turn_tools": {"tools": [{"type": "function", "function": WEATHER}], "reasoning_effort": "high", "messages": [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "", "reasoning_content": "Nothing to say."},
        {"role": "user", "content": "Paris?"}]},
    "reasoning_only_turn": {"reasoning_effort": "low", "messages": [
        {"role": "user", "content": "Hi"}, {"role": "assistant", "content": None, "reasoning_content": "Nothing to say."},
        {"role": "user", "content": "Again?"}]},
    "double_encoded_arguments": {"tools": [{"type": "function", "function": WEATHER}], "messages": [
        {"role": "user", "content": "Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "get_weather", "arguments": json.dumps(json.dumps({"city": "Paris", "unit": "celsius"}))}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "18 C"}]},
    "effort_max": {"reasoning_effort": "max", "messages": [{"role": "user", "content": "q"}]},
    "effort_xhigh": {"reasoning_effort": "xhigh", "messages": [{"role": "user", "content": "q"}]},
    "effort_medium": REQUESTS["medium"],
    "effort_low": REQUESTS["low"],
    "effort_none": REQUESTS["none"],
    "thinking_default_effort": REQUESTS["kwargs_on"],
}
RELEASE_EFFORT = {None: None, "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max", "maximum": "max"}


def release_prompt(req):
    """The release's encoder on a client's OpenAI body as it was sent (gen: it always ends a user turn with the prompt)."""
    word = req.get("reasoning_effort") or (req.get("chat_template_kwargs") or {}).get("reasoning_effort")
    on = (req.get("chat_template_kwargs") or {}).get("enable_thinking")
    thinking = on is not False and word not in ("none", "off") and (word is not None or on is True)
    msgs = [dict(m, role="system" if m["role"] == "developer" else m["role"]) for m in copy.deepcopy(req["messages"])]
    if req.get("tools"):
        if msgs[0]["role"] == "system":
            msgs[0]["tools"] = req["tools"]
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": req["tools"]})
    return enc.encode_messages(msgs, thinking_mode="thinking" if thinking else "chat",
                               reasoning_effort=RELEASE_EFFORT[word] if thinking else None)


@needs_encoder
class ReleaseEndToEnd(unittest.TestCase):
    """A client's body through the server's path for this family is the release's encoding of that body."""

    def test_openai_bodies(self):
        for name, req in EDGE.items():
            with self.subTest(request=name):
                messages, tools, kwargs = openai_to_messages(copy.deepcopy(req), DIALECT)
                self.assertEqual(TPL.render(messages, tools=tools, **kwargs), release_prompt(req))

    def test_anthropic_bodies(self):
        # the Anthropic shape of results_out_of_order and of a late system message (its blocks joined by a blank line)
        body = {"system": [{"type": "text", "text": "Use tools."}, {"type": "text", "text": "Be brief."}],
                "tools": [{"name": WEATHER["name"], "description": WEATHER["description"], "input_schema": WEATHER["parameters"]}],
                "thinking": {"type": "enabled", "budget_tokens": 1024}, "messages": [
                    {"role": "user", "content": "Weather in Paris and Rome?"},
                    {"role": "assistant", "content": [{"type": "thinking", "thinking": "Both."},
                                                      {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}},
                                                      {"type": "tool_use", "id": "t2", "name": "get_weather", "input": {"city": "Rome"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "Rome 21 C"},
                                                 {"type": "tool_result", "tool_use_id": "t1", "content": "Paris 18 C"}]},
                    {"role": "user", "content": [{"type": "text", "text": "Line one."}, {"type": "text", "text": "Line two."}]}]}
        messages, tools, kwargs = anthropic_to_messages(copy.deepcopy(body), True, DIALECT)
        # the release's input for the same conversation (Anthropic's budget under 2K is the server's "low")
        ref = [{"role": "system", "content": "Use tools.\n\nBe brief.", "tools": wrap_tools(tools)},
               {"role": "user", "content": "Weather in Paris and Rome?"},
               {"role": "assistant", "content": "", "reasoning_content": "Both.", "tool_calls": [
                   {"id": "t1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                   {"id": "t2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
               {"role": "tool", "tool_call_id": "t2", "content": "Rome 21 C"},
               {"role": "tool", "tool_call_id": "t1", "content": "Paris 18 C"},
               {"role": "user", "content": "Line one.\n\nLine two."}]
        self.assertEqual(kwargs, {"reasoning_effort": "low"})
        self.assertEqual(TPL.render(messages, tools=tools, **kwargs),
                         enc.encode_messages(ref, thinking_mode="thinking", reasoning_effort="low"))
        self.assertLess(TPL.render(messages, tools=tools, **kwargs).index("Paris 18 C"),
                        TPL.render(messages, tools=tools, **kwargs).index("Rome 21 C"))

    def test_empty_arguments_are_no_arguments(self):
        # A deliberate difference: "" (a client's spelling of a call with no arguments) is a call with no parameters, as
        # the model wrote it; the release's encoder would write a parameter "arguments" with an empty value, which the
        # model never wrote (its parser gives "{}" for such a call).
        req = {"tools": [{"type": "function", "function": WEATHER}], "messages": [
            {"role": "user", "content": "Go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                 "function": {"name": "get_weather", "arguments": ""}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}]}
        messages, tools, kwargs = openai_to_messages(copy.deepcopy(req), DIALECT)
        ours = TPL.render(messages, tools=tools, **kwargs)
        self.assertIn(f'<{D} invoke name="get_weather">\n\n</{D} invoke>', ours)
        theirs = release_prompt(req)
        self.assertIn(f'<{D} parameter name="arguments" string="true"></{D} parameter>', theirs)
        self.assertEqual(ours, theirs.replace(f'<{D} parameter name="arguments" string="true"></{D} parameter>', ""))
        # and what the model writes for a call without parameters reads back as "{}"
        completion = f'\n\n<{D} calls>\n<{D} invoke name="get_weather">\n</{D} invoke>\n</{D} calls>' + EOS   # chat mode
        self.assertEqual(enc.parse_message_from_completion_text(completion, "chat")["tool_calls"][0]["function"]["arguments"], "{}")

    def test_responses_bodies(self):
        # the Responses API's path: a late developer message stays a system message, text parts join by a blank line, the
        # effort word is the release's (high 75); its tool results were always put in the order of their calls
        from serve import responses as R
        body = {"instructions": "Rules.", "reasoning": {"effort": "high"}, "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Part one."},
                                                            {"type": "input_text", "text": "Part two."}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Ok."}]},
            {"type": "message", "role": "developer", "content": "Be brief."},
            {"role": "user", "content": "Go"}]}
        messages, kwargs = R.input_messages(copy.deepcopy(body), DIALECT), R.template_kwargs(body, {}, DIALECT)
        self.assertEqual(kwargs, {"reasoning_effort": "high"})
        self.assertEqual(TPL.render(messages, tools=None, **kwargs), enc.encode_messages(
            [S("Rules."), U("Part one.\n\nPart two."), A("Ok."), S("Be brief."), U("Go")], thinking_mode="thinking",
            reasoning_effort="high"))
        # and the default dialect as it was: the late developer message a user turn, the parts joined with nothing
        self.assertEqual([m["role"] for m in R.input_messages(copy.deepcopy(body))], ["system", "user", "assistant", "user", "user"])
        self.assertEqual(R.input_messages(copy.deepcopy(body))[1]["content"], "Part one.Part two.")

    def test_the_default_dialect_is_unchanged(self):
        # every other family's conversion: no ids, late system messages as user turns, parts joined with nothing
        messages, _, kwargs = openai_to_messages(copy.deepcopy(EDGE["results_out_of_order"]))
        self.assertNotIn("tool_call_id", messages[2])
        self.assertNotIn("id", messages[1]["tool_calls"][0])
        self.assertEqual(kwargs, {"reasoning_effort": "xhigh"})
        messages, _, _ = openai_to_messages(copy.deepcopy(EDGE["late_system"]))
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user", "user"])
        messages, _, _ = openai_to_messages(copy.deepcopy(EDGE["system_parts"]))
        self.assertEqual(messages[0]["content"], "Rule one.Rule two.")


Q = [{"role": "user", "content": "q"}]
THINKS = BOS + EFFORT.format("{}") + "<｜User｜>q<｜Assistant｜><think>"
CHATS = BOS + "<｜User｜>q<｜Assistant｜></think>"


class ThinkingSwitches(unittest.TestCase):
    """DSV41_BACKLOG #45: the switches a client turns thinking on or off with, through Service.with_shared and the converters
    (as the routes call them), with the config's level "max" (strata-dsv41.json's): vLLM's chat_template_kwargs `thinking`,
    Anthropic's thinking enabled without a budget and adaptive, and enable_thinking true without a level."""

    def service(self, effort="max", family=SPEC):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, TPL if family is SPEC else ChatTemplate(X / "serve" / "chat_template.jinja"),
                      family=family)
        svc.config_effort = effort
        return svc

    def convert(self, body, api="openai", effort="max", family=SPEC):
        svc = self.service(effort, family)
        req = svc.with_shared(copy.deepcopy(body), api)
        if api == "openai":
            return openai_to_messages(req, svc.dialect)
        return anthropic_to_messages(req, svc.anthropic_think_unasked, svc.dialect)

    def prompt(self, body, api="openai", effort="max"):
        messages, tools, kwargs = self.convert(body, api, effort)
        return TPL.render(messages, tools=tools, **kwargs)

    def test_the_three_bodies_with_the_config_level(self):
        for body, api, want in (
                ({"messages": Q, "chat_template_kwargs": {"thinking": False}}, "openai", CHATS),
                ({"messages": Q, "chat_template_kwargs": {"thinking": True}}, "openai", THINKS.format(100)),
                ({"messages": Q, "thinking": {"type": "enabled"}}, "anthropic", THINKS.format(100)),
                ({"messages": Q, "thinking": {"type": "adaptive"}}, "anthropic", THINKS.format(100)),
                # (c): enable_thinking true names no level either (it got the template's 75)
                ({"messages": Q, "chat_template_kwargs": {"enable_thinking": True}}, "openai", THINKS.format(100))):
            with self.subTest(body=body):
                self.assertEqual(self.prompt(body, api), want)

    def test_a_level_of_none_leaves_the_clients_on_at_the_default(self):
        # the config's "none" does not turn off a request that asks for thinking; it thinks at the template's level (75)
        for body, api in (({"messages": Q, "chat_template_kwargs": {"thinking": True}}, "openai"),
                          ({"messages": Q, "chat_template_kwargs": {"enable_thinking": True}}, "openai"),
                          ({"messages": Q, "thinking": {"type": "enabled"}}, "anthropic"),
                          ({"messages": Q, "thinking": {"type": "adaptive"}}, "anthropic")):
            for effort in ("none", None):
                with self.subTest(body=body, effort=effort):
                    self.assertEqual(self.prompt(body, api, effort), THINKS.format(75))
        self.assertEqual(self.prompt({"messages": Q, "chat_template_kwargs": {"thinking": False}}, "openai", "none"), CHATS)
        self.assertEqual(self.prompt({"messages": Q}, "openai", "none"), CHATS)

    def test_the_requests_own_level_and_off_win(self):
        for body, api, want in (
                ({"messages": Q, "chat_template_kwargs": {"thinking": True, "reasoning_effort": "low"}}, "openai", THINKS.format(50)),
                ({"messages": Q, "reasoning_effort": "low", "chat_template_kwargs": {"thinking": True}}, "openai", THINKS.format(50)),
                ({"messages": Q, "reasoning_effort": "high", "chat_template_kwargs": {"thinking": False}}, "openai", CHATS),
                ({"messages": Q, "chat_template_kwargs": {"thinking": True, "enable_thinking": False}}, "openai", CHATS),
                ({"messages": Q, "thinking": {"type": "enabled"}, "output_config": {"effort": "low"}}, "anthropic", THINKS.format(50)),
                ({"messages": Q, "thinking": {"type": "enabled", "budget_tokens": 1024}}, "anthropic", THINKS.format(50)),
                ({"messages": Q, "thinking": {"type": "disabled"}}, "anthropic", CHATS)):
            with self.subTest(body=body):
                self.assertEqual(self.prompt(body, api), want)

    def test_a_reasoning_object_or_output_config_without_a_level(self):
        # the audit of 2026-10-24: a reasoning object names a level only by its effort (enabled false turns thinking off),
        # an output_config only by its effort - without one the config's level applies, as when the field is left out
        # (they rendered no thinking: with_shared kept the level out and the template thinks only when asked)
        for body, api, want in (
                ({"messages": Q, "reasoning": {"enabled": True}}, "openai", THINKS.format(100)),
                ({"messages": Q, "reasoning": {"exclude": False}}, "openai", THINKS.format(100)),
                ({"messages": Q, "reasoning": {"enabled": False}}, "openai", CHATS),
                ({"messages": Q, "reasoning": {"effort": "low"}}, "openai", THINKS.format(50)),
                ({"messages": Q, "output_config": {"other": 1}}, "anthropic", THINKS.format(100)),
                ({"messages": Q, "output_config": {"other": 1, "effort": "low"}}, "anthropic", THINKS.format(50)),
                ({"messages": Q, "thinking": {"type": "enabled"}, "output_config": {"other": 1}}, "anthropic",
                 THINKS.format(100))):
            with self.subTest(body=body):
                self.assertEqual(self.prompt(body, api), want)
        svc = self.service()
        req = svc.with_shared({"messages": Q, "output_config": {"other": 1}}, "anthropic")
        self.assertEqual(req["output_config"], {"other": 1, "effort": "max"})   # the client's keys kept

    def test_the_default_dialect_is_unchanged(self):
        # every other family (here Qwen's): `thinking` is no switch, Anthropic's enabled without a budget sets nothing, and
        # enable_thinking true keeps the config's level out, as before
        qwen = None
        _, _, kwargs = self.convert({"messages": Q, "chat_template_kwargs": {"thinking": False}}, family=qwen)
        self.assertEqual(kwargs, {"reasoning_effort": "xhigh"})
        _, _, kwargs = self.convert({"messages": Q, "chat_template_kwargs": {"enable_thinking": True}}, family=qwen)
        self.assertEqual(kwargs, {"enable_thinking": True})
        for kind in ("enabled", "adaptive"):
            svc = self.service(family=qwen)
            req = svc.with_shared({"messages": Q, "thinking": {"type": kind}}, "anthropic")
            self.assertNotIn("output_config", req)
            self.assertEqual(anthropic_to_messages(req, True, svc.dialect)[2], {})


# encoding.py's response_format_template, written after a system message's text (and its tools) by render_message: "\n\n" +
# "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}", schema=to_json(response_format)
# (json.dumps with ensure_ascii=False)
RF = "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
SCHEMA_FMT = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
    "type": "object", "properties": {"x": {"type": "integer"}, "note": {"type": "string", "description": "élan"}},
    "required": ["x"]}}}


class ResponseFormat(unittest.TestCase):
    """DSV41_BACKLOG #56: a request's response_format as the release encodes it - the `## Response Format:` section of the
    first system message (Dialect.response_format) - not serve/structured.py's generic directive; the answer is checked as
    before.  The expected prompts are the strings encoding.py's render_message writes for them."""

    def chat(self, req, dialect=DIALECT):
        from serve.structured import prepare_format
        messages, tools, kwargs = openai_to_messages(copy.deepcopy(req), dialect)
        messages, validator = prepare_format(req.get("response_format"), messages, section=dialect.response_format)
        return TPL.render(messages, tools=tools, **kwargs), validator

    def test_the_releases_section(self):
        def js(f):
            return json.dumps(f, ensure_ascii=False)
        system = [{"role": "system", "content": "Answer in JSON."}, {"role": "user", "content": "q"}]
        for fmt in (SCHEMA_FMT, {"type": "json_object"}):
            with self.subTest(fmt=fmt["type"]):
                prompt, _ = self.chat({"messages": system, "response_format": fmt})
                self.assertEqual(prompt, BOS + "<｜System｜>Answer in JSON." + RF + js(fmt) + "<｜User｜>q<｜Assistant｜></think>")
                # no system message: the encoder reads the format from one, so an empty one goes first (as for tools)
                prompt, _ = self.chat({"messages": Q, "response_format": fmt})
                self.assertEqual(prompt, BOS + "<｜System｜>" + RF + js(fmt) + "<｜User｜>q<｜Assistant｜></think>")
                # thinking: the effort opens the system message, then its text and the format
                prompt, _ = self.chat({"messages": system, "response_format": fmt, "reasoning_effort": "max"})
                self.assertEqual(prompt, BOS + EFFORT.format(100) + "Answer in JSON." + RF + js(fmt) +
                                 "<｜User｜>q<｜Assistant｜><think>")
                self.assertNotIn("OUTPUT FORMAT REQUIREMENT", prompt)
        prompt, validator = self.chat({"messages": Q, "response_format": {"type": "text"}})
        self.assertEqual((prompt, validator), (CHATS, None))

    def test_the_answer_is_checked_as_before(self):
        from serve.structured import StructuredOutputError, validated_json
        _, validator = self.chat({"messages": Q, "response_format": SCHEMA_FMT})
        self.assertEqual(validated_json('{"x": 1}', validator, "stop"), '{"x":1}')
        for bad, finish in (("[1]", "stop"), ("not json", "stop"), ('{"x": 1}', "length")):
            with self.subTest(answer=bad, finish=finish), self.assertRaises(StructuredOutputError):
                validated_json(bad, validator, finish)
        with self.assertRaises(ValueError):                  # a format the server refuses is still a 400
            self.chat({"messages": Q, "response_format": {"type": "json_schema", "json_schema": {"name": "x"}}})

    def test_responses_with_tools(self):
        from serve import responses as R
        from serve.structured import prepare_format
        body = {"input": "q", "text": {"format": {"type": "json_object"}},
                "tools": [{"type": "function", "name": WEATHER["name"], "description": WEATHER["description"],
                           "parameters": WEATHER["parameters"]}]}
        tools = R.request_tools(body)[0]
        messages, _ = prepare_format(R.text_format(body), R.input_messages(copy.deepcopy(body), DIALECT), with_tools=True,
                                     section=DIALECT.response_format)
        prompt = TPL.render(messages, tools=tools)
        # render_tools' text ends "...to invoke tool calls.\n", and the format follows it on the same system message
        self.assertTrue(prompt.startswith(BOS + "<｜System｜>\n\n## Tools\n\n"))
        self.assertTrue(prompt.endswith("schemas to invoke tool calls.\n" + RF + '{"type": "json_object"}<｜User｜>q<｜Assistant｜></think>'))
        self.assertNotIn("OUTPUT FORMAT REQUIREMENT", prompt)

    def test_through_the_routes(self):
        import urllib.request
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        tok = ByteTokenizer()
        engine = MockEngine(tok, '{"x": 1}')
        svc = Service(engine, tok, TPL, model_name=SPEC.model_name, family=SPEC)
        svc.stop_ids = set(tok.encode("<|im_end|>", parse_special=True))     # the stub's end of a reply
        httpd = serve(svc, port=0)
        try:
            for path, body in (("/v1/chat/completions", {"messages": Q, "response_format": {"type": "json_object"}}),
                               ("/v1/responses", {"input": "q", "text": {"format": {"type": "json_object"}}})):
                with self.subTest(path=path):
                    r = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                               data=json.dumps(dict(body, model="m")).encode(),
                                               headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(r, timeout=30) as resp:
                        self.assertEqual(resp.status, 200)
                        self.assertIn('{\\"x\\":1}', resp.read().decode())
                    self.assertEqual(tok.decode(engine.last_prompt),
                                     BOS + "<｜System｜>" + RF + '{"type": "json_object"}<｜User｜>q<｜Assistant｜></think>')
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_default_dialect_keeps_the_directive(self):
        from serve.structured import prepare_format
        messages, _, _ = openai_to_messages({"messages": [{"role": "system", "content": "Rules."}, *Q]})
        out, _ = prepare_format({"type": "json_object"}, messages)
        self.assertTrue(out[0]["content"].startswith("Rules.\n\nOUTPUT FORMAT REQUIREMENT: Return exactly one JSON object"))
        self.assertNotIn("response_format", out[0])


class PastReasoning(unittest.TestCase):
    """How a past assistant turn's reasoning reaches the prompt (DSV41_BACKLOG #56).  With tools every turn keeps its reasoning
    (the release's drop_thinking is off then), so the turn renders as <think>REASONING</think>ANSWER."""

    def prompt(self, assistant, dialect=DIALECT):
        req = {"tools": [{"type": "function", "function": WEATHER}], "chat_template_kwargs": {"enable_thinking": True},
               "messages": [{"role": "user", "content": "q1"}, dict(assistant, role="assistant"),
                            {"role": "user", "content": "q2"}]}
        messages, tools, kwargs = openai_to_messages(copy.deepcopy(req), dialect)
        return TPL.render(messages, tools=tools, **kwargs)

    @staticmethod
    def turn(text):
        return "<｜User｜>q1<｜Assistant｜><think>" + text + EOS + "<｜User｜>q2<｜Assistant｜><think>"

    def test_the_reasoning_field(self):
        # vLLM's name for a past turn's reasoning (its V4.1 renderer maps it to reasoning_content): it was not read
        self.assertIn(self.turn("r1</think>a1"), self.prompt({"content": "a1", "reasoning": "r1"}))
        # a reasoning_content of its own is the turn's
        self.assertIn(self.turn("rc1</think>a1"), self.prompt({"content": "a1", "reasoning": "r1", "reasoning_content": "rc1"}))
        self.assertIn(self.turn("</think>a1"), self.prompt({"content": "a1", "reasoning": None}))
        # a turn of reasoning alone is a turn (FamilySpec.reasoning_turns)
        self.assertIn(self.turn("r1</think>"), self.prompt({"content": "", "reasoning": "r1"}))
        # the default dialect (every other family) as before: not read
        self.assertIn(self.turn("</think>a1"), self.prompt({"content": "a1", "reasoning": "r1"}, Dialect()))

    def test_inline_think_in_the_content(self):
        # a client that sends the reasoning inline: "<think>R</think>answer" as content rendered
        # "<think></think><think>R</think>answer"; the block is the turn's reasoning, both parts as written
        self.assertIn(self.turn("R</think>answer"), self.prompt({"content": "<think>R</think>answer"}))
        self.assertIn(self.turn("\nR\n</think>\n\nanswer"), self.prompt({"content": "  <think>\nR\n</think>\n\nanswer"}))
        self.assertIn(self.turn("R</think>"), self.prompt({"content": "<think>R</think>"}))
        self.assertIn(self.turn("</think>answer"), self.prompt({"content": "<think></think>answer"}))
        # only when the turn has no reasoning of its own (then the content is as sent)
        self.assertIn(self.turn("own</think><think>R</think>answer"),
                      self.prompt({"content": "<think>R</think>answer", "reasoning_content": "own"}))
        self.assertIn(self.turn("r1</think><think>R</think>answer"),
                      self.prompt({"content": "<think>R</think>answer", "reasoning": "r1"}))
        # a block that is not whole, or not at the start, is the content's text
        self.assertIn(self.turn("</think>see <think>R</think>"), self.prompt({"content": "see <think>R</think>"}))
        self.assertIn(self.turn("</think><think>R"), self.prompt({"content": "<think>R"}))
        # without thinking the turn's reasoning is not written (the encoder's chat mode), so neither is the block
        messages, _, kwargs = openai_to_messages({"messages": [{"role": "user", "content": "q1"},
                                                               {"role": "assistant", "content": "<think>R</think>answer"},
                                                               {"role": "user", "content": "q2"}]}, DIALECT)
        self.assertEqual(TPL.render(messages, **kwargs),
                         BOS + "<｜User｜>q1<｜Assistant｜></think>answer" + EOS + "<｜User｜>q2<｜Assistant｜></think>")
        # the default dialect (every other family) as before
        self.assertIn(self.turn("</think><think>R</think>answer"), self.prompt({"content": "<think>R</think>answer"}, Dialect()))

    def test_inline_think_on_messages(self):
        # Anthropic's messages: a string, or text blocks, of an assistant turn with no thinking block
        for content in ("<think>R</think>answer", [{"type": "text", "text": "<think>R</think>answer"}]):
            with self.subTest(content=content):
                body = {"thinking": {"type": "enabled", "budget_tokens": 1024}, "messages": [
                    {"role": "user", "content": "q1"}, {"role": "assistant", "content": content},
                    {"role": "user", "content": "q2"}]}
                messages = anthropic_to_messages(copy.deepcopy(body), True, DIALECT)[0]
                self.assertEqual((messages[1]["reasoning_content"], messages[1]["content"]), ("R", "answer"))
                self.assertEqual(anthropic_to_messages(copy.deepcopy(body), True, Dialect())[0][1],
                                 {"role": "assistant", "content": "<think>R</think>answer"})
        # a thinking block of its own wins: the text is as sent
        body = {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "own"}, {"type": "text", "text": "<think>R</think>answer"}]}]}
        self.assertEqual(anthropic_to_messages(body, True, DIALECT)[0][1],
                         {"role": "assistant", "content": "<think>R</think>answer", "reasoning_content": "own"})
        # a user's text is never touched
        self.assertEqual(anthropic_to_messages({"messages": [{"role": "user", "content": "<think>R</think>q"}]}, True,
                                               DIALECT)[0], [{"role": "user", "content": "<think>R</think>q"}])


class ServerDefaults(unittest.TestCase):
    """What the server says a request that names no sampling gets - GET /props' default_generation_settings - with a stub
    tokenizer (none of it reads the vocabulary)."""

    def service(self, sampling, full_vocab=True, family=SPEC):
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok")
        engine.full_vocab = full_vocab
        return Service(engine, tok, TPL, model_name=SPEC.model_name, family=family, sampling_defaults=sampling)

    def props(self, svc):
        import urllib.request
        from serve.server import serve
        httpd = serve(svc, port=0)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/props", timeout=30) as r:
                return json.loads(r.read())["default_generation_settings"]["params"]
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_chat_defaults(self):
        # moex: GET /health's "defaults", which the Chat page starts from: the config's sampling, the family's top_k 0 and the
        # config's thinking level - not the settings the page shared with other apps; a plain config names none.  It reads no
        # vocabulary, so it runs on the stub (DSV41_BACKLOG #57: it was skipped without the deepseek4 tokenizer)
        svc = self.service({"temperature": 1.0, "top_p": 1.0})
        svc.config_effort = "max"
        want = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "reasoning_effort": "max"}
        self.assertEqual(svc.chat_defaults(), want)
        svc.shared = {"temperature": 0.6, "reasoning_effort": "low"}
        self.assertEqual(svc.chat_defaults(), want)
        self.assertEqual(self.service({}, full_vocab=False, family=None).chat_defaults(), {})

    def test_props_top_k(self):
        # DSV41_BACKLOG #56: strata-dsv41.json's sampling block names no top_k; the engine samples the whole vocabulary
        # (top_k 0, sampling_keys) and /health's "defaults" said so, /props left it out
        release = {"temperature": 1.0, "top_p": 1.0}
        self.assertEqual(self.props(self.service(release)), {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "n_predict": -1})
        self.assertEqual(self.props(self.service(dict(release, top_k=20)))["top_k"], 20)      # the config's own
        shared = self.service(release)
        shared.shared = {"top_k": 40, "max_tokens": 99}                                       # the Chat settings shared
        self.assertEqual(self.props(shared), {"temperature": 1.0, "top_p": 1.0, "top_k": 40, "n_predict": 99})
        # an engine that cuts (every other family's) as before: no top_k when none is set
        self.assertEqual(self.props(self.service(release, full_vocab=False)), {"temperature": 1.0, "top_p": 1.0,
                                                                                "n_predict": -1})


def tokenizer_or_skip(test):
    try:
        return SPEC.load_tokenizer(SPEC.tokenizer_path())
    except Exception as e:  # noqa: BLE001 - the directory (or the regex module) is not on this machine
        test.skipTest(f"no deepseek4 tokenizer here ({type(e).__name__}: {e})")


class Tokens(unittest.TestCase):
    def test_control_tokens(self):
        tok = tokenizer_or_skip(self)
        for text, tid in (("<｜System｜>", 128799), ("<｜User｜>", 128803), ("<｜Assistant｜>", 128804), (D, 128825),
                          ("<think>", 128821), ("</think>", 128822), (BOS, 0), (EOS, 1)):
            self.assertEqual(tok.encode(text, parse_special=True), [tid], text)

    def test_server_prompt_tokens(self):
        tok = tokenizer_or_skip(self)
        from serve.server import MockEngine, Service
        svc = Service(MockEngine(tok, "ok"), tok, SPEC.load_template(SPEC.tokenizer_path()), model_name=SPEC.model_name,
                      family=SPEC)
        req = copy.deepcopy(REQUESTS["tools"])
        req["messages"][1]["content"] = "Weather in Paris? (typed: <｜User｜> <｜DSML｜ calls>)"
        messages, tools, kwargs = openai_to_messages(req)
        ids = svc.encode_prompt(messages, tools, kwargs)
        plain = copy.deepcopy(REQUESTS["tools"])
        ids_plain = svc.encode_prompt(*openai_to_messages(plain))
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[-2:], [128804, 128821])                    # <｜Assistant｜><think>
        self.assertEqual(ids.count(128799), 1)                          # the system prompt's <｜System｜>
        # the typed control texts stay text: no token more than the template's own
        for tid in (128803, 128825):
            self.assertEqual(ids.count(tid), ids_plain.count(tid))
        self.assertTrue(svc.thinking_on(kwargs))


# ------------------------------------------------------------------------------------------------ 3. the parser
PARSER_TOOLS = [WEATHER, ADD, WRITE, LOOKUP, {"name": "foo"}]

TURNS = {
    "text": {"reasoning_content": "Simple arithmetic: 2+2=4.", "content": "4"},
    "reference_call": {"reasoning_content": "  reason  ", "content": "summary",
                       "tool_calls": [call("lookup", '{"query":"value","limit":2}', type="function")]},
    "multiline": {"reasoning_content": "Line one.\nLine two with `code` and \"quotes\".\n\n- a list item",
                  "content": "Answer:\n\n```python\nprint('hi')\n```\nDone."},
    "all_types": {"reasoning_content": "write and add", "content": "Doing both.",
                  "tool_calls": [call("write_file", ARGS), call("add", {"a": -7, "b": 0, "tags": ["x", "y z"]})]},
    "no_parameters": {"reasoning_content": "", "content": "", "tool_calls": [call("foo", {})]},
    "json_like_strings": {"reasoning_content": "r", "content": "check",
                          "tool_calls": [call("get_weather", {"city": "123", "unit": "true"})]},
    "unicode": {"reasoning_content": "思考中…", "content": "你好 \U0001f600 café"},
    "namespaced": {"reasoning_content": "r", "content": "",
                   "tool_calls": [call("lookup", {"query": "v"}, namespace="search"), call("lookup", {"query": "w"})]},
    "three_calls": {"reasoning_content": "many", "content": "Three.",
                    "tool_calls": [call("get_weather", {"city": "Paris"}), call("get_weather", {"city": "Rome"}),
                                   call("add", {"a": 2, "b": 3})]},
}


def events(text, thinking, pieces, stream=True):
    p = FP.make_parser("deepseek41", thinking, PARSER_TOOLS, stream_tools=stream)
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


class ParserCheck(unittest.TestCase):
    def check_text(self, text, thinking, want):
        """`text` (the model's output, without the end-of-sentence token the engine stops on) parses into `want` in
        every chunking; streamed, each call's argument pieces join into its arguments."""
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


@needs_encoder
class ParserMatchesEncoder(ParserCheck):
    """The encoder's assistant turn (render_message) -> parse_message_from_completion_text, and -> the family parser."""

    def test_turns(self):
        for name, turn in TURNS.items():
            for mode in ("thinking", "chat"):
                with self.subTest(turn=name, mode=mode):
                    completion = enc.render_message(1, [U("go"), dict(turn, role="assistant")], thinking_mode=mode)
                    ref = enc.parse_message_from_completion_text(completion, thinking_mode=mode)
                    # a namespaced call is NAMESPACE::NAME in the text; the release's parser splits it, Strata's keeps it
                    want = (ref["reasoning_content"], ref["content"],
                            [(("%s::" % c["namespace"] if c.get("namespace") is not None else "") + c["function"]["name"],
                              json.loads(c["function"]["arguments"])) for c in ref["tool_calls"]])
                    self.assertTrue(completion.endswith(EOS))
                    self.check_text(completion[:-len(EOS)], mode == "thinking", want)


class ParserCases(ParserCheck):
    def test_template_turn_reads_back(self):
        """What the template writes for a past turn (after the prompt it would have followed) parses back into it."""
        for name, turn in TURNS.items():
            for thinking in (True, False):
                with self.subTest(turn=name, thinking=thinking):
                    asst = dict(turn, role="assistant")
                    prefix = render([U("go")], thinking=thinking, enable_thinking=thinking)
                    full = render([U("go"), asst], gen=False, enable_thinking=thinking)
                    self.assertTrue(full.startswith(prefix) and full.endswith(EOS))
                    calls = [(("%s::" % c["namespace"]) if c.get("namespace") else "") + c["function"]["name"]
                             for c in turn.get("tool_calls", [])]
                    args = [json.loads(c["function"]["arguments"]) if isinstance(c["function"]["arguments"], str)
                            else c["function"]["arguments"] for c in turn.get("tool_calls", [])]
                    self.check_text(full[len(prefix):-len(EOS)], thinking,
                                    (turn["reasoning_content"] if thinking else "", turn["content"], list(zip(calls, args))))

    def test_call_without_tools_is_text(self):
        # moex (DSV41_BACKLOG #65): a request that declared no tools gets a call block the model writes anyway as text, not
        # a tool_calls finish for a tool it never offered; with tools the same text is a call
        text = (f'r</think>answer\n\n<{D} calls>\n<{D} invoke name="lookup">\n'
                f'<{D} parameter name="query" string="true">value</{D} parameter>\n</{D} invoke>\n</{D} calls>')
        for stream in (True, False):
            with self.subTest(stream=stream):
                p = FP.make_parser("deepseek41", True, None, stream_tools=stream)
                evs = []
                for piece in text:
                    evs += p.feed(piece)
                evs += p.finish()
                self.assertFalse([e for e in evs if e.kind in ("tool_start", "tool_call")])
                self.assertEqual(summary(evs)[1], text.split("</think>", 1)[1])
        self.assertEqual(summary(events(text, True, list(text)))[2], [("lookup", {"query": "value"})])

    def test_v4_unspaced_dsml_is_no_call(self):
        # the release's parser refuses V4's tags; this one reads them as text
        v4 = (f'reason</think>summary\n\n<{D}tool_calls>\n<{D}invoke name="lookup">\n'
              f'<{D}parameter name="query" string="true">value</{D}parameter>\n</{D}invoke>\n</{D}tool_calls>')
        evs = events(v4, True, list(v4))
        self.assertFalse([e for e in evs if e.kind in ("tool_start", "tool_call")])
        self.assertEqual(summary(evs)[1], v4.split("</think>", 1)[1])

    def test_newlines_around_the_answer(self):
        # the answer as the release's parser gives it (the audit of 2026-10-19): its newlines after </think> kept, and of
        # those before the call block only the blank line of "\n\n<｜DSML｜ calls" dropped (the other families' machine
        # drops them all, Syntax.exact_answer)
        cases = {f'r</think>\n\nanswer\n\n\n<{D} calls>\n<{D} invoke name="foo">\n\n</{D} invoke>\n</{D} calls>':
                 ("r", "\n\nanswer\n", [("foo", {})]),
                 f'r</think>\n\n<{D} calls>\n<{D} invoke name="foo">\n\n</{D} invoke>\n</{D} calls>': ("r", "", [("foo", {})]),
                 "r</think>\n\nanswer\n": ("r", "\n\nanswer\n", []),
                 "r</think>\n": ("r", "\n", [])}
        for text, want in cases.items():
            self.check_text(text, True, want)
            if enc is not None:
                ref = enc.parse_message_from_completion_text(text + EOS, thinking_mode="thinking")
                self.assertEqual((ref["reasoning_content"], ref["content"]), want[:2], text)

    def test_a_fence_closed_right_before_the_end(self):
        # DSV41_BACKLOG #56: the "```" that closes a code block right before </think> is three backticks, which the quoted-tag
        # rule (#537) counted as an open quote - so an answer that starts with a backtick stayed reasoning.  A fence line
        # opens no quote; the quotes the rule is for stay quotes.
        cases = {"Plan:\n```py\nx=1\n```</think>```py\nprint(1)\n```": ("Plan:\n```py\nx=1\n```", "```py\nprint(1)\n```", []),
                 "Code:\n  ```\n  a\n  ```</think>`b` is the answer.": ("Code:\n  ```\n  a\n  ```", "`b` is the answer.", []),
                 "Plan:\n````\nx\n````</think>`y`": ("Plan:\n````\nx\n````", "`y`", []),
                 "the parser ends at `</think>` and keeps going</think>Answer.":
                     ("the parser ends at `</think>` and keeps going", "Answer.", []),
                 "write ```</think>``` here</think>Done.": ("write ```</think>``` here", "Done.", [])}
        for text, want in cases.items():
            self.check_text(text, True, want)
            if enc is not None and "write `" not in text and "at `" not in text:   # the release's parser: the first </think>
                ref = enc.parse_message_from_completion_text(text + EOS, thinking_mode="thinking")
                self.assertEqual((ref["reasoning_content"], ref["content"]), want[:2], text)

    def test_a_reasoning_that_ends_on_an_apostrophe(self):
        # the audit of 2026-10-24: an apostrophe never quotes </think> (it cannot tell a quote from "I'll") - a reasoning
        # that ends on ' and an answer that starts with ' split at the tag, as the release's parser does (the first </think>)
        cases = {"The word is 'yes'</think>'Yes' is right.": ("The word is 'yes'", "'Yes' is right.", []),
                 "It said 'no'</think>'No' was its answer.\n": ("It said 'no'", "'No' was its answer.\n", [])}
        for text, want in cases.items():
            self.check_text(text, True, want)
            if enc is not None:
                ref = enc.parse_message_from_completion_text(text + EOS, thinking_mode="thinking")
                self.assertEqual((ref["reasoning_content"], ref["content"]), want[:2], text)

    def test_call_inside_the_reasoning_is_rescued(self):
        # as V4: the model may leave its thinking by writing the block
        text = f'need weather\n\n<{D} calls>\n<{D} invoke name="get_weather">\n' \
               f'<{D} parameter name="city" string="true">Rome</{D} parameter>\n</{D} invoke>\n</{D} calls>'
        p = FP.make_parser("deepseek41", True, PARSER_TOOLS, stream_tools=True)
        evs = p.feed(text) + p.finish()
        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                         [("get_weather", {"city": "Rome"})])
        self.assertEqual(p.rescued, 1)

    def test_a_cut_inside_a_call_sends_no_markup(self):
        # moex (upstream Xolotl #106): a turn cut (max tokens, cancelled, failed) anywhere in a call's markup sends none of it
        # as the answer; a call announced before the cut stays unfinished (#211); a natural stop keeps the held text as before
        head = "r</think>answer"
        text = (head + f'\n\n<{D} calls>\n<{D} invoke name="lookup">\n'
                f'<{D} parameter name="query" string="true">value</{D} parameter>\n</{D} invoke>\n</{D} calls>')
        whole = text.index(f'</{D} invoke>') + len(f'</{D} invoke>')   # streamed, the call is reported as it ends
        for k in range(len(head), len(text)):
            for stream in (True, False):
                for reason in ("length", "cancel", "error"):
                    with self.subTest(cut=k, stream=stream, reason=reason):
                        p = FP.make_parser("deepseek41", True, PARSER_TOOLS, stream_tools=stream)
                        evs = p.feed(text[:k])
                        evs += p.finish(reason)
                        content = "".join(e.text for e in evs if e.kind == "content")
                        self.assertNotIn(D, content)
                        self.assertTrue("answer\n\n".startswith(content) and content.startswith("answer"), content)
                        self.assertEqual([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
                                         [("lookup", {"query": "value"})] if stream and k >= whole else [])
        p = FP.make_parser("deepseek41", True, PARSER_TOOLS, stream_tools=True)
        evs = p.feed(head + f'\n\n<{D} calls>\n<{D} inv') + p.finish()
        self.assertIn(f'<{D} calls>', "".join(e.text for e in evs if e.kind == "content"))

    def test_forced_call(self):
        forced = SPEC.forced_call({"type": "function", "function": {"name": "get_weather"}}, PARSER_TOOLS)
        self.assertEqual(forced, f'\n\n<{D} calls>\n<{D} invoke name="get_weather">\n')
        # the template's own text for such a call starts with it
        turn = render([U("go"), A("", tool_calls=[call("get_weather", {"city": "Rome"})])], gen=False)
        self.assertTrue(turn[len(render([U("go")])):].startswith(forced))
        p = FP.make_parser("deepseek41", False, PARSER_TOOLS, stream_tools=True)
        evs = p.feed(forced)
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_start"], ["get_weather"])
        evs += p.feed(f'<{D} parameter name="city" string="true">Rome</{D} parameter>\n</{D} invoke>\n</{D} calls>')
        evs += p.finish()
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"city": "Rome"}])
        self.assertEqual(SPEC.forced_call("required", PARSER_TOOLS), f'\n\n<{D} calls>\n<{D} invoke name="')
        self.assertIsNone(SPEC.forced_call("auto", PARSER_TOOLS))


if __name__ == "__main__":
    unittest.main(verbosity=1)
