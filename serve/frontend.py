"""serve/frontend.py - plan v0.3 P8: the request/response layer between API clients and the engine.

Everything here is text: no model, no GPU. The engine consumes token ids and produces text deltas; this module

  * normalizes OpenAI Chat Completions and Anthropic Messages requests into the chat template's message form,
  * renders the model's own Jinja chat template (pack `tokenizer/chat_template.jinja`) exactly as Hugging Face
    does (checked against the pack's `chat_golden.json`, 10 cases incl. thinking and tools),
  * parses the streamed output incrementally into reasoning (`<think>...</think>`), content and tool calls in
    the template's XML form (`<tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>`),
    never emitting a partial tag to the client.

Design note: plan v0.3 describes a single C++ process. The frontend is Python first because the template is Jinja
and the formats change often; the engine boundary is token ids in, text deltas out, which keeps a later C++ port
(or a pybind wrapper) a local change.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ template
class TemplateRequestError(jinja2.exceptions.TemplateError, ValueError):
    """The template refused the request's messages (e.g. "No user query found in messages."): a ValueError, so the
    client gets a 400 with the template's message instead of a dropped connection (#365)."""


class ChatTemplate:
    """The model's chat template, rendered with the same Jinja settings as transformers' apply_chat_template."""

    def __init__(self, path: str | Path):
        def raise_exception(message):
            raise TemplateRequestError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        self.source = Path(path).read_text(encoding="utf-8")
        self.template = env.from_string(self.source)
        self.caps = self._detect_caps()

    keep_reasoning_turns = False   # moex: a family whose encoder renders a turn of reasoning alone (FamilySpec.reasoning_turns)

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        # Do not use completed empty assistant turns as examples for the next reply. Keep the final message.
        # (#843.  STRATA_KEEP_EMPTY_TURNS=1 renders them as before.)
        messages = [m for i, m in enumerate(messages) if _rendered(messages, i, self.keep_reasoning_turns)]
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **kwargs)

    def _detect_caps(self) -> dict[str, bool]:
        """llama.cpp's capability names, checked at load time against this template and Strata's tool-call format.
        These are rendering hints, not a guarantee that the model will follow a request."""
        def render(messages, tools=None):
            try:
                return self.render(messages, tools=tools)
            except Exception as exc:                # noqa: BLE001 - a hint only: never stop the server for a probe
                # A template may reject a role or feature (a custom one may fail in any way): the feature is off,
                # discovery still answers for the others, and the start-up goes on.
                LOGGER.debug("chat template capability probe failed: %s", exc, exc_info=True)
                return ""

        user = {"role": "user", "content": "strata_caps_user"}
        tools = [{"name": f"strata_caps_call_{i}", "description": "strata_caps_description",
                  "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}}
                 for i in range(2)]
        tool_prompt = render([user], tools)

        def calls_supported(count):
            calls = [{"name": f"strata_caps_call_{i}", "arguments": {"arg": f"strata_caps_arg_{i}"}}
                     for i in range(count)]
            messages = [user, {"role": "assistant", "content": "",
                               "tool_calls": [{"function": call} for call in calls]}]
            replies = [f"strata_caps_result_{i}" for i in range(count)]
            messages += [{"role": "tool", "content": reply} for reply in replies] + [user]
            prompt = render(messages, tools)
            # Instructions include example XML and literal tag names, which are not model output. Parse only
            # complete calls to our probe functions, using the same body parser as OutputParser.
            bodies = re.findall(r"<tool_call>\s*(<function=strata_caps_call_\d+>.*?</function>)\s*</tool_call>",
                                prompt, re.S)
            try:
                parsed = [parse_tool_call(body) for body in bodies]
            except ValueError:
                return False
            return [{"name": call.name, "arguments": call.arguments} for call in parsed] == calls \
                and all(reply in prompt for reply in replies)

        history = [user, {"role": "assistant", "content": "strata_caps_answer",
                          "reasoning_content": "strata_caps_reasoning"}, user]
        return {"supports_tools": all(s in tool_prompt for s in
                                      ("strata_caps_call_0", "strata_caps_description", "<tool_call>", "<function=")),
                "supports_tool_calls": calls_supported(1),
                "supports_system_role": "strata_caps_system" in render(
                    [{"role": "system", "content": "strata_caps_system"}, user]),
                "supports_parallel_tool_calls": calls_supported(2),
                "supports_preserve_reasoning": "strata_caps_reasoning" in render(history)}


# ------------------------------------------------------------------------------------------------ requests
# The content-part types the readers take as text (a part with no type that has a "text" field reads as text too)
TEXT_PARTS = ("text", "input_text", None)


def _text_of(content, join: str = "") -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return join.join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in
                     TEXT_PARTS)


IMAGE_PARTS = ("image_url", "input_image", "image")

# Thinking levels.  The model's template knows low, medium and xhigh (its default; "high" means xhigh), and
# enable_thinking=false for none.  Clients spell these many ways; everything maps onto those four.
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "maximum": "xhigh"}


def effort_kwargs(value, efforts: dict | None = None) -> dict:
    """A reasoning effort as given by a client -> the template's kwargs.  Unknown values are a 400, not a crash.
    `efforts`: a family's own table in place of EFFORT (Dialect.efforts)."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    table = efforts or EFFORT
    key = str(value).strip().lower()
    if key not in table:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium or high")
    level = table[key]
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level}


@dataclass(frozen=True)
class Dialect:
    """How a family's template takes a client's messages, where it differs from every other family's (moex; the audit of
    2026-10-19).  The default is the conversion every template has always had; DeepSeek-V4.1's (serve/families.py) is the
    release's encoder's (encoding/encoding.py), whose template reads what the default drops:

      * call_ids - a tool message keeps its tool_call_id and a call its id: the encoder puts a turn's tool results in the
        order of the calls before them (sort_tool_results_by_call_order), which the template does by those ids;
      * late_system - a system message after the first stays one (the encoder writes it as <｜System｜>...) instead of
        becoming a user turn (_late_system_to_user);
      * text_join - a message's text parts joined by this (the encoder: a blank line; the default: nothing);
      * efforts - the client's effort words -> the request's levels (the default: EFFORT, where high is xhigh);
      * thinking_switches - the thinking switches as the release's servers read them (DSV41_BACKLOG #45): chat_template_kwargs'
        `thinking` (vLLM's V4.1 switch) exactly as `enable_thinking`, Anthropic's thinking "enabled" without a budget and
        "adaptive" as on, and a request that turns thinking on without naming a level gets the shared or the config's
        (Service.with_shared);
      * response_format - a request's response_format goes on the first system message, where the template writes the
        release's own section for it (`## Response Format:` and the format as JSON), in place of the generic directive
        (serve/structured.py prepare_format, DSV41_BACKLOG #56);
      * reasoning_field - a past assistant turn's reasoning is also read from `reasoning` when it has no reasoning_content
        (vLLM's name for it, which its V4.1 renderer maps; #56);
      * inline_think - an assistant message whose content opens with a whole <think>...</think> block has that block as its
        reasoning when it has none of its own (_inline_think; #56);
      * think_split - (open, close): the family's own think tags, an assistant message that carries no reasoning of its own
        and whose text holds the close tag split as ERNIE-5.5's encoder splits it (_split_think) - before the #537 marking,
        which would leave the tags as text the template cannot split on;
      * thoughts_field - a past assistant turn's reasoning is read from `thoughts` first, even empty (ERNIE-5.5's encoder);
      * parallel_option - a request that allows one call (one_call_only: `parallel_tool_calls: false`, Anthropic's
        `disable_parallel_tool_use`) reaches the template as parallel_tool_calls False and its reply keeps its first call
        only (FirstCallOnly), as ERNIE-5.5's server constrains it to one."""
    call_ids: bool = False
    late_system: bool = False
    text_join: str = ""
    efforts: dict | None = None
    thinking_switches: bool = False
    response_format: bool = False
    reasoning_field: bool = False
    inline_think: bool = False
    think_split: tuple = ()
    thoughts_field: bool = False
    parallel_option: bool = False

    @property
    def switches(self) -> tuple:
        """The chat_template_kwargs keys that turn thinking on or off."""
        return ("enable_thinking", "thinking") if self.thinking_switches else ("enable_thinking",)

    def thinking_unleveled(self, thinking) -> bool:
        """Anthropic's `thinking` turns thinking on without naming a level - {"type": "enabled"} without budget_tokens, or
        {"type": "adaptive"} - and this dialect reads it so (thinking_switches; the default reads neither as on)."""
        return self.thinking_switches and isinstance(thinking, dict) and (
            thinking.get("type") == "adaptive" or (thinking.get("type") == "enabled" and not thinking.get("budget_tokens")))


DEFAULT_DIALECT = Dialect()


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K low, under 8K medium, else high."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "medium" if n < 8192 else "xhigh"}


def _has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in content)


def _rendered(messages: list, i: int, keep_reasoning: bool = False) -> bool:
    """#843: whether ChatTemplate.render keeps messages[i] - anything but a completed assistant turn with nothing in
    it (no text, no picture, no call; with `keep_reasoning`, the family's reasoning_turns, no reasoning either), and the
    last message always (STRATA_KEEP_EMPTY_TURNS=1: every message)."""
    m = messages[i]
    return (i == len(messages) - 1 or os.environ.get("STRATA_KEEP_EMPTY_TURNS") == "1" or
            not (isinstance(m, dict) and m.get("role") == "assistant" and not _text_of(m.get("content")).strip()
                 and not _has_image(m.get("content")) and not m.get("tool_calls")
                 and not (keep_reasoning and m.get("reasoning_content"))))


def _image_source(part: dict) -> str:
    """An image part's source as one string: a data: URL, an http(s) URL or a local file path.
    OpenAI: {"type": "image_url", "image_url": {"url": ...}} (or "image_url": "..."), Responses-style
    {"type": "input_image", "image_url": ...}; Anthropic: {"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": ...}} or {"source": {"type": "url", "url": ...}}."""
    if part.get("type") == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _parts_of(content, join: str = ""):
    """Message content for the template: a string when there is no image (unchanged behaviour), otherwise the
    template's list form - text items and image items, in order - whose image items carry their source."""
    if not _has_image(content):
        return _text_of(content, join)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in IMAGE_PARTS:
            items.append({"type": "image", "source": _image_source(part)})
        elif part.get("type") in TEXT_PARTS and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


# The blocks a messages request may carry: text, the image parts, thinking, tool_use and tool_result.
# redacted_thinking is allowed only to be dropped without an error: another server's encrypted reasoning, which
# carries nothing this server can read - the same reason responses.py's _reasoning_text gives for another server's
# encrypted_content.
ANTHROPIC_BLOCKS = ("text",) + IMAGE_PARTS + ("thinking", "tool_use", "tool_result", "redacted_thinking")


def _check_parts(content, path: str, kinds):
    """A request's content parts (on chat) or content blocks (on messages), looked at while the request is converted.
    A part whose type this server does not read (a file, audio, a document, Anthropic's tool_reference,
    search_result, server_tool_use / web_search_tool_result blocks, ...) is left out of the prompt exactly as it always
    was, with a debug line: real clients (Claude Code, Codex, opencode, Cline, Continue, Open WebUI) send such parts
    next to text the model can answer from, so refusing them would break requests that work.  A "refusal" part is read
    as its text, as /v1/responses does.  Content that is not a list, and a part that is not an object, passes through
    unchanged."""
    if not isinstance(content, list):
        return content
    out = content
    for j, part in enumerate(content):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "refusal":
            if out is content:
                out = list(content)
            out[j] = {"type": "text", "text": part.get("refusal") or ""}
        elif kind not in kinds:
            LOGGER.debug("%s[%d].type: content part of type %r is not read by this server; left out of the prompt",
                         path, j, kind)
    return out


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the rendered conversation, in prompt order (the template renders one
    <|vision_start|><|image_pad|><|vision_end|> per image item, message by message)."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


def tool_results_in_call_order(messages: list[dict], dialect: Dialect = DEFAULT_DIALECT,
                               keep_reasoning: bool = False) -> list[dict]:
    """moex (DSV41_BACKLOG #44): with Dialect.call_ids, `messages` with each turn's tool results in the order of the
    calls before them, exactly as DeepSeek-V4.1's encoder puts them (encoding.py: merge_tool_messages, then
    sort_tool_results_by_call_order) and its template does (deepseek41.jinja, whose own sort then moves nothing).  The
    pictures are collected after it (images_of; the release: "Collect images only after tool results are in their final
    prompt order"): in message order, results that came in another order got each other's pictures.

    A turn is a run of tool and user messages as the template merges them - a tool message joins the turn being built, a
    user message one without a task - among the messages ChatTemplate.render keeps (`keep_reasoning`: its
    keep_reasoning_turns; a turn it leaves out ends no run).  The order is that of the last assistant message with calls
    (their ids; a call's index), an id it does not know sorts as 0, and the sort is stable.  The same list when nothing
    moves, and always without call_ids (every other family)."""
    if not dialect.call_ids:
        return messages
    out, order, turn, task = list(messages), {}, None, None

    def flush(turn):
        results = [i for i in turn if messages[i].get("role") == "tool"]
        if len(results) > 1 and order:
            for at, i in zip(results, sorted(results, key=lambda i: order.get(messages[i].get("tool_call_id", ""), 0))):
                out[at] = messages[i]

    for i, m in enumerate(messages):
        if not _rendered(messages, i, keep_reasoning):
            continue
        role = m.get("role")
        if turn is not None and (role == "tool" or role == "user" and task is None):
            turn.append(i)
            continue
        if turn is not None:
            flush(turn)
            turn = None
        if role in ("user", "tool"):
            turn, task = [i], (m.get("task") if role == "user" else None)
        elif role == "assistant" and m.get("tool_calls"):
            order = {}
            for k, c in enumerate(m["tool_calls"]):
                fn = c.get("function") if isinstance(c, dict) else None
                cid = (c.get("id") if isinstance(c, dict) else None) or (fn.get("id") if isinstance(fn, dict) else None)
                if cid:
                    order[cid] = k
    if turn is not None:
        flush(turn)
    return out if any(a is not b for a, b in zip(out, messages)) else messages


# #537: a literal <think> / </think> inside a message's text is plain text, not the model's reasoning markers.  The
# tokenizer matches those two strings as their special tokens everywhere (GGUF token type 4, as llama.cpp does), so
# a user quoting "</think>" used to hand the model a real end-of-reasoning token.  Before the template is rendered
# they are swapped for these private-use characters, and the server encodes the spans they mark as ordinary text.
THINK_TAGS = {"<think>": "\U000F0E01", "</think>": "\U000F0E02"}
# #554: the vision markers the same way.  They are special tokens only where the template writes them for an image
# item; the same strings in a message's text (an agent reading chat_template.jinja, a tool result quoting it) became
# the same special ids, so text with <|vision_start|><|image_pad|> before a picture took that picture's embeddings
VISION_TAGS = {"<|vision_start|>": "\U000F0E03", "<|image_pad|>": "\U000F0E04", "<|vision_end|>": "\U000F0E05",
               "<|video_pad|>": "\U000F0E06"}
# #2026: `</tool_call>` the same way.  A user asking for the string verbatim handed the model the real end-of-call token
# (type 4, matched everywhere); it copied that token into its answer and the reply came back empty.  An assistant
# message keeps its own: a client replaying the model's raw call text sends the tokens the model wrote.
TOOL_LITERAL_TAGS = {"</tool_call>": "\U000F0E07"}
LITERAL_TAGS = {**THINK_TAGS, **VISION_TAGS, **TOOL_LITERAL_TAGS}
THINK_MARKS = {v: k for k, v in LITERAL_TAGS.items()}
CONTROL_MARK0 = 0xF0E10      # the control tokens' marks start here, clear of the two sets above


def literal_tags(controls) -> dict[str, str]:
    """LITERAL_TAGS plus a mark for each control token's text (`controls`: the tokenizer's CONTROL literals,
    <|im_start|>, <|im_end|>, <|endoftext|>, ...).  The rendered prompt is encoded with those literals parsed, so one
    written inside a message - a file an agent reads, a pasted chat template - opened or ended a turn there.  Longest
    first, as the tokenizer matches them: a literal inside a longer one is not marked before it."""
    tags = {**LITERAL_TAGS, **{c: chr(CONTROL_MARK0 + k) for k, c in enumerate(c for c in controls
                                                                                if c not in LITERAL_TAGS)}}
    return dict(sorted(tags.items(), key=lambda t: -len(t[0])))


def _mark(text: str, tags: dict[str, str] = LITERAL_TAGS) -> str:
    for tag, mark in tags.items():
        text = text.replace(tag, mark)
    return text


def _mark_deep(v, tags: dict[str, str] = LITERAL_TAGS):
    if isinstance(v, str):
        return _mark(v, tags)
    if isinstance(v, dict):
        return {k: _mark_deep(x, tags) for k, x in v.items()}
    if isinstance(v, list):
        return [_mark_deep(x, tags) for x in v]
    return v


def _has_tag(v, tags: dict[str, str] = LITERAL_TAGS) -> bool:
    if isinstance(v, str):
        return any(tag in v for tag in tags)
    if isinstance(v, dict):
        return any(_has_tag(x, tags) for x in v.values())
    if isinstance(v, list):
        return any(_has_tag(x, tags) for x in v)
    return False


def _has_tag_m(v, tags, memo, variant) -> bool:   # moex: _has_tag through a LiteralMemo (serve/prompt_memo.py)
    if isinstance(v, str):
        return memo.has_tag(v, tags, variant)
    if isinstance(v, dict):
        return any(_has_tag_m(x, tags, memo, variant) for x in v.values())
    if isinstance(v, list):
        return any(_has_tag_m(x, tags, memo, variant) for x in v)
    return False


def _mark_deep_m(v, tags, memo, variant):           # moex: _mark_deep through a LiteralMemo
    if isinstance(v, str):
        return memo.mark(v, tags, variant)
    if isinstance(v, dict):
        return {k: _mark_deep_m(x, tags, memo, variant) for k, x in v.items()}
    if isinstance(v, list):
        return [_mark_deep_m(x, tags, memo, variant) for x in v]
    return v


def mark_think_literals(messages: list[dict], tools: list[dict] | None, tags: dict[str, str] = LITERAL_TAGS,
                        assistant_keeps=TOOL_LITERAL_TAGS, memo=None):
    """#537: (messages, tools) with every literal <think> / </think> (#554: and vision marker) in their text swapped for
    LITERAL_TAGS' marks (or `tags`': literal_tags() adds the control tokens' texts), and
    whether there was one (None: no change, the same objects back - a prompt without them renders as it always did).
    An assistant message whose content opens with a whole <think>...</think> block (clients that send the reasoning
    inline) keeps that one block as the model's markers, as before.  `assistant_keeps`: the tags an assistant message
    keeps as written (#2026: Qwen's `</tool_call>`); a model family passes none - its own call tags (<tool_call>,
    <arg_key>, ...) are all marked, so its `</tool_call>` alone becoming the real token would leave raw call text half
    text, half tokens, and turn a `</tool_call>` quoted inside a past call's argument into the end of that call.
    `memo` (moex, serve/prompt_memo.LiteralMemo bound to `tags`): the per-string tests and replacements remembered across
    requests - the same results."""
    if memo is not None and memo.tags is not tags:
        memo = None
    if memo is None:
        has, deep, mark1 = (lambda v, t, w: _has_tag(v, t)), (lambda v, t, w: _mark_deep(v, t)), (lambda s, t, w: _mark(s, t))
    else:
        has = lambda v, t, w: _has_tag_m(v, t, memo, w)                     # noqa: E731
        deep = lambda v, t, w: _mark_deep_m(v, t, memo, w)                  # noqa: E731
        mark1 = memo.mark
    if not has(messages, tags, "all") and not has(tools, tags, "all"):
        return messages, tools, False
    out = []
    own = {t: v for t, v in tags.items() if t not in assistant_keeps}      # what an assistant message keeps as written
    own_key = ("own", tuple(assistant_keeps))
    for m in messages:
        m = dict(m)
        content = m.get("content")
        mine, which = (own, own_key) if m.get("role") == "assistant" else (tags, "all")
        for k, v in m.items():
            if k != "role":
                m[k] = deep(v, mine, which)
        if m.get("role") == "assistant" and isinstance(content, str) and content.lstrip().startswith("<think>") \
                and "</think>" in content:
            i, j = content.index("<think>") + len("<think>"), content.index("</think>")
            m["content"] = (content[:i] + mark1(content[i:j], own, own_key) + "</think>" +
                            mark1(content[j + len("</think>"):], own, own_key))
        out.append(m)
    return out, deep(tools, tags, "all"), True


_THINK_MARK_RE = re.compile("|".join(THINK_MARKS))


def _char_ranges(chars) -> str:
    """moex: a regex class of single characters as runs of consecutive code points ([a-c...]): the same set as listing
    them, without re's per-character test of a long list (slow past the BMP)."""
    cps, parts, i = sorted({ord(c) for c in chars}), [], 0
    while i < len(cps):
        j = i
        while j + 1 < len(cps) and cps[j + 1] == cps[j] + 1:
            j += 1
        parts.append(re.escape(chr(cps[i])) if i == j else re.escape(chr(cps[i])) + "-" + re.escape(chr(cps[j])))
        i = j + 1
    return "[" + "".join(parts) + "]"


def unmark_think_literals(prompt: str, tags: dict[str, str] = LITERAL_TAGS) -> tuple[str, list[tuple[int, int]]]:
    """The rendered prompt with THINK_TAGS' marks turned back into the tags' text, and the (start, end) spans of those
    tags in it: the server encodes them as ordinary text (the tokenizer's encode_plain_spans)."""
    marks = THINK_MARKS if tags is LITERAL_TAGS else {v: k for k, v in tags.items()}
    # moex: every mark is one character (literal_tags: consecutive code points), so a class of their ranges finds the same
    # matches as the alternation - which, over literal_tags' ~1,290 marks, took 5.5 s to scan an ~880K-token prompt
    # (2026-11-04); the ranges, ~20 ms
    if tags is LITERAL_TAGS:
        mark_re = _THINK_MARK_RE
    elif all(len(k) == 1 for k in marks):
        mark_re = re.compile(_char_ranges(marks))
    else:
        mark_re = re.compile("|".join(map(re.escape, marks)))
    out, spans, pos, n = [], [], 0, 0
    for m in mark_re.finditer(prompt):
        out.append(prompt[pos:m.start()])
        n += m.start() - pos
        tag = marks[m.group(0)]
        out.append(tag)
        spans.append((n, n + len(tag)))
        n += len(tag)
        pos = m.end()
    out.append(prompt[pos:])
    return "".join(out), spans


def _inline_think(out: dict) -> dict:
    """moex (DSV41_BACKLOG #56; Dialect.inline_think): an assistant message whose content opens with a whole <think>...</think>
    block - a client that sends the reasoning inline - has that block as its reasoning_content when it has none of its own,
    and the rest as its content, both as written.  A template that writes a past turn's reasoning itself (DeepSeek-V4.1's)
    rendered such a turn "<think></think><think>R</think>answer"."""
    content = out.get("content")
    if out.get("role") != "assistant" or out.get("reasoning_content") or not isinstance(content, str):
        return out
    head = content.lstrip()
    if head.startswith("<think>") and "</think>" in head:
        end = head.index("</think>")
        if end > len("<think>"):
            out["reasoning_content"] = head[len("<think>"):end]
        out["content"] = head[end + len("</think>"):]
    return out


def _reasoning_given(m: dict, dialect: Dialect) -> bool:
    """moex (Dialect.think_split): the client's message carries its own reasoning - a string in a field the family reads,
    even an empty one (ERNIE-5.5's encoder then splits nothing: isinstance(reasoning, str))."""
    return (isinstance(m.get("reasoning_content"), str) or (dialect.thoughts_field and isinstance(m.get("thoughts"), str))
            or (dialect.reasoning_field and isinstance(m.get("reasoning"), str)))


def _split_think(out: dict, open_tag: str, close_tag: str) -> dict:
    """moex (Dialect.think_split): an assistant message without reasoning of its own whose text holds the family's close
    tag, split as ERNIE-5.5's encoder splits it (encoding_ernie.py _Renderer.assistant): the reasoning from after the last
    open tag before the first close tag (or from the start) up to that close tag, the content after the LAST close tag -
    both as written, any tag left inside them text (the #537 marking), as the encoder's slices are.  Run before the
    marking: marked, the tags are text the template cannot split on, and the turn rendered as an empty think block
    followed by the tags as text."""
    content = out.get("content")
    if out.get("role") != "assistant" or not isinstance(content, str) or close_tag not in content:
        return out
    first = content.index(close_tag)
    start = content.rfind(open_tag, 0, first)
    out["reasoning_content"] = content[start + len(open_tag) if start >= 0 else 0:first]
    out["content"] = content[content.rindex(close_tag) + len(close_tag):]
    return out


def one_call_only(req: dict) -> bool:
    """moex (Dialect.parallel_option): the request allows at most one tool call - OpenAI's and the Responses API's
    `parallel_tool_calls: false`, Anthropic's `tool_choice.disable_parallel_tool_use: true`."""
    choice = req.get("tool_choice")
    return req.get("parallel_tool_calls") is False or (
        isinstance(choice, dict) and choice.get("disable_parallel_tool_use") is True)


def _late_system_to_user(messages: list[dict], dialect: Dialect = DEFAULT_DIALECT) -> list[dict]:
    """The chat template takes a system message only at the start ("System message must be at the beginning").
    Clients also send them mid-conversation - Claude Code's hook context as {"role": "system"} after the first user
    turn, some OpenAI clients a late "developer" message (issue #56) - so those become user messages, in place:
    merging them into the first one would change the prompt's start and cost the conversation cache every turn.
    A family whose template writes them (Dialect.late_system: DeepSeek-V4.1's <｜System｜>) gets them as they are."""
    if dialect.late_system:
        return messages
    return [dict(m, role="user") if m.get("role") == "system" and i > 0 else m for i, m in enumerate(messages)]


def _object_list(value, name: str) -> list[dict]:
    """#460: a request's "messages" (or a message's "tool_calls") as a list of objects.  Some clients send the array
    double-encoded, as a JSON string, which used to be iterated character by character and crashed on m.get: such a
    string is decoded.  Anything that is still not a list of objects is a ValueError, which the server answers with
    a 400 naming the field.  None (or no field) is an empty list, as a missing field always was."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(f"{name} must be a list of objects (a string was sent that is not JSON)") from None
    if not isinstance(value, list) or not all(isinstance(m, dict) for m in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


_TOOL_ORDERS: "OrderedDict[frozenset, tuple]" = OrderedDict()   # a set of tool names -> the order it first came in
_TOOL_ORDERS_LOCK = threading.Lock()
_TOOL_ORDERS_MAX = 64


def _stable_tool_order(tools: list[dict], name_of) -> list[dict]:
    """The same set of tools comes back in the order it was first seen.  Clients send their tools in the order their
    MCP servers finished registering, which can differ from one request (or one client restart) to the next, and the
    template renders the tool list before the system prompt: one swap moves the shared prefix back thousands of
    tokens and the whole conversation is read again (opencode issue 23571).  A request's first sight of a tool set
    is untouched (so its prompt is byte for byte what the client sent); later requests with the same names in
    another order are put back in that first order.  Duplicate names, or a single tool: left as sent."""
    if len(tools) < 2:
        return tools
    names = [name_of(t) for t in tools]
    if len(set(names)) != len(names):
        return tools
    key = frozenset(names)
    with _TOOL_ORDERS_LOCK:
        order = _TOOL_ORDERS.get(key)
        if order is None:
            _TOOL_ORDERS[key] = tuple(names)
            while len(_TOOL_ORDERS) > _TOOL_ORDERS_MAX:
                _TOOL_ORDERS.popitem(last=False)
            return tools
        _TOOL_ORDERS.move_to_end(key)
    pos = {n: k for k, n in enumerate(order)}
    return sorted(tools, key=lambda t: pos[name_of(t)])


def _tool_list(value, wrapper: str | None) -> list[dict]:
    """#592: a request's "tools" as a list of tool objects, each with a name - in the OpenAI shape
    {"type": "function", "function": {"name": ...}} (`wrapper` "function"; a bare {"name": ...} is still taken), or
    in the Anthropic shape {"name": ...} (`wrapper` None).  A value that is not (a string such as "auto", a list of
    names, an object without a name) is a ValueError - a 400 naming the field - where it used to take the request
    thread down with no reply at all.  No value (or an empty one) is no tools, as always.  A tool's schema -
    "parameters" in the OpenAI shape, "input_schema" in the Anthropic one - is an object when it is there, for the
    same reason: a string or a list passed and raised later instead, on the model's first call of that tool."""
    if not value:
        return []
    shape = ('{"type": "function", "function": {"name": ..., "parameters": {...}}}' if wrapper else
             '{"name": ..., "input_schema": {...}}')
    try:
        tools = _object_list(value, "tools")
    except ValueError:
        raise ValueError(f"tools must be a list of tool objects, each {shape}") from None
    key = "parameters" if wrapper else "input_schema"
    for i, t in enumerate(tools):
        fn = t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"tools[{i}] has no name: each tool must be {shape}")
        schema = fn.get(key)
        if schema is not None and not isinstance(schema, dict):
            # it reached parse_tool_call and the stream parser as the tool's schema, whose .get("properties") raised
            # AttributeError in the request thread - after the 200 and whatever the model had said before the call
            raise ValueError(f'tools[{i}] ({fn["name"]}): "{key}" must be an object (the JSON schema of its '
                             f"parameters), not {type(schema).__name__}")
    return _stable_tool_order(tools, lambda t: (t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t)["name"])


def tool_arguments(raw) -> dict:
    """A call's arguments from a client's history -> the mapping the template renders (both APIs).  A JSON object
    (as text or already parsed) is used as it is; empty is {}; anything else (cut-off or malformed JSON, a list, a
    number) is kept as text under "arguments" instead of refusing the whole conversation (#510).  moex: text that is
    JSON of JSON (a client that encoded the arguments twice) is read twice, as DeepSeek-V4.1's encoder reads it
    (encode_arguments_to_dsml) - the call the model wrote, where it used to land whole under "arguments"."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        value = raw
        for _ in range(2):
            try:
                value = json.loads(value)
            except ValueError:
                break
            if not isinstance(value, str):
                break
        return value if isinstance(value, dict) else {"arguments": raw}
    return {"arguments": json.dumps(raw, ensure_ascii=False)}


def openai_to_messages(req: dict, dialect: Dialect = DEFAULT_DIALECT) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs).  `dialect`: the family's (Dialect)."""
    messages = []
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        role = m.get("role")
        if role == "developer":
            role = "system"
        # an image is read in a user, tool or assistant message; the system prompt takes text only, as _text_of reads
        images = role in ("user", "tool", "assistant")
        content = _check_parts(m.get("content"), f"messages[{i}].content",
                               TEXT_PARTS + IMAGE_PARTS if images else TEXT_PARTS)
        out = {"role": role, "content": _parts_of(content, dialect.text_join) if images else
               _text_of(content, dialect.text_join)}
        if dialect.thoughts_field and role == "assistant" and isinstance(m.get("thoughts"), str):
            out["reasoning_content"] = m["thoughts"]            # moex: ERNIE-5.5's encoder reads `thoughts` first, even empty
        elif m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        elif dialect.reasoning_field and role == "assistant" and isinstance(m.get("reasoning"), str) and m["reasoning"]:
            out["reasoning_content"] = m["reasoning"]           # moex (DSV41_BACKLOG #56): vLLM's (and OpenRouter's) field
        if dialect.inline_think:
            _inline_think(out)
        if dialect.think_split and not _reasoning_given(m, dialect):
            _split_think(out, *dialect.think_split)
        if dialect.call_ids and role == "tool" and isinstance(m.get("tool_call_id"), str):
            out["tool_call_id"] = m["tool_call_id"]
        if m.get("tool_calls"):
            calls = []
            for c in _object_list(m["tool_calls"], "tool_calls"):
                fn = c.get("function", c)
                if not isinstance(fn, dict):
                    raise ValueError("tool_calls must be a list of objects (each with a \"function\" object)")
                call = {"function": {"name": fn.get("name"), "arguments": tool_arguments(fn.get("arguments"))}}
                if dialect.call_ids and isinstance(c.get("id"), str):
                    call["id"] = c["id"]
                calls.append(call)
            out["tool_calls"] = calls
        messages.append(out)
    tools = [t.get("function", t) if t.get("type") == "function" else t
             for t in _tool_list(req.get("tools"), "function")] or None
    kwargs = {}
    # OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
    kwargs.update(effort_kwargs(req.get("reasoning_effort") or reasoning.get("effort"), dialect.efforts))
    if reasoning.get("enabled") is False:   # moex (the audit of 2026-10-24): OpenRouter's {"enabled": false}, read nowhere before
        kwargs = {"enable_thinking": False}
    # the vLLM / llama.cpp convention: {"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
    # moex: `enable_thinking: true` is kept too - a no-op for Qwen (thinking is its default), the switch that turns it on
    # for a family whose template defaults to off (DeepSeek-V4: families.Deepseek4Spec.thinking_on).  moex (DSV41_BACKLOG
    # #45): a dialect with thinking_switches reads vLLM's V4.1 `thinking` the same way (it was ignored: with the config's
    # effort, {"thinking": false} rendered max thinking)
    switches = dialect.switches
    for k, v in (req.get("chat_template_kwargs") or {}).items():
        if k in switches and not v:
            kwargs = {"enable_thinking": False}
        elif k in switches and kwargs.get("enable_thinking") is not False:
            kwargs["enable_thinking"] = True
        elif k == "reasoning_effort" and kwargs.get("enable_thinking") is not False:
            kwargs.update(effort_kwargs(v, dialect.efforts))
    if dialect.parallel_option and one_call_only(req):
        kwargs["parallel_tool_calls"] = False      # moex: the template's option, as ERNIE-5.5's vLLM plugin passes it
    return _late_system_to_user(messages, dialect), tools, kwargs


BILLING_HEADER = "x-anthropic-billing-header:"


def pin_billing_stamp(system: str) -> str:
    """Claude Code starts its system prompt with `x-anthropic-billing-header: cc_version=2.1.170.bf4;
    cc_entrypoint=sdk-cli; cch=b145e;`.  cch changes on EVERY request and the version's 4th part on every session, so
    the prompt changed ~22K tokens in (after the tool list) on every turn and the conversation cache could only reuse
    up to its last 16K checkpoint: half of every agent prompt was read again.  Both stamps are pinned to f's, as
    llama.cpp does (ggml-org/llama.cpp#21793); only a header at the very start of the system text is touched, and
    only inside its first 160 characters."""
    if not system.startswith(BILLING_HEADER):
        return system
    s = list(system)
    cch = system.find("cch=", len(BILLING_HEADER))
    if 0 <= cch <= 160:
        v, end = cch + 4, system.find(";", cch + 4)
        if end > v and end - v <= 16:
            s[v:end] = "f" * (end - v)
    cv = system.find("cc_version=")
    if 0 <= cv <= 160:
        v, end = cv + len("cc_version="), system.find(";", cv)
        if v < end and end - v <= 64:
            parts = system[v:end].split(".")
            if len(parts) > 3:
                tail = v + len(".".join(parts[:3])) + 1
                s[tail:end] = "f" * (end - tail)
    return "".join(s)


def anthropic_to_messages(req: dict, think_unasked: bool = True,
                          dialect: Dialect = DEFAULT_DIALECT) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs).  `think_unasked`: a request
    without "thinking", an effort or a budget gets the template's default (it thinks), as through 0.1.31; False
    renders it without thinking (#278, the config's "anthropic_thinking": "on_request").  `dialect`: the family's: with
    call_ids a tool_use's id and a tool_result's tool_use_id become the OpenAI shape's call id and tool_call_id."""
    messages = []
    join = dialect.text_join
    system = req.get("system")
    if system:
        messages.append({"role": "system",
                         "content": pin_billing_stamp(_text_of(_check_parts(system, "system", TEXT_PARTS), join))})
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        content = _check_parts(m.get("content"), f"messages[{i}].content", ANTHROPIC_BLOCKS)
        if isinstance(content, str):
            out = {"role": m["role"], "content": content}
            if dialect.think_split:                     # no thinking block: the text is read as the encoder reads it
                _split_think(out, *dialect.think_split)
            messages.append(_inline_think(out) if dialect.inline_think else out)
            continue
        if m.get("role") == "user" and _has_image(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            messages.append({"role": "user", "content": _parts_of(content, join)})
            continue
        text, reasoning, calls, parts = [], [], [], []
        for j, block in enumerate(content or []):
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
                parts.append(block)
            elif kind in IMAGE_PARTS:
                parts.append(block)
            elif kind == "thinking":
                reasoning.append(block.get("thinking", ""))
            elif kind == "tool_use":
                call = {"function": {"name": block.get("name"), "arguments": tool_arguments(block.get("input"))}}
                if dialect.call_ids and isinstance(block.get("id"), str):
                    call["id"] = block["id"]
                calls.append(call)
            elif kind == "tool_result":
                # A tool's image (Claude Code's Read of a picture) reaches the encoder like a user's, as the OpenAI
                # path's tool messages already do; a text-only result is one string, as before.
                result = {"role": "tool", "content": _parts_of(_check_parts(
                    block.get("content"), f"messages[{i}].content[{j}].content", TEXT_PARTS + IMAGE_PARTS), join)}
                if dialect.call_ids and isinstance(block.get("tool_use_id"), str):
                    result["tool_call_id"] = block["tool_use_id"]
                messages.append(result)
        if text or calls or reasoning or parts:
            # an image sent beside tool results stays in this turn instead of being dropped
            out = {"role": m["role"], "content": _parts_of(parts, join) if _has_image(parts) else join.join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            if dialect.think_split and not reasoning:   # a thinking block is the turn's own reasoning: no split then
                _split_think(out, *dialect.think_split)
            messages.append(_inline_think(out) if dialect.inline_think else out)
    tools = [{"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
             for t in _tool_list(req.get("tools"), None)] or None
    kwargs = {}
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort:
        kwargs.update(effort_kwargs(effort, dialect.efforts))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    elif dialect.thinking_unleveled(thinking):
        # moex (DSV41_BACKLOG #45): "enabled" without a budget, or "adaptive", asks for thinking, which a template that
        # thinks only when asked (DeepSeek-V4.1) did not render.  The shared or the config's level comes in as output_config
        # (Service.with_shared, the branch above); without one the template's default level
        kwargs["enable_thinking"] = True
    elif thinking is None and not req.get("reasoning_budget_tokens") and not think_unasked:
        # Opt-in (the config's "anthropic_thinking": "on_request"; the default thinks as 0.1.31 did, since a
        # client that never asks would otherwise lose the thinking on every turn).  Anthropic's thinking is
        # opt-in there. Claude Code's helper calls (a session title, a topic check) ask for none
        # and allow a few dozen tokens, which the model otherwise spent thinking and answered with no text at all.
        # A config's reasoning_effort still applies: Service.with_shared sets output_config before this runs.  A
        # request that gives its own reasoning_budget_tokens (#123) asks for thinking, so it thinks as before.
        kwargs["enable_thinking"] = False
    if dialect.parallel_option and one_call_only(req):
        kwargs["parallel_tool_calls"] = False      # moex: disable_parallel_tool_use, the template's option as on the OpenAI path
    return _late_system_to_user(messages, dialect), tools, kwargs


# ------------------------------------------------------------------------------------------------ output parser
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_call"
    text: str = ""
    call: ToolCall | None = None


class FirstCallOnly:
    """moex (Dialect.parallel_option): a reply's events with every call after the first complete one left out - a request
    that allows one call (one_call_only) gets one, as ERNIE-5.5's server constrains its decoding to one invoke (vLLM's
    ERNIE tool parser: stop_after_first).  Text and reasoning pass as written.  `dropped`: the calls left out."""
    CALL_KINDS = ("tool_start", "tool_args", "tool_call")

    def __init__(self):
        self.done, self.dropped = False, 0

    def __call__(self, evs: list) -> list:
        out = []
        for ev in evs:
            if ev.kind in self.CALL_KINDS and self.done:
                self.dropped += ev.kind == "tool_call"
                continue
            if ev.kind == "tool_call":
                self.done = True
            out.append(ev)
        return out


THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"


PARAM_END = "</parameter>"
FUNC_START = "<function="
FUNC_END = "</function>"


def param_end(text: str, final: bool = False) -> int:
    """Where a parameter value in `text` ends: the first `</parameter>` followed (after whitespace) by the next
    `<parameter=` or `</function>` - the same text inside a value (a file that documents the call format, #210) is
    part of the value.  -1: none yet; -2: a candidate whose follower has not arrived (streaming; `final` accepts it)."""
    at = text.find(PARAM_END)
    while at >= 0:
        after = text[at + len(PARAM_END):].lstrip()
        if after.startswith(("<parameter=", FUNC_END)):
            return at
        if not after or "<parameter=".startswith(after) or FUNC_END.startswith(after):
            return at if final else -2
        at = text.find(PARAM_END, at + 1)
    return -1


def call_end(text: str) -> int:
    """Where a tool call's body ends (#210): the `</tool_call>` after the call's own `</function>`, found by walking
    its parameters with param_end, so a value may contain either tag.  -1: not complete yet.  A body that is not in
    the call format ends at the first `</tool_call>`, as before."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith("<function="):
        return text.find(CALL_END) if not "<function=".startswith(s) else -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        if s.startswith("<parameter="):
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            end = param_end(text[gt + 1:])
            if end < 0:
                return -1
            pos = gt + 1 + end + len(PARAM_END)
        elif s.startswith(FUNC_END):
            rest = text[pos + len(FUNC_END):]
            s = rest.lstrip()
            if s.startswith(CALL_END):
                return pos + len(FUNC_END) + len(rest) - len(s)
            return -1 if CALL_END.startswith(s) else text.find(CALL_END, pos)
        elif not s or "<parameter=".startswith(s) or FUNC_END.startswith(s):
            return -1
        else:
            return text.find(CALL_END, pos)


def tool_choice_of(tool_choice) -> tuple[str, str | None]:
    """A client's `tool_choice` in either API's shape -> ("auto" | "none" | "required" | "named", name).  OpenAI:
    "auto" / "none" / "required", {"type": "function", "function": {"name": N}} or the flat {"type": "function",
    "name": N}; Anthropic: {"type": "auto" | "any" | "none"} and {"type": "tool", "name": N}.  A value it does not
    know is ("unknown", None)."""
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        return (tool_choice, None) if tool_choice in ("auto", "none", "required") else ("unknown", None)
    if isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        if kind in ("auto", "none"):
            return kind, None
        if kind in ("any", "required"):
            return "required", None
        if kind in ("function", "tool"):
            fn = tool_choice.get("function")
            name = (fn.get("name") if isinstance(fn, dict) else None) or tool_choice.get("name")
            if isinstance(name, str) and name:
                return "named", name
    return "unknown", None


def forced_call(tool_choice, tools: list[dict] | None) -> str | None:
    """`tool_choice` -> the text that opens the call the reply must make, or None (the model decides).
    There is no grammar here: the server writes this opening itself, so the model can only go on with a call.
    "required" / Anthropic "any": any of the tools; a named function: that one.  A value it cannot honour (an
    unknown shape, a name that is not one of the tools, "required" with no tools) is logged and acts as "auto",
    not a 400: a client's odd choice must not stop its request.  "none" is handled by the caller (no tools)."""
    kind, name = tool_choice_of(tool_choice)
    names = {t.get("name") for t in tools or [] if isinstance(t, dict)}
    if kind in ("auto", "none"):
        return None
    if kind == "required" and len(names) == 1:     # one tool to call: name it, the model cannot invent another
        kind, name = "named", next(iter(names))
    if kind == "required" and names:
        return CALL_START + "\n<function="
    if kind == "named" and name in names:
        return CALL_START + f"\n<function={name}>\n"
    print(f"[strata] tool_choice {json.dumps(tool_choice)[:200]} is not supported here (or names no tool of the "
          "request): the model decides, as with \"auto\"", flush=True)
    return None


PARAM_START = "<parameter="


def function_end(text: str) -> int:
    """Where a `<function=NAME>...</function>` at the start of `text` ends (just past `</function>`), walking its
    parameters with param_end like call_end does; -1: not complete (or not in the call format)."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith(FUNC_START):
        return -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        if s.startswith(PARAM_START):
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            end = param_end(text[gt + 1:])
            if end < 0:
                return -1
            pos = gt + 1 + end + len(PARAM_END)
        elif s.startswith(FUNC_END):
            return pos + len(FUNC_END)
        else:
            return -1


_JSON_FENCE = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n\s*", re.I)


def json_tool_call(body: str, schemas: dict) -> ToolCall | None:
    """`{"name": N, "arguments": {...}}` (or "parameters", or the arguments as a JSON string) of a declared tool -> its
    call; anything else -> None.  #2037: the object may sit in a Markdown fence (```json ... ```)."""
    body = body.strip()
    m = _JSON_FENCE.match(body)
    if m:
        body = body[m.end():].rstrip()
        body = body[:-3] if body.endswith("```") else body
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or obj["name"] not in schemas:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if args is None:
        args = {}
    return ToolCall(name=obj["name"], arguments=args) if isinstance(args, dict) else None


def parse_tool_call(body: str, schema: dict | None = None) -> ToolCall:
    """`<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>...</function>` -> ToolCall. Values are JSON-decoded
    when the tool's schema says the parameter is not a string (or, without a schema, when they parse as JSON
    objects/arrays/numbers/booleans)."""
    body = body.strip()
    if not body.startswith("<function=") or ">" not in body:
        raise ValueError("malformed tool call: " + body[:80])
    name = body[len("<function="):body.index(">")].strip()      # `<function= capture>`: the model's stray space
    rest = body[body.index(">") + 1:]
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    args = {}
    while "<parameter=" in rest:
        rest = rest[rest.index("<parameter=") + len("<parameter="):]
        pname = rest[:rest.index(">")]
        rest = rest[rest.index(">") + 1:]
        end = param_end(rest, final=True)
        value = rest[:end] if end >= 0 else rest
        rest = rest[end + len(PARAM_END):] if end >= 0 else ""
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        declared = (props.get(pname) or {}).get("type")
        if declared == "string":
            args[pname] = value
        else:
            try:
                args[pname] = json.loads(value)
            except ValueError:
                args[pname] = value
    return ToolCall(name=name, arguments=args)


RCALL_MAX = 32768        # the most of a `<tool_call>` inside the reasoning that is held while waiting for its end (#804)


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 recover: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False
        self.schemas = {t.get("name"): t for t in tools or []}
        # stream_tools: a tool call is also reported while it is being written - "tool_start" (its name and id) as
        # soon as the name is known, then "tool_args" pieces of its JSON arguments (string parameters character by
        # character; other types whole, once complete) - before the final "tool_call".  Without it, a client sees
        # nothing until the call is complete, which for a large file write can be many minutes.
        self.stream_tools = stream_tools
        # #804/#1058: calls found inside the reasoning wait here as [raw text, ToolCall | None] until the turn shows
        # they were acts: only whitespace (or more calls) after them, then the end of the turn or `</think>`.  The
        # reasoning text before them is tracked (code fence, inline code, current line) to tell an act from a quote.
        self.pending: list[list] = []
        self.rescued = 0             # calls delivered from the reasoning
        self.refused = 0             # declared calls kept as reasoning text (quoted, or the turn was cut)
        self.fence, self.line, self.ticks = "", "", 0
        # recover (opt-in, the config's "tool_call_recovery"): a declared tool's call written next to the template's
        # form is the call - `<parameter=NAME>` as the opener, JSON inside <tool_call>, a bare <function=NAME> at the
        # start of a line outside code, or a second call in the same <tool_call> (which otherwise merges into one).
        self.recover = recover
        self.bare = False            # the call being read opened with <function= alone (no <tool_call> around it)
        self.batch = False           # the call being finished is followed by another in the same wrapper
        self._reset_scan()

    def _ok_at(self, p: int) -> bool:
        """self.buf[p] would open a call: at the start of a line, outside a code fence and inline code."""
        snap = (self.fence, self.line, self.ticks)
        self._track(self.buf[:p])
        ok = self._opener_ok()
        self.fence, self.line, self.ticks = snap
        return ok

    def _follower(self, after: str) -> str:
        """recover: what a `<tool_call>` followed (after whitespace) by `after` opens: "call" (`<function=`), "param"
        (a declared tool's name written as `<parameter=NAME>`), "json" (`{`), "wait" (not decided yet) or "text"."""
        if after.startswith(FUNC_START):
            return "call"
        if after.startswith(PARAM_START):
            gt = after.find(">")
            if gt < 0:
                return "wait" if "\n" not in after else "text"
            return "param" if after[len(PARAM_START):gt] in self.schemas else "text"
        if after.startswith("{"):
            return "json" if self.schemas else "text"
        if after.startswith("```") or (after and "```".startswith(after)):
            # #2037: the JSON call inside a Markdown fence (```json ... ```) in the explicit wrapper
            m = _JSON_FENCE.match(after)
            if m:
                rest = after[m.end():]
                if rest.startswith("{"):
                    return "json" if self.schemas else "text"
                return "wait" if not rest else "text"
            return "wait" if "\n" not in after else "text"
        if not after or FUNC_START.startswith(after) or PARAM_START.startswith(after):
            return "wait"
        return "text"

    def _bare_opener(self) -> tuple[int, bool]:
        """recover: the first `<function=` in self.buf that opens a call without the wrapper (start of a line, outside
        code, a declared tool) -> (position, decided); (-1, False) for none.  A name still arriving is undecided."""
        p = self.buf.find(FUNC_START)
        while p >= 0:
            if self._ok_at(p):
                gt = self.buf.find(">", p)
                name = self.buf[p + len(FUNC_START):gt if gt >= 0 else len(self.buf)]
                if gt < 0 and not any(c.isspace() for c in name):
                    return p, False
                if gt >= 0 and name in self.schemas:
                    return p, True
            p = self.buf.find(FUNC_START, p + 1)
        return -1, False

    @property
    def line(self) -> str:
        """Materialize the current line only for a decision, newline, or state snapshot."""
        if self._line_blocks is not None:
            self._line_blocks.extend(self._line_parts)
            self._line_parts = ["".join(self._line_blocks)]
            self._line_blocks = None
        elif len(self._line_parts) > 1:
            self._line_parts = ["".join(self._line_parts)]
        return self._line_parts[0] if self._line_parts else ""

    @line.setter
    def line(self, text: str):
        self._line_parts = [text] if text else []
        self._line_blocks = None

    def _track(self, text: str) -> str:
        """Follow the reasoning text that has gone out: the open code fence, the current line, and the backticks of
        the current paragraph.  Returns the text."""
        # Most token fragments stay on the current line. Fence and paragraph
        # bookkeeping happens only when a newline completes that line.
        if "\n" not in text:
            if text:
                self._line_parts.append(text)
                if len(self._line_parts) == 64:
                    # Release small fragment objects in bounded groups. Completed
                    # blocks are never recopied until the full line is requested.
                    if self._line_blocks is None:
                        self._line_blocks = []
                    self._line_blocks.append("".join(self._line_parts))
                    self._line_parts.clear()
            return text
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                line = self.line + part if self._line_parts or self._line_blocks else part
                self._line_parts.clear()
                self._line_blocks = None
                s = line.lstrip()
                if self.fence:
                    if s.startswith(self.fence * 3):
                        self.fence = ""
                elif s[:3] in ("```", "~~~") and (s[0] * 3) not in s[3:]:
                    self.fence = s[0]
                elif not s:
                    self.ticks = 0
                else:
                    self.ticks += line.count("`")
            else:
                if part:
                    self._line_parts.append(part)
        return text

    def _opener_ok(self) -> bool:
        """The reasoning text so far puts a `<tool_call>` at the start of a line, outside a code fence and outside
        inline code."""
        return not self.fence and not self.line.strip() and self.ticks % 2 == 0

    def _in_code(self) -> bool:
        """The text so far leaves the next character inside a code fence or inline code."""
        return bool(self.fence) or (self.ticks + self.line.count("`")) % 2 == 1

    def _quoted_think_end(self, i: int) -> bool | None:
        """#537: the `</think>` at self.buf[i] is the model quoting the tag while it thinks, not the end of the
        thinking: written right after a quote or backtick and followed by anything but a line break, as in `</think>`
        or "</think>...".  The end of the thinking is followed by a line break.  None while the character after the
        tag has not arrived."""
        before = self.buf[i - 1] if i else self.line[-1:]
        if before not in ('"', "'", "`"):
            return False
        after = self.buf[i + len(THINK_END):]
        if not after:
            return None
        return after[0] not in "\r\n"

    def _release(self, deliver: bool) -> list[Event]:
        """Settle the calls waiting in self.pending: events for real calls, or all of it back as reasoning text."""
        out = []
        for raw, call in self.pending:
            if call is not None and deliver:
                out.append(Event("tool_call", call=call))
                self.rescued += 1
            elif call is None or not deliver:
                if call is not None:
                    self.refused += 1
                if raw:
                    out.append(Event("reasoning", self._track(raw)))
        self.pending = []
        return out

    def _reset_scan(self):
        self.sp = 0                  # how much of self.buf (the call body) the scanner has consumed
        self.ss = "name"             # name -> between -> str|raw -> ... -> done
        self.scall = None            # the ToolCall being streamed (its id is reused by the final event)
        self.sfirst = True
        self.sval_started = False
        self.sdeclared = {}

    def _scan(self) -> list[Event]:
        """Advance the streaming view of the call body in self.buf (see stream_tools)."""
        out = []

        def args(s):
            if s:
                out.append(Event("tool_args", s, call=self.scall))
        while True:
            rest = self.buf[self.sp:]
            if self.ss == "name":
                a = rest.find("<function=")
                b = rest.find(">", a + 10) if a >= 0 else -1
                if b < 0:
                    return out
                name = rest[a + 10:b]
                self.scall = ToolCall(name=name, arguments={})
                props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
                self.sdeclared = {k: (v or {}).get("type") for k, v in props.items()}
                out.append(Event("tool_start", call=self.scall))
                args("{")
                self.sp += b + 1
                self.ss = "between"
            elif self.ss == "between":
                stripped = rest.lstrip()
                self.sp += len(rest) - len(stripped)
                if stripped.startswith("<parameter="):
                    b = stripped.find(">")
                    if b < 0:
                        return out
                    pname = stripped[11:b]
                    args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                    self.sfirst = False
                    self.sp += b + 1
                    if self.sdeclared.get(pname) == "string":
                        args('"')
                        self.ss, self.sval_started = "str", False
                    else:
                        self.ss = "raw"
                elif stripped.startswith("</function>"):
                    args("}")
                    self.sp += len("</function>")
                    self.ss = "done"
                else:
                    return out          # a tag still arriving (or trailing text): wait
            elif self.ss == "str":
                if not self.sval_started:
                    if not rest:
                        return out
                    if rest[0] == "\n":          # the value's leading newline is not part of it
                        self.sp += 1
                        rest = rest[1:]
                    self.sval_started = True
                end = param_end(rest)
                if end >= 0:
                    value = rest[:end]
                    if value.endswith("\n"):
                        value = value[:-1]
                    args(json.dumps(value)[1:-1] + '"')
                    self.sp += end + len(PARAM_END)
                    self.ss = "between"
                    continue
                # an undecided </parameter> (-2) is held from its start, like a tag still arriving
                safe = rest.find(PARAM_END) if end == -2 else len(rest) - self._hold(rest, (PARAM_END,))
                if safe > 0 and rest[safe - 1] == "\n":   # may be the trailing newline before </parameter>
                    safe -= 1
                if safe > 0:
                    args(json.dumps(rest[:safe])[1:-1])
                    self.sp += safe
                return out
            elif self.ss == "raw":
                end = param_end(rest)
                if end < 0:
                    return out
                value = rest[:end]
                if value.startswith("\n"):
                    value = value[1:]
                if value.endswith("\n"):
                    value = value[:-1]
                try:
                    v = json.loads(value)
                except ValueError:
                    v = value
                args(json.dumps(v, ensure_ascii=False))
                self.sp += end + len(PARAM_END)
                self.ss = "between"
            else:
                return out

    def _close_scan(self) -> list[Event]:
        """A streamed call that ended without a clean </function>: close its JSON so clients can still parse it."""
        out = []
        if self.scall is None or self.ss == "done":
            return out
        tail = ""
        if self.ss == "str":
            tail += '"'
        elif self.ss == "raw":
            tail += json.dumps(self.buf[self.sp:].strip("\n"))
        tail += "}"
        out.append(Event("tool_args", tail, call=self.scall))
        self.ss = "done"
        return out

    def _hold(self, text: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
        best = 0
        for tag in tags:
            for n in range(1, len(tag)):
                if text.endswith(tag[:n]):
                    best = max(best, n)
        return best

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while True:
            if self.state == "rcall":
                # #804: a `<tool_call>` inside the reasoning, with tools declared.  It is held whole (never streamed
                # as a call) until it ends: a declared name is then a tool call, anything else stays reasoning text.
                body = self.buf[len(CALL_START):]
                end, think = call_end(body), body.find(THINK_END)
                if end >= 0 and (think < 0 or think >= end):
                    call = None
                    try:
                        name = body[:end].strip()[len("<function="):].split(">", 1)[0].strip()
                        if name in self.schemas:
                            call = parse_tool_call(body[:end], self.schemas.get(name))
                    except ValueError:
                        pass
                    raw = self.buf[:len(CALL_START) + end + len(CALL_END)]
                    if call is not None or self.pending:
                        self.pending.append([raw, call])    # settled by what follows it (see __init__)
                    else:
                        out.append(Event("reasoning", self._track(raw)))
                    self.buf = body[end + len(CALL_END):]
                    self.state = "reasoning"
                elif think >= 0:                     # the thinking ended inside it: it never was a call
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf[:len(CALL_START) + think])))
                    self.buf = body[think:]
                    self.state = "reasoning"
                elif len(body) > RCALL_MAX:          # a tag in the prose that never closes: stop holding the thinking back
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf)))
                    self.buf = ""
                    self.state = "reasoning"
                else:
                    return out
            elif self.state == "reasoning":
                if self.pending:
                    # calls wait for what follows: more calls or whitespace keep them, `</think>` makes them acts,
                    # any other text means they were quoted
                    stripped = self.buf.lstrip()
                    if len(stripped) < len(self.buf):
                        self.pending[-1][0] += self.buf[:len(self.buf) - len(stripped)]
                        self.buf = stripped
                    if not self.buf:
                        return out
                    if self.buf.startswith(CALL_START):
                        self.state = "rcall"
                    elif self.buf.startswith(THINK_END):
                        out += self._release(True)
                    elif CALL_START.startswith(self.buf) or THINK_END.startswith(self.buf):
                        return out
                    else:
                        out += self._release(False)
                    continue
                i = self.buf.find(THINK_END)
                tool = self.buf.find(CALL_START) if self.schemas else -1
                if tool >= 0 and (i < 0 or tool < i):
                    if tool:
                        out.append(Event("reasoning", self._track(self.buf[:tool])))
                    if self._opener_ok():
                        self.buf = self.buf[tool:]
                        self.state = "rcall"
                    else:                            # mid-sentence, in a fence or in inline code: a quote
                        out.append(Event("reasoning", self._track(CALL_START)))
                        self.buf = self.buf[tool + len(CALL_START):]
                    continue
                if i < 0:
                    keep = self._hold(self.buf, (THINK_END, CALL_START) if self.schemas else (THINK_END,))
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self._track(self.buf[:len(self.buf) - keep])))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                quoted = self._quoted_think_end(i)
                if quoted is None:                   # held like a partial tag until the character after it is here
                    if i:
                        out.append(Event("reasoning", self._track(self.buf[:i])))
                        self.buf = self.buf[i:]
                    return out
                if quoted:                           # #537: a quoted `</think>` is reasoning text
                    out.append(Event("reasoning", self._track(self.buf[:i + len(THINK_END)])))
                    self.buf = self.buf[i + len(THINK_END):]
                    continue
                if i:
                    out.append(Event("reasoning", self._track(self.buf[:i])))
                self.buf = self.buf[i + len(THINK_END):]
                self.state, self.lead = "content", True
                self.fence, self.line, self.ticks = "", "", 0     # the answer's own text starts here
            elif self.state == "content":
                if self.lead:                                   # newlines right after </think> or a call
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                i = self.buf.find(CALL_START)
                b, decided = self._bare_opener() if self.recover and self.schemas else (-1, False)
                if b >= 0 and (i < 0 or b < i):
                    # recover: <function=NAME> alone at the start of a line - the call without its <tool_call>.
                    # Until the name has arrived it is held.
                    j = b
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0 and self.buf[:j].strip():
                        out.append(Event("content", self._track(self.buf[:j])))
                    if not decided:
                        self.buf = self.buf[j:]
                        return out
                    self.buf = self.buf[b:]
                    self.state, self.bare = "call", True
                    continue
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    j = len(self.buf) - self._hold(self.buf, (CALL_START,))
                    if self.recover and self.schemas:            # a line that may be becoming a bare <function=
                        k = self.buf.rfind("\n") + 1
                        tail = self.buf[k:]
                        if tail and FUNC_START.startswith(tail) and self._ok_at(k):
                            j = min(j, k)
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                # #1058: an opener inside a code fence or inline code is text (a quoted example), never a call.  The
                # text before it decides, so streamed and whole outputs agree.  Mid-sentence openers still count.
                snap = (self.fence, self.line, self.ticks)
                self._track(self.buf[:i])
                in_code = self._in_code()
                self.fence, self.line, self.ticks = snap
                if in_code:
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                # A call is `<tool_call>` and then (after whitespace) `<function=`; the tag with anything else after
                # it is prose that names the format ("I'll use a <tool_call> block") - content, not a malformed call
                # that ends the request.  Until its follower has arrived it is held, like a partial tag.
                after = self.buf[i + len(CALL_START):].lstrip()
                kind = self._follower(after) if self.recover else None
                if kind in ("param", "json"):
                    if i and self.buf[:i].strip():
                        out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                    rest = self.buf[i + len(CALL_START):]
                    if kind == "param":                  # <parameter=NAME> as the opener: the call's <function=NAME>
                        at = len(rest) - len(after)
                        rest = rest[:at] + FUNC_START + after[len(PARAM_START):]
                    self.buf = rest
                    self.state = "call" if kind == "param" else "json"
                    continue
                if kind == "wait":
                    after = ""
                if after and not after.startswith(FUNC_START) and not FUNC_START.startswith(after):
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                if not after.startswith(FUNC_START):
                    j = i
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                if i and self.buf[:i].strip():
                    out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                self.buf = self.buf[i + len(CALL_START):]
                self.state = "call"
            elif self.state == "json":
                # recover: `{"name": ..., "arguments": {...}}` inside the wrapper is the call once its </tool_call> has
                # arrived and it names a declared tool; anything else is the text it is
                i = self.buf.find(CALL_END)
                if i < 0:
                    return out
                call = json_tool_call(self.buf[:i].strip(), self.schemas)
                if call is None:
                    out.append(Event("content", self._track(CALL_START + self.buf[:i + len(CALL_END)])))
                else:
                    out.append(Event("tool_call", call=call))
                self.buf = self.buf[i + len(CALL_END):]
                self.state, self.lead = "content", call is not None
            else:
                drop = None
                if self.bare:                       # no wrapper: the call ends at its </function>; a stray
                    i = function_end(self.buf)      # </tool_call> after it is the wrapper's closer, dropped
                    if i >= 0:
                        after = self.buf[i:].lstrip()
                        if after.startswith(CALL_END):
                            drop = len(self.buf[i:]) - len(after) + len(CALL_END)
                        elif after and not CALL_END.startswith(after):
                            drop = 0
                else:
                    i = call_end(self.buf)
                    if i >= 0:
                        drop = len(CALL_END)
                    if self.recover:                # a batch in one wrapper: the next call follows this one's
                        f = function_end(self.buf)  # </function> (or opens as <parameter=NAME>)
                        nxt = self.buf[f:].lstrip() if f >= 0 else ""
                        if nxt.startswith(PARAM_START) and self._follower(nxt) == "param":
                            at = len(self.buf) - len(nxt)
                            self.buf = self.buf[:at] + FUNC_START + nxt[len(PARAM_START):]
                            nxt = self.buf[at:]
                        if nxt.startswith(FUNC_START):
                            i, drop, self.batch = f, 0, True
                if self.stream_tools:
                    if i >= 0:
                        whole, self.buf = self.buf, self.buf[:i]     # scan only the body
                        out += self._scan()
                        out += self._close_scan()
                        self.buf = whole
                    else:
                        out += self._scan()
                if i < 0 or drop is None:
                    return out
                body = self.buf[:i]
                self.buf = self.buf[i + drop:]
                batch, self.batch = self.batch, False
                out += self._finish_call(body)
                if batch and out[-1].kind == "tool_call":
                    self.state, self.lead = "call", False

    def finish(self, reason: str | None = None) -> list[Event]:
        """End of generation: flush whatever is held (an unterminated tool call is returned as content; one that was
        already announced stays unfinished: its JSON is not closed and no "tool_call" follows it, #211).  `reason` is
        how the turn ended: calls waiting from the reasoning become real calls only on a natural stop (None = stop);
        a turn cut by max tokens (or cancelled, or failed) keeps them as reasoning text (#1058)."""
        out = []
        if self.pending:
            out += self._release(reason in (None, "stop") and self.state == "reasoning" and not self.buf)
        if self.state == "call" and self.bare and function_end(self.buf) >= 0:
            i = function_end(self.buf)          # recover: a bare call is whole at its </function>
            body, rest = self.buf[:i], self.buf[i:].strip()
            self.buf = ""
            out += self._finish_call(body)
            if rest and not CALL_END.startswith(rest):
                out.append(Event("content", self._track(rest)))
            return out
        if self.state == "call" and self.stream_tools and self.scall is not None:
            out += self._scan()                 # the output ended inside a call that was already announced
            if self.ss == "done":               # only its </tool_call> is missing: the call itself is whole
                out.append(Event("tool_call", call=self.scall))
            self.buf = ""
            self._reset_scan()
            return out
        if self.state == "reasoning" and self.buf == THINK_END:   # the output ends on a held `</think>`: its end
            self.buf, self.state = "", "content"
        if self.buf:
            # an unfinished call inside the reasoning (#804) is reasoning text, never a call
            kind = {"reasoning": "reasoning", "rcall": "reasoning", "content": "content"}.get(self.state, "content")
            text = CALL_START + self.buf if self.state in ("call", "json") and not self.bare else self.buf
            out.append(Event(kind, text))
            self.buf = ""
        return out

    def _finish_call(self, body: str) -> list[Event]:
        """A whole call body in the template's form -> its tool_call event.  With recover, one that still does not
        parse is the text it is instead of an error that ends the request."""
        name = body.strip()[len(FUNC_START):].split(">", 1)[0].strip()
        bare, self.bare = self.bare, False
        try:
            call = parse_tool_call(body, self.schemas.get(name))
        except ValueError:
            if not self.recover:
                raise
            self._reset_scan()
            self.state = "content"
            return [Event("content", self._track(body if bare else CALL_START + body + CALL_END))]
        if self.scall is not None:
            call.id = self.scall.id
        self._reset_scan()
        self.state, self.lead = "content", True
        return [Event("tool_call", call=call)]
