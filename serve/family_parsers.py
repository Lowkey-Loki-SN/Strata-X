"""serve/family_parsers.py - a family's model text -> reasoning, content and tool calls (the events serve/server.py reads).

The Qwen path reads the model's text with serve/frontend.py's OutputParser: a state machine over `reasoning` / `content` /
`call` (and `rcall`, a call inside the reasoning), with the reasoning-call rescue of #804/#1058, the code-fence guard of
#1058, and a call that is also reported while it is written (`tool_start`, `tool_args`, then `tool_call`).  A family beyond
Qwen needs the same machine over another CALL SYNTAX, so this module is that machine with the syntax taken out of it:

    FamilyParser(syntax, thinking, tools, stream_tools)        same interface and events as OutputParser
    Syntax                                                     what a call block looks like: opener, calls, parameters, closer
    Reader                                                     one call block read incrementally (the grammar, once)

The syntaxes (the formats are the ones each family's own chat template writes for a past call, which is what the model was
trained on; llama.cpp's parsers agree):

    qwen   <tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n</function>\\n</tool_call>
           (the reference: moex/tests/test_family_parsers.py checks this machine against OutputParser event for event)
    mimo2  <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>             values exact
    glm5   <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>                 whitespace between tags allowed
    deepseek4  \\n\\n<｜DSML｜tool_calls>\\n<｜DSML｜invoke name="NAME">\\n<｜DSML｜parameter name="K" string="true|false">V
               </｜DSML｜parameter>\\n</｜DSML｜invoke>\\n</｜DSML｜tool_calls>      several invokes per block
    deepseek41 the same with a space before each tag name: <｜DSML｜ calls>, <｜DSML｜ invoke name=...>, <｜DSML｜ parameter ...>
               (serve/test_deepseek41.py checks it against the release's parse_message_from_completion_text)
    hy4    <tool_calls:TK><tool_call:TK>NAME<arg_key:TK>K</arg_key:TK><arg_value:TK>V</arg_value:TK></tool_call:TK></tool_calls:TK>
           (TK the GGUF vocabulary's suffix, :6124c78e; several calls per block; the thinking's tags carry it too)
    ernie55  <ernie:tool_call>\n<invoke name="NAME"><parameter name="K">V</parameter></invoke>\n</ernie:tool_call>
             (several invokes per block; serve/test_ernie55.py checks it against the release's decoding_ernie.py)

A string value is the text as written; any other value is JSON (`string="false"` says so in DeepSeek's text; for the others
the tool's schema does, and without a schema a value that parses as JSON is JSON).  Nothing here touches the Qwen path.
"""
from __future__ import annotations

import json
import re

from serve.frontend import Event, ToolCall, tool_choice_of

THINK_START = "<think>"
THINK_END = "</think>"
WAIT = -2                       # undecided: the text so far is a prefix of what would decide it
RCALL_MAX = 32768               # the most of a call inside the reasoning that is held while waiting for its end (#804)


def starts(s: str, tag: str) -> int:
    """1 if `s` starts with `tag`, WAIT if `s` is a proper prefix of it (more text decides), else 0."""
    if s.startswith(tag):
        return 1
    return WAIT if tag.startswith(s) else 0


def hold(text: str, tags) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
    best = 0
    for tag in tags:
        for n in range(1, len(tag)):
            if text.endswith(tag[:n]):
                best = max(best, n)
    return best


# ------------------------------------------------------------------------------------------------ call syntaxes
class Syntax:
    """The call block of a family: `opener` calls `closer`; a call is a head, parameters and (where it has one) a tail."""
    key = ""
    opener = "<tool_call>"
    closer = "</tool_call>"
    tail = ""                    # ends one call inside the block ("" = the closer ends it)
    multi = False                # a block may hold several calls
    vend = ""                    # ends a parameter's value
    follow: tuple = ()           # what may come (after whitespace) after a value's end tag: the next parameter, or the call's end
    pad = False                  # a string value is written on lines of its own: one newline off each end (Qwen's format)
    opens_think = False          # the model writes its own `<think>` (MiMo): the output starts undecided
    call_lead = ""               # what the forced call's opening is made of (see forced())
    # moex (Hy4-flash): the tags around the thinking (a family whose vocabulary spells them with a suffix has its own)
    think_start = THINK_START
    think_end = THINK_END
    # moex: the answer is the text as written - its newlines after </think> kept, and of those before a call only the
    # blank line that belongs to the call's opening ("\n\n" + opener) dropped (DeepSeek-V4.1: the release's parser, whose
    # content ends exactly at "\n\n<｜DSML｜ calls"); off: every newline there dropped, as the families' machine always did
    exact_answer = False
    # moex (DSV41_BACKLOG #65): a call block in the answer is read as a call only when the request declared tools - a
    # model that writes one unasked gets its text back, not a tool_calls finish for a tool the client never offered
    # (the thinking state always asked for schemas); off: every family's answer state as before
    calls_need_tools = False

    def head(self, s: str):
        """`s` starts at where a call's head should: -> (characters used, function name) | None (not a head) | WAIT."""
        raise NotImplementedError

    def param(self, s: str):
        """`s` starts at a parameter's head: -> (characters used, name, "string" | "json" | None) | None | WAIT."""
        raise NotImplementedError

    def lead(self, after: str) -> int:
        """`after` is the text after the opener (leading whitespace gone): 1 it is a call, 0 it is prose that names the
        tag, WAIT not yet known."""
        raise NotImplementedError

    def forced(self, name: str | None) -> str:
        """The text that opens the call a reply must make (`name` None: whichever tool the model picks)."""
        raise NotImplementedError


class QwenSyntax(Syntax):
    key = "qwen"
    tail = "</function>"
    vend = "</parameter>"
    follow = ("<parameter=", "</function>")
    pad = True

    def head(self, s):
        r = starts(s, "<function=")
        if r != 1:
            return r or None
        gt = s.find(">")
        return (gt + 1, s[10:gt]) if gt >= 0 else WAIT

    def param(self, s):
        r = starts(s, "<parameter=")
        if r != 1:
            return r or None
        gt = s.find(">")
        return (gt + 1, s[11:gt], None) if gt >= 0 else WAIT

    def lead(self, after):
        return starts(after, "<function=") if after else WAIT

    def forced(self, name):
        return "<tool_call>\n<function=" + (f"{name}>\n" if name else "")


class MimoSyntax(QwenSyntax):
    """`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`: the template writes no newline
    around anything, so the values are exact."""
    key = "mimo2"
    pad = False
    opens_think = True

    def forced(self, name):
        return "<tool_call><function=" + (f"{name}>" if name else "")


class GlmSyntax(Syntax):
    """`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`: the name is the text up to the first
    tag (or whitespace); the template writes the pairs back to back, GLM-4.5 wrote a newline between them, both are read."""
    key = "glm5"
    vend = "</arg_value>"
    follow = ("<arg_key>", "</tool_call>")
    _name = re.compile(r"[^\s<>=\"'/]+")

    def head(self, s):
        if not s:
            return WAIT
        m = re.match(r"[^\s<]+", s)
        if not m:
            return None
        return WAIT if m.end() == len(s) else (m.end(), m.group())

    def param(self, s):
        r = starts(s, "<arg_key>")
        if r != 1:
            return r or None
        i = s.find("</arg_key>")
        if i < 0:
            return WAIT
        rest = s[i + 10:]
        stripped = rest.lstrip()
        r = starts(stripped, "<arg_value>")
        if r != 1:
            return r or None
        return (i + 10 + len(rest) - len(stripped) + 11, s[9:i].strip(), None)

    def lead(self, after):
        if not after:
            return WAIT
        m = self._name.match(after)
        if not m:
            return 0
        r = after[m.end():].lstrip()
        if not r:
            return WAIT                 # the name may go on, or its first tag has not come
        for tag in ("<arg_key>", "</tool_call>"):
            got = starts(r, tag)
            if got:
                return got
        return 0

    def forced(self, name):
        return "<tool_call>" + (name or "")


DSML = "｜DSML｜"


class DeepseekSyntax(Syntax):
    """DeepSeek-V4's DSML: one `tool_calls` block of `invoke`s, each with `parameter`s that say `string="true"` (text as
    written) or `string="false"` (JSON)."""
    key = "deepseek4"
    opener = "<" + DSML + "tool_calls>"
    closer = "</" + DSML + "tool_calls>"
    tail = "</" + DSML + "invoke>"
    multi = True
    vend = "</" + DSML + "parameter>"
    follow = ("<" + DSML + "parameter", "</" + DSML + "invoke>")
    INVOKE = "<" + DSML + 'invoke name="'
    PARAM = "<" + DSML + 'parameter name="'

    def head(self, s):
        r = starts(s, self.INVOKE)
        if r != 1:
            return r or None
        q = s.find('"', len(self.INVOKE))
        if q < 0:
            return WAIT
        gt = s.find(">", q)
        return (gt + 1, s[len(self.INVOKE):q]) if gt >= 0 else WAIT

    def param(self, s):
        r = starts(s, self.PARAM)
        if r != 1:
            return r or None
        q = s.find('"', len(self.PARAM))
        if q < 0:
            return WAIT
        gt = s.find(">", q)
        if gt < 0:
            return WAIT
        m = re.search(r'string="(true|false)"', s[q:gt])
        return (gt + 1, s[len(self.PARAM):q], None if m is None else ("string" if m.group(1) == "true" else "json"))

    def lead(self, after):
        return starts(after, "<" + DSML + "invoke") if after else WAIT

    def forced(self, name):
        return "\n\n" + self.opener + "\n" + self.INVOKE + (f'{name}">\n' if name else "")


class Deepseek41Syntax(DeepseekSyntax):
    """DeepSeek-V4.1's DSML: V4's grammar with a space before each tag name (`<｜DSML｜ calls>`, `<｜DSML｜ invoke name=`,
    `<｜DSML｜ parameter name=`, encoding.py's tool_calls_block_name etc.).  V4's unspaced tags are no call here: the
    release's parser refuses them too."""
    key = "deepseek41"
    opener = "<" + DSML + " calls>"
    closer = "</" + DSML + " calls>"
    tail = "</" + DSML + " invoke>"
    vend = "</" + DSML + " parameter>"
    follow = ("<" + DSML + " parameter", "</" + DSML + " invoke>")
    INVOKE = "<" + DSML + ' invoke name="'
    PARAM = "<" + DSML + ' parameter name="'
    exact_answer = True          # the answer as parse_message_from_completion_text gives it (the audit of 2026-10-19)
    calls_need_tools = True      # the audit of 2026-10-20 (#65)

    def lead(self, after):
        return starts(after, "<" + DSML + " invoke") if after else WAIT


HY4_TK = ":6124c78e"   # the suffix of Hy4-flash's tags in the GGUF's vocabulary and template (the HF release's is ":opensource")


class Hy4Syntax(Syntax):
    """Hy4-flash (arch hy_v4_flash): `<tool_calls:TK><tool_call:TK>NAME<arg_key:TK>K</arg_key:TK><arg_value:TK>V</arg_value:TK>
    ...</tool_call:TK>...</tool_calls:TK>` - GLM-5's pairs inside one block of calls, every tag with the vocabulary's suffix
    (the GGUF's template writes a past call so; vLLM's hy_v4 parser reads the same).  The thinking is `<think:TK>` ..
    `</think:TK>`."""
    key = "hy4"
    opener = "<tool_calls" + HY4_TK + ">"
    closer = "</tool_calls" + HY4_TK + ">"
    tail = "</tool_call" + HY4_TK + ">"
    multi = True
    vend = "</arg_value" + HY4_TK + ">"
    follow = ("<arg_key" + HY4_TK + ">", "</tool_call" + HY4_TK + ">")
    think_start = "<think" + HY4_TK + ">"
    think_end = "</think" + HY4_TK + ">"
    CALL = "<tool_call" + HY4_TK + ">"
    KEY, KEY_END, VALUE = "<arg_key" + HY4_TK + ">", "</arg_key" + HY4_TK + ">", "<arg_value" + HY4_TK + ">"

    def head(self, s):
        r = starts(s, self.CALL)
        if r != 1:
            return r or None
        rest = s[len(self.CALL):]
        lt = rest.find("<")
        if lt < 0:
            return WAIT                  # the name may go on
        name = rest[:lt].strip()
        if not name or any(c.isspace() for c in name):
            return None
        return (len(self.CALL) + lt, name)

    def param(self, s):
        r = starts(s, self.KEY)
        if r != 1:
            return r or None
        i = s.find(self.KEY_END)
        if i < 0:
            return WAIT
        rest = s[i + len(self.KEY_END):]
        stripped = rest.lstrip()
        r = starts(stripped, self.VALUE)
        if r != 1:
            return r or None
        return (i + len(self.KEY_END) + len(rest) - len(stripped) + len(self.VALUE), s[len(self.KEY):i].strip(), None)

    def lead(self, after):
        return starts(after, self.CALL) if after else WAIT

    def forced(self, name):
        return self.opener + self.CALL + (name or "")


class ErnieSyntax(Syntax):
    """ERNIE-5.5-Flash (arch ernie55): `<ernie:tool_call>\\n<invoke name="NAME"><parameter name="K">V</parameter>...</invoke>\\n
    ...</ernie:tool_call>` - one block of invokes, each value as written (a string) or JSON, which the tool's schema decides
    (the release's encoding_ernie.py writes a past call so; its decoding_ernie.py reads `<invoke\\s+name="..">` and
    `<parameter\\s+name="..">` the same way).  The thinking is `<ernie:think>` .. `</ernie:think>`; the answer is the text
    between `</ernie:think>` and the next marker, as written."""
    key = "ernie55"
    opener = "<ernie:tool_call>"
    closer = "</ernie:tool_call>"
    tail = "</invoke>"
    multi = True
    vend = "</parameter>"
    follow = ("<parameter", "</invoke>")
    think_start = "<ernie:think>"
    think_end = "</ernie:think>"
    exact_answer = True          # decoding_ernie.py's content: the text as written up to the call block
    calls_need_tools = True      # a block without declared tools is its text (as DeepSeek-V4.1's, #65)
    _invoke = re.compile(r'<invoke\s+name="')
    _param = re.compile(r'<parameter\s+name="')

    @staticmethod
    def _could_be(s: str, word: str) -> bool:
        """`s` is a proper start of `<WORD\\s+name="` (more text decides)"""
        full = "<" + word
        if len(s) <= len(full):
            return full.startswith(s)
        if not s.startswith(full):
            return False
        rest = s[len(full):]
        gap = len(rest) - len(rest.lstrip())
        return gap > 0 and 'name="'.startswith(rest[gap:])

    def _tag(self, s: str, rx, word: str):
        m = rx.match(s)
        if m is None:   # not (yet) the tag: WAIT while `s` could still become it
            return WAIT if self._could_be(s, word) else None
        q = s.find('"', m.end())
        if q < 0:
            return WAIT
        gt = s.find(">", q)
        if gt < 0:
            return WAIT
        if s[q + 1:gt].strip():
            return None   # (the release's tag closes right after the quote)
        return gt + 1, s[m.end():q]

    def head(self, s):
        r = self._tag(s, self._invoke, "invoke")
        return r if r is None or r == WAIT else (r[0], r[1])

    def param(self, s):
        r = self._tag(s, self._param, "parameter")
        return r if r is None or r == WAIT else (r[0], r[1], None)

    def lead(self, after):
        return starts(after, "<invoke") if after else WAIT

    def forced(self, name):
        return self.opener + "\n" + '<invoke name="' + (f'{name}">' if name else "")


SYNTAXES = {s.key: s for s in (QwenSyntax(), MimoSyntax(), GlmSyntax(), DeepseekSyntax(), Deepseek41Syntax(), Hy4Syntax(),
                               ErnieSyntax())}


def value_end(x: Syntax, text: str, final: bool = False) -> int:
    """Where a parameter value in `text` ends: the first end tag followed (after whitespace) by what may follow it - the
    same text inside a value (a file that documents the call format, #210) is part of the value.  -1: none yet; WAIT: a
    candidate whose follower has not arrived (streaming; `final` accepts it)."""
    at = text.find(x.vend)
    while at >= 0:
        after = text[at + len(x.vend):].lstrip()
        if after.startswith(x.follow):
            return at
        if not after or any(f.startswith(after) for f in x.follow):
            return at if final else WAIT
        at = text.find(x.vend, at + 1)
    return -1


def padded(x: Syntax, value: str, leading: bool = True) -> str:
    """A string value as the call's text has it: a pad syntax puts the value on lines of its own."""
    if x.pad:
        if leading and value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
    return value


# ------------------------------------------------------------------------------------------------ one block, read
class Reader:
    """One call block (the text after its opener) read as it grows.  `advance(body)` takes the whole body so far and
    returns the events its new text allows; `end` is where the closer starts once the block is complete (None before),
    `calls` the calls read.  The events are OutputParser's: `tool_start` when a call's name is known, `tool_args` pieces
    of its JSON arguments (a string value character by character, anything else whole once complete) and `tool_call`
    when it ends - the last only with `stream` (without it the caller builds the events from `calls`)."""

    def __init__(self, x: Syntax, schemas: dict, stream: bool):
        self.x, self.schemas, self.stream = x, schemas, stream
        self.pos = 0
        self.st = "head"                 # head -> params -> (str | raw) -> params ... -> close | done
        self.calls: list[ToolCall] = []
        self.cur: ToolCall | None = None
        self.declared: dict = {}
        self.first = True
        self.pname = ""
        self.vstart = 0
        self.started = False
        self.end: int | None = None
        self.bad = False                 # the text broke the grammar: `end` is the first closer (or None until it arrives)
        self.announced = False           # a tool_start went out

    # -- helpers
    def _args(self, out, s):
        if s and self.stream:
            out.append(Event("tool_args", s, call=self.cur))

    def _finish_call(self, out):
        call, self.cur = self.cur, None
        self.calls.append(call)
        if self.stream:
            out.append(Event("tool_call", call=call))
        self.first = True

    def _fallback(self, body: str) -> bool:
        """The grammar broke: the block ends at the first closer, if it is there yet."""
        at = body.find(self.x.closer, self.pos)
        self.bad = True
        if at < 0:
            return False
        self.end, self.st = at, "done"
        return True

    # -- the machine
    def advance(self, body: str) -> list[Event]:
        out: list[Event] = []
        x = self.x
        while self.st != "done":
            rest = body[self.pos:]
            if self.st in ("head", "params", "close"):
                stripped = rest.lstrip()
                self.pos += len(rest) - len(stripped)
                rest = stripped
            if self.st == "head":
                if rest.startswith(x.closer):
                    self.end, self.st = self.pos, "done"
                    break
                if rest and x.closer.startswith(rest):
                    return out                  # the closer still arriving (after a block's last call)
                r = x.head(rest)
                if r == WAIT:
                    return out
                if r is None:
                    if not rest:
                        return out
                    self._fallback(body)
                    return out
                used, name = r
                self.cur = ToolCall(name=name, arguments={})
                props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
                self.declared = {k: (v or {}).get("type") for k, v in props.items()}
                self.announced = True
                if self.stream:
                    out.append(Event("tool_start", call=self.cur))
                self._args(out, "{")
                self.pos += used
                self.st = "params"
            elif self.st == "params":
                if x.tail and rest.startswith(x.tail):
                    self._args(out, "}")
                    self.pos += len(x.tail)
                    self._finish_call(out)
                    self.st = "head" if x.multi else "close"
                    continue
                if not x.tail and rest.startswith(x.closer):
                    self._args(out, "}")
                    self._finish_call(out)
                    self.end, self.st = self.pos, "done"
                    break
                r = x.param(rest)
                if r == WAIT:
                    return out
                if r is None:
                    if not rest or (x.tail and x.tail.startswith(rest)) or x.closer.startswith(rest):
                        return out          # a tag still arriving (or trailing text): wait
                    self._fallback(body)
                    return out
                used, pname, kind = r
                self.pname = pname
                self._args(out, ("" if self.first else ",") + json.dumps(pname) + ":")
                self.first = False
                self.pos += used
                self.vstart, self.started = self.pos, False
                is_string = kind == "string" or (kind is None and self.declared.get(pname) == "string")
                if is_string:
                    self._args(out, '"')
                    self.st = "str"
                else:
                    self.st = "raw"
            elif self.st == "str":
                if not self.started:
                    if not rest:
                        return out
                    if x.pad and rest[0] == "\n":       # the value's leading newline is not part of it
                        self.pos += 1
                        rest = rest[1:]
                        self.vstart = self.pos
                    self.started = True
                end = value_end(x, rest)
                if end >= 0:
                    value = padded(x, rest[:end], leading=False)
                    self._args(out, json.dumps(value)[1:-1] + '"')
                    self.cur.arguments[self.pname] = padded(x, body[self.vstart:self.pos + end], leading=False)
                    self.pos += end + len(x.vend)
                    self.st = "params"
                    continue
                # an undecided end tag (WAIT) is held from its start, like a tag still arriving
                safe = rest.find(x.vend) if end == WAIT else len(rest) - hold(rest, (x.vend,))
                if x.pad and safe > 0 and rest[safe - 1] == "\n":    # may be the trailing newline before the end tag
                    safe -= 1
                if safe > 0:
                    self._args(out, json.dumps(rest[:safe])[1:-1])
                    self.pos += safe
                return out
            elif self.st == "raw":
                end = value_end(x, rest)
                if end < 0:
                    return out
                value = padded(x, rest[:end])
                try:
                    v = json.loads(value)
                except ValueError:
                    v = value
                self._args(out, json.dumps(v, ensure_ascii=False))
                self.cur.arguments[self.pname] = v
                self.pos += end + len(x.vend)
                self.st = "params"
            elif self.st == "close":
                if rest.startswith(x.closer):
                    self.end, self.st = self.pos, "done"
                    break
                if not rest or x.closer.startswith(rest):
                    return out
                self._fallback(body)
                return out
        return out

    def close(self, body: str) -> list[Event]:
        """The block ended without a clean end (its closer came after a broken call, or the text stopped): close the
        JSON of an announced call so clients can still parse it, and finish the call with what it has."""
        out: list[Event] = []
        if self.cur is None:
            return out
        tail = ""
        if self.st == "str":
            tail += '"'
            self.cur.arguments[self.pname] = padded(self.x, body[self.vstart:], leading=False)
        elif self.st == "raw":
            raw = body[self.pos:].strip("\n")
            tail += json.dumps(raw)
            self.cur.arguments[self.pname] = raw
        tail += "}"
        self._args(out, tail)
        self._finish_call(out)
        return out


# ------------------------------------------------------------------------------------------------ the parser
class FamilyParser:
    """OutputParser's interface (feed, finish, state, buf, pending, rescued, refused) over a family's call syntax."""

    def __init__(self, syntax: Syntax, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False):
        self.x = syntax
        self.state = ("open" if syntax.opens_think else "reasoning") if thinking else "content"
        self.buf = ""
        self.lead = False
        self.schemas = {t.get("name"): t for t in tools or []}
        self.stream_tools = stream_tools
        self.pending: list[list] = []
        self.rescued = 0
        self.refused = 0
        self.fence, self.line, self.ticks = "", "", 0
        self.reader: Reader | None = None
        self.rreader: Reader | None = None

    # -- the reasoning text that has gone out (code fence, inline code, current line), as OutputParser tracks it
    @property
    def line(self) -> str:
        """The current line, joined only when a decision, a newline or a snapshot needs it (upstream 2a91a144: one long line
        streamed in small pieces no longer re-copies itself on every piece)."""
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
        if "\n" not in text:   # most pieces stay on the current line: kept as parts, joined in blocks of 64
            if text:
                self._line_parts.append(text)
                if len(self._line_parts) == 64:
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

    def _quoted_think_end(self, i: int) -> bool | None:
        """#537 for the families: the `</think>` at self.buf[i] is the model quoting the tag while it thinks when the same
        quote mark or backtick encloses it (`</think>`, "</think>") and that mark opens a quotation on its line.  Upstream's
        rule (a quote before it, no line break after it) is Qwen's: Qwen's template writes a line break after its real
        </think>, GLM-5's, MiMo's and DeepSeek-V4's write the answer right after it - so a reasoning that ends on a quote
        would keep the whole answer as reasoning.  An apostrophe never quotes it (moex, the audit of 2026-10-24): it cannot
        tell an opening quote from "I'll", and with the enclosure alone a reasoning that ended on ' before an answer that
        began with ' kept the answer as reasoning, where the release's parser splits at the first </think>.  A backtick
        opens no quote on a fence line (moex, DSV41_BACKLOG #56): "```" closing a code block right before the tag is three
        backticks, an odd count, and an answer that opened with a fence of its own stayed reasoning.  None while the
        character after the tag has not arrived."""
        line = (self.line + self.buf[:i]).rsplit("\n", 1)[-1]
        before = line[-1:]
        if before not in ('"', "`"):
            return False
        if before == "`" and line.strip().startswith("```"):
            return False
        after = self.buf[i + len(self.x.think_end):]
        if not after:
            return None
        if after[0] != before:
            return False
        return line.count(before) % 2 == 1

    def _opener_ok(self) -> bool:
        return not self.fence and not self.line.strip() and self.ticks % 2 == 0

    def _in_code(self) -> bool:
        return bool(self.fence) or (self.ticks + self.line.count("`")) % 2 == 1

    def _release(self, deliver: bool) -> list[Event]:
        out = []
        for raw, calls in self.pending:
            if calls and deliver:
                for call in calls:
                    out.append(Event("tool_call", call=call))
                    self.rescued += 1
            else:
                if calls:
                    self.refused += len(calls)
                if raw:
                    out.append(Event("reasoning", self._track(raw)))
        self.pending = []
        return out

    def _calls_of(self, rd: Reader, body: str) -> list[ToolCall] | None:
        """The calls of a block found inside the reasoning when every function is a declared tool (else None: prose)."""
        if rd.bad or not rd.calls or any(c.name not in self.schemas for c in rd.calls):
            return None
        return rd.calls

    # -- the machine
    def feed(self, delta: str) -> list[Event]:
        x = self.x
        self.buf += delta
        out: list[Event] = []
        tags = (x.think_end, x.opener) if self.schemas else (x.think_end,)
        while True:
            if self.state == "open":
                # the model opens its own <think> (MiMo): undecided until the first characters are in
                s = self.buf.lstrip()
                if not s:
                    return out
                if s.startswith(x.think_start):
                    self.buf = s[len(x.think_start):]
                    self.state = "reasoning"
                    continue
                if x.think_start.startswith(s):
                    return out
                self.state = "content"
            elif self.state == "rcall":
                # #804: an opener inside the reasoning, with tools declared.  It is held whole (never streamed as a
                # call) until it ends: declared names make tool calls, anything else stays reasoning text.
                body = self.buf[len(x.opener):]
                if self.rreader is None:
                    self.rreader = Reader(x, self.schemas, stream=False)
                self.rreader.advance(body)
                end, think = self.rreader.end, body.find(x.think_end)
                if end is not None and (think < 0 or think >= end):
                    calls = self._calls_of(self.rreader, body)
                    raw = self.buf[:len(x.opener) + end + len(x.closer)]
                    if calls is not None or self.pending:
                        self.pending.append([raw, calls])    # settled by what follows it
                    else:
                        out.append(Event("reasoning", self._track(raw)))
                    self.buf = body[end + len(x.closer):]
                    self.state, self.rreader = "reasoning", None
                elif think >= 0:                     # the thinking ended inside it: it never was a call
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf[:len(x.opener) + think])))
                    self.buf = body[think:]
                    self.state, self.rreader = "reasoning", None
                elif len(body) > RCALL_MAX:          # an opener in the prose that never closes: stop holding the thinking back
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf)))
                    self.buf = ""
                    self.state, self.rreader = "reasoning", None
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
                    if self.buf.startswith(x.opener):
                        self.state = "rcall"
                    elif self.buf.startswith(x.think_end):
                        out += self._release(True)
                    elif x.opener.startswith(self.buf) or x.think_end.startswith(self.buf):
                        return out
                    else:
                        out += self._release(False)
                    continue
                i = self.buf.find(x.think_end)
                tool = self.buf.find(x.opener) if self.schemas else -1
                if tool >= 0 and (i < 0 or tool < i):
                    if tool:
                        out.append(Event("reasoning", self._track(self.buf[:tool])))
                    if self._opener_ok():
                        self.buf = self.buf[tool:]
                        self.state = "rcall"
                    else:                            # mid-sentence, in a fence or in inline code: a quote
                        out.append(Event("reasoning", self._track(x.opener)))
                        self.buf = self.buf[tool + len(x.opener):]
                    continue
                if i < 0:
                    keep = hold(self.buf, tags)
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self._track(self.buf[:len(self.buf) - keep])))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                quoted = self._quoted_think_end(i)
                if quoted is None:                   # held like a partial tag until the character after it is in
                    if i:
                        out.append(Event("reasoning", self._track(self.buf[:i])))
                        self.buf = self.buf[i:]
                    return out
                if quoted:                           # #537: a quoted `</think>` is reasoning text
                    out.append(Event("reasoning", self._track(self.buf[:i + len(x.think_end)])))
                    self.buf = self.buf[i + len(self.x.think_end):]
                    continue
                if i:
                    out.append(Event("reasoning", self._track(self.buf[:i])))
                self.buf = self.buf[i + len(self.x.think_end):]
                self.state, self.lead = "content", True
                self.fence, self.line, self.ticks = "", "", 0     # the answer's own text starts here
            elif self.state == "content":
                if self.lead and not x.exact_answer:            # newlines right after </think> or a call
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                i = self.buf.find(x.opener) if self.schemas or not x.calls_need_tools else -1
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    j = len(self.buf) - hold(self.buf, (x.opener,))
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                # #1058: an opener inside a code fence or inline code is text (a quoted example), never a call.
                snap = (self.fence, self.line, self.ticks)
                self._track(self.buf[:i])
                in_code = self._in_code()
                self.fence, self.line, self.ticks = snap
                if in_code:
                    out.append(Event("content", self._track(self.buf[:i + len(x.opener)])))
                    self.buf = self.buf[i + len(x.opener):]
                    continue
                # A call is the opener and then what the syntax says a call begins with; the tag with anything else
                # after it is prose that names the format - content, not a malformed call that ends the request.
                # Until its follower has arrived it is held, like a partial tag.
                after = self.buf[i + len(x.opener):].lstrip()
                verdict = x.lead(after)
                if verdict == 0:
                    out.append(Event("content", self._track(self.buf[:i + len(x.opener)])))
                    self.buf = self.buf[i + len(x.opener):]
                    continue
                if verdict == WAIT:
                    j = i
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                head = self.buf[:i]
                if x.exact_answer:                   # the call's own opening is "\n\n" + opener: the rest is the answer's
                    head = head[:-2] if head.endswith("\n\n") else head
                    if head:
                        out.append(Event("content", self._track(head)))
                elif i and head.strip():
                    out.append(Event("content", self._track(head.rstrip("\n"))))
                self.buf = self.buf[i + len(x.opener):]
                self.state = "call"
            else:
                if self.reader is None:
                    self.reader = Reader(x, self.schemas, stream=self.stream_tools)
                rd = self.reader
                out += rd.advance(self.buf)
                if rd.end is None:
                    return out
                body = self.buf[:rd.end]
                self.buf = self.buf[rd.end + len(x.closer):]
                if rd.bad or rd.cur is not None:
                    out += rd.close(body)           # a call that did not end cleanly: closed with what it has
                if not self.stream_tools:
                    for call in rd.calls:
                        out.append(Event("tool_call", call=call))
                self.reader = None
                self.state, self.lead = "content", True

    def finish(self, reason: str | None = None) -> list[Event]:
        """End of generation: flush whatever is held (an unterminated call is returned as content; one that was already
        announced stays unfinished: its JSON is not closed and no "tool_call" follows it, #211).  `reason` is how the turn
        ended: calls waiting from the reasoning become real calls only on a natural stop (None = stop); a turn cut by max
        tokens (or cancelled, or failed) keeps them as reasoning text (#1058)."""
        out = []
        if self.state == "reasoning" and self.buf == self.x.think_end:   # #537: the output ends on a held `</think>`: its end
            self.buf, self.state = "", "content"
        if self.pending:
            out += self._release(reason in (None, "stop") and self.state == "reasoning" and not self.buf)
        if self.state == "call" and self.stream_tools and self.reader is not None and self.reader.announced:
            rd = self.reader
            out += rd.advance(self.buf)         # the output ended inside a call that was already announced; a call that
            self.buf = ""                       # ended was reported as it ended, one that did not stays unfinished
            self.reader = None
            return out
        if self.buf and self.state in ("content", "call") and reason not in (None, "stop"):
            # moex (upstream Xolotl #106, Delta 730cfb42): a turn cut (max tokens, cancelled, failed) while a call's markup was
            # held - an opener still arriving, or a call whose name had not come yet - sends none of it: in the answer's text
            # an agent read the half-written markup as the reply.  What the answer holds here is only ever that (and the
            # newlines before it); a natural stop keeps it as text, as before
            self.buf, self.reader = "", None
            return out
        if self.buf:
            # an unfinished call inside the reasoning (#804) is reasoning text, never a call
            kind = {"reasoning": "reasoning", "rcall": "reasoning", "content": "content"}.get(self.state, "content")
            text = self.buf if self.state != "call" else self.x.opener + self.buf
            out.append(Event(kind, text))
            self.buf = ""
        return out


# ------------------------------------------------------------------------------------------------ entry points
def make_parser(name: str, thinking: bool, tools=None, stream_tools: bool = True) -> FamilyParser:
    """The parser of a family (serve/families.py `parser_name`: "glm5", "deepseek4", "deepseek41", "mimo2"; "qwen" is the
    reference)."""
    return FamilyParser(SYNTAXES[name], thinking=thinking, tools=tools, stream_tools=stream_tools)


def forced_call(name: str, tool_choice, tools: list[dict] | None) -> str | None:
    """`tool_choice` -> the text that opens the call the reply must make, or None (the model decides): as
    serve.frontend.forced_call, in the family's syntax.  The text is complete (it carries the blank line a template puts
    in front of a call, where it does): the server writes it as it is."""
    x = SYNTAXES[name]
    kind, want = tool_choice_of(tool_choice)
    names = {t.get("name") for t in tools or [] if isinstance(t, dict)}
    if kind in ("auto", "none"):
        return None
    if kind == "required" and len(names) == 1:     # one tool to call: name it, the model cannot invent another
        kind, want = "named", next(iter(names))
    if kind == "required" and names:
        return x.forced(None)
    if kind == "named" and want in names:
        return x.forced(want)
    print(f"[strata] tool_choice {json.dumps(tool_choice)[:200]} is not supported here (or names no tool of the "
          "request): the model decides, as with \"auto\"", flush=True)
    return None
