"""serve/families.py - the registry of the model families the front end can serve.

One family key per model family, one object per key that holds everything the Python front end must know about the model
and that is not in the engine: which tokenizer and pre-tokenizer, which chat template and what to feed it, which token ids
end a reply, how the BOS is handled, how thinking is switched, how the output is parsed into reasoning, content and tool
calls, which sampling the model asks for, and the facts the server needs about the engine's vocabulary.

    "family": "glm5" | "deepseek4" | "deepseek41" | "mimo2" | "hy4" | "ernie55"     in strata-<model>.json (the run config)

A config without `family` (or with "qwen3.8-flash-next") is the Qwen3.8-Flash-Next path exactly as before: serve/server.py
then never touches this module's objects (`from_config` returns None) and uses tools/strata_tokenizer.py, serve/frontend.py's
ChatTemplate and OutputParser.  Everything the Qwen path does stays what it was; a family only adds branches.

The facts below were read from the GGUFs' own metadata and from llama.cpp (the glm5next-mtp tree) and are checked against
it: moex/docs/FAMILIES.md says where each one comes from, moex/tools/verify_tokenizer.py checks the tokenizers and
moex/tests/ the rest.  DeepSeek-V4.1's come from its release (tokenizer.json, encoding/encoding.py, inference/
generate.py): no GGUF of it carries a template or sampling, and serve/test_deepseek41.py checks its template and parser
against the release's encoder.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

QWEN_KEY = "qwen3.8-flash-next"
THINK_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n</think>"


@dataclass(frozen=True)
class FamilySpec:
    key: str
    title: str
    model_name: str                   # the name the API reports (the config's `model_name` overrides it)
    arch: str                         # the GGUF's general.architecture
    pre: str                          # tokenizer.ggml.pre: which pre-tokenizer llama.cpp (and tools/strata_tokenizer_families) uses
    tokenizer_dir: str                # default directory under $STRATA_X_DATA/tokenizers
    template_file: str                # the GGUF's chat template, packaged: serve/family_templates/<file> (the fallback)
    n_vocab: int                      # tokens in the vocabulary (the engine's output width)
    bos_id: int | None                # the BOS token id llama.cpp reports (None: the GGUF has none)
    add_bos: bool                     # whether the tokenizer adds it (always False here: the template writes it as text)
    eos_ids: tuple[int, ...]          # the ids that end a reply: the engine's --eos-ids and the server's stop ids
    eog_ids: tuple[int, ...]          # llama.cpp's whole end-of-generation set (a superset: also FIM tokens, `</s>`)
    turn_token_ids: tuple[int, ...]   # the token that opens a turn (the cache's checkpoint is taken in front of the last one)
    thinking_default: bool            # what a request that says nothing gets
    default_sampling: dict = field(default_factory=dict)       # the GGUF's general.sampling.*, in the server's key names
    image_marker_ids: dict = field(default_factory=dict)       # the vision markers in the vocabulary (the counts depend on the encoder)
    # moex: images.  The template writes <start><pad><end> per picture; the server expands the pad that follows `start` to
    # one pad per image token (as Qwen's <|vision_start|><|image_pad|>, serve/server.py prepare) and the engine (GENI) puts
    # the encoder's rows at those pads.  (start text, pad text, pad id); None: the family takes no images.  `vision_ready`:
    # this family's engine reads images (GENI) - the config's "vision" is refused until it does.
    image_tokens: tuple | None = None
    vision_ready: bool = False
    # moex: a family whose template writes no start token (DeepSeek-V4, V4.1): how many lead pad rows its encoder puts in front
    # of an image's rows - mainline mtmd's deepseek4v writes 3 (the server keeps 3 - (P % 4) of them, server.py prepare);
    # DeepSeek-V4.1's encoder (serve/vision_dsv41.py) writes the span alone, 0
    image_lead_pads: int = 3
    # moex (DSV41_BACKLOG #49): an image encoder that does not start (its interpreter or script missing, it ends before
    # READY, its weights do not load) leaves the server running text-only instead of stopping it - DeepSeek-V4.1's runs on
    # another venv's torch (the config's vision "python").  False: the server stops, or the request fails, as always
    vision_optional: bool = False
    # moex (DSV41_PLAN §54): the engine itself encodes this family's pictures, on its first GPU, when the config's vision
    # block says "gpu": true (strata-x's VENC: serve/server.py EngineVision) - DeepSeek-V4.1's; otherwise an encoder process
    vision_engine: bool = False
    # moex: this family's engine samples the whole vocabulary when a request's top_k is 0 (no cut) and top_p is 1 - the
    # server then sends top_k=0 instead of the widest list, 64 (strata-x's kernels::SamplerParams::full_vocab)
    full_vocab: bool = False
    # moex: how the server reads a client's messages for this family - serve.frontend.Dialect's fields (empty: the default,
    # every family's conversion as it was) - and "reasoning_turns": a past assistant turn with reasoning alone is a turn,
    # not an empty one the renderer drops (ChatTemplate.render, #843)
    dialect: dict = field(default_factory=dict)
    reasoning_turns: bool = False
    # moex (DSV41_BACKLOG #46): the reasoning loop guard (the config's reasoning_loop_recovery) also reads the calls the parser
    # holds inside the thinking (its `pending`): a thinking that repeats call blocks there left it no text to measure.  Only a
    # block that repeats one held before it counts (serve/server.py held_repeats): distinct calls are no loop
    loop_guard_held: bool = False
    reasoning_wrap_up: str = THINK_WRAP_UP
    template_vars: dict = field(default_factory=dict)          # variables the template reads that are not messages/tools
    template_filters: dict = field(default_factory=dict)
    wrap_tools: bool = True           # hand the template tools as {"type": "function", "function": {...}}
    neutralize_skip: tuple[str, ...] = ()
    parser_name: str = ""

    # ---- tokenizer and template
    def tokenizer_path(self, configured: str | Path | None = None) -> Path:
        import os
        # $STRATA_X_DATA, by default the folder Strata-X-data beside this checkout (moex/env.sh)
        return Path(configured) if configured else Path(os.environ.get(
            "STRATA_X_DATA") or ROOT.parent / "Strata-X-data") / "tokenizers" / self.tokenizer_dir

    def load_tokenizer(self, path: str | Path):
        import strata_tokenizer_families as F
        return F.FamilyTokenizer.from_dir(path)

    def load_template(self, path: str | Path | None = None):
        from serve.family_template import FamilyChatTemplate
        exported = Path(path) / "chat_template.jinja" if path else None
        src = exported if exported and exported.exists() else ROOT / "serve" / "family_templates" / self.template_file
        return FamilyChatTemplate(src, self)

    # ---- request dialect -> this template (see FamilyChatTemplate)
    def thinking_kwargs(self, kwargs: dict) -> dict:
        """Strata's request kwargs (`enable_thinking`, `reasoning_effort` low / medium / xhigh) -> the template's variables."""
        raise NotImplementedError

    def generation_tail(self, kwargs: dict) -> str:
        """Text the family appends to the rendered generation prompt (where the template has no switch of its own)."""
        return ""

    def thinking_on(self, kwargs: dict) -> bool:
        """Whether the prompt a request with these kwargs renders makes the model think (the server starts its output
        parser from this, so it must agree with `thinking_kwargs` / `generation_tail`)."""
        return kwargs.get("enable_thinking", self.thinking_default) is not False

    # ---- output
    def new_parser(self, thinking: bool, tools=None, stream_tools: bool = True):
        from serve.family_parsers import make_parser
        return make_parser(self.parser_name, thinking, tools, stream_tools)

    def exact_answer(self) -> bool:
        """The parser gives the answer as the model wrote it (serve.family_parsers Syntax.exact_answer: DeepSeek-V4.1)."""
        from serve.family_parsers import SYNTAXES
        return bool(getattr(SYNTAXES.get(self.parser_name), "exact_answer", False))

    def forced_call(self, tool_choice, tools):
        """`tool_choice` -> the text that opens the call the reply must make, or None (as serve.frontend.forced_call)."""
        from serve.family_parsers import forced_call
        return forced_call(self.parser_name, tool_choice, tools)

    # ---- images
    def image_rows(self, size, vision: dict) -> int | None:
        """moex (DSV41_BACKLOG #56): the rows the family's image encoder gives a picture of `size` (width, height; None: a
        picture whose size is not known - the most one may cost), without running it (`vision`: the config's vision
        block).  None: only the encoder knows (every family but DeepSeek-V4.1), and count_tokens counts the placeholder."""
        return None


# ------------------------------------------------------------------------------------------------ GLM-5.3
@dataclass(frozen=True)
class Glm5(FamilySpec):
    """GLM-5.3-Flash (arch glm5next).  The template starts `[gMASK]<sop>` and always says `Reasoning Effort: Max|High|Low`
    in a system turn; its generation prompt is `<|assistant|><think>`: an OPEN think, always, with no template switch."""

    def thinking_kwargs(self, kwargs):
        out = {}
        level = kwargs.get("reasoning_effort")
        if level == "low":
            out["reasoning_effort"] = "low"
        elif level == "medium":
            out["reasoning_effort"] = "high"          # the template knows low, high and (its default) max
        return out

    def generation_tail(self, kwargs):
        # no `enable_thinking` in this template: thinking off is the empty think block the template itself writes for a
        # past turn without reasoning (`<think></think>`), so the prompt is closed here
        return "" if self.thinking_on(kwargs) else "</think>"


# ------------------------------------------------------------------------------------------------ DeepSeek-V4-Flash
@dataclass(frozen=True)
class Deepseek4(FamilySpec):
    """DeepSeek-V4-Flash (arch deepseek4).  The template starts with `bos_token`; `thinking` is a variable whose default is
    false: the generation prompt is `<｜Assistant｜></think>` (off) or `<｜Assistant｜><think>` (on).  reasoning_effort
    `high` / `max` add a paragraph in front of the system prompt, in thinking mode only."""

    def thinking_on(self, kwargs):
        # off unless the request asks: an effort (low .. max) is such a request, `enable_thinking: false` ends it
        if kwargs.get("enable_thinking") is False:
            return False
        return bool(kwargs.get("enable_thinking") is True or "reasoning_effort" in kwargs or self.thinking_default)

    def thinking_kwargs(self, kwargs):
        on = self.thinking_on(kwargs)
        out = {"thinking": on}
        level = kwargs.get("reasoning_effort")
        if on and level == "medium":
            out["reasoning_effort"] = "high"          # the template knows high and max; low is "no extra text", its default
        elif on and level == "xhigh":
            out["reasoning_effort"] = "max"
        return out


# ------------------------------------------------------------------------------------------------ DeepSeek-V4.1-Flash
@dataclass(frozen=True)
class Deepseek41(Deepseek4):
    """DeepSeek-V4.1-Flash (arch deepseek41).  V4's vocabulary (byte for byte) and V4's switch: `thinking` is off unless the
    request asks, as in V4 and in the release's own generate.py (whose default mode is "chat").  The prompt is V4.1's: the
    template is a port of the release's encoder (encoding/encoding.py), where `<｜System｜>` opens the system prompt and
    any system message later in the conversation, a call is DSML with a space before each tag name (`<｜DSML｜ calls>`),
    and the effort is a number written in front of the first message, in thinking mode only: `Reasoning Effort: N (range
    1-100, ...)`, 75 when the request names none.

    Strata's levels map onto the encoder's three aliases (low 50, high 75, max 100): low is low, medium and high are high,
    xhigh and max are max.  The family's Dialect (serve/frontend.py) reads a client's words as the release does - its "high"
    is the release's high (75), its "max" or "xhigh" the max (100) - where the default dialect turns a client's high into the
    server's xhigh.  An integer 1-100 is passed as it is (frontend.effort_kwargs refuses numbers from a client today: only a
    direct caller can give one)."""

    EFFORTS = {"low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}

    def thinking_kwargs(self, kwargs):
        on = self.thinking_on(kwargs)
        out = {"thinking": on}
        level = kwargs.get("reasoning_effort")
        if on and level is not None:
            out["reasoning_effort"] = self.effort(level)
        return out

    def effort(self, level):
        """A request's effort -> the template's: a level as one of the encoder's aliases, an integer 1-100 as it is."""
        if isinstance(level, int) and not isinstance(level, bool):
            if 1 <= level <= 100:
                return level
        elif isinstance(level, str) and level in self.EFFORTS:
            return self.EFFORTS[level]
        raise ValueError(f"unknown reasoning effort {level!r}: DeepSeek-V4.1 takes low, medium, high, xhigh, max or "
                         "a whole number from 1 to 100")

    def load_template(self, path=None):
        # the tokenizer directory is V4's, and the chat_template.jinja exported beside it is V4's template: V4.1's is
        # always the packaged one (the release has no Jinja template, only the encoder this one is a port of)
        from serve.family_template import FamilyChatTemplate
        return FamilyChatTemplate(ROOT / "serve" / "family_templates" / self.template_file, self)

    def image_rows(self, size, vision):
        """The release's image plan (serve/vision_dsv41_plan.py, as serve/vision_dsv41.py runs it) for the model's
        inference/config.json (the vision block's "model") and the block's "max_tokens": the span's rows, its delimiters
        included - or the most a picture may cost (1,024) when its size is not known.  None without that config."""
        from serve.vision_dsv41_plan import image_rows, max_tokens_of
        cfg = _vision_config(vision.get("model"))
        if cfg is None:
            return None
        limit = int(vision.get("max_tokens") or 0)
        return max_tokens_of(cfg, limit) if size is None else image_rows(size[0], size[1], cfg, limit)


# ------------------------------------------------------------------------------------------------ MiMo-V2.6-Flash
@dataclass(frozen=True)
class Mimo2(FamilySpec):
    """MiMo-V2.6-Flash (arch mimo2).  ChatML-like, but the roles are written as they come (`tool` results are their own
    `<|im_start|>tool` turns) and the assistant turn is `<think>REASONING</think>CONTENT`.  The generation prompt is plain
    `<|im_start|>assistant\\n`: the model opens its own `<think>`; `enable_thinking=false` prefills `<think></think>`."""

    def thinking_kwargs(self, kwargs):
        return {"enable_thinking": False} if kwargs.get("enable_thinking") is False else {}


# ------------------------------------------------------------------------------------------------ Hy4-flash
HY4_TK = ":6124c78e"   # the suffix of Hy4-flash's special tokens in the GGUF's vocabulary (the HF release's is ":opensource")
HY4_WRAP_UP = THINK_WRAP_UP[:-len("</think>")] + "</think" + HY4_TK + ">"


@dataclass(frozen=True)
class Hy4(FamilySpec):
    """Hy4-flash (arch hy_v4_flash; moex/docs/HY4_PORT.md).  Turns are `<｜hy_start｜>ROLE<｜hy_middle｜>TEXT<｜hy_end｜>`, every
    tag with the vocabulary's suffix; the template always writes a system turn (one of its own when the request has none)
    that ends `<｜reasoning_mode｜>reasoning_effort:EFFORT`, EFFORT no_think, high (its default) or max, and its generation
    prompt opens the thinking, `<think:TK>` (no_think: the empty `<think:TK></think:TK>`).

    The template is the release's (tencent/Hy4-flash d88c9dfd chat_template.jinja), its tag suffix the GGUF vocabulary's
    (:6124c78e for :opensource - the same ids, all 24 suffixed tokens checked); AngelSlim's GGUF template also takes "low",
    which the release's refuses and its model card does not name (it gives high and no_think).  Strata's levels map onto
    the release's: low and medium are high (its default - it has no lower level that thinks), xhigh and max are max;
    thinking off is no_think."""

    EFFORTS = {"low": "high", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}

    def load_template(self, path=None):
        # always the packaged template: the tokenizer directory's chat_template.jinja is the GGUF's (AngelSlim's, with "low")
        from serve.family_template import FamilyChatTemplate
        return FamilyChatTemplate(ROOT / "serve" / "family_templates" / self.template_file, self)

    def thinking_on(self, kwargs):
        return kwargs.get("enable_thinking", self.thinking_default) is not False

    def thinking_kwargs(self, kwargs):
        if not self.thinking_on(kwargs):
            return {"reasoning_effort": "no_think"}
        level = kwargs.get("reasoning_effort")
        if level is None:
            return {}
        if level not in self.EFFORTS:
            raise ValueError(f"unknown reasoning effort {level!r}: Hy4-flash takes low, medium, high, xhigh or max")
        return {"reasoning_effort": self.EFFORTS[level]}

    def image_rows(self, size, vision):
        """The release's image plan (serve/vision_hy4_plan.py, as serve/vision_hy4.py runs it) for the processor_config.json
        the vision block names as its "model" and the block's "max_tokens": the picture's rows, the newline column included
        - or the most a picture may cost when its size is not known."""
        from serve.vision_hy4_plan import image_rows, max_rows, processor_limits
        lim = processor_limits(vision.get("model"))
        limit = int(vision.get("max_tokens") or 0)
        return max_rows(limit, lim) if size is None else image_rows(size[0], size[1], limit, lim)


ERNIE_WRAP_UP = THINK_WRAP_UP[:-len("</think>")] + "</ernie:think>"


@dataclass(frozen=True)
class Ernie55(FamilySpec):
    """ERNIE-5.5-Flash (arch ernie55; moex/docs/ERNIE55_PLAN.md).  The release writes its chat format in Python
    (encoding_ernie.py render_chat) and reads replies in Python (decoding_ernie.py parse_completion) - no Jinja template
    ships - so the packaged template (family_templates/ernie55.jinja) is that renderer and the `ernie55` parser that decoder;
    serve/test_ernie55.py checks both against the release's code.  Turns are `<|ernie:im_start|>ROLE\\nTEXT<|ernie:im_end|>`
    after `<|ERNIE_MASK|>`; the system turn's `<ernie:global_setting>` carries `reasoning_effort=EFFORT` (low / medium / high,
    the release's default high) while thinking is on, and the generation prompt opens the thinking, `<ernie:think>` (off:
    the empty `<ernie:think></ernie:think>`).  Strata's levels map onto the release's three: xhigh and max are its high."""

    EFFORTS = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}

    def load_template(self, path=None):
        # always the packaged template (the release has none to export)
        from serve.family_template import FamilyChatTemplate
        return FamilyChatTemplate(ROOT / "serve" / "family_templates" / self.template_file, self)

    def thinking_kwargs(self, kwargs):
        out = self._thinking(kwargs)
        if kwargs.get("parallel_tool_calls") is False:
            # moex (Dialect.parallel_option): the template's option - no `parallel_tool_calls=True` line, as the release's
            # vLLM plugin renders a request with parallel_tool_calls false (options.parallel_tool_calls)
            out["parallel_tool_calls"] = False
        return out

    def _thinking(self, kwargs):
        if not self.thinking_on(kwargs):
            return {"enable_thinking": False}
        level = kwargs.get("reasoning_effort")
        if level is None:
            return {}
        if level not in self.EFFORTS:
            raise ValueError(f"unknown reasoning effort {level!r}: ERNIE-5.5 takes low, medium, high, xhigh or max")
        return {"reasoning_effort": self.EFFORTS[level]}


# ------------------------------------------------------------------------------------------------ the registry
def _from_json(value):
    import json
    return json.loads(value)


def _call_arguments(value):
    """A past call's arguments as DeepSeek-V4.1's encoder reads them (encode_arguments_to_dsml): a mapping as it is, text
    parsed as JSON (twice: text that is JSON of JSON), and anything that is still not a mapping kept whole under
    "arguments"."""
    import json
    args = value
    for _ in range(2):
        if not isinstance(args, str):
            break
        try:
            args = json.loads(args)
        except ValueError:
            break
    return args if isinstance(args, dict) else {"arguments": value}


_VISION_CONFIGS: dict = {}


def _vision_config(path):
    """moex (DSV41_BACKLOG #56): a model's inference/config.json with the vision_* keys its image plan reads, read once;
    None when it cannot be read or has no vision tower."""
    if path not in _VISION_CONFIGS:
        import json
        try:
            cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            cfg = None
        keys = ("vision_patch_size", "vision_downsample_ratio", "vision_max_n_token", "vision_min_pixels")
        ok = isinstance(cfg, dict) and cfg.get("vision_n_layers") and all(isinstance(cfg.get(k), int) for k in keys)
        _VISION_CONFIGS[path] = cfg if ok else None
    return _VISION_CONFIGS[path]


FAMILIES: dict[str, FamilySpec] = {}


def register(spec: FamilySpec) -> FamilySpec:
    FAMILIES[spec.key] = spec
    return spec


register(Glm5(
    key="glm5", title="GLM-5.3-Flash", model_name="glm-5.3-flash", arch="glm5next", pre="glm5", tokenizer_dir="glm5", template_file="glm5.jinja",
    n_vocab=154880, bos_id=154822, add_bos=False,
    eos_ids=(154820, 154827, 154829), eog_ids=(154820, 154827, 154829),     # <|endoftext|> <|user|> <|observation|>
    turn_token_ids=(154826, 154827, 154828, 154829),                       # <|system|> <|user|> <|assistant|> <|observation|>
    thinking_default=True, default_sampling={"temperature": 1.0, "top_p": 0.95},
    image_marker_ids={"begin_of_image": 154830, "image": 154854, "end_of_image": 154831, "begin_of_video": 154832,
                      "video": 154855, "end_of_video": 154833, "begin_of_audio": 154834, "end_of_audio": 154835},
    # mainline mtmd's glm5v: <|begin_of_image|> rows <|end_of_image|> (tools/mtmd/mtmd.cpp PROJECTOR_TYPE_GLM5V)
    image_tokens=("<|begin_of_image|>", "<|image|>", 154854), vision_ready=True,
    reasoning_wrap_up=THINK_WRAP_UP, template_vars={"bos_token": "[gMASK]", "eos_token": "<|endoftext|>"},
    neutralize_skip=("/nothink",), parser_name="glm5"))

register(Deepseek4(
    key="deepseek4", title="DeepSeek-V4-Flash", model_name="deepseek-v4-flash", arch="deepseek4", pre="joyai-llm", tokenizer_dir="deepseek4",
    template_file="deepseek4.jinja", n_vocab=129280, bos_id=0, add_bos=False,
    eos_ids=(1,), eog_ids=(1,),                                           # <｜end▁of▁sentence｜>
    turn_token_ids=(128804,),                                             # <｜Assistant｜>
    thinking_default=False, default_sampling={"temperature": 1.0, "top_p": 1.0},
    image_marker_ids={"deepseek_image": 129264, "rl_image_pad": 129262, "rl_image_start": 129263},
    # mainline mtmd's deepseek4v adds NO vocab tokens: the encoder's rows carry their own start / newline / end sentinels and
    # 0-3 lead pads (START at a position = 3 mod 4); its text side reads the image span non-causally, routes it with
    # exp_probs_b_vl and skips the hash layers' tid2eid for it.  The template's placeholder is <｜deepseek_image｜> alone (no
    # start marker).  The server keeps 3 - (P % 4) of the record's lead pads (server.py prepare), the engine reads each span
    # as one non-causal batch routed as an image (src/x/session.cpp run_request, src/x/dsv4.cpp route).
    image_tokens=(None, "<\uff5cdeepseek_image\uff5c>", 129264), vision_ready=True,
    reasoning_wrap_up=THINK_WRAP_UP,
    template_vars={"bos_token": "<\uff5cbegin\u2581of\u2581sentence\uff5c>", "eos_token": "<\uff5cend\u2581of\u2581sentence\uff5c>"},
    template_filters={"from_json": _from_json}, parser_name="deepseek4"))

register(Deepseek41(
    key="deepseek41", title="DeepSeek-V4.1-Flash", model_name="deepseek-v4.1-flash", arch="deepseek41", pre="joyai-llm",
    tokenizer_dir="deepseek4",                                            # V4.1's tokenizer.json is V4's, byte for byte
    template_file="deepseek41.jinja", n_vocab=129280, bos_id=0, add_bos=False,
    eos_ids=(1,), eog_ids=(1,),                                           # <｜end▁of▁sentence｜>
    # <｜System｜> <｜User｜> <｜Assistant｜>: the last is always the generation prompt's <｜Assistant｜> (V4's checkpoint);
    # the first past the root's 2048 tokens can be a turn's start, so the root can hold a system prompt alone
    turn_token_ids=(128799, 128803, 128804),
    thinking_default=False,
    default_sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0},      # generate.py / the model card: T 1.0, no cut
    full_vocab=True,
    # the release's encoder (encoding/encoding.py) where the server's default conversion loses something (the audit of
    # 2026-10-19, DSV41_BACKLOG #35): tool results in the order of their calls (by id), a later system message as
    # <｜System｜>, a message's text parts joined by a blank line, and its effort words - low 50, high 75, max 100 (a client's
    # "high" used to become the server's xhigh: 100); medium is the release's default, high (75).  The re-audit (#45): vLLM's
    # V4.1 switch `thinking` in chat_template_kwargs, Anthropic's thinking enabled without a budget or adaptive, and the config's
    # level for a request that turns thinking on without one; (#56) a response_format as the release's `## Response Format:`,
    # a past turn's reasoning under vLLM's name `reasoning` or inline in its content (<think>R</think>answer)
    dialect={"call_ids": True, "late_system": True, "text_join": "\n\n",
             "efforts": {"none": None, "off": None, "minimal": None, "disabled": None, "false": None, "low": "low",
                         "medium": "medium", "high": "high", "xhigh": "max", "max": "max", "maximum": "max"},
             "thinking_switches": True, "response_format": True, "reasoning_field": True, "inline_think": True},
    reasoning_turns=True, loop_guard_held=True,
    image_marker_ids={"deepseek_image": 129264, "rl_image_pad": 129262, "rl_image_start": 129263},
    # moex vision (2026-10-19): the template writes <｜deepseek_image｜> for a picture, as the release's encoder does, and the
    # server repeats it once per row of the image's span (serve/vision_dsv41.py: the release's ViT and aligner, the span's
    # delimiter rows included, no lead pads); the engine reads the span causally with exp_probs_b_vl and no engram (V4: non-
    # causally, moex/docs/DSV41_PORT.md)
    image_tokens=(None, "<\uff5cdeepseek_image\uff5c>", 129264), vision_ready=True, image_lead_pads=0,
    vision_optional=True, vision_engine=True,
    reasoning_wrap_up=THINK_WRAP_UP,
    template_vars={"bos_token": "<\uff5cbegin\u2581of\u2581sentence\uff5c>", "eos_token": "<\uff5cend\u2581of\u2581sentence\uff5c>"},
    template_filters={"call_arguments": _call_arguments}, parser_name="deepseek41"))

register(Mimo2(
    key="mimo2", title="MiMo-V2.6-Flash", model_name="mimo-v2.6-flash", arch="mimo2", pre="qwen2", tokenizer_dir="mimo2", template_file="mimo2.jinja",
    n_vocab=152576, bos_id=None, add_bos=False,
    eos_ids=(151645, 151643),                                             # <|im_end|> <|endoftext|>
    eog_ids=(128247, 151643, 151645, 151662, 151663, 151664),             # + `</s>` and the FIM pad / repo / file-sep tokens
    turn_token_ids=(151644,),                                             # <|im_start|>
    thinking_default=True, default_sampling={"temperature": 1.0, "top_p": 0.95},
    image_marker_ids={"vision_start": 151652, "vision_end": 151653, "image_pad": 151655, "video_pad": 151656,
                      "audio_pad": 151669, "mimo_audio_start": 151673, "mimo_audio_end": 151674,
                      "mimo_video_start": 151670, "mimo_video_end": 151671},
    # mainline mtmd's mimovl: <|vision_start|> rows <|vision_end|> (tools/mtmd/mtmd.cpp PROJECTOR_TYPE_MIMOVL)
    image_tokens=("<|vision_start|>", "<|image_pad|>", 151655), vision_ready=True,
    reasoning_wrap_up=THINK_WRAP_UP, template_vars={"eos_token": "<|im_end|>"}, parser_name="mimo2"))


register(Hy4(
    key="hy4", title="Hy4-flash", model_name="hy4-flash", arch="hy_v4_flash", pre="hy_v4", tokenizer_dir="hy4",
    template_file="hy4.jinja", n_vocab=120832, bos_id=120000, add_bos=False,
    eos_ids=(120025,), eog_ids=(120025,),                                 # <｜hy_end:TK｜> (llama.cpp's whole EOG set)
    turn_token_ids=(120000,),                                             # <｜hy_start:TK｜>
    thinking_default=True,
    # generation_config.json and the model card: T 0.9, top_p 1.0, no top-k (the GGUF's general.sampling the same)
    default_sampling={"temperature": 0.9, "top_p": 1.0, "top_k": 0}, full_vocab=True,
    image_marker_ids={"image_start": 120118, "image_end": 120119, "image_pad": 120120, "image_newline": 120121,
                      "video_start": 120122, "video_end": 120123, "video_pad": 120124},
    # moex vision (2026-11-18): the template writes <image_start><image_pad><image_end> a picture; the server repeats the pad
    # once per row of the encoder's record (serve/vision_hy4.py: the release's tower and projector, each merged row's cells
    # then the newline row, no extra begin / end rows: cat_extra_token 0), and the engine reads the rows causally with the
    # release's M-RoPE (vLLM's get_mrope_input_positions; src/x/hy4.cpp hy4_mpos_set).  The encoder runs on llama.cpp's venv's
    # torch: a server whose encoder does not start goes on with text alone
    # (with the vision block's "gpu": true the engine encodes them on a GPU's free VRAM: src/x/vision.cpp, strata/x/hy4_vision.hpp)
    image_tokens=("<\uff5chy_image\u2581start" + HY4_TK + "\uff5c>", "<\uff5chy_image\u2581pad" + HY4_TK + "\uff5c>", 120120),
    vision_ready=True, image_lead_pads=0, vision_optional=True, vision_engine=True,
    reasoning_wrap_up=HY4_WRAP_UP,
    template_vars={"bos_token": "<\uff5chy_start" + HY4_TK + "\uff5c>", "eos_token": "<\uff5chy_end" + HY4_TK + "\uff5c>"},
    parser_name="hy4"))


register(Ernie55(
    key="ernie55", title="ERNIE-5.5-Flash", model_name="ernie-5.5-flash", arch="ernie55", pre="ernie55", tokenizer_dir="ernie55",
    template_file="ernie55.jinja", n_vocab=201216,
    bos_id=2, add_bos=False,                                              # <|ERNIE_MASK|>: the template writes it as text
    # <|ernie:im_end|> ends a reply; generation_config.json's eos also holds <|ERNIE_MASK|> (decoding_ernie.py ends at either)
    eos_ids=(8, 2), eog_ids=(8, 2),
    turn_token_ids=(7,),                                                  # <|ernie:im_start|>
    thinking_default=True,                                                # the release's default: thinking on, effort high
    default_sampling={"temperature": 1.0, "top_p": 0.95},                 # generation_config.json
    # no top-k cut, as the release's servers sample (vLLM's top_k 0 / SGLang's cookbook: T 1.0 and top_p 0.95 only): a
    # request that names no positive top_k reaches the engine as top_k 0 - the whole vocabulary's 0.95 nucleus, as vLLM cuts
    # it (strata-x's full_vocab, DSV41_BACKLOG #56) - never the engine's default 20 (which cut it at 21% of positions)
    full_vocab=True,
    # the server's request path as the release's encoder reads a conversation (serve/test_ernie55.py ServerPath): a system
    # message after the first stays a system turn, a past turn's reasoning is also read from vLLM's / OpenRouter's
    # `reasoning`, and a past assistant turn of reasoning alone is rendered (encoding_ernie.py renders every assistant turn);
    # moex (2026-11-25): a turn's reasoning written inline is split as the encoder splits it, `thoughts` is read first, and
    # a request that allows one call is told so (no parallel_tool_calls=True line) and gets its first call only
    dialect={"late_system": True, "reasoning_field": True, "think_split": ("<ernie:think>", "</ernie:think>"),
             "thoughts_field": True, "parallel_option": True},
    reasoning_turns=True,
    reasoning_wrap_up=ERNIE_WRAP_UP,
    template_filters={"from_json": _from_json}, parser_name="ernie55"))


def get(key: str | None) -> FamilySpec | None:
    """The family of a config's `family` key; None for the Qwen3.8-Flash-Next default (and for no key)."""
    if key in (None, "", QWEN_KEY):
        return None
    if key not in FAMILIES:
        raise ValueError('unknown "family" %r: use %s' % (key, ", ".join(sorted([QWEN_KEY, *FAMILIES]))))
    return FAMILIES[key]


def from_config(cfg: dict) -> FamilySpec | None:
    return get(cfg.get("family"))


def check_config(cfg: dict, spec: FamilySpec) -> None:
    """The run-config keys that exist only for the Qwen path are refused for a family, by name, before anything loads
    (ValueError; the server turns it into a start-up error)."""
    bad = []
    if cfg.get("vision") and not spec.vision_ready:
        bad.append('"vision" (this family\'s engine does not read images yet)')
    if cfg.get("effort_position", "start") != "start":
        bad.append('"effort_position": "end" (it is Qwen\'s trailing effort turn)')
    if cfg.get("reasoning_loop_recovery") in (True, "recover"):
        bad.append('"reasoning_loop_recovery": "recover" (it rewrites Qwen\'s effort sentence; "stop" works)')
    # engine features the family engines do not have yet: the server would pass flags they reject (exit 2 at start)
    if isinstance(cfg.get("parallel"), int) and not isinstance(cfg.get("parallel"), bool) and cfg["parallel"] > 1:
        bad.append('"parallel" (no batch slots in this engine yet: one request at a time)')
    for key in ("vram_elastic", "split_skip_if_fits"):
        if cfg.get(key) is True:
            bad.append('"%s" (not in this engine yet)' % key)
    # upstream 0.1.43's data-parallel replicas cut "gpu" into groups and start one engine on each: these models need every
    # card for one engine, and the replicas' path has not been run with a family engine
    if cfg.get("replicas") not in (None, False, 0, 1):
        bad.append('"replicas" / --replicas (one engine holds the model on every card)')
    if bad:
        raise ValueError('"family": %r does not support %s' % (spec.key, "; ".join(bad)))
