"""serve/family_template.py - a model family's chat template, rendered in the front end's own Jinja environment.

The Qwen path renders serve/frontend.py's ChatTemplate (the pack's chat_template.jinja) with the sandboxed Jinja settings
of transformers' apply_chat_template.  A family beyond Qwen (GLM-5, DeepSeek-V4, MiMo-V2) renders ITS template the same
way: `FamilyChatTemplate` takes the settings from a ChatTemplate (it overlays that template's environment, so a change
to the front end's settings reaches the families by construction) and adds only what the family's template needs and
Strata's ChatTemplate does not provide:

  * template variables the template reads (`bos_token` for DeepSeek-V4, whose template starts with it),
  * filters the template calls (`from_json`, DeepSeek-V4),
  * the tools in the form the template reads (the OpenAI wrapper `{"type": "function", "function": {...}}`: DeepSeek-V4's
    template silently skips a tool without it; GLM accepts both; llama.cpp passes the wrapper to every template),
  * the thinking switch.  Strata's request layer speaks one dialect (frontend.effort_kwargs: `enable_thinking: false`, or
    `reasoning_effort` low / medium / xhigh); each template has its own: DeepSeek-V4 `thinking` (off by default) and
    `reasoning_effort` high / max; GLM-5.3 has no switch at all (always an open `<think>`) and only `reasoning_effort`
    low / high; MiMo `enable_thinking is false`.  `FamilySpec.thinking_kwargs` translates, and where the template cannot
    express a switch the prompt gets its tail from the family (`generation_tail`).

Nothing here is used by the Qwen path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate  # noqa: E402


def environment():
    """The front end's Jinja environment (ChatTemplate's own), as an overlay we may add filters to."""
    base = ChatTemplate(ROOT / "serve" / "chat_template.jinja").template.environment
    env = base.overlay()
    env.filters = dict(base.filters)
    env.globals = dict(base.globals)
    return env


def wrap_tools(tools):
    """Tools in the OpenAI form `{"type": "function", "function": {...}}`; a tool already in it is left as it is."""
    if not tools:
        return tools
    out = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
            out.append(t)
        else:
            out.append({"type": "function", "function": t})
    return out


class FamilyChatTemplate(ChatTemplate):
    """`render(messages, tools=None, add_generation_prompt=True, **kwargs)` as ChatTemplate's, for a family.

    `kwargs` are Strata's request kwargs (`enable_thinking`, `reasoning_effort`); the spec turns them into the template's
    variables.  `spec` is a serve.families.FamilySpec (anything with the attributes used below)."""

    def __init__(self, path: str | Path, spec):
        env = environment()
        for name, fn in getattr(spec, "template_filters", {}).items():
            env.filters[name] = fn
        self.source = Path(path).read_text(encoding="utf-8")
        self.template = env.from_string(self.source)
        self.spec = spec
        self.keep_reasoning_turns = bool(getattr(spec, "reasoning_turns", False))
        self.caps = self._family_caps()

    def _family_caps(self) -> dict[str, bool]:
        """llama.cpp's chat_template_caps for /props (upstream 0.1.41's ChatTemplate._detect_caps, which this __init__
        does not run: it looks for Qwen's <tool_call><function=...> syntax, which these templates do not write).  Whether
        the template renders tool definitions, past calls (one and two) with their results, a system turn and a past
        turn's reasoning - each probe's own markers found in the prompt.  Hints only: a probe that raises is a no."""
        def render(messages, tools=None):
            try:
                return self.render(messages, tools=tools)
            except Exception:   # noqa: BLE001 - a hint only: never stop the server for a probe
                return ""

        user = {"role": "user", "content": "strata_caps_user"}
        # the shapes the server hands render(): flat tool definitions (wrap_tools wraps them per family) and past calls as
        # frontend.py builds them from a client's history ({"function": {"name", "arguments": a mapping}})
        tools = [{"name": f"strata_caps_call_{i}", "description": "strata_caps_description",
                  "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}} for i in range(2)]
        tool_prompt = render([user], tools)

        def calls_supported(count):
            calls = [{"function": {"name": f"strata_caps_call_{i}", "arguments": {"arg": f"strata_caps_arg_{i}"}}}
                     for i in range(count)]
            messages = [user, {"role": "assistant", "content": "", "tool_calls": calls}]
            messages += [{"role": "tool", "content": f"strata_caps_result_{i}"} for i in range(count)]
            prompt = render(messages + [user], tools)
            return all(f"strata_caps_arg_{i}" in prompt and f"strata_caps_result_{i}" in prompt for i in range(count))

        history = [user, {"role": "assistant", "content": "strata_caps_answer",
                          "reasoning_content": "strata_caps_reasoning"}, user]
        return {"supports_tools": "strata_caps_call_0" in tool_prompt and "strata_caps_description" in tool_prompt,
                "supports_tool_calls": calls_supported(1),
                "supports_system_role": "strata_caps_system" in render(
                    [{"role": "system", "content": "strata_caps_system"}, user]),
                "supports_parallel_tool_calls": calls_supported(2),
                "supports_preserve_reasoning": "strata_caps_reasoning" in render(history)}

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        spec = self.spec
        variables = dict(spec.template_vars)
        variables.update(spec.thinking_kwargs(kwargs))
        tools = wrap_tools(tools) if spec.wrap_tools else tools
        prompt = super().render(messages, tools=tools, add_generation_prompt=add_generation_prompt, **variables)
        if add_generation_prompt:
            prompt += spec.generation_tail(kwargs)
        return prompt


def from_json(value):
    return json.loads(value)
