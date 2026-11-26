"""moex/tests/test_family_props.py - GET /props with each family's template (GLM-5, DeepSeek-V4, MiMo).

Upstream 0.1.41 added "chat_template_caps": svc.template.caps to /props; FamilyChatTemplate never set `caps` (its
__init__ does not run ChatTemplate's), so every /props raised AttributeError and a client polling it filled the
server's console with tracebacks.  The family template now probes its own capabilities (family_template.py
_family_caps): what each template renders, not Qwen's <tool_call><function=...> syntax that upstream's probe reads.
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest
import urllib.request

X = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(X))
sys.path.insert(0, str(X / "tools"))

from serve import families  # noqa: E402
from serve.server import MockEngine, Service, serve  # noqa: E402

# supports_preserve_reasoning: DeepSeek-V4's template drops a past turn's reasoning; GLM's and MiMo's keep it
PRESERVE_REASONING = {"glm5": True, "deepseek4": False, "mimo2": True}


class FamilyProps(unittest.TestCase):
    def test_props_answers_with_the_family_caps(self):
        for key, preserve in PRESERVE_REASONING.items():
            with self.subTest(family=key):
                spec = families.get(key)
                tok = spec.load_tokenizer(spec.tokenizer_path())
                svc = Service(MockEngine(tok, "hi", max_context=4096), tok, spec.load_template(spec.tokenizer_path()),
                              model_name=spec.model_name, family=spec)
                httpd = serve(svc, port=0)
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/props", timeout=30) as r:
                        self.assertEqual(r.status, 200)
                        caps = json.loads(r.read())["chat_template_caps"]
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                for name in ("supports_tools", "supports_tool_calls", "supports_system_role",
                             "supports_parallel_tool_calls"):
                    self.assertIs(caps[name], True, name)
                self.assertIs(caps["supports_preserve_reasoning"], preserve)


if __name__ == "__main__":
    unittest.main()
