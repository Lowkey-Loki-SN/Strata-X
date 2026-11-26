"""moex: a streamed request whose prepare() runs long (its pictures encoded again after the server's start: ~15 s each)
starts its stream after PREPARE_BEAT_S and sends keep-alive comments until prepare() returns - Node's fetch gives up on a
response whose headers take more than 300 s - and an error prepare() raises after that goes out as the API's stream
error; a quick prepare() and a non-streamed request answer as before.

    python -m unittest serve.test_prepare_keepalive -v
"""
import http.client
import json
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]


class PrepareKeepAlive(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        for name, v in (("PREPARE_BEAT_S", 0.1), ("KEEPALIVE_S", 0.1)):
            p = mock.patch.object(server, name, v)
            p.start()
            self.addCleanup(p.stop)
        self.real_prepare = self.svc.prepare

    def slow_prepare(self, delay, fail=None):
        def prepare(*a, **k):
            time.sleep(delay)
            if fail is not None:
                raise fail
            return self.real_prepare(*a, **k)
        p = mock.patch.object(self.svc, "prepare", prepare)
        p.start()
        self.addCleanup(p.stop)

    def post(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        self.addCleanup(c.close)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, r.read()

    def test_a_slow_prepare_streams_keep_alives_then_the_answer(self):
        self.slow_prepare(0.6)
        status, out = self.post("/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        head, _, rest = out.partition(b"data: ")
        self.assertGreaterEqual(head.count(b": keep-alive"), 2, out)   # before the first chunk
        self.assertIn(b"[DONE]", rest)
        self.assertIn(b'"finish_reason": "stop"', rest)

    def test_a_slow_prepare_on_the_messages_api(self):
        self.slow_prepare(0.6)
        status, out = self.post("/v1/messages", {"stream": True, "max_tokens": 8,
                                                 "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        head, _, rest = out.partition(b"event: message_start")
        self.assertGreaterEqual(head.count(b": keep-alive"), 2, out)
        self.assertIn(b"event: message_stop", rest)

    def test_an_error_after_the_stream_started_is_a_stream_error(self):
        self.slow_prepare(0.4, ValueError("the prompt is too long"))
        status, out = self.post("/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)                                  # (the headers had gone out)
        self.assertIn(b": keep-alive", out)
        self.assertIn(b'"type": "invalid_request_error"', out)
        self.assertIn(b"the prompt is too long", out)
        self.assertTrue(out.rstrip().endswith(b"data: [DONE]"), out)
        status, out = self.post("/v1/messages", {"stream": True, "max_tokens": 8,
                                                 "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertIn(b"event: error", out)
        self.assertIn(b'"type": "invalid_request_error"', out)

    def test_a_slow_prepare_on_the_responses_api(self):
        # the audit of 2026-10-24: /v1/responses had no beat - its headers waited for prepare() as the other routes' did
        self.slow_prepare(0.6)
        status, out = self.post("/v1/responses", {"stream": True, "input": "hi"})
        self.assertEqual(status, 200)
        head, _, rest = out.partition(b"event: response.created")
        self.assertGreaterEqual(head.count(b": keep-alive"), 2, out)
        self.assertIn(b"event: response.completed", rest)

    def test_an_error_after_the_stream_started_on_the_responses_api(self):
        self.slow_prepare(0.4, ValueError("the prompt exceeds the context"))
        status, out = self.post("/v1/responses", {"stream": True, "input": "hi"})
        self.assertEqual(status, 200)                                  # (the headers had gone out)
        self.assertIn(b": keep-alive", out)
        head, _, data = out.partition(b"event: error\ndata: ")
        self.assertNotIn(b"response.created", head)
        body = json.loads(data.split(b"\n\n")[0])
        self.assertEqual((body["type"], body["code"], body["param"]), ("error", "context_length_exceeded", "input"))
        self.assertIn("the prompt exceeds the context", body["message"])

    def test_a_quick_prepare_answers_as_before(self):
        self.slow_prepare(0.0, ValueError("the prompt is too long"))
        status, out = self.post("/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 400)                                  # an HTTP error, nothing streamed
        self.assertNotIn(b"keep-alive", out)
        self.assertIn(b"the prompt is too long", out)

    def test_a_slow_unstreamed_request_waits_for_its_json(self):
        self.slow_prepare(0.4)
        status, out = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertNotIn(b"keep-alive", out)
        self.assertEqual(json.loads(out)["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
