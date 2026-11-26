#!/usr/bin/env python3
"""moex/tests/test_family_images.py - the server's image half for the family engines, without an engine or an encoder.

    python3 -B -m unittest moex/tests/test_family_images.py      (or run the file)

Service.prepare() with a stand-in encoder (each picture: a file and its number of rows): one pad per image row behind the
family's start marker (GLM <|begin_of_image|><|image|>, MiMo <|vision_start|><|image_pad|>), the end marker after the
run, a marker typed as TEXT inside a message staying text, two pictures in order, the engine's --image-pad flag, and the
"vision" config refused for a family whose engine reads no images (DeepSeek-V4).

DeepSeek-V4.1 (moex, DSV41_BACKLOG #57): one <｜deepseek_image｜> per row of the encoder's span (no start or end marker),
several pictures in order, the placeholder typed as text staying text, a user's and tools' pictures in one turn; pictures
in tool results that came out of call order each get their own placeholder, in the OpenAI and the Anthropic shape, as the
release's encoder collects them, with the prompt unchanged (#44); count_tokens counts a picture's rows by the release's
plan, the length of the prompt prepare() builds (#56); its optional encoder (#49): one that does not start leaves the server
reading text, reporting no images, and answering a picture with a 400 that says why; other families' requests fail as
before.
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import base64
import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import threading
import dataclasses
import unittest
import urllib.error
import urllib.request

X = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(X))
sys.path.insert(0, str(X / "tools"))

from unittest import mock  # noqa: E402

from serve import families  # noqa: E402
from serve import server as S  # noqa: E402
from serve import test_deepseek41 as D41  # noqa: E402  - its conversations, the release's encoder (or None)
from serve.frontend import (Dialect, anthropic_to_messages, images_of, openai_to_messages,  # noqa: E402
                            tool_results_in_call_order)

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class FakeVision:
    """Vision's interface for prepare(): encode(source) -> (embeddings file, rows); `dir` for the combined file.
    `rows`: a list, taken in turn, or (moex) a mapping of a picture's source to its rows - each file then holds its
    source and a bar, so the combined file shows the order the engine gets the pictures in."""

    def __init__(self, rows):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="fake-vision-"))
        self.rows = dict(rows) if isinstance(rows, dict) else list(rows)
        self.calls = 0
        self.seen = []

    def encode(self, source):
        n = self.rows[source] if isinstance(self.rows, dict) else self.rows[self.calls % len(self.rows)]
        self.calls += 1
        self.seen.append(source)
        p = self.dir / f"img{self.calls}.sve"
        p.write_bytes(f"{source}|".encode() if isinstance(self.rows, dict) else b"\0" * 20)
        return p, n


class FakeEngine:
    max_context = 32768


def service(key, rows):
    spec = families.get(key)
    tok = spec.load_tokenizer(spec.tokenizer_path())
    svc = S.Service(FakeEngine(), tok, spec.load_template(spec.tokenizer_path()), model_name=spec.model_name,
                    vision=FakeVision(rows), family=spec)
    return svc, tok, spec


def request(*parts):
    content = []
    for p in parts:
        content.append({"type": "image_url", "image_url": {"url": PNG}} if p == "IMG" else {"type": "text", "text": p})
    return {"messages": [{"role": "user", "content": content}]}


class FamilyImages(unittest.TestCase):
    def expand(self, key, rows, *parts):
        svc, tok, spec = service(key, rows)
        msgs, tools, kw = openai_to_messages(request(*parts), svc.dialect)   # the family's (glm5, mimo2: the default)
        ids, _, _ = svc.prepare(msgs, tools, kw, max_new=16)
        path = getattr(svc.embeddings, "path", None)
        svc.drop_embeddings()
        return ids, tok, spec, path

    def check_runs(self, key, start_id, pad_id, end_id):
        ids, tok, spec, path = self.expand(key, [5], "IMG", "Describe this image.")
        self.assertIsNotNone(path, "prepare() names a combined embeddings file for GENI")
        j = ids.index(start_id)
        self.assertEqual(ids[j + 1:j + 6], [pad_id] * 5)
        self.assertEqual(ids[j + 6], end_id)
        self.assertEqual(ids.count(pad_id), 5)
        # two pictures, in order, each its own run
        ids2, _, _, _ = self.expand(key, [3, 7], "IMG", "and", "IMG", "compare them")
        runs, cur = [], 0
        for t in ids2 + [-1]:
            if t == pad_id:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        self.assertEqual(runs, [3, 7])
        # the marker typed as text inside the message is text (its literal pieces), not a pad
        text = spec.image_tokens[1]
        ids3, _, _, _ = self.expand(key, [4], "IMG", f"what does {text} mean?")
        self.assertEqual(ids3.count(pad_id), 4)

    def test_glm5(self):
        self.check_runs("glm5", 154830, 154854, 154831)

    def test_mimo2(self):
        self.check_runs("mimo2", 151652, 151655, 151653)

    def test_deepseek41(self):
        # moex (DSV41_BACKLOG #57): no start or end marker - the template's <｜deepseek_image｜> is the picture's place,
        # once per row of the encoder's span (its delimiter rows are rows of the record: no lead pads to trim)
        ids, tok, spec, path = self.expand("deepseek41", [5], "IMG", "Describe this image.")
        self.assertIsNotNone(path, "prepare() names a combined embeddings file for GENI")
        self.assertEqual(runs(ids, DS_PAD), [5])
        j = ids.index(DS_PAD)
        self.assertEqual(tok.decode(ids[:j]).rsplit("<｜User｜>", 1)[1], "")   # the user's turn starts with the picture
        self.assertTrue(tok.decode(ids[j + 5:]).startswith("\n\nDescribe this image."))   # the parts joined as the release does
        # several pictures, in order, each its own run - two of them side by side
        ids2, _, _, _ = self.expand("deepseek41", [3, 7, 4], "IMG", "and", "IMG", "IMG", "compare them")
        self.assertEqual(runs(ids2, DS_PAD), [3, 7, 4])
        # the placeholder typed as text inside the message is text, not a picture's place
        ids3, _, _, _ = self.expand("deepseek41", [4], "IMG", f"what does {spec.image_tokens[1]} mean?")
        self.assertEqual(runs(ids3, DS_PAD), [4])
        self.assertIn(spec.image_tokens[1], tok.decode(ids3))

    def test_deepseek41_user_and_tool_pictures(self):
        # a user's picture, two tool results with pictures that came out of call order, and a user's picture after them in
        # the same turn: each run where the template puts its picture, with that picture's rows (item #44)
        svc, tok, spec = service("deepseek41", {"u1.png": 2, "a.png": 3, "b.png": 7, "u2.png": 5})
        img = lambda f: {"type": "image_url", "image_url": {"url": f}}             # noqa: E731
        body = {"tools": [{"type": "function", "function": READ}], "messages": [
            {"role": "user", "content": [img("u1.png"), {"type": "text", "text": "Read a.png and b.png too."}]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": c, "type": "function", "function": {"name": "Read", "arguments": json.dumps({"file_path": f})}}
                for c, f in (("c1", "a.png"), ("c2", "b.png"))]},
            {"role": "tool", "tool_call_id": "c2", "content": [img("b.png")]},
            {"role": "tool", "tool_call_id": "c1", "content": [img("a.png")]},
            {"role": "user", "content": [img("u2.png"), {"type": "text", "text": "And this one?"}]}]}
        msgs, tools, kw = openai_to_messages(body, svc.dialect)
        ids, _, _ = svc.prepare(msgs, tools, kw, max_new=16)
        data = pathlib.Path(svc.embeddings.path).read_bytes()
        svc.drop_embeddings()
        self.assertEqual(runs(ids, DS_PAD), [2, 3, 7, 5])
        self.assertEqual(data, b"u1.png|a.png|b.png|u2.png|")
        self.assertEqual(svc.vision.calls, 4)                                       # each picture encoded once

    def test_engine_flag(self):
        cfg = {"vision": {"exe": "x", "mmproj": "y", "model": "z"}, "args": []}
        self.assertEqual(S.family_engine_args(families.get("glm5"), cfg)[-2:], ["--image-pad", "154854"])
        self.assertEqual(S.family_engine_args(families.get("mimo2"), cfg)[-2:], ["--image-pad", "151655"])
        self.assertEqual(S.family_engine_args(families.get("deepseek41"), cfg)[-2:], ["--image-pad", "129264"])
        self.assertNotIn("--image-pad", S.family_engine_args(families.get("glm5"), {"args": []}))

    def test_vision_config(self):
        # all of them read images now (DeepSeek-V4 since 77306402, its span read non-causally); one that does not is refused
        for key in ("glm5", "mimo2", "deepseek4", "deepseek41"):
            families.check_config({"vision": {"exe": "x"}}, families.get(key))      # accepted
        blind = dataclasses.replace(families.get("glm5"), vision_ready=False)
        with self.assertRaises(ValueError):
            families.check_config({"vision": {"exe": "x"}}, blind)


DS_PAD = 129264                                         # DeepSeek-V4.1's <｜deepseek_image｜>
READ = {"name": "Read", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}


def runs(ids, pad):
    """The lengths of the runs of `pad` in `ids`, in order (one per picture)."""
    out, cur = [], 0
    for t in list(ids) + [-1]:
        if t == pad:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    return out


def tool_pictures(order, anthropic=False, files=("a.png", "b.png")):
    """A client's body: two Read calls (c1 reads a.png, c2 b.png), their results - each a picture - in `order`, then a
    question; in the OpenAI shape or Anthropic's (ids t1, t2)."""
    ids = ["t1", "t2"] if anthropic else ["c1", "c2"]
    file = dict(zip(ids, files))
    if anthropic:
        picture = lambda f: {"type": "image", "source": {"type": "url", "url": f}}             # noqa: E731
        return {"tools": [{"name": "Read", "description": READ["description"], "input_schema": READ["parameters"]}],
                "messages": [{"role": "user", "content": "Compare a.png and b.png."},
                             {"role": "assistant", "content": [{"type": "tool_use", "id": t, "name": "Read",
                                                                "input": {"file_path": file[t]}} for t in ids]},
                             {"role": "user", "content": [
                                 {"type": "tool_result", "tool_use_id": ids[k], "content": [
                                     {"type": "text", "text": file[ids[k]] + ":"}, picture(file[ids[k]])]}
                                 for k in order] + [{"type": "text", "text": "Which is larger?"}]}]}
    picture = lambda f: {"type": "image_url", "image_url": {"url": f}}                         # noqa: E731
    return {"tools": [{"type": "function", "function": READ}], "messages": [
        {"role": "user", "content": "Compare a.png and b.png."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": c, "type": "function", "function": {"name": "Read", "arguments": json.dumps({"file_path": file[c]})}}
            for c in ids]},
        *[{"role": "tool", "tool_call_id": ids[k],
           "content": [{"type": "text", "text": file[ids[k]] + ":"}, picture(file[ids[k]])]} for k in order],
        {"role": "user", "content": "Which is larger?"}]}


class ToolPictureOrder(unittest.TestCase):
    """moex (DSV41_BACKLOG #44): DeepSeek-V4.1's template puts a turn's tool results in the order of their calls; the
    server puts the tool messages in that order before it collects the pictures (tool_results_in_call_order), so each
    placeholder gets its own picture - results that came in another order used to get each other's - and the prompt is
    what it was."""

    @classmethod
    def setUpClass(cls):
        cls.svc, cls.tok, cls.spec = service("deepseek41", {"a.png": 3, "b.png": 7, "c.png": 2})

    def prepared(self, msgs, tools, kw):
        ids, _, _ = self.svc.prepare(msgs, tools, kw, max_new=16)
        path = self.svc.embeddings.path
        data = pathlib.Path(path).read_bytes() if path else b""
        self.svc.drop_embeddings()
        return ids, data

    def test_each_placeholder_gets_its_own_picture(self):
        for anthropic in (False, True):
            convert = (lambda r: anthropic_to_messages(r, True, self.svc.dialect)) if anthropic else \
                (lambda r: openai_to_messages(r, self.svc.dialect))
            for order in ((0, 1), (1, 0)):
                with self.subTest(anthropic=anthropic, order=order):
                    msgs, tools, kw = convert(tool_pictures(order, anthropic))
                    ids, data = self.prepared(msgs, tools, kw)
                    prompt = self.tok.decode(ids)
                    self.assertLess(prompt.index("a.png:"), prompt.index("b.png:"))   # the template's call order
                    self.assertEqual(runs(ids, DS_PAD), [3, 7])          # a.png's rows at a.png's place
                    self.assertEqual(data, b"a.png|b.png|")              # and the engine gets them in that order

    def test_the_prompt_is_the_templates(self):
        # the template sorts the results itself, so the prompt is the same with the server's sort or without it; a
        # conversation already in order is handed back as it is
        msgs, tools, kw = openai_to_messages(tool_pictures((1, 0)), self.svc.dialect)
        moved = tool_results_in_call_order(msgs, self.svc.dialect, True)
        self.assertEqual([m.get("tool_call_id") for m in moved], [None, None, "c1", "c2", None])
        render = self.svc.template.render
        self.assertEqual(render(moved, tools=tools, **kw), render(msgs, tools=tools, **kw))
        ordered, _, _ = openai_to_messages(tool_pictures((0, 1)), self.svc.dialect)
        self.assertIs(tool_results_in_call_order(ordered, self.svc.dialect, True), ordered)
        self.assertEqual(self.svc.encode_prompt(ordered, tools, kw), self.svc.encode_prompt(moved, tools, kw))
        # every other family: no call ids, nothing moves
        plain, _, _ = openai_to_messages(tool_pictures((1, 0)))
        self.assertIs(tool_results_in_call_order(plain, Dialect(), True), plain)
        self.assertIs(tool_results_in_call_order(msgs, Dialect(), True), msgs)

    def test_turns_as_the_template_sees_them(self):
        D = self.svc.dialect
        img = lambda f: [{"type": "image", "source": f}]                             # noqa: E731
        call = lambda *cs: {"role": "assistant", "content": "", "tool_calls": [      # noqa: E731
            {"id": c, "function": {"name": "Read", "arguments": {}}} for c in cs]}
        tool = lambda c, f: {"role": "tool", "tool_call_id": c, "content": img(f)}   # noqa: E731
        user = {"role": "user", "content": "go"}
        empty = {"role": "assistant", "content": ""}
        thought = {"role": "assistant", "content": "", "reasoning_content": "r"}
        cases = [
            # an id the calls do not have sorts as the first call (stable)
            ([user, call("c1", "c2"), tool("zz", "c.png"), tool("c2", "b.png"), tool("c1", "a.png")],
             ["c.png", "a.png", "b.png"]),
            # an empty assistant turn between them is left out by the renderer (#843): one turn for the template
            ([user, call("c1", "c2"), tool("c2", "b.png"), empty, tool("c1", "a.png"), user], ["a.png", "b.png"]),
            # a turn with reasoning alone is kept for this family: two turns of one result each, nothing moves
            ([user, call("c1", "c2"), tool("c2", "b.png"), thought, tool("c1", "a.png"), user], ["b.png", "a.png"]),
            # the order is the last calls' (a later turn without calls keeps it), per turn
            ([user, call("c1", "c2"), tool("c2", "b.png"), tool("c1", "a.png"), {"role": "assistant", "content": "ok"},
              tool("c2", "c.png"), tool("c1", "a.png")], ["a.png", "b.png", "a.png", "c.png"]),
        ]
        for msgs, want in cases:
            with self.subTest(msgs=[m.get("tool_call_id") or m["role"] for m in msgs]):
                moved = tool_results_in_call_order(msgs, D, True)
                self.assertEqual(images_of(moved), want)
                ids, data = self.prepared(msgs, None, {})
                self.assertEqual(runs(ids, DS_PAD), [{"a.png": 3, "b.png": 7, "c.png": 2}[f] for f in want])
                self.assertEqual(data, "".join(f + "|" for f in want).encode())
        # STRATA_KEEP_EMPTY_TURNS=1 renders the empty turn: then it ends the run, and nothing moves
        with mock.patch.dict(os.environ, {"STRATA_KEEP_EMPTY_TURNS": "1"}):
            msgs = cases[1][0]
            self.assertIs(tool_results_in_call_order(msgs, D, True), msgs)
            self.assertEqual(runs(self.prepared(msgs, None, {})[0], DS_PAD), [7, 3])

    @unittest.skipIf(D41.enc is None, "the release's encoder is not here (DSV41_ENCODING)")
    def test_the_release_collects_them_in_this_order(self):
        # the release's encoder (encoding.py encode_messages): its pictures, collected after its sort, are ours
        for anthropic in (False, True):
            for order in ((0, 1), (1, 0)):
                with self.subTest(anthropic=anthropic, order=order):
                    body = tool_pictures(order, anthropic, files=("x/a.png", "x/b.png"))
                    msgs, tools, kw = (anthropic_to_messages(body, True, D41.DIALECT) if anthropic else
                                       openai_to_messages(body, D41.DIALECT))
                    _, media = D41.enc.encode_messages(D41.reference_input(msgs, tools), thinking_mode="chat",
                                                       return_multi_modal_data=True)
                    self.assertEqual(images_of(tool_results_in_call_order(msgs, D41.DIALECT, True)),
                                     [r["source"] for r in media["images"]])
                    self.assertEqual([r["source"] for r in media["images"]], ["x/a.png", "x/b.png"])

    def test_every_template_case_renders_as_before(self):
        # serve/test_deepseek41.py's conversations, every request variant: the sort moves only results out of call
        # order, and the prompt never changes
        for name, case in D41.CASES.items():
            moved = tool_results_in_call_order(case["messages"], D41.DIALECT, True)
            out_of_order = name in ("tool_results_out_of_order", "tool_results_unknown_ids")
            self.assertEqual(moved is not case["messages"], out_of_order, name)
            for variant, kwargs in D41.VARIANTS.items():
                for gen in (True, False):
                    with self.subTest(case=name, variant=variant, gen=gen):
                        self.assertEqual(D41.render(moved, case.get("tools"), gen, **kwargs),
                                         D41.render(case["messages"], case.get("tools"), gen, **kwargs))


# DeepSeek-V4.1's vision keys, as in the release's inference/config.json (what its image plan reads)
VISION_KEYS = {"vision_n_layers": 32, "vision_patch_size": 14, "vision_downsample_ratio": 3, "vision_max_n_token": 1024,
               "vision_min_pixels": 295936, "vision_max_wh_ratio": None}
RELEASE = pathlib.Path(os.environ.get("DSV41_REF", _os.path.join(_MODELS, 'dsv41-1017/reference/inference')))


def picture(width, height, fmt="PNG", orientation=None) -> bytes:
    from PIL import Image
    im = Image.new("RGB", (width, height), (40, 90, 200))
    out = io.BytesIO()
    if orientation is None:
        im.save(out, format=fmt)
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        im.save(out, format=fmt, exif=exif)
    return out.getvalue()


def data_url(data: bytes, mime="image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


class PictureTokens(unittest.TestCase):
    """moex (DSV41_BACKLOG #56): count_tokens counted a picture as its one placeholder token; for DeepSeek-V4.1 it counts
    the rows the encoder gives it - the release's plan for its size (serve/vision_dsv41_plan.py, which the encoder runs),
    read from the picture's header - so the count is the length of the prompt prepare() builds.  Other families: as
    before."""

    @classmethod
    def setUpClass(cls):
        d = pathlib.Path(tempfile.mkdtemp(prefix="dsv41-config-"))
        (d / "config.json").write_text(json.dumps({"dim": 5120, **VISION_KEYS}), encoding="utf-8")
        cls.vision_cfg = {"exe": "vision_dsv41.py", "mmproj": "dense", "model": str(d / "config.json")}
        cls.dir = d

    def service(self, key, rows, cfg=True):
        spec = families.get(key)
        tok = spec.load_tokenizer(spec.tokenizer_path())
        vision = FakeVision(rows)
        if cfg:
            vision.cfg = dict(self.vision_cfg)
        return S.Service(FakeEngine(), tok, spec.load_template(spec.tokenizer_path()), model_name=spec.model_name,
                         vision=vision, family=spec), spec

    def test_the_plan_is_the_releases(self):
        if not (RELEASE / "image_processor.py").is_file():
            self.skipTest(f"the release's image_processor.py is not in {RELEASE} (DSV41_REF)")
        import importlib.util
        import types
        stub = types.ModuleType("torch")                  # its plan needs no torch; its dataclass names torch.Tensor
        stub.Tensor = object
        with mock.patch.dict(sys.modules, {"torch": stub}):
            spec = importlib.util.spec_from_file_location("dsv41_ref_image_processor_plan", RELEASE / "image_processor.py")
            ref = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ref)
        cfg = json.loads((RELEASE / "config.json").read_text(encoding="utf-8"))
        self.assertEqual({k: cfg[k] for k in VISION_KEYS}, VISION_KEYS)
        from serve import vision_dsv41_plan as P
        sides = (1, 2, 13, 14, 15, 41, 42, 43, 100, 150, 200, 333, 480, 543, 640, 1000, 1023, 1600, 1800, 2400, 4032,
                 9000, 30000)
        for cap in (0, 300, 1024):
            args = types.SimpleNamespace(**{k: v for k, v in cfg.items() if k.startswith("vision_")})
            if cap:
                args.vision_max_n_token = min(cap, cfg["vision_max_n_token"])
            for w in sides:
                for h in sides:
                    n_h, n_w, _, _ = ref.plan_image_grid(w, h, args)
                    self.assertEqual(P.image_rows(w, h, cfg, cap), ref.num_image_tokens(n_h, n_w), (w, h, cap))

    def test_the_count_is_the_prompts_length(self):
        jpeg = picture(1600, 300, "JPEG", orientation=6)                 # a phone's portrait photo, stored turned
        from PIL import Image
        self.assertEqual(Image.open(io.BytesIO(S.Vision.normalize(jpeg))).size, (300, 1600))
        self.assertEqual(S.Vision.normalized_size(jpeg), (300, 1600))  # what the encoder is handed: upright
        sources = {data_url(picture(200, 150)): (200, 150), data_url(picture(2400, 1800)): (2400, 1800),
                   data_url(picture(640, 480)): (640, 480), data_url(jpeg, "image/jpeg"): (300, 1600)}
        from serve import vision_dsv41_plan as P
        rows = {src: P.image_rows(w, h, VISION_KEYS) for src, (w, h) in sources.items()}
        # the spans: 200x150 is scaled up to the minimum pixels (12 x 15 cells), 640x480 is 12 x 16, the upright photo
        # 300x1600 39 x 8, and 2400x1800 is cut down to 1512x1134 (8,748 patches, 27 x 36 cells): 27 * 37 + 2 = 1,001
        self.assertEqual(sorted(rows.values()), [194, 206, 353, 1001])
        svc, spec = self.service("deepseek41", rows)
        srcs = list(sources)
        body = {"messages": [
            {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                      "data": srcs[0].split(",", 1)[1]}},
                                         {"type": "text", "text": "What is in these?"},
                                         {"type": "image", "source": {"type": "url", "url": srcs[1]}}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "p"}},
                                              {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "j"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "image", "source": {"type": "url",
                                                                                                      "url": srcs[3]}}]},
                {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "image", "source": {"type": "url",
                                                                                                      "url": srcs[2]}}]}]}]}
        msgs, tools, kw = anthropic_to_messages(body, True, svc.dialect)
        counted = svc.count_tokens(msgs, tools, kw)
        ids, _, _ = svc.prepare(msgs, tools, kw, max_new=16)
        svc.drop_embeddings()
        self.assertEqual(counted, len(ids))
        self.assertEqual(counted, len(svc.encode_prompt(msgs, tools, kw)) + sum(rows.values()) - 4)
        self.assertEqual(sorted(runs(ids, DS_PAD)), sorted(rows.values()))

    def test_pictures_it_cannot_see(self):
        svc, spec = self.service("deepseek41", [5])

        def count(src, files=True):
            msgs = [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "source": src}]}]
            return svc.count_tokens(msgs, None, {}, files=files) - len(svc.encode_prompt(msgs, None, {}))

        with mock.patch.object(S.Vision, "download", side_effect=AssertionError("count_tokens fetched a URL")):
            self.assertEqual(count("https://example.com/a.png"), 1023)     # not fetched: the most a picture costs
        self.assertEqual(count("data:image/png;base64,AAAA"), 0)          # not a picture: its placeholder
        self.assertEqual(count("/nonexistent/picture.png"), 0)
        f = self.dir / "local.png"
        f.write_bytes(picture(640, 480))
        self.assertEqual(count(str(f)), 206 - 1)                           # a file on this PC: read
        self.assertEqual(count(str(f), files=False), 0)                    # unless a page of another site named it
        svc.vision.cfg = dict(self.vision_cfg, max_tokens=300)             # the vision block's cap (--max-tokens)
        self.assertEqual(count("https://example.com/a.png"), 300 - 1)
        self.assertEqual(count(data_url(picture(2400, 1800))), 282 - 1)    # cut to 784x588: 14 x 19 cells

    def test_other_families_and_no_encoder_count_as_before(self):
        src = data_url(picture(640, 480))
        msgs = [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "source": src}]}]
        for key, cfg in (("glm5", True), ("mimo2", True), ("deepseek41", False)):
            with self.subTest(family=key, cfg=cfg):
                svc, spec = self.service(key, [7], cfg)
                self.assertEqual(svc.count_tokens(msgs, None, {}), len(svc.encode_prompt(msgs, None, {})))
        svc, spec = self.service("deepseek41", [7])
        svc.vision = None                                                   # no encoder (or one that did not start)
        self.assertEqual(svc.count_tokens(msgs, None, {}), len(svc.encode_prompt(msgs, None, {})))

    def test_count_tokens_over_http(self):
        spec = families.get("deepseek41")
        tok = spec.load_tokenizer(spec.tokenizer_path())
        vision = FakeVision([5])
        vision.cfg = dict(self.vision_cfg)
        svc = S.Service(S.MockEngine(tok, "x", max_context=1 << 16), tok, spec.load_template(), model_name=spec.model_name,
                        vision=vision, family=spec)
        body = {"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(picture(2400, 1800)).decode()}},
            {"type": "text", "text": "Describe it."}]}]}
        msgs, tools, kw = anthropic_to_messages(json.loads(json.dumps(body)), True, svc.dialect)
        want = len(svc.encode_prompt(msgs, tools, kw)) + 1001 - 1
        httpd = S.serve(svc, port=0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages/count_tokens",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                self.assertEqual(json.loads(r.read()), {"input_tokens": want})
        finally:
            httpd.shutdown()
            httpd.server_close()


class Restartable(S.MockEngine):
    """A mock engine that is unloaded until a request starts it (alive / unload / restart, as StrataEngine)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.running, self.unloaded, self.starts = False, True, 0

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def restart(self):
        self.running, self.unloaded = True, False
        self.starts += 1


class Unstartable:
    """An image encoder that never starts (its interpreter gone): Vision's alive / restart / unload / shutdown."""

    def __init__(self):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="fake-vision-"))
        self.restarts, self.shut = 0, False

    def alive(self):
        return False

    def restart(self):
        self.restarts += 1
        raise OSError(2, "No such file or directory", "/nonexistent/venv/bin/python")

    def unload(self):
        pass

    def shutdown(self):
        self.shut = True


class OptionalEncoder(unittest.TestCase):
    """moex (DSV41_BACKLOG #49): DeepSeek-V4.1's image encoder (on another venv's torch) is optional: when it does not start
    the server reads text, reports no images and answers a picture with a 400 that says why; other families as before."""

    WHY = "the vision encoder did not start: [Errno 2] No such file or directory: '/nonexistent/venv/bin/python'"

    def get(self, base, path):
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return json.loads(r.read())

    def post(self, base, body):
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_text_only_after_a_failed_start(self):
        spec = families.get("deepseek41")
        tok = spec.load_tokenizer(spec.tokenizer_path())
        svc = S.Service(S.MockEngine(tok, "fine", max_context=4096), tok, spec.load_template(), model_name=spec.model_name,
                        vision=FakeVision([5]), family=spec)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            svc.vision_failed(self.WHY)
        self.assertEqual(out.getvalue(), S.vision_off_line(self.WHY) + "\n")
        self.assertIsNone(svc.vision)
        httpd = S.serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            self.assertIs(self.get(base, "/health")["images"], False)
            self.assertEqual(self.get(base, "/v1/models")["data"][0]["architecture"]["input_modalities"], ["text"])
            self.assertIs(self.get(base, "/props")["modalities"]["vision"], False)
            self.assertEqual(self.get(base, "/v1/status")["vision"]["error"], self.WHY)
            status, body = self.post(base, request("IMG", "What is this?"))
            self.assertEqual(status, 400)
            self.assertIn(self.WHY, body["error"]["message"])
            self.assertIn("reads text only", body["error"]["message"])
            status, body = self.post(base, {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8})
            self.assertEqual(status, 200)                           # text as before
            self.assertTrue(body["choices"][0]["message"]["content"].startswith("fine"))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_a_lazy_start_that_fails(self):
        for key, optional in (("deepseek41", True), ("glm5", False), ("mimo2", False)):
            with self.subTest(family=key):
                spec = families.get(key)
                tok = spec.load_tokenizer(spec.tokenizer_path())
                engine, vision = Restartable(tok, "fine", max_context=4096), Unstartable()
                svc = S.Service(engine, tok, spec.load_template(spec.tokenizer_path()), model_name=spec.model_name,
                                vision=vision, family=spec)
                with contextlib.redirect_stdout(io.StringIO()):
                    if optional:
                        svc.load()                                  # the engine starts, for text
                    else:
                        with self.assertRaises(S.EngineDied):       # as always: the request fails, the engine waits
                            svc.load()
                self.assertEqual((engine.starts, vision.restarts), (1 if optional else 0, 1))
                self.assertEqual(svc.vision is None, optional)
                self.assertEqual(vision.shut, optional)
                if optional:
                    self.assertTrue(svc.vision_error.startswith("the vision encoder did not start: "), svc.vision_error)
                    self.assertIn("/nonexistent/venv/bin/python", svc.vision_error)
                    with contextlib.redirect_stdout(io.StringIO()):
                        svc.load()                                  # loaded, and no second try of the encoder
                    self.assertEqual((engine.starts, vision.restarts), (1, 1))
                    msgs, tools, kw = openai_to_messages(request("IMG", "and this?"))
                    with self.assertRaisesRegex(ValueError, "cannot read images: the vision encoder did not start"):
                        svc.prepare(msgs, tools, kw, max_new=16)
                else:
                    self.assertIsNone(svc.vision_error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
