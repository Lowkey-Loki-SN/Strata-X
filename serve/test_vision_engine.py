"""moex (DSV41_PLAN §54): the image encoder in the engine, the server's side - the weights' manifest and a picture's patches
file (serve/vision_dsv41_plan.py), the VENC exchange (StrataEngine.vision_encode) and EngineVision's cache, one VENC for a
request's new pictures, its refusals.  No GPU, no model: a fake engine answers VENC.  (The patches against torch's, bit for
bit: moex/tools/dsv41/checks/vision_patches_parity.py; the GPU's rows against the CPU encoder's: vision_gpu_parity.py.)

    python -m unittest serve.test_vision_engine -v
"""
import io
import json
import queue
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import serve.server as server
from serve import vision_dsv41_plan as plan
from serve.server import ClientGone, EngineDied, EngineSilent, EngineVision, StrataEngine

CFG = {"vision_n_layers": 1, "vision_dim": 1024, "vision_n_heads": 16, "vision_inter_dim": 2816, "vision_patch_size": 14,
       "vision_downsample_ratio": 3, "vision_max_n_token": 1024, "vision_min_pixels": 295936, "vision_max_wh_ratio": None,
       "vision_rope_theta": 10000}


def png(w, h, seed=0) -> bytes:
    a = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3)).astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(a, "RGB").save(b, "PNG")
    return b.getvalue()


def write_dense(d: Path, cfg=CFG, dtype="BF16", drop=()):
    """A release-like dense directory: the encoder's tensors (tiny shapes: the manifest does not check them) in two
    safetensors files and model.safetensors.index.json."""
    names = [n for n in plan.vision_tensor_names(cfg) if n not in drop]
    files = {"model-1.safetensors": names[: len(names) // 2], "model-2.safetensors": names[len(names) // 2:]}
    wmap = {}
    for fname, ns in files.items():
        head, data, off = {}, b"", 0
        for k, n in enumerate(ns):
            shape = [2, k + 1]
            nb = 2 * 2 * (k + 1)
            head[n] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nb]}
            data += bytes(range(nb % 256)) + b"\0" * (nb - nb % 256)
            off += nb
            wmap[n] = fname
        hj = json.dumps(head).encode()
        (d / fname).write_bytes(struct.pack("<Q", len(hj)) + hj + data)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap}))
    return files


class Manifest(unittest.TestCase):
    def test_every_tensor_by_its_file_offset(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write_dense(d)
            out = d / "m.svm"
            plan.write_manifest(out, d, CFG)
            lines = out.read_text().splitlines()
            self.assertEqual(lines[0], "SVM1")
            self.assertEqual(len(lines) - 1, len(plan.vision_tensor_names(CFG)))
            for line in lines[1:]:
                tag, name, off, nb, shape, path = line.split(" ", 5)
                with open(path, "rb") as f:                       # the offset is the file's: header + its data offset
                    hlen = struct.unpack("<Q", f.read(8))[0]
                    h = json.loads(f.read(hlen))[name]
                self.assertEqual((tag, int(off), int(nb)), ("T", 8 + hlen + h["data_offsets"][0],
                                                             h["data_offsets"][1] - h["data_offsets"][0]))
                self.assertEqual(shape, "x".join(map(str, h["shape"])))
                self.assertTrue(Path(path).is_absolute())

    def test_a_missing_or_wrong_tensor_refuses(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write_dense(d, drop=("vision.blocks.0.mlp.w2.weight",))
            with self.assertRaisesRegex(ValueError, "mlp.w2.weight: not in"):
                plan.write_manifest(d / "m.svm", d, CFG)
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write_dense(d, dtype="F32")
            with self.assertRaisesRegex(ValueError, "BF16 expected"):
                plan.write_manifest(d / "m.svm", d, CFG)


class Patches(unittest.TestCase):
    def test_the_file_is_its_header_and_the_bf16_patches(self):
        a, vh, vw, lh, lw = plan.patches_of(png(640, 480), CFG, 1024)
        self.assertEqual((a.dtype, a.shape), (np.uint16, (vh * vw, 588)))
        self.assertEqual((lh, lw), (-(-vh // 3), -(-vw // 3)))
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.svp"
            plan.write_patches(p, a, vh, vw, lh, lw)
            b = p.read_bytes()
        self.assertEqual(struct.unpack("<6i", b[:24]), (plan.SVP_MAGIC, vh, vw, lh, lw, 588))
        self.assertEqual(b[24:], a.tobytes())

    def test_grey_padding_rounds_as_torch_does(self):
        # a black picture wider than its plan pads with grey 127: (127/255 - 0.5)/0.5 = -0.0039215684 (mantissa 0x008080, past
        # the halfway point) -> bf16 0xbb81, torch's own (checked: ((tensor(127.) / 255 - 0.5) / 0.5).to(bfloat16))
        b = io.BytesIO()
        Image.new("RGB", (900, 300), (0, 0, 0)).save(b, "PNG")
        a, vh, vw, _, _ = plan.patches_of(b.getvalue(), CFG, 1024)
        vals = set(np.unique(a).tolist())
        self.assertEqual(vals, {0xBF80, 0xBB81})                 # -1.0 (black) and the grey's bf16


class FakeProc:
    def __init__(self):
        self.sent = []
        self.killed = False

        class In:
            def write(_, s):
                self.sent.append(s)

            def flush(_):
                pass

        self.stdin = In()

    def poll(self):
        return 1 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def fake_engine(lines):
    e = StrataEngine.__new__(StrataEngine)
    e.proc = FakeProc()
    e.lines = queue.Queue()
    for x in lines:
        e.lines.put(x)
    e.ended = False
    import threading
    e.wlock = threading.Lock()
    return e


class VencExchange(unittest.TestCase):
    def test_the_answer(self):
        e = fake_engine(["INFO something\n", "VOK 206,968 1234.5 free\n"])
        rows, ms, how = e.vision_encode([("/a.svp", "/a.sve"), ("/b.svp", "/b.sve")], 5.0)
        self.assertEqual((rows, ms, how), ([206, 968], 1234.5, "free"))
        self.assertEqual(e.proc.sent, ["VENC 2 /a.svp /a.sve /b.svp /b.sve\n"])

    def test_a_refusal_keeps_the_engine(self):
        e = fake_engine(["ERR the image encoder's workspace fits neither ...\n"])
        with self.assertRaisesRegex(ValueError, "workspace fits neither"):
            e.vision_encode([("/a.svp", "/a.sve")], 5.0)
        self.assertFalse(e.ended)

    def test_out_of_step_ends_it(self):
        for bad in ("T 5\n", "VOK 206 nan free\n", "VOK 206,10 1.0 free\n"):
            e = fake_engine([bad])
            with self.assertRaises(EngineDied):
                e.vision_encode([("/a.svp", "/a.sve")], 5.0)
            self.assertTrue(e.ended and e.proc.killed, bad)

    def test_an_engine_that_ends_or_says_nothing(self):
        e = fake_engine([None])
        with self.assertRaisesRegex(EngineDied, "stopped unexpectedly"):
            e.vision_encode([("/a.svp", "/a.sve")], 5.0)
        e = fake_engine([])
        with self.assertRaises(EngineSilent):
            e.vision_encode([("/a.svp", "/a.sve")], 0.2)
        self.assertTrue(e.ended)

    def test_a_path_with_a_space_is_refused_before_anything_is_sent(self):
        e = fake_engine([])
        with self.assertRaises(ValueError):
            e.vision_encode([("/a b.svp", "/a.sve")], 5.0)
        self.assertEqual(e.proc.sent, [])


class FakeVencEngine:
    """Answers vision_encode as the engine would: each rows file an SVE1 record of the picture's span (its rows from the
    patches file's header), or a refusal."""

    def __init__(self, refuse=None):
        self.calls = []
        self.refuse = refuse

    def vision_encode(self, pairs, timeout):
        self.calls.append([a for a, _ in pairs])
        if self.refuse:
            raise ValueError(self.refuse)
        rows = []
        for svp, sve in pairs:
            h = struct.unpack("<6i", Path(svp).read_bytes()[:24])
            n = plan.num_image_tokens(h[3], h[4])
            Path(sve).write_bytes(struct.pack("<5i", 0x31455653, n, n, 1, 4) + b"\0" * (16 * n))
            rows.append(n)
        return rows, 12.0, "free"


class EngineVisionCache(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        d = Path(self.t.name)
        write_dense(d)
        (d / "config.json").write_text(json.dumps(CFG))
        self.v = EngineVision({"gpu": True, "mmproj": str(d), "model": str(d / "config.json")})
        self.addCleanup(self.v.shutdown)
        self.eng = FakeVencEngine()
        self.v.attach(self.eng)
        self.ensured = 0

        def ensure():
            self.ensured += 1
        self.v.ensure = ensure

    def test_the_engine_args_name_the_manifest(self):
        args = self.v.engine_args()
        self.assertEqual(args[0], "--vision-weights")
        self.assertEqual(Path(args[1]).read_text().splitlines()[0], "SVM1")

    def test_new_pictures_go_in_one_call_and_are_cached(self):
        a, b = png(640, 480, 1), png(1920, 1080, 2)
        hold = []
        got = self.v.encode_all([a, b, a], hold=hold)
        self.assertEqual(len(self.eng.calls), 1)                  # one VENC for the request's new pictures
        self.assertEqual(len(self.eng.calls[0]), 2)               # the same picture twice: encoded once
        self.assertEqual([r for _, r in got], [206, 968, 206])
        self.assertEqual(got[0], got[2])
        self.assertTrue(all(p.exists() for p, _ in got))
        self.assertEqual(len(hold), 3)
        self.assertEqual(self.ensured, 1)
        again = self.v.encode_all([b])                            # the next turn: from the cache, no call
        self.assertEqual(len(self.eng.calls), 1)
        self.assertEqual(again[0], got[1])
        self.assertEqual(list(self.v.dir.glob("*.svp")), [])      # the patches files go once encoded
        self.v.release(hold)

    def test_a_tool_picture_that_cannot_be_read_becomes_its_error(self):
        out = self.v.encode_all([png(640, 480, 3), b"not a picture"], tolerate={1})
        self.assertIsInstance(out[0], tuple)
        self.assertIsInstance(out[1], ValueError)
        with self.assertRaises(ValueError):
            self.v.encode_all([b"not a picture"])

    def test_an_engine_refusal(self):
        self.v.attach(FakeVencEngine(refuse="the image encoder's workspace fits neither ..."))
        with self.assertRaisesRegex(ValueError, "could not be read: the image encoder's workspace"):
            self.v.encode_all([png(640, 480, 4)])
        out = self.v.encode_all([png(640, 480, 5)], tolerate={0})  # a tool's: the error in its place
        self.assertIsInstance(out[0], ValueError)
        self.assertEqual(list(self.v.dir.glob("*.svp")), [])

    def test_a_client_that_left(self):
        import threading
        ev = threading.Event()
        ev.set()
        with self.assertRaises(ClientGone):
            self.v.encode_all([png(640, 480, 6)], cancel=ev)

    def test_encode_alone(self):
        p, n = self.v.encode(png(640, 480, 7))
        self.assertEqual(n, 206)
        self.assertTrue(p.exists())

    def test_start_vision_picks_it_for_a_family_whose_engine_encodes(self):
        class Fam:
            vision_engine = True
            vision_optional = True
        d = Path(self.t.name)
        cfg = {"gpu": True, "mmproj": str(d), "model": str(d / "config.json"), "exe": "x"}
        v, why = server.start_vision(cfg, family=Fam())
        self.addCleanup(v.shutdown)
        self.assertIsInstance(v, EngineVision)
        self.assertIsNone(why)
        cfg["model"] = str(d / "missing.json")                    # an optional family goes on without it
        v2, why2 = server.start_vision(cfg, family=Fam())
        self.assertIsNone(v2)
        self.assertIn("the vision encoder did not start", why2)


if __name__ == "__main__":
    unittest.main()
