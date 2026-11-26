"""serve/quant_check.py against the engine's rules, on synthetic GGUF headers (no model files, no GPU, no network).

Run: python serve/test_quant_check.py
"""
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import quant_check as QC  # noqa: E402

F32, F16, Q4_0, Q8_0, Q4_K, Q6_K, IQ3_XXS, IQ4_XS, BF16, TQ2_0, MXFP4 = 0, 1, 2, 8, 12, 14, 18, 23, 30, 35, 39


def gguf(path, kv, tensors):
    """A GGUF v3 header: kv {key: str | int}, tensors [(name, shape, type)] (no data: the check reads headers only)."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = [struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kv))]
    for k, v in kv.items():
        out.append(s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str) else struct.pack("<II", 4, v)))
    for name, shape, t in tensors:
        out.append(s(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", t, 0))
    Path(path).write_bytes(b"".join(out))


def layer(i, exp=IQ3_XXS, up=None, down=None, dense=Q8_0, router=F32):
    return [(f"blk.{i}.ffn_gate_exps.weight", [64, 32, 8], exp), (f"blk.{i}.ffn_up_exps.weight", [64, 32, 8], up or exp),
            (f"blk.{i}.ffn_down_exps.weight", [32, 64, 8], down or exp), (f"blk.{i}.attn_q_a.weight", [64, 32], dense),
            (f"blk.{i}.ffn_gate_inp.weight", [64, 8], router), (f"blk.{i}.attn_norm.weight", [64], F32)]


class Check(unittest.TestCase):
    def run_check(self, arch, tensors, kv=None, env=None, args=None):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.gguf"
            gguf(p, {"general.architecture": arch, f"{arch}.block_count": 2, **(kv or {})}, tensors)
            return QC.check_local(p, env, args)

    def base(self, **kw):
        return [("token_embd.weight", [64, 100], Q6_K), ("output.weight", [64, 100], Q6_K)] + layer(0, **kw) + layer(1, **kw)

    def test_a_clean_set_runs(self):
        rep = self.run_check("glm5-next", self.base())
        self.assertEqual((rep.errors, rep.notes), ([], []))
        self.assertEqual(rep.model, "GLM-5.3-Flash")
        self.assertIn("runs: every weight", "\n".join(rep.lines()))

    def test_unknown_architecture(self):
        rep = self.run_check("qwen35moe", self.base())
        self.assertIn("is not one strata-x runs", rep.errors[0])

    def test_bf16_projections(self):
        # GLM, MiMo, DeepSeek-V4: F32 or a GPU-kernel type (src/x/engine.cpp proj_ok); Hy4, ERNIE, DeepSeek-V4.1 take BF16
        for arch in ("glm5next", "mimo2", "deepseek4"):
            with self.subTest(arch=arch):
                rep = self.run_check(arch, self.base(dense=BF16))
                self.assertTrue(any("are BF16" in e for e in rep.errors), rep.errors)
        for arch in ("hy_v4_flash", "ernie55", "deepseek41"):
            with self.subTest(arch=arch):
                self.assertEqual(self.run_check(arch, self.base(dense=BF16)).errors, [])

    def test_f16_projections_everywhere(self):
        for arch in QC.ARCHES:
            with self.subTest(arch=arch):
                self.assertTrue(any("are F16" in e for e in self.run_check(arch, self.base(dense=F16)).errors))

    def test_deepseek4_router_and_f32_only_weights(self):
        self.assertEqual(self.run_check("deepseek4", self.base(router=BF16)).errors, [])      # its router: BF16 or F32
        rep = self.run_check("deepseek4", self.base() + [("blk.0.indexer.proj.weight", [64, 64], Q8_0)])
        self.assertTrue(any("F32 only there" in e for e in rep.errors), rep.errors)          # ggml-org's MXFP4 file
        self.assertEqual(self.run_check("deepseek4", self.base() + [("blk.0.indexer.proj.weight", [64, 64], F32)]).errors, [])

    def test_expert_rules(self):
        rep = self.run_check("glm5next", self.base(up=IQ4_XS))
        self.assertTrue(any("gate and up experts of different types" in e for e in rep.errors))
        self.assertTrue(any("F32 experts" in e for e in self.run_check("mimo2", self.base(exp=F32)).errors))
        self.assertTrue(any("BF16 experts" in e for e in self.run_check("deepseek41", self.base(exp=BF16)).errors))
        rep = self.run_check("glm5next", self.base(exp=Q4_0))
        self.assertEqual(rep.errors, [])
        self.assertTrue(any("no batched prompt kernel" in n and "Q4_0" in n for n in rep.notes), rep.notes)
        rep = self.run_check("glm5next", self.base(exp=TQ2_0))
        self.assertTrue(any("no GPU kernel" in n for n in rep.notes) and any("no batched prompt" in n for n in rep.notes))
        self.assertEqual(self.run_check("glm5next", self.base(exp=MXFP4, down=MXFP4)).notes, [])
        self.assertTrue(any("Q5_1" in n for n in self.run_check("glm5next", self.base(exp=7)).notes))

    def test_deepseek41_sets(self):
        comp = [("blk.0.attn_compressor_kv.weight", [64, 64], BF16)]
        rep = self.run_check("deepseek41", self.base() + comp)                      # the command line: no config, a note
        self.assertTrue(any("STRATA_DSV41_BF16_W=1" in n for n in rep.notes) and not rep.errors)
        rep = self.run_check("deepseek41", self.base() + comp, args=[])             # the server: a config without it
        self.assertTrue(any("STRATA_DSV41_BF16_W=1" in e for e in rep.errors))
        self.assertEqual(self.run_check("deepseek41", self.base() + comp, env={"STRATA_DSV41_BF16_W": "1"}, args=[]).errors, [])
        q8 = [("blk.0.indexer.proj.weight", [64, 64], Q8_0)]                          # F32 or BF16 only there
        self.assertTrue(any("F32 or BF16 only" in e for e in self.run_check("deepseek41", self.base() + q8).errors))
        self.assertEqual(self.run_check("deepseek41", self.base() + [("blk.0.indexer.attn_q_b.weight", [64, 64], BF16)],
                                        env={"STRATA_DSV41_BF16_W": "1"}, args=[]).errors, [])
        engram = lambda t: [("per_layer_token_embd.weight", [64, 1000], t)]   # noqa: E731
        self.assertTrue(any("Engram table" in e for e in self.run_check("deepseek41", self.base() + engram(Q6_K)).errors))
        self.assertTrue(any("Engram table" in e for e in self.run_check("deepseek41", self.base() + engram(BF16)).errors))
        for ok in (67, 43, 20):
            self.assertEqual(self.run_check("deepseek41", self.base() + engram(ok)).errors, [])
        self.assertEqual(self.run_check("deepseek41", self.base() + engram(Q6_K), args=["--pack", "dir"]).errors, [])
        self.assertEqual(self.run_check("deepseek41", self.base() + engram(Q6_K),
                                        env={"STRATA_DSV41_ENGRAM_GGUF": "f.gguf"}).errors, [])

    def test_hy4_engram(self):
        engram = lambda t: [("per_layer_token_embd.weight", [64, 1000], t)]   # noqa: E731
        self.assertEqual(self.run_check("hy_v4_flash", self.base() + engram(43)).errors, [])
        self.assertEqual(self.run_check("hy_v4_flash", self.base() + engram(Q8_0)).errors, [])
        self.assertTrue(any("Engram table" in e for e in self.run_check("hy_v4_flash", self.base() + engram(BF16)).errors))

    def test_drafter_blocks_are_not_held_against_the_set(self):
        # GLM's MTP block (the last nextn_predict_layers blocks) loads only with --mtp: its types are not the trunk's rules
        mtp = [("blk.2.attn_q_a.weight", [64, 32], BF16), ("blk.2.ffn_gate_exps.weight", [64, 32, 8], F32)]
        rep = self.run_check("glm5next", self.base() + mtp, kv={"glm5next.block_count": 3, "glm5next.nextn_predict_layers": 1})
        self.assertEqual(rep.errors, [])

    def test_split_sets(self):
        with tempfile.TemporaryDirectory() as d:
            kv = {"general.architecture": "ernie55", "ernie55.block_count": 2}
            gguf(Path(d) / "s-00001-of-00003.gguf", kv, self.base()[:4])
            gguf(Path(d) / "s-00003-of-00003.gguf", {}, self.base()[4:])
            rep = QC.check_local(Path(d) / "s-00001-of-00003.gguf")
            self.assertTrue(any("not there" in e and "s-00002-of-00003.gguf" in e for e in rep.errors), rep.errors)
            gguf(Path(d) / "s-00002-of-00003.gguf", {}, [])
            rep = QC.check_local(Path(d) / "s-00003-of-00003.gguf")      # any shard names the set
            self.assertEqual((rep.files, rep.errors), (3, []))

    def test_a_dense_only_file(self):
        dense = [t for t in self.base() if "_exps" not in t[0]]
        rep = self.run_check("deepseek41", dense)                                    # the command line: a note
        self.assertTrue(any("no routed experts" in n for n in rep.notes) and not rep.errors)
        self.assertTrue(any("no routed experts" in e for e in self.run_check("deepseek41", dense, args=[]).errors))
        for extra in (["--pack", "p"], ["--experts-gguf", "x.gguf"]):
            self.assertEqual(self.run_check("deepseek41", dense, args=extra).errors, [])

    def test_a_hugging_face_set_by_its_headers(self):
        """check_hf over a fake Hugging Face: the tree listing and range requests served from synthetic files (no
        network); the reader fetches only header bytes, and a set is found from any one shard's name."""
        import io
        import json as _json
        with tempfile.TemporaryDirectory() as d:
            kv = {"general.architecture": "hy_v4_flash", "hy_v4_flash.block_count": 2}
            (Path(d) / "Q4").mkdir()
            gguf(Path(d) / "Q4/h-00001-of-00002.gguf", kv, self.base()[:4])
            gguf(Path(d) / "Q4/h-00002-of-00002.gguf", {}, self.base()[4:] + [("per_layer_token_embd.weight", [64, 9], BF16)])
            served = []

            def urlopen(req, timeout=None):
                url = req.full_url
                if "/api/models/" in url:
                    files = [{"path": f"Q4/{p.name}"} for p in sorted((Path(d) / "Q4").iterdir())]
                    return io.BytesIO(_json.dumps(files).encode())
                path = Path(d) / url.split("/resolve/main/", 1)[1]
                a, b = map(int, req.headers["Range"].split("=")[1].split("-"))
                data = path.read_bytes()[a:b + 1]
                served.append(len(data))
                return io.BytesIO(data)
            real = QC.urllib.request.urlopen
            QC.urllib.request.urlopen = urlopen
            try:
                out = QC.check_hf("org/repo", "Q4/h-00002-of-00002.gguf")
            finally:
                QC.urllib.request.urlopen = real
        self.assertEqual([k for k, _ in out], ["h"])
        rep = out[0][1]
        self.assertEqual((rep.model, rep.files), ("Hy4-flash", 2))
        self.assertTrue(any("Engram table" in e for e in rep.errors))      # BF16: Hy4 reads it as F8_E4M3 or Q8_0
        self.assertTrue(served and max(served) <= 256 << 10)                 # header bytes only, in small windows

    def test_model_arg(self):
        self.assertEqual(QC.model_arg({"args": ["--serve", "-m", "/m/a.gguf"]}), "/m/a.gguf")
        self.assertEqual(QC.model_arg({"args": ["-m", "a.gguf"], "cwd": "/x"}), os.path.join("/x", "a.gguf"))
        self.assertIsNone(QC.model_arg({"args": ["--serve"]}))


if __name__ == "__main__":
    unittest.main(verbosity=1)
