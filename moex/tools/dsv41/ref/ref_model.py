"""moex/tools/dsv41/ref/ref_model.py - DeepSeek-V4.1's own model.py, run one layer at a time from the native pack.

The release's inference/model.py is imported with ref_kernels as its `kernel` module (pure torch, CPU). The full
model cannot sit in RAM (289 GB of experts, 203 GB of engram rows), so the runner builds one Block at a time:
dense weights from the pack's dense safetensors (the release's bytes, renamed and converted the way convert.py
does for one rank), the 384 experts from the pack's record file (ggml MXFP4, turned back into the release's FP4
layout), and engram rows read by number from the pack's table files. The model-wide state that crosses layers
(model.shared_attn: compressed KV, index keys, top-k lists, candidates) lives in model.py itself.

    from ref_model import Reference
    ref = Reference(pack_dir, dense_dir, reference_dir, tokenizer_dir, max_seq_len=4096)
    out = ref.prefill(token_ids, layers=range(40), on_layer=callback)
"""
from __future__ import annotations

import gc
import importlib
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fp4_repack  # noqa: E402
import ref_kernels  # noqa: E402

RECORD = 18_800_640


def import_model(reference_dir: pathlib.Path):
    """model.py with our kernels; its own engram.py / vision.py / image_processor.py from the release."""
    sys.modules["kernel"] = ref_kernels
    if str(reference_dir) not in sys.path:
        sys.path.insert(0, str(reference_dir))
    return importlib.import_module("model")


class Pack:
    """The native pack: experts/L*.bin records, engram tables, dense safetensors, plus the release's index."""

    def __init__(self, pack_dir: pathlib.Path, dense_dir: pathlib.Path, index_path: pathlib.Path):
        from safetensors import safe_open
        self.pack_dir, self.dense_dir = pack_dir, dense_dir
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self._safe_open = safe_open
        self._open = {}

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._open:
            self._open[shard] = self._safe_open(str(self.dense_dir / shard), framework="pt", device="cpu")
        return self._open[shard].get_tensor(name)

    def names(self, prefix: str) -> list[str]:
        return [n for n in self.weight_map if n.startswith(prefix) and ".experts." not in n and ".engram.embed." not in n]

    def expert(self, layer: int, e: int, mtp: bool = False) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        path = self.pack_dir / "experts" / (f"M{layer}.bin" if mtp else f"L{layer:02d}.bin")
        with open(path, "rb") as f:
            f.seek(e * RECORD)
            rec = f.read(RECORD)
            os.posix_fadvise(f.fileno(), e * RECORD, RECORD, os.POSIX_FADV_DONTNEED)
        return {k: (torch.from_numpy(w.copy()), torch.from_numpy(s.copy()))
                for k, (w, s) in fp4_repack.split_record(rec).items()}

    def engram_files(self, layer: int) -> tuple[pathlib.Path, pathlib.Path]:
        d = self.pack_dir / "engram"
        return d / f"L{layer:02d}.weight.bin", d / f"L{layer:02d}.scale.bin"


def make_engram_embedding_class(nn, pack: Pack):
    """Stands in for model.ParallelEngramEmbedding: rows read by number from the pack instead of a 98 GB tensor.
    Same arithmetic as the reference: FP8 rows x E8M0 scales per 32, to BF16."""

    class PackEngramEmbedding(nn.Module):
        layer_for_rows: dict[int, int] = {}

        def __init__(self, num_embeddings: int, dim: int):
            super().__init__()
            self.num_embeddings, self.dim, self.block_size = num_embeddings, dim, 32
            self.layer = PackEngramEmbedding.layer_for_rows[num_embeddings]
            wpath, spath = pack.engram_files(self.layer)
            if not wpath.exists() or not spath.exists():
                raise FileNotFoundError(f"engram table for layer {self.layer} not in the pack yet: {wpath}")
            self.w = np.memmap(wpath, dtype=np.uint8, mode="r").reshape(num_embeddings, dim)
            self.s = np.memmap(spath, dtype=np.uint8, mode="r").reshape(num_embeddings, dim // 32)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            flat = indices.reshape(-1).numpy()
            if flat.size and (flat.min() < 0 or flat.max() >= self.num_embeddings):
                raise IndexError("engram row out of range")
            uniq, inv = np.unique(flat, return_inverse=True)
            w = torch.from_numpy(np.asarray(self.w[uniq])).view(torch.float8_e4m3fn).float()
            s = torch.from_numpy(np.asarray(self.s[uniq])).view(torch.float8_e8m0fnu).float()
            v = (w.unflatten(-1, (-1, self.block_size)) * s.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
            return v[torch.from_numpy(inv)].reshape(*indices.shape, self.dim)

    return PackEngramEmbedding


class Reference:
    def __init__(self, pack_dir, dense_dir, reference_dir, tokenizer_dir, index_path, max_seq_len: int = 4096,
                 no_engram: bool = False, threads: int | None = None, index_topk: int = 0):
        self.M = M = import_model(pathlib.Path(reference_dir))
        torch.set_default_dtype(torch.bfloat16)
        if threads:
            torch.set_num_threads(threads)
        cfg = json.loads((pathlib.Path(reference_dir) / "config.json").read_text())
        self.args = args = M.ModelArgs(**cfg)
        args.max_batch_size, args.max_seq_len = 1, max_seq_len
        if index_topk:   # a check only (ref_run.py --index-topk): every indexer keeps this many rows, not the release's 512
            args.index_topk = index_topk
        # Transformer.__init__ sets these globals; one rank, FP8 dense weights
        M.world_size, M.rank = 1, 0
        M.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        self.pack = Pack(pathlib.Path(pack_dir), pathlib.Path(dense_dir), pathlib.Path(index_path))
        self.no_engram = no_engram
        self.layout = None if no_engram else M.EngramLayout.from_args(args)
        if self.layout is not None:
            cls = make_engram_embedding_class(torch.nn, self.pack)
            cls.layer_for_rows = {n: lid for lid, n in zip(self.layout.layer_ids, self.layout.num_embeddings)}
            M.ParallelEngramEmbedding = cls
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
            self.engram_hash = M.NgramHashState(args, self.layout, tok)

    # -- weights --------------------------------------------------------------------------------------------
    def _load_module(self, module: torch.nn.Module, prefix: str, skip_experts: bool = True) -> None:
        """Copy the release's tensors under `prefix` into `module`, converting as convert.py does for mp=1."""
        own = dict(module.named_parameters())
        want = set(own)
        if skip_experts:
            want = {k for k in want if not (k.startswith("ffn.experts.") or k.startswith("engram.embed."))}
        got = set()
        for name in self.pack.names(prefix):
            key = name[len(prefix):]
            if key.endswith("wo_a.scale"):
                continue  # consumed by the BF16 conversion of wo_a.weight below
            if key.startswith("engram.") and getattr(module, "engram", None) is None:
                continue  # --no-engram: the layer is built without its engram
            if key not in own:
                raise KeyError(f"{name}: no parameter {key!r} in the module")
            t = self.pack.tensor(name)
            if key.endswith("wo_a.weight"):  # convert.py: wo_a dequantized to BF16 with its 32x32 scales
                s = self.pack.tensor(name.replace("weight", "scale")).float()
                t = (t.float().unflatten(0, (-1, 32)).unflatten(-1, (-1, 32)) * s[:, None, :, None]).flatten(2, 3).flatten(0, 1)
                t = t.to(torch.bfloat16)
            p = own[key]
            if tuple(p.shape) != tuple(t.shape):
                raise ValueError(f"{name}: shape {tuple(t.shape)} vs parameter {tuple(p.shape)}")
            with torch.no_grad():
                p.data = t.to(p.dtype) if p.dtype != t.dtype else t
            got.add(key)
        # the scale of wo_a has no parameter after the BF16 conversion
        missing = sorted(want - got - {k for k in want if k.endswith("wo_a.scale")})
        if missing:
            raise KeyError(f"{prefix}: parameters not in the pack: {missing[:8]}{' ...' if len(missing) > 8 else ''}")

    def load_expert(self, moe, layer: int, e: int, mtp: bool = False) -> None:
        ex = moe.experts[e]
        parts = self.pack.expert(layer, e, mtp)
        for name in ("w1", "w2", "w3"):
            w, s = parts[name]
            lin = getattr(ex, name)
            lin.weight.data = w.view(torch.float4_e2m1fn_x2)
            lin.scale.data = s.view(torch.float8_e8m0fnu)
            lin.weight.scale = lin.scale
        ex.loaded = True

    def _load_experts(self, moe, layer: int, mtp: bool = False) -> None:
        for e, ex in enumerate(moe.experts):
            if ex is not None:
                self.load_expert(moe, layer, e, mtp)

    def block(self, layer: int, experts: bool = True):
        """Layer `layer` as model.Block with the release's weights; experts=False leaves the routed experts unloaded
        (load_expert reads one on demand: a few decode steps route to a few dozen of the 384)."""
        M = self.M
        t0 = time.time()
        with torch.device("cpu"):
            b = M.Block(layer, self.args, self.layout)
        self._load_module(b, f"layers.{layer}.")
        if experts:
            self._load_experts(b.ffn, layer)
        b.eval()
        b.load_seconds = time.time() - t0
        return b

    def embed(self):
        e = self.M.ParallelEmbedding(self.args.vocab_size, self.args.dim)
        e.weight.data = self.pack.tensor("embed.weight")
        return e

    def head(self):
        h = self.M.ParallelHead(self.args.vocab_size, self.args.dim, self.args.norm_eps, self.args.hc_eps)
        h.weight.data = self.pack.tensor("head.weight").float()
        n = self.M.RMSNorm(self.args.dim, self.args.norm_eps)
        n.weight.data = self.pack.tensor("norm.weight")
        return n, h

    # -- forward --------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def prefill(self, ids: torch.Tensor, layers=None, on_layer=None, logits: bool = True, images=None) -> dict:
        """One prefill pass (start_pos 0) over `layers` (default all 40), as Transformer.forward runs it.
        on_layer(i, h_in, h_out, info) sees every layer; returns logits [S, vocab] (fp32) when all layers ran.
        `images` (moex vision): [(start, rows [n, dim])] - each image's span rows (serve/vision_dsv41.py's record: the ViT and
        aligner rows with the image_start / image_newline / image_end rows, what merge_image_embeddings writes); its positions
        are image tokens for the engram (no n-gram, no engram) and the gates (bias_vl), as Transformer.forward's masks make them."""
        M, args = self.M, self.args
        layers = list(range(args.n_layers)) if layers is None else list(layers)
        if layers != list(range(layers[0], layers[-1] + 1)) or layers[0] != 0:
            raise ValueError("layers must be a contiguous range starting at 0 (attention state crosses layers)")
        ids = ids.view(1, -1).long()
        image_mask = None
        if images:
            image_mask = torch.zeros_like(ids, dtype=torch.bool)
            for start, rows in images:
                image_mask[0, start:start + rows.shape[0]] = True
        engram_mask = None if image_mask is None else ~image_mask
        hashes = None if self.layout is None else self.engram_hash(ids, 0, engram_mask)
        h = self.embed()(ids)
        for start, rows in images or ():
            h[0, start:start + rows.shape[0]] = rows.to(h.dtype)
        h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
        info = {}
        for i in layers:
            blk = self.block(i)
            t0 = time.time()
            routes = []
            gate_fwd = blk.ffn.gate.forward

            def capture(x, image_mask=None, _f=gate_fwd):
                w, idx = _f(x, image_mask)
                routes.append(idx.clone())
                return w, idx

            blk.ffn.gate.forward = capture
            h_in = h
            if blk.engram is not None:
                h = blk.engram(h, hashes[:, :, blk.engram.layer_hash_index, :], engram_mask)
            h, pre_mix = blk(h, 0, pre_mix, image_mask)
            info[i] = dict(load_s=blk.load_seconds, run_s=time.time() - t0)
            if on_layer is not None:
                on_layer(i, h_in, h, dict(info[i], routes=routes[0] if routes else None, pre_mix=pre_mix))
            del blk
            gc.collect()
        out = dict(info=info, h=h, pre_mix=pre_mix)
        if logits and layers[-1] == args.n_layers - 1:
            norm, head = self.head()
            hf = M.Block.hc_pre(None, h, pre_mix)
            out["logits"] = head(norm(hf), full_logits=True)[0]
        return out
