"""moex/tools/dsv41/ref/ref_run.py - run DeepSeek-V4.1's reference model on the CPU from the native pack.

    python ref_run.py --text FILE|--ids FILE.npy [--layers 0-39] [--gemm-act fp8|fp32] [--no-engram]
                      --out DIR [--max-seq-len 4096] [--threads 20] [--index-topk N]

Writes to DIR: ids.npy; per layer hNN.npy (the [S, 4, 5120] residual after the layer, fp32) and routesNN.npy (the
[S, 6] routed expert ids); logits.npy ([S, 129280] fp32) when all 40 layers ran; run.json (settings and timings).
--no-engram skips layers 1 and 14's engram additions: not the real model, only for like-for-like checks while the
tables are missing. --index-topk N keeps N rows in every indexer instead of the release's 512 (N >= the token count:
dense attention) - not the real model either, a check of what the selection costs. The GPUs are never touched (torch is a
CPU build here; CUDA_VISIBLE_DEVICES is cleared anyway).
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import json
import os
import pathlib
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ref_model import Reference  # noqa: E402

D = pathlib.Path(_os.path.join(_MODELS, 'dsv41-1017'))


def parse_layers(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=pathlib.Path)
    src.add_argument("--ids", type=pathlib.Path)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--gemm-act", choices=["fp8", "fp32"], default="fp8")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--index-topk", type=int, default=0)
    ap.add_argument("--pack", type=pathlib.Path, default=D / "native")
    ap.add_argument("--dense", type=pathlib.Path, default=D / "dense")
    ap.add_argument("--reference", type=pathlib.Path, default=D / "reference" / "inference")
    ap.add_argument("--tokenizer", type=pathlib.Path, default=D / "hf-small")
    ap.add_argument("--index", type=pathlib.Path, default=D / "index.json")
    a = ap.parse_args()
    os.environ["DSV41_REF_GEMM_ACT"] = a.gemm_act
    a.out.mkdir(parents=True, exist_ok=True)

    if a.ids:
        ids = np.load(a.ids).astype(np.int64).reshape(-1)[: a.max_tokens]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(a.tokenizer))
        ids = np.asarray(tok.encode(a.text.read_text(), add_special_tokens=False)[: a.max_tokens], dtype=np.int64)
    np.save(a.out / "ids.npy", ids)
    layers = parse_layers(a.layers)
    print(f"{len(ids)} tokens, layers {layers[0]}-{layers[-1]}, GEMM activations {a.gemm_act}"
          f"{', NO ENGRAM' if a.no_engram else ''}{f', index top-k {a.index_topk}' if a.index_topk else ''}", flush=True)

    ref = Reference(a.pack, a.dense, a.reference, a.tokenizer, a.index, max_seq_len=a.max_seq_len,
                    no_engram=a.no_engram, threads=a.threads, index_topk=a.index_topk)

    def on_layer(i, h_in, h_out, info):
        np.save(a.out / f"h{i:02d}.npy", h_out[0].float().numpy())
        if info["routes"] is not None:
            np.save(a.out / f"routes{i:02d}.npy", info["routes"].numpy().astype(np.int16))
        print(f"  layer {i:2d}: load {info['load_s']:5.1f} s, run {info['run_s']:6.1f} s", flush=True)

    t0 = time.time()
    out = ref.prefill(torch.from_numpy(ids), layers=layers, on_layer=on_layer)
    if "logits" in out:
        np.save(a.out / "logits.npy", out["logits"].float().numpy())
    (a.out / "run.json").write_text(json.dumps(dict(
        tokens=len(ids), layers=[layers[0], layers[-1]], gemm_act=a.gemm_act, no_engram=a.no_engram,
        index_topk=a.index_topk or None, seconds=round(time.time() - t0, 1),
        info={str(k): v for k, v in out["info"].items()}), indent=1))
    print(f"done in {time.time() - t0:.0f} s -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
