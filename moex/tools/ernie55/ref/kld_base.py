"""moex/tools/ernie55/ref/kld_base.py - the reference model (model.py, fp32) over one explicit token sequence of any length,
and its distributions at positions [T/2, T - 1) as a KLD base (llama-perplexity's format, one chunk of T tokens; strata-x
--kld reads it, compares its own rows and with --kld-save writes its rows in the same format: kld_compare.py compares bases).
Weights: the FP8 release exactly, or with --gguf the routed experts of a GGUF set (ernie55_expert_dq: the weights the engine
computes with), the rest the release's - the engine's own error, apart from the quantization's.

    ~/AI/vLLM/.venv/bin/python -I moex/tools/ernie55/ref/kld_base.py --ids A.ids[,B.ids] --out X.kld [--gguf G.gguf]
        [--threads N] [--resume]

A.ids: comma-separated ids; several files are joined in order (a rendered prompt, then a reply). The pass is layer by layer
over the whole sequence (each layer's cache lives only while it runs); X.kld.state keeps the streams every 4 layers and after
the last, so --resume continues. Measured: 49,583 tokens in 7,447 s (2026-11-27, 30 threads, beside other jobs; ~140 s an
HCA layer, ~400 s a DSA layer); RAM ~35 GB at that length; the base 10 GB.
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import pathlib
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M  # noqa: E402
from imatrix import kld_rows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="comma-separated id files, joined in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", default="", help="the routed experts from this GGUF set (any of its files)")
    ap.add_argument("--tool", default=_os.path.join(_REPO, "build-dsv41-dev", "ernie55_expert_dq"))
    ap.add_argument("--model", default=_os.path.join(_MODELS, "HF/ERNIE-5.5-Flash-Preview-FP8"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    ids: list[int] = []
    for f in a.ids.split(","):
        ids += [int(x) for x in pathlib.Path(f).read_text().split(",") if x.strip()]
    T = len(ids)
    m = M.Model(a.model)
    c = m.cfg
    dq = None
    if a.gguf:
        from oracle_dump import ExpertDQ  # noqa: E402
        dq = ExpertDQ(pathlib.Path(a.tool), a.gguf, keep=0)   # (layer by layer: each expert is read once a layer)
        m.expert_src = dq
    state = pathlib.Path(a.out + ".state")
    t0 = time.time()
    with torch.inference_mode():
        first = 0
        if a.resume and state.exists():
            st = torch.load(state)
            R, first = st["R"], st["next"]
            assert R.shape[0] == T, "the state is of another sequence"
            print(f"resuming at layer {first}", flush=True)
        else:
            R = m.embed(ids)
        pos = torch.arange(T)
        for il in range(first, c.n_layer):
            t1 = time.time()
            R = m.attn_half(il, R, pos, M.LayerCache())
            R = m.ffn_half(il, R)
            m.clear()
            if (il + 1) % 4 == 0 and il + 1 < c.n_layer:
                torch.save({"R": R, "next": il + 1}, str(state) + ".tmp")
                pathlib.Path(str(state) + ".tmp").replace(state)
            print(f"layer {il}: {time.time() - t1:.0f} s", flush=True)
        if first < c.n_layer:   # (the last layer's streams too: a failure writing the rows resumes here)
            torch.save({"R": R, "next": c.n_layer}, str(state) + ".tmp")
            pathlib.Path(str(state) + ".tmp").replace(state)
        fr = T // 2
        with open(a.out + ".tmp", "wb") as f:
            f.write(b"_logits_" + struct.pack("<3i", T, c.vocab, 1))
            f.write(np.asarray(ids, dtype="<i4").tobytes())
            for s in range(fr, T - 1, 1024):
                e = min(s + 1024, T - 1)
                f.write(kld_rows(m.logits(m.final_norm(R[s:e]))).tobytes())
        pathlib.Path(a.out + ".tmp").replace(a.out)
    if dq:
        dq.close()
    state.unlink(missing_ok=True)
    print(f"{a.out}: 1 chunk of {T}, positions {fr}-{T - 2} scored, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
