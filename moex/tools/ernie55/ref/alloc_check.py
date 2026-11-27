"""moex/tools/ernie55/ref/alloc_check.py - a per-layer recipe against a uniform set, by the model's own output, before the
recipe is written: the reference model (model.py, fp32, exact FP8) over a few chunks of text in three variants run at once
- the exact model; the set (every layer's routed experts from GGUF A); the candidate (layers named in --from-b from GGUF B,
layers named in --fakequant from the converter's own quantization in the types given, the rest from A) - and each set's
distributions compared with the exact model's at positions [T/2, T - 1) of every chunk (mean KL, the same top token).

    ~/AI/vLLM/.venv/bin/python -I moex/tools/ernie55/ref/alloc_check.py --a A.gguf [--b B.gguf --from-b 4-9,30]
        [--fakequant 1-3,40-42:iq4_xs,iq4_xs --imatrix I.imx] --text F:OFFSET,... [--len 2048] [--threads N]
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import pathlib
import sys
import time

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M  # noqa: E402
from oracle_dump import ExpertDQ  # noqa: E402
from recipe_check import FakeQ  # noqa: E402

BOS = 2


def layers_of(spec: str) -> set[int]:
    out: set[int] = set()
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


class Mixed:
    """expert_src for the candidate: per layer from B, from the fake-quant or from A"""

    def __init__(self, a, b, b_layers, fq, fq_layers, fq_types):
        self.a, self.b, self.bl, self.fq, self.fl, self.ft = a, b, b_layers, fq, fq_layers, fq_types

    def __call__(self, il, e):
        if il in self.fl:
            return self.fq.get(il, e, *self.ft)
        if il in self.bl:
            return self.b(il, e)
        return self.a(il, e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the uniform set (any of its files)")
    ap.add_argument("--b", default="", help="a second set some layers come from")
    ap.add_argument("--from-b", default="")
    ap.add_argument("--fakequant", default="", help="LAYERS:GU_TYPE,D_TYPE")
    ap.add_argument("--imatrix", default="")
    ap.add_argument("--cand", action="append", default=[], help="NAME=b:LAYERS;fq:LAYERS:GU,D - more candidates, one pass")
    ap.add_argument("--text", required=True, help="FILE:OFFSET,... - one chunk from each file")
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--model", default=_os.path.join(_MODELS, "HF/ERNIE-5.5-Flash-Preview-FP8"))
    ap.add_argument("--tool", default=_os.path.join(_REPO, "build-dsv41-dev", "ernie55_expert_dq"))
    ap.add_argument("--conv", default=_os.path.join(_REPO, "build-dsv41-dev", "convert_gguf_ernie55"))
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(str(pathlib.Path(a.model) / "tokenizer.json"))
    chunks = []
    for spec in a.text.split(","):
        f, off = spec.rsplit(":", 1)
        ids = tok.encode(pathlib.Path(f).read_text(encoding="utf-8", errors="replace"), add_special_tokens=False).ids
        chunks.append([BOS] + ids[int(off):int(off) + a.len - 1])
    m = M.Model(a.model)
    c = m.cfg
    da = ExpertDQ(pathlib.Path(a.tool), a.a, keep=0)
    db = ExpertDQ(pathlib.Path(a.tool), a.b, keep=0) if a.b else None
    b_layers = layers_of(a.from_b) if db else set()
    fq, fq_layers, fq_types = None, set(), ("", "")
    if a.fakequant:
        spec, _, types = a.fakequant.rpartition(":")
        fq_layers, fq_types = layers_of(spec), tuple(types.split(","))
        fq = FakeQ(a.conv, a.model, a.imatrix, max(1, a.threads))
    moe = set(range(c.first_dense, c.n_layer))
    cands = {}   # name -> (expert_src, the layers where it differs from the set)
    if b_layers or fq_layers:
        cands["cand"] = (Mixed(da, db, b_layers, fq, fq_layers, fq_types), sorted((b_layers | fq_layers) & moe))
    for spec in a.cand:
        name, _, parts = spec.partition("=")
        bl, fl, ft = set(), set(), ("", "")
        for part in filter(None, parts.split(";")):
            kind, _, rest = part.partition(":")
            if kind == "b":
                bl = layers_of(rest)
            elif kind == "fq":
                lay, _, types = rest.rpartition(":")
                fl, ft = layers_of(lay), tuple(types.split(","))
        if fl and fq is None:
            fq = FakeQ(a.conv, a.model, a.imatrix, max(1, a.threads))
        cands[name] = (Mixed(da, db, bl, fq, fl, ft), sorted((bl | fl) & moe))
    C, T = len(chunks), a.len
    pos = torch.arange(T)
    t0 = time.time()
    with torch.inference_mode():
        R = {"exact": torch.stack([m.embed(ch) for ch in chunks])}
        for il in range(c.n_layer):
            t1 = time.time()
            for v in list(R):
                for ci in range(C):
                    R[v][ci] = m.attn_half(il, R[v][ci], pos, M.LayerCache())
            if il == c.first_dense:
                R["set"] = R["exact"].clone()
            for name, (_, dif) in cands.items():
                if dif and il == dif[0]:
                    R[name] = R["set"].clone()
            # the exact rows, the set's (and the candidate's while it still is the set) through A, the candidate's own
            groups = [("exact", None), ("set", da if il in moe else None)]
            for v, src in groups:
                if v not in R:
                    continue
                m.expert_src = src
                R[v] = m.ffn_half(il, R[v].reshape(-1, c.hc, c.hidden)).view(C, T, c.hc, c.hidden)
            for name, (src, _) in cands.items():
                if name in R:
                    m.expert_src = src if il in moe else None
                    R[name] = m.ffn_half(il, R[name].reshape(-1, c.hc, c.hidden)).view(C, T, c.hc, c.hidden)
            m.expert_src = None
            m.clear()
            here = [n for n, (_, d) in cands.items() if il in d]
            print(f"layer {il}: {time.time() - t1:.0f} s{(' (differs here: ' + ', '.join(here) + ')') if here else ''}", flush=True)
        fr = T // 2
        out = {v: [0.0, 0] for v in R if v != "exact"}
        n = 0
        for ci in range(C):
            for s in range(fr, T - 1, 256):
                e = min(s + 256, T - 1)
                ref = torch.log_softmax(m.logits(m.final_norm(R["exact"][ci, s:e])), -1)
                p = ref.exp()
                for v in out:
                    lv = torch.log_softmax(m.logits(m.final_norm(R[v][ci, s:e])), -1)
                    out[v][0] += float((p * (ref - lv)).sum())
                    out[v][1] += int((ref.argmax(-1) == lv.argmax(-1)).sum())
                n += e - s
    print(f"{C} chunks of {T}, {n} positions scored, {time.time() - t0:.0f} s")
    for name, (_, d) in cands.items():
        print(f"  {name}: differs from the set at layers {d}")
    for v, (k, s) in out.items():
        print(f"  {v:5s} mean KL {k / n:.5f}  same top {100 * s / n:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
