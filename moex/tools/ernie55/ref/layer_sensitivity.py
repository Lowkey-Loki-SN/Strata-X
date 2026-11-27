"""moex/tools/ernie55/ref/layer_sensitivity.py - which layers' routed-expert quantization the model's output feels: the
reference model (model.py, fp32, exact FP8) over a few chunks of text, run at once as variants - the exact model; one
variant a group of MoE layers whose routed experts come from a GGUF set (ernie55_expert_dq: the set's own weights), every
other layer exact; and one with the set's experts in every layer - each compared with the exact model's distributions at
positions [T/2, T - 1) of every chunk (mean KL(exact || variant)). A group's number is what quantizing those layers alone
costs; the groups' sum against the all-layers variant says how far the costs add.

    ~/AI/vLLM/.venv/bin/python -I moex/tools/ernie55/ref/layer_sensitivity.py --gguf G.gguf --text A.txt:OFFSET,B.txt:OFFSET
        [--len 2048] [--group 3] [--threads N]

A variant shares the exact model's streams until its first quantized layer (the attention half of that layer too), so the
work is about half of variants x layers. The variants' streams stay in RAM ([chunks, T, 4, 4096] fp32 each).
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

BOS = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the set whose routed experts are tried (any of its files)")
    ap.add_argument("--text", required=True, help="FILE:OFFSET,... - one chunk from each file, OFFSET tokens in")
    ap.add_argument("--len", type=int, default=2048, help="tokens a chunk (with its BOS)")
    ap.add_argument("--group", type=int, default=3, help="MoE layers a group")
    ap.add_argument("--model", default=_os.path.join(_MODELS, "HF/ERNIE-5.5-Flash-Preview-FP8"))
    ap.add_argument("--tool", default=_os.path.join(_REPO, "build-dsv41-dev", "ernie55_expert_dq"))
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
        assert len(chunks[-1]) == a.len, f"{f}: too short"
    m = M.Model(a.model)
    c = m.cfg
    moe = list(range(c.first_dense, c.n_layer))
    groups = [moe[i:i + a.group] for i in range(0, len(moe), a.group)]
    names = ["exact"] + [f"layers {g[0]}-{g[-1]}" for g in groups] + ["all layers"]
    quant_at = [set()] + [set(g) for g in groups] + [set(moe)]       # the layers where a variant takes the set's experts
    first_q = [None] + [g[0] for g in groups] + [moe[0]]
    dq = ExpertDQ(pathlib.Path(a.tool), a.gguf, keep=0)
    C, T = len(chunks), a.len
    pos = torch.arange(T)
    t0 = time.time()
    with torch.inference_mode():
        R = {0: torch.stack([m.embed(ch) for ch in chunks])}         # variant -> [C, T, 4, 4096]; absent: the exact one's
        for il in range(c.n_layer):
            t1 = time.time()
            for v in list(R):                                          # the attention half: each live variant, each chunk
                for ci in range(C):
                    R[v][ci] = m.attn_half(il, R[v][ci], pos, M.LayerCache())
            for v in range(1, len(names)):                             # a variant starts here: the exact streams, so far
                if first_q[v] == il:
                    R[v] = R[0].clone()
            for use_set in (False, True):                              # the FFN half: the exact experts' rows at once,
                vs = [v for v in R if (il in quant_at[v]) == use_set]  # then the set's
                if not vs:
                    continue
                X = torch.cat([R[v].reshape(-1, c.hc, c.hidden) for v in vs])
                m.expert_src = dq if use_set else None
                X = m.ffn_half(il, X)
                m.expert_src = None
                for k, v in enumerate(vs):
                    R[v] = X[k * C * T:(k + 1) * C * T].view(C, T, c.hc, c.hidden)
            m.clear()
            print(f"layer {il}: {time.time() - t1:.0f} s, {len(R)} variants live", flush=True)
        # the scored rows: KL(exact || variant), the same top token
        fr = T // 2
        kl = {v: 0.0 for v in range(1, len(names))}
        same = {v: 0 for v in range(1, len(names))}
        n = 0
        for ci in range(C):
            for s in range(fr, T - 1, 256):
                e = min(s + 256, T - 1)
                ref = torch.log_softmax(m.logits(m.final_norm(R[0][ci, s:e])), -1)
                p = ref.exp()
                for v in range(1, len(names)):
                    lv = torch.log_softmax(m.logits(m.final_norm(R[v][ci, s:e])), -1)
                    kl[v] += float((p * (ref - lv)).sum())
                    same[v] += int((ref.argmax(-1) == lv.argmax(-1)).sum())
                n += e - s
    dq.close()
    print(f"{C} chunks of {T}, {n} positions scored, {time.time() - t0:.0f} s; {a.gguf.rsplit('/', 1)[-1]}'s routed experts")
    tot = 0.0
    for v in range(1, len(names)):
        if v < len(names) - 1:
            tot += kl[v] / n
        print(f"  {names[v]:14s} mean KL {kl[v] / n:.5f}  same top {100 * same[v] / n:.2f}%")
    print(f"  the groups' sum {tot:.5f} against all layers at once {kl[len(names) - 1] / n:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
