"""moex/tools/ernie55/ref/imatrix.py - from one pass of the reference model over two sets of text: the routed experts'
importance matrix (calibration text; convert_gguf_ernie55 --imatrix, the ERN55IMX format: model.py Imatrix.save) and the
quality yardstick's base (held-out text; strata-x --kld, llama-perplexity's --kl-divergence-base format).

Layer-major: every chunk's streams go through layer L before any goes through layer L + 1, so each layer's weights are
read from the drive once for both sets (a token-major pass would read all 270 GB of experts per chunk). The streams of
every chunk stay in RAM between layers ([chunks, T, 4, 4096] fp32: 4.3 GB for 16 x 4,096); a layer's caches live only
while its chunks run. The attention runs a chunk at a time (each chunk its own sequence: <|ERNIE_MASK|> and T - 1 tokens
of one text); mHC's FFN half and the MoE run over a set's rows at once (each row on its own), so each routed expert is
dequantized once a layer a set. The statistics are model.py's (per expert: the sum of squared latent inputs per column for
gate / up, of squared SiTU-GLU outputs for down, and the row count), in fp64, from the calibration set only.

    ~/AI/vLLM/.venv/bin/python -I moex/tools/ernie55/ref/imatrix.py --text A.txt:6,B.txt --chunks 16 --len 4096
        [--ids-chunks CHAT.ids] --out ernie55.imx [--kld-text C.txt:4,D.txt --kld-chunks 16 --kld-len 1024 --kld-out ernie55.kld]
        [--threads N] [--layers N] [--resume]

FILE:N takes N chunks from FILE (spread over its length); the files without a count share what is left. OUT is rewritten
after every layer, OUT.state (both sets' streams and the statistics) every 6 layers, so --resume continues from the last
state. The KLD base scores positions [len / 2, len - 1) of each evaluation chunk, as llama-perplexity's.
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
from tokenizers import Tokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M  # noqa: E402

# the token a sequence starts with: <|ERNIE_MASK|> - the tokenizer's post-processor puts it in front of every encoded text
# (tokenizer.json TemplateProcessing) and the chat format opens with it (encoding_ernie.py render); config.json's
# bos_token_id (1, <|ERNIE_BEGIN|>) is not what either writes
BOS = 2


def take_chunks(tok: Tokenizer, spec: str, n_chunks: int, length: int) -> list[list[int]]:
    """FILE[:N],... -> n_chunks chunks of `length` tokens (BOS first), each file's spread over its length"""
    files = [(x.rsplit(":", 1)[0], int(x.rsplit(":", 1)[1])) if ":" in x and x.rsplit(":", 1)[1].isdigit() else (x, -1)
             for x in spec.split(",")]
    left = n_chunks - sum(k for _, k in files if k >= 0)
    free = [i for i, (_, k) in enumerate(files) if k < 0]
    if left < 0 or (left > 0 and not free):
        raise SystemExit("the files' chunk counts do not add up to the chunks asked for")
    for r, i in enumerate(free):
        files[i] = (files[i][0], left // len(free) + (1 if r < left % len(free) else 0))
    chunks: list[list[int]] = []
    span = length - 1
    for f, k in files:
        ids = tok.encode(pathlib.Path(f).read_text(encoding="utf-8", errors="replace"), add_special_tokens=False).ids
        if len(ids) < k * span:
            raise SystemExit(f"{f}: {len(ids)} tokens, {k * span} needed for its {k} chunks")
        for j in range(k):
            o = j * (len(ids) - span) // max(k - 1, 1) if k > 1 else 0
            chunks.append([BOS] + ids[o:o + span])
        print(f"{f}: {len(ids)} tokens, {k} chunks of {length}", flush=True)
    return chunks


def kld_rows(lg: torch.Tensor) -> np.ndarray:
    """strata-x's kld_encode (llama-perplexity's base rows): per row the scale and the min log-probability as two floats,
    then a uint16 per vocabulary entry (padded to an even count)"""
    lg = lg.to(torch.float32)
    mx = lg.max(-1).values
    mn = torch.maximum(lg.min(-1).values, mx - 16.0)
    lse = torch.log(torch.exp(lg - mx[:, None]).sum(-1, dtype=torch.float64)).to(torch.float32)
    scale = torch.clamp((mx - mn) / 65535.0, min=1e-12)
    q = torch.round((torch.maximum(lg, mn[:, None]) - mn[:, None]) / scale[:, None]).clamp(0, 65535).to(torch.int32)
    V = lg.shape[1]
    nv = 2 * ((V + 1) // 2) + 4
    out = np.zeros((lg.shape[0], nv), dtype=np.uint16)
    head = np.stack([scale.numpy(), (mn - mx - lse).numpy()], 1).astype("<f4")
    out[:, :4] = head.view(np.uint16).reshape(-1, 4)
    out[:, 4:4 + V] = q.numpy().astype(np.uint16)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="", help="the calibration text files (FILE:N: N chunks from it)")
    ap.add_argument("--ids-chunks", default="", help="calibration chunks as token ids: a file, one chunk a line (comma-"
                    "separated ids, the first --len of each line taken); for text with the chat format's control tokens")
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--len", type=int, default=4096, help="tokens a calibration chunk (with its BOS)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kld-text", default="", help="the held-out text files for the KLD base (FILE:N)")
    ap.add_argument("--kld-ids-chunks", default="", help="held-out chunks as token ids (one a line, the first --kld-len "
                    "taken), after --kld-text's")
    ap.add_argument("--kld-chunks", type=int, default=16)
    ap.add_argument("--kld-len", type=int, default=1024)
    ap.add_argument("--kld-out", default="")
    ap.add_argument("--model", default=_os.path.join(_MODELS, 'HF/ERNIE-5.5-Flash-Preview-FP8'))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0, help="stop after this many layers (a test); 0: all")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(str(pathlib.Path(a.model) / "tokenizer.json"))
    calib = take_chunks(tok, a.text, a.chunks, a.len) if a.text else []
    if a.ids_chunks:
        for ln, line in enumerate(pathlib.Path(a.ids_chunks).read_text().splitlines()):
            ids = [int(x) for x in line.split(",") if x.strip()]
            if not ids:
                continue
            if len(ids) < a.len:
                raise SystemExit(f"{a.ids_chunks}: line {ln + 1} has {len(ids)} ids, {a.len} needed")
            calib.append(ids[:a.len])
        print(f"{a.ids_chunks}: {len(calib) - (a.chunks if a.text else 0)} chunks of {a.len} token ids", flush=True)
    if not calib and not (a.kld_out and (a.kld_text or a.kld_ids_chunks)):
        raise SystemExit("no calibration chunks (--text or --ids-chunks), and no KLD base asked for")
    kld = bool((a.kld_text or a.kld_ids_chunks) and a.kld_out)
    evalc = take_chunks(tok, a.kld_text, a.kld_chunks, a.kld_len) if kld and a.kld_text else []
    if kld and a.kld_ids_chunks:
        n0 = len(evalc)
        for ln, line in enumerate(pathlib.Path(a.kld_ids_chunks).read_text().splitlines()):
            ids = [int(x) for x in line.split(",") if x.strip()]
            if not ids:
                continue
            if len(ids) < a.kld_len:
                raise SystemExit(f"{a.kld_ids_chunks}: line {ln + 1} has {len(ids)} ids, {a.kld_len} needed")
            evalc.append(ids[:a.kld_len])
        print(f"{a.kld_ids_chunks}: {len(evalc) - n0} held-out chunks of {a.kld_len} token ids", flush=True)
    m = M.Model(a.model)
    c = m.cfg
    n_layer = a.layers or c.n_layer
    out = pathlib.Path(a.out)
    state = pathlib.Path(str(out) + ".state")
    first = 0
    imx = M.Imatrix(c.n_layer, c.n_expert, c.latent, c.ff_exp)
    if a.resume and state.exists():   # (the statistics as they were at the state: OUT may already hold later layers)
        st = torch.load(state)
        R, Re, first = st["R"], st["Re"], st["next"]
        imx.gu, imx.dn, imx.cnt = st["gu"], st["dn"], st["cnt"]
        print(f"resuming at layer {first}", flush=True)
    else:
        with torch.inference_mode():
            R = torch.stack([m.embed(ch) for ch in calib]) if calib else None       # [C, T, 4, 4096]; none: a KLD base only
            Re = torch.stack([m.embed(ch) for ch in evalc]) if evalc else None     # [Ce, Te, 4, 4096]
    pos, pos_e = torch.arange(a.len), torch.arange(a.kld_len)
    t0 = time.time()

    def run_set(il: int, X: torch.Tensor, p: torch.Tensor, stats: bool) -> torch.Tensor:
        for ci in range(X.shape[0]):   # the attention: each chunk its own sequence
            X[ci] = m.attn_half(il, X[ci], p, M.LayerCache())
        m.imatrix = imx if stats else None
        X = m.ffn_half(il, X.view(-1, c.hc, c.hidden)).view(X.shape)   # the FFN over the set's rows at once
        m.imatrix = None
        return X

    with torch.inference_mode():
        for il in range(first, n_layer):
            t1 = time.time()
            if R is not None:
                R = run_set(il, R, pos, True)
            if Re is not None:
                Re = run_set(il, Re, pos_e, False)
            m.clear()
            if R is not None:
                imx.save(out)
            if (il + 1) % 6 == 0 and il + 1 < n_layer:   # (the streams: 4.3 GB at 16 x 4,096, every 6 layers)
                torch.save({"R": R, "Re": Re, "next": il + 1, "gu": imx.gu, "dn": imx.dn, "cnt": imx.cnt}, state)
            cnt = imx.cnt[il]
            seen = cnt[cnt > 0]
            print(f"layer {il}: {time.time() - t1:.0f} s; experts seen {int(seen.numel())} of {c.n_expert}, rows a seen expert "
                  f"min {int(seen.min()) if seen.numel() else 0} median {int(seen.median()) if seen.numel() else 0}", flush=True)
        if kld and n_layer == c.n_layer:   # the base: the evaluation chunks' scored rows
            first_row = a.kld_len // 2
            V = c.vocab
            with open(a.kld_out + ".tmp", "wb") as f:
                f.write(b"_logits_" + struct.pack("<3i", a.kld_len, V, len(evalc)))
                f.write(np.asarray(evalc, dtype="<i4").tobytes())
                for ci in range(len(evalc)):
                    h = m.final_norm(Re[ci, first_row:a.kld_len - 1])
                    f.write(kld_rows(m.logits(h)).tobytes())
            pathlib.Path(a.kld_out + ".tmp").replace(a.kld_out)
            print(f"{a.kld_out}: {len(evalc)} chunks of {a.kld_len}, positions {first_row}-{a.kld_len - 2} scored", flush=True)
    if n_layer == c.n_layer:
        state.unlink(missing_ok=True)
    print(f"{out}: {len(calib)} chunks of {a.len} tokens through {n_layer - first} layers in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
