"""moex/tools/ernie55/ref/run.py - the reference model answering one chat turn, greedily, on the CPU: the prompt read in
one call, then one token a step through the caches, until an end token (ids 2 / 8) or --max-new.

    ~/AI/vLLM/.venv/bin/python -I moex/tools/ernie55/ref/run.py "What is 17 * 23?" [--think] [--max-new 48]
                                        [--model DIR] [--index FILE] [--tokenizer FILE] [--threads N] [--ids-out FILE]

The prompt is the release's format (encoding_ernie.py `render`, one user turn, no system message): the system block with
the global setting (reasoning_effort=high when thinking), the user turn, the assistant prefix and <ernie:think> - closed
at once when thinking is off. Tokenized with the release's tokenizer.json (HF tokenizers), the markers as their ids.
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import pathlib
import sys
import time

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M  # noqa: E402

META = pathlib.Path(_os.path.join(_DATA, 'ernie55-meta'))


def prompt_text(user: str, think: bool) -> str:
    gs = "<ernie:global_setting>" + ("\nreasoning_effort=high" if think else "") + "\n</ernie:global_setting>"
    return ("<|ERNIE_MASK|><|ernie:im_start|>system\n" + gs + "<|ernie:im_end|>\n<|ernie:im_start|>user\n" + user +
            "<|ernie:im_end|>\n<|ernie:im_start|>assistant\n" + ("<ernie:think>" if think else "<ernie:think></ernie:think>"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--model", default=_os.path.join(_MODELS, 'HF/ERNIE-5.5-Flash-Preview-FP8'))
    ap.add_argument("--index", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ids-out", default=None, help="write the prompt and generated ids here (one comma list a line)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    tp = pathlib.Path(a.tokenizer) if a.tokenizer else pathlib.Path(a.model) / "tokenizer.json"
    if not tp.exists():
        tp = META / "tokenizer.json"
    tok = Tokenizer.from_file(str(tp))
    text = prompt_text(a.user, a.think)
    ids = tok.encode(text, add_special_tokens=False).ids
    print(f"prompt: {len(ids)} tokens: {ids}", flush=True)
    m = M.Model(a.model, a.index)
    cache = m.new_cache()
    t0 = time.time()
    with torch.inference_mode():
        logits = m.forward(ids, cache)
        print(f"read in {time.time() - t0:.1f} s", flush=True)
        out = []
        for _ in range(a.max_new):
            nxt = int(torch.argmax(logits))
            out.append(nxt)
            print(f"  {nxt:6d} {tok.decode([nxt], skip_special_tokens=False)!r}  (top logit {float(logits.max()):.3f})", flush=True)
            if nxt in (2, 8):
                break
            t1 = time.time()
            logits = m.forward([nxt], cache)
            print(f"         step {time.time() - t1:.1f} s", flush=True)
    print("reply:", tok.decode(out, skip_special_tokens=False), flush=True)
    if a.ids_out:
        pathlib.Path(a.ids_out).write_text(",".join(map(str, ids)) + "\n" + ",".join(map(str, out)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
