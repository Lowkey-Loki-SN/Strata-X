"""moex/tools/dsv41/ref/ref_image.py - DeepSeek-V4.1's image path against the release's model.py (moex vision, milestone 3).

    python ref_image.py prepare OUT_DIR [--image PNG]     (the llama.cpp venv: torch + PIL; no GPU)
    python ref_image.py reference OUT_DIR [--threads 20]  (model.py on the CPU from the native pack, ~30 min)
    python ref_image.py compare OUT_DIR ENGINE_LOG        (the engine's first-token tops against the reference's log-probs)

prepare: a picture (a generated one: "7319" in red on white, unless --image) through serve/vision_dsv41.py (bf16: the release's
numbers, test_vision_dsv41.py) -> OUT_DIR/image.sve; a user turn [the picture, a question] rendered by the server's deepseek41
template in chat mode and tokenized, its <｜deepseek_image｜> repeated once per row of the span -> OUT_DIR/ids.npy, span.json.
reference: Reference.prefill(ids, images=[(start, rows)]) - the span's rows over the embeddings, the image positions' engram masked
and routed with bias_vl, as Transformer.forward does - and the log-probabilities at every position from the span's last row to
the prompt's end -> OUT_DIR/ref_logprobs.npy.  compare: the engine's STRATA_X_FIRST_TOP lines (one request per prefix, run by
serve_image_probe.py) put against them: per position the top token, and the reference's log-probability of the engine's top 8."""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import io
import json
import os
import pathlib
import re
import sys

X = pathlib.Path(__file__).resolve().parents[4]
D = pathlib.Path(_os.path.join(_MODELS, 'dsv41-1017'))


def prepare(out: pathlib.Path, image: pathlib.Path | None):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    sys.path[:0] = [str(X), str(X / "tools")]
    import serve.vision_dsv41 as V
    from serve import families
    out.mkdir(parents=True, exist_ok=True)
    if image is None:
        im = Image.new("RGB", (640, 480), (255, 255, 255))
        d = ImageDraw.Draw(im)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120)
        d.rectangle([40, 40, 600, 440], outline=(0, 0, 0), width=6)
        d.text((120, 160), "7319", fill=(200, 0, 0), font=font)
        b = io.BytesIO()
        im.save(b, format="PNG")
        data = b.getvalue()
    else:
        data = image.read_bytes()
    (out / "image.png").write_bytes(data)
    cfg = json.loads((D / "reference/inference/config.json").read_text())
    torch.set_num_threads(int(os.environ.get("THREADS", "20")))
    enc = V.Encoder(D / "dense", cfg, torch.bfloat16)
    p, h, w, lh, lw = V.load_image(data, cfg, cfg["vision_max_n_token"])
    with torch.inference_mode():
        rows = enc.span(enc.features(p, h, w), lh, lw)
    V.write_sve(out / "image.sve", rows)
    scfg = json.loads((X / "strata-dsv41.json").read_text())
    fam = families.from_config(scfg)
    tok = fam.load_tokenizer(pathlib.Path(scfg["tokenizer"]))
    tpl = fam.load_template()
    msgs = [{"role": "user", "content": [{"type": "image", "source": "x"},
                                          {"type": "text", "text": "What number is written in this picture? Answer with the number only."}]}]
    text = tpl.render(msgs, tools=None, add_generation_prompt=True, **{"enable_thinking": False})
    ids = tok.encode(text, parse_special=True)
    pad = 129264
    assert ids.count(pad) == 1, ids.count(pad)
    start = ids.index(pad)
    n = rows.shape[0]
    ids = ids[:start] + [pad] * n + ids[start + 1:]
    np.save(out / "ids.npy", np.array(ids, dtype=np.int64))
    (out / "span.json").write_text(json.dumps({"start": start, "rows": n, "grid": [lh, lw], "patches": [h, w], "tokens": len(ids)}))
    print(f"{len(ids)} tokens, the span [{start}, {start + n}) ({lh}x{lw} + delimiters); text: {text!r}")


def reference(out: pathlib.Path, threads: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DSV41_REF_GEMM_ACT"] = "fp32"
    import numpy as np
    import torch
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from ref_model import Reference
    sys.path[:0] = [str(X)]
    span = json.loads((out / "span.json").read_text())
    ids = torch.from_numpy(np.load(out / "ids.npy"))
    raw = (out / "image.sve").read_bytes()
    n, width = int.from_bytes(raw[4:8], "little"), int.from_bytes(raw[16:20], "little")
    rows = torch.from_numpy(np.frombuffer(raw[20:], dtype=np.float32).reshape(n, width).copy())
    ref = Reference(D / "native", D / "dense", D / "reference" / "inference", D / "hf-small", D / "index.json",
                    max_seq_len=4096, threads=threads)
    res = ref.prefill(ids, images=[(span["start"], rows)],
                      on_layer=lambda i, a, b, info: print(f"  layer {i:2d}: load {info['load_s']:5.1f} s, run {info['run_s']:5.1f} s", flush=True))
    lg = res["logits"].float()
    first = span["start"] + span["rows"] - 1
    lp = torch.log_softmax(lg[first:], dim=-1).numpy()
    np.save(out / "ref_logprobs.npy", lp)
    print(f"reference log-probs at positions {first}..{len(ids) - 1} -> {out / 'ref_logprobs.npy'}")


def compare(out: pathlib.Path, log: pathlib.Path):
    import numpy as np
    span = json.loads((out / "span.json").read_text())
    lp = np.load(out / "ref_logprobs.npy")
    first = span["start"] + span["rows"] - 1
    tops = re.findall(r"first-token top at position (\d+):(.*)", log.read_text())
    agree = n = 0
    worst = 0.0
    for pos_s, rest in tops:
        pos = int(pos_s)
        if pos < first or pos - first >= lp.shape[0]:
            continue
        pairs = [(int(a), float(b)) for a, b in (x.split(":") for x in rest.split("|")[0].split())]   # (the logits hash after |)
        ref = lp[pos - first]
        eng_top = pairs[0][0]
        ref_top = int(np.argmax(ref))
        d = max(abs(ref[i] - v) for i, v in pairs[:4])
        worst = max(worst, d)
        agree += eng_top == ref_top
        n += 1
        print(f"  position {pos}: engine top {eng_top} ({pairs[0][1]:.3f}), reference top {ref_top} ({ref[ref_top]:.3f}); "
              f"the engine's top-4 log-probs vs the reference's: max |diff| {d:.3f}")
    print(f"image path vs model.py: top token equal at {agree} of {n} positions after the image; worst top-4 log-prob diff {worst:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "reference", "compare"))
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("log", type=pathlib.Path, nargs="?")
    ap.add_argument("--image", type=pathlib.Path)
    ap.add_argument("--threads", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.out, a.image)
    elif a.cmd == "reference":
        reference(a.out, a.threads)
    else:
        compare(a.out, a.log)


if __name__ == "__main__":
    main()
