"""moex/tools/dsv41/test_fp4_repack.py - fp4_repack.py on real DeepSeek-V4.1 experts, bit for bit.

    python test_fp4_repack.py --index model.safetensors.index.json [--experts layers.0.ffn.experts.0 ...]

Range-reads each expert's w1/w2/w3 weights and scales from the release on Hugging Face (~19 MB per expert), then
checks that ggml's own MXFP4 decoding (gguf-py) of the repacked blocks gives exactly the floats of the release's
decoding, and that the record is 18,800,640 bytes (whole 4 KiB pages).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fp4_repack  # noqa: E402

REPO = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/"
DEFAULT = ["layers.0.ffn.experts.0", "layers.20.ffn.experts.191", "layers.39.ffn.experts.383",
           "mtp.2.ffn.experts.127"]


def fetch(url: str, a: int, b: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if len(data) != b - a:
        raise IOError(f"short read {len(data)} of {b - a} from {url}")
    return data


_headers: dict[str, tuple[int, dict]] = {}


def header(shard: str) -> tuple[int, dict]:
    if shard not in _headers:
        n = struct.unpack("<Q", fetch(REPO + shard, 0, 8))[0]
        _headers[shard] = (8 + n, json.loads(fetch(REPO + shard, 8, 8 + n)))
    return _headers[shard]


def tensor(index: dict, name: str) -> np.ndarray:
    shard = index["weight_map"][name]
    base, h = header(shard)
    t = h[name]
    a, b = t["data_offsets"]
    dt = {"I8": np.int8, "U8": np.uint8, "F8_E8M0": np.uint8}[t["dtype"]]
    return np.frombuffer(fetch(REPO + shard, base + a, base + b), dtype=dt).reshape(t["shape"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, type=pathlib.Path)
    ap.add_argument("--experts", nargs="*", default=DEFAULT)
    a = ap.parse_args()
    from gguf.quants import MXFP4  # ggml's MXFP4 decoding, as gguf-py implements it

    index = json.loads(a.index.read_text())
    fails = 0
    for ex in a.experts:
        parts = {}
        for w in ("w1", "w3", "w2"):
            parts[w] = (tensor(index, f"{ex}.{w}.weight"), tensor(index, f"{ex}.{w}.scale"))
        rec = fp4_repack.expert_record(*parts["w1"], *parts["w3"], *parts["w2"])
        line = [f"{ex}: record {len(rec):,} B ({len(rec) / 4096:.2f} pages)"]
        ok = len(rec) == 18_800_640
        for w, (wt, sc) in parts.items():
            ref = fp4_repack.dequant_reference(wt, sc)
            blocks = fp4_repack.repack(wt, sc).reshape(-1, fp4_repack.BLOCK_BYTES)
            got = MXFP4.dequantize_blocks(blocks).reshape(ref.shape)
            same = np.array_equal(ref.view(np.uint32), got.astype(np.float32).view(np.uint32))
            ok = ok and same
            line.append(f"{w} {wt.shape[0]}x{wt.shape[1] * 2} {'bit-exact' if same else 'DIFFERENT'} "
                        f"(scales {int(sc.min())}..{int(sc.max())})")
        print("  ".join(line))
        fails += not ok
    print("PASS" if not fails else f"FAIL ({fails})")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
