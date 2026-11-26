"""verify_pack_downs.py PACK [K] - a pack written by gguf_pack.py checked on disk and against the release.

    python -I verify_pack_downs.py ~/AI/Models/dsv41-1017/ud-iq3xxs [K]

Every one of the 40 layers in experts.tsv; each record file exactly 384 records of its stride, the stride its types' blob
rounded up to 4 KiB; and every MXFP4 down (ggml type 39) compared byte for byte with the native records' down part - K
experts a layer (4 by default: the first, the last and two between). Unsloth's UD quants keep MXFP4 downs in some
layers; equal bytes say they are the release's own (fp4_repack.py writes the release's FP4 as ggml block_mxfp4, as
llama.cpp's conversion does). Reads ~2 x K x 6.3 MB a MXFP4 layer through the page cache, dropped after each file.
gguf_pack.py --verify N is the other check: the records against the GGUF on Hugging Face.
"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gguf_pack import N_EMBD, N_EXPERT, N_FF, N_LAYER, TYPE_NAME, row_bytes  # noqa: E402

NATIVE = pathlib.Path(_os.path.join(_MODELS, 'dsv41-1017/native'))
SRC = 18_800_640   # a native record: [gate | up | down], all MXFP4, no padding
MX = 39


def main() -> int:
    pack = pathlib.Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    layers = {}
    for line in open(pack / "experts.tsv"):
        if line.startswith("#") or not line.strip():
            continue
        il, ty, stride, rel = line.split()
        gu, _, d = ty.partition(",")
        layers[int(il)] = (int(gu), int(d or gu), int(stride), rel)
    bad = same = 0
    k = max(2, k)
    picks = sorted({round(i * (N_EXPERT - 1) / (k - 1)) for i in range(k)})   # the first, the last, the rest spread
    for il in range(N_LAYER):
        if il not in layers:
            print(f"layer {il}: missing from experts.tsv")
            bad += 1
            continue
        gu, d, stride, rel = layers[il]
        blob = 2 * N_FF * row_bytes(gu, N_EMBD) + N_EMBD * row_bytes(d, N_FF)
        if stride != (blob + 4095) // 4096 * 4096:
            print(f"layer {il}: stride {stride:,} for a {blob:,}-byte blob")
            bad += 1
        size = os.path.getsize(pack / rel)
        if size != N_EXPERT * stride:
            print(f"layer {il}: {size:,} bytes, not {N_EXPERT} x {stride:,}")
            bad += 1
            continue
        if d != MX:
            continue
        gu_new, gu_src, d_len = N_FF * row_bytes(gu, N_EMBD), N_FF * row_bytes(MX, N_EMBD), N_EMBD * row_bytes(MX, N_FF)
        with open(NATIVE / "experts" / f"L{il:02d}.bin", "rb") as fs, open(pack / rel, "rb") as fn:
            for e in picks:
                fs.seek(e * SRC + 2 * gu_src)
                fn.seek(e * stride + 2 * gu_new)
                if fs.read(d_len) != fn.read(d_len):
                    print(f"layer {il} ({TYPE_NAME[gu]} / MXFP4) expert {e}: the down differs from the release's")
                    bad += 1
                else:
                    same += 1
            for f in (fs, fn):
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    mx_layers = sum(1 for v in layers.values() if v[1] == MX)
    total = sum(N_EXPERT * v[2] for v in layers.values())
    print(f"{len(layers)} layers, {total / 1e9:.2f} GB of records; {mx_layers} MXFP4-down layers, {same} downs equal to the "
          f"release's; {'all checks pass' if not bad else f'{bad} problems'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
