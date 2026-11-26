#!/usr/bin/env python3
"""moex/tools/gguf_kv_patch.py - a GGUF's metadata rewritten into a NEW file, every tensor byte copied unchanged.

    python3 -I gguf_kv_patch.py IN.gguf OUT.gguf --preset glm5v
    python3 -I gguf_kv_patch.py IN.gguf OUT.gguf --set-str KEY=VALUE --set-f32 KEY=VALUE --set-u32 KEY=VALUE --drop KEY

Only the key/value section changes: the tensor infos (names, shapes, types, offsets relative to the data section) and the
data section are copied byte for byte; the data section starts at the next multiple of the file's alignment after the new
header (tensor offsets are relative to it, so they stay valid).  The output must not exist (never overwrites).

--preset glm5v: GLM-5.3-Flash's projector as buun-llama-cpp's converter wrote it ("glm5next", clip.vision.swiglu_limit,
no pixel budget) -> what mainline llama.cpp's converter writes and its mtmd requires (conversion/qwen3vl.py
Glm5NextVisionModel: projector_type "glm5v", clip.vision.swiglu_clamp, clip.vision.image_min_pixels / image_max_pixels
from the processor's min/max_image_tokens 16 / 8000 x (patch 14 x merge 2)^2).  The math is the same in both trees
(models/glm4v.cpp vs models/glm5next-vision.cpp): only names and the stored budget differ.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(13)
SCALAR = {T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h", T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f",
          T_BOOL: "<?", T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d"}


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> bytes:
        return self.take(self.u64())


def read_value_raw(r: Reader, vtype: int) -> tuple[bytes, object]:
    """The value's exact bytes (as stored after the type word) and its decoded form."""
    if vtype in SCALAR:
        fmt = SCALAR[vtype]
        b = r.take(struct.calcsize(fmt))
        return b, struct.unpack(fmt, b)[0]
    if vtype == T_STRING:
        s = r.string()
        return struct.pack("<Q", len(s)) + s, s.decode("utf-8", "replace")
    if vtype == T_ARRAY:
        et = r.u32()
        n = r.u64()
        parts = [struct.pack("<IQ", et, n)]
        vals = []
        for _ in range(n):
            b, v = read_value_raw(r, et)
            parts.append(b)
            vals.append(v)
        return b"".join(parts), vals
    raise ValueError(f"unknown GGUF value type {vtype}")


def gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_bytes(key: str, vtype: int, value) -> bytes:
    out = gguf_string(key) + struct.pack("<I", vtype)
    if vtype == T_STRING:
        return out + gguf_string(value)
    return out + struct.pack(SCALAR[vtype], value)


PRESETS = {
    "glm5v": {
        "require": {"clip.projector_type": "glm5next"},
        "set": [("clip.projector_type", T_STRING, "glm5v"),
                ("clip.vision.swiglu_clamp", T_FLOAT32, 10.0),
                ("clip.vision.image_min_pixels", T_UINT32, 16 * 28 * 28),     # 12544
                ("clip.vision.image_max_pixels", T_UINT32, 8000 * 28 * 28)],  # 6272000
        "drop": ["clip.vision.swiglu_limit"],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--set-str", action="append", default=[])
    ap.add_argument("--set-f32", action="append", default=[])
    ap.add_argument("--set-u32", action="append", default=[])
    ap.add_argument("--drop", action="append", default=[])
    a = ap.parse_args()
    if os.path.exists(a.dst):
        print(f"{a.dst} exists: refusing to overwrite", file=sys.stderr)
        return 2
    sets: list[tuple[str, int, object]] = []
    drops: list[str] = list(a.drop)
    require: dict = {}
    if a.preset:
        p = PRESETS[a.preset]
        sets += p["set"]
        drops += p["drop"]
        require = p["require"]
    sets += [(kv.split("=", 1)[0], T_STRING, kv.split("=", 1)[1]) for kv in a.set_str]
    sets += [(kv.split("=", 1)[0], T_FLOAT32, float(kv.split("=", 1)[1])) for kv in a.set_f32]
    sets += [(kv.split("=", 1)[0], T_UINT32, int(kv.split("=", 1)[1])) for kv in a.set_u32]

    with open(a.src, "rb") as f:
        r = Reader(f)
        if r.take(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        version = r.u32()
        if version < 2:
            raise SystemExit(f"GGUF version {version} is not supported")
        n_tensors, n_kv = r.u64(), r.u64()
        kvs: list[tuple[str, int, bytes, object]] = []
        for _ in range(n_kv):
            key = r.string().decode("utf-8")
            vtype = r.u32()
            raw, val = read_value_raw(r, vtype)
            kvs.append((key, vtype, raw, val))
        tinfo_start = f.tell()
        for _ in range(n_tensors):          # walk the tensor infos to find where they end; copied verbatim below
            r.string()
            nd = r.u32()
            r.take(8 * nd)
            r.u32()
            r.u64()
        tinfo_end = f.tell()
        f.seek(tinfo_start)
        tinfo = f.read(tinfo_end - tinfo_start)
        align = 32
        for key, vtype, raw, val in kvs:
            if key == "general.alignment":
                align = int(val)
        data_start = (tinfo_end + align - 1) // align * align
        size = os.path.getsize(a.src)

        have = {k: v for k, _, _, v in kvs}
        for k, want in require.items():
            if have.get(k) != want:
                raise SystemExit(f"{a.src}: {k} is {have.get(k)!r}, the preset expects {want!r}")
        out_kvs: list[bytes] = []
        set_keys = {k for k, _, _ in sets}
        for key, vtype, raw, val in kvs:
            if key in drops or key in set_keys:
                continue
            out_kvs.append(gguf_string(key) + struct.pack("<I", vtype) + raw)
        for key, vtype, value in sets:
            out_kvs.append(kv_bytes(key, vtype, value))
        header = b"GGUF" + struct.pack("<IQQ", version, n_tensors, len(out_kvs)) + b"".join(out_kvs) + tinfo
        pad = (-len(header)) % align
        tmp = a.dst + ".part"
        with open(tmp, "wb") as o:
            o.write(header + b"\0" * pad)
            f.seek(data_start)
            left = size - data_start
            while left > 0:
                chunk = f.read(min(left, 64 << 20))
                if not chunk:
                    raise SystemExit("short read of the data section")
                o.write(chunk)
                left -= len(chunk)
        os.rename(tmp, a.dst)
    print(f"wrote {a.dst}: {len(out_kvs)} keys (was {n_kv}), {n_tensors} tensors, data section {size - data_start} bytes "
          f"copied unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
