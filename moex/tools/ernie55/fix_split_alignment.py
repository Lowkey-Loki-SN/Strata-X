"""moex/tools/ernie55/fix_split_alignment.py - adds general.alignment to the later files of a GGUF set convert_gguf_ernie55 wrote
before 2026-11-25, in place.

The converter pads every file's header to 4,096 bytes and places the tensors 4,096-aligned, but only the first file carried
`general.alignment`: a reader takes GGUF's default 32 for the others (gguf_reader.hpp, llama.cpp's gguf.cpp), so it read every
tensor of files 2.. at the wrong offset (the converter's own re-read used its own header size and did not see it).  This
rewrites only the header of each later file: the same key-value pairs plus general.alignment = 4096 (u32), the same tensor
infos, then zeros up to where the data already starts - refused when the larger header would not end before that.  The data
is not touched.

    python3 -I fix_split_alignment.py FILE-00002-of-0000K.gguf [...]
"""
import os
import struct
import sys

ALIGN = 4096
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # gguf value types: their scalar sizes


def parse(b: bytes):
    """-> (n_tensors, n_kv, end of the KVs, end of the tensor infos, the KV keys, the tensor offsets)"""
    if b[:4] != b"GGUF":
        raise SystemExit("not a GGUF")
    ver, = struct.unpack_from("<I", b, 4)
    if ver != 3:
        raise SystemExit(f"GGUF version {ver}")
    n_t, n_kv = struct.unpack_from("<QQ", b, 8)
    p = 24

    def string(p):
        n, = struct.unpack_from("<Q", b, p)
        return b[p + 8:p + 8 + n].decode("utf-8"), p + 8 + n

    def value(t, p):
        if t == 8:   # string
            return string(p)[1]
        if t == 9:   # array: element type, count, elements
            et, n = struct.unpack_from("<IQ", b, p)
            p += 12
            for _ in range(n):
                p = value(et, p)
            return p
        return p + SIZES[t]

    keys = []
    for _ in range(n_kv):
        k, p = string(p)
        t, = struct.unpack_from("<I", b, p)
        p = value(t, p + 4)
        keys.append(k)
    kv_end = p
    offs = []
    for _ in range(n_t):
        _, p = string(p)
        nd, = struct.unpack_from("<I", b, p)
        p += 4 + 8 * nd
        p += 4   # type
        off, = struct.unpack_from("<Q", b, p)
        offs.append(off)
        p += 8
    return n_t, n_kv, kv_end, p, keys, offs


def main() -> int:
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            b = f.read(8 << 20)
        n_t, n_kv, kv_end, info_end, keys, offs = parse(b)
        if "general.alignment" in keys:
            print(f"{path}: has general.alignment already")
            continue
        data = (info_end + ALIGN - 1) // ALIGN * ALIGN   # where the converter put the data (its header padded to 4,096)
        if any(o % ALIGN for o in offs) or min(offs) != 0:
            raise SystemExit(f"{path}: tensor offsets not the converter's 4,096-aligned layout")
        key = b"general.alignment"
        kv = struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, ALIGN)
        head = bytearray(b[:info_end])
        struct.pack_into("<Q", head, 16, n_kv + 1)
        head = head[:kv_end] + kv + head[kv_end:]
        if len(head) > data:
            raise SystemExit(f"{path}: the header would grow past the data at {data} ({len(head)} bytes): not patched")
        head += b"\0" * (data - len(head))
        assert len(head) == data
        fd = os.open(path, os.O_WRONLY)
        try:
            if os.pwrite(fd, bytes(head), 0) != len(head):
                raise SystemExit(f"{path}: short write")
            os.fsync(fd)
        finally:
            os.close(fd)
        with open(path, "rb") as f:
            _, n_kv2, _, info_end2, keys2, offs2 = parse(f.read(8 << 20))
        ok = n_kv2 == n_kv + 1 and "general.alignment" in keys2 and offs2 == offs and (info_end2 + ALIGN - 1) // ALIGN * ALIGN == data
        print(f"{path}: general.alignment added ({info_end} -> {info_end2} header bytes, the data at {data}){'' if ok else ' - CHECK FAILED'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
