#!/usr/bin/env python3
"""moex/tools/hy4/hy4_slice.py - the first K layers of a Hy4-flash (hy_v4_flash) GGUF as a complete, smaller model.

    python3 -I moex/tools/hy4/hy4_slice.py <file 1> <file 2> <out.gguf> [--layers 6 | 0-3,11] [--engram-primes 1009,...]

For the op-level checks of moex/docs/HY4_PORT.md (llama.cpp's CPU build as the oracle, strata-x's probe mode) before, or
without, the whole 168 GB set: layers 0 .. K-1 (with the default 6: the dense layer, two sparse layers with the indexer, three
sliding-window layers, a third sparse one, and the Engram's first layer) - or a list of layers, renumbered from 0 in their
order (0-3,11: the Engram's first two modules, the second on a later GPU with --split 4) - the embedding, the head and its
stream reduce.  The list must start at layer 0 (the dense layer).

What changes from the source, nothing else:
- `hy_v4_flash.block_count` = K and every per-layer array (head counts, the patterns, the SwiGLU limits) cut to K; no NextN
  block (`nextn_predict_layers` 0); no split keys.
- The Engram keeps the modules whose layer is below K.  Their buckets are re-sized to small primes (default 16 primes
  around 1000), the offsets rebased from 0 and the table cut to their rows - the first rows of the source's table.  The
  hash, the lookups and the math are the real ones over a smaller table, so llama.cpp and strata-x compare op for op on
  the same metadata; only the rows a token lands on differ from the full model's.

The source files may still be downloading (hf_fetch.py's .part files grow from the start): the tensors read must be there.
Standard library only.
"""
import argparse
import struct
import sys

# ggml type -> (elements a block, bytes a block); the types a hy_v4_flash GGUF holds
TYPES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
         16: (256, 66), 17: (256, 74), 18: (256, 98), 20: (32, 18), 23: (256, 136), 30: (1, 2), 39: (32, 17), 40: (64, 36),
         43: (1, 1)}
# GGUF value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.path = path

    def rd(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise SystemExit(f"{self.path}: short read in the header")
        return struct.unpack(fmt, b)[0]

    def s(self):
        return self.f.read(self.rd("<Q")).decode("utf-8")

    def val(self, t):
        if t in SCALAR:
            return t, self.rd(SCALAR[t])
        if t == STR:
            return t, self.s()
        if t == ARR:
            et, n = self.rd("<I"), self.rd("<Q")
            return t, (et, [self.val(et)[1] for _ in range(n)])
        raise SystemExit(f"{self.path}: unknown value type {t}")

    def header(self):
        if self.f.read(4) != b"GGUF":
            raise SystemExit(f"{self.path}: not a GGUF")
        ver, nt, nkv = self.rd("<I"), self.rd("<Q"), self.rd("<Q")
        kv = {}
        for _ in range(nkv):
            k = self.s()
            kv[k] = self.val(self.rd("<I"))
        tens = []
        for _ in range(nt):
            name = self.s()
            nd = self.rd("<I")
            dims = [self.rd("<Q") for _ in range(nd)]
            tens.append({"name": name, "dims": dims, "type": self.rd("<I"), "off": self.rd("<Q")})
        align = kv.get("general.alignment", (U32, 32))[1]
        data = (self.f.tell() + align - 1) // align * align
        return ver, kv, tens, data


def nbytes(t):
    be, bb = TYPES[t["type"]]
    n = 1
    for d in t["dims"]:
        n *= d
    if t["dims"][0] % be:
        raise SystemExit(f"{t['name']}: rows of {t['dims'][0]} are not whole blocks of type {t['type']}")
    return n // be * bb


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")

    def w(self, fmt, v):
        self.f.write(struct.pack(fmt, v))

    def s(self, x):
        b = x.encode("utf-8")
        self.w("<Q", len(b))
        self.f.write(b)

    def val(self, t, v):
        if t in SCALAR:
            self.w(SCALAR[t], v)
        elif t == STR:
            self.s(v)
        elif t == ARR:
            et, items = v
            self.w("<I", et)
            self.w("<Q", len(items))
            for it in items:
                self.val(et, it)
        else:
            raise SystemExit(f"unknown value type {t}")


def primes_near(n, count):
    out, x = [], n
    while len(out) < count:
        if x > 1 and all(x % p for p in range(2, int(x ** 0.5) + 1)):
            out.append(x)
        x += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file1")
    ap.add_argument("file2")
    ap.add_argument("out")
    ap.add_argument("--layers", default="6", help="K (layers 0 .. K-1) or a list like 0-3,11")
    ap.add_argument("--engram-primes", default="", help="the Engram's bucket counts, one a column (default: primes from 1009)")
    a = ap.parse_args()
    if "," in a.layers or "-" in a.layers:
        keep_l = []
        for part in a.layers.split(","):
            lo, _, hi = part.partition("-")
            keep_l += list(range(int(lo), int(hi or lo) + 1))
    else:
        keep_l = list(range(int(a.layers)))
    if keep_l != sorted(set(keep_l)) or keep_l[0] != 0:
        raise SystemExit("--layers: ascending, distinct, from layer 0")
    K = len(keep_l)
    new_of = {old: i for i, old in enumerate(keep_l)}
    r1, r2 = Reader(a.file1), Reader(a.file2)
    ver, kv, t1, d1 = r1.header()
    _, kv2, t2, d2 = r2.header()
    arch = kv["general.architecture"][1]
    if arch != "hy_v4_flash":
        raise SystemExit(f"{a.file1}: architecture {arch}, not hy_v4_flash")
    pre = arch + "."
    n_all = kv[pre + "block_count"][1]
    nextn = kv.get(pre + "nextn_predict_layers", (U32, 0))[1]
    n_layer = n_all - nextn
    if keep_l[-1] >= n_layer:
        raise SystemExit(f"--layers {a.layers}: the model has {n_layer}")
    out = {}
    for k, (t, v) in kv.items():
        if k.startswith("split."):
            continue
        if t == ARR and k.startswith(pre) and len(v[1]) in (n_all, n_layer) and not k.startswith(pre + "ple."):
            v = (v[0], [v[1][i] for i in keep_l])   # a per-layer array (43 with the NextN block, 42 without)
        out[k] = (t, v)
    out[pre + "block_count"] = (U32, K)
    out[pre + "nextn_predict_layers"] = (U32, 0)
    out["general.name"] = (STR, f"Hy4-flash layers {a.layers} (moex hy4_slice)")
    # the Engram: the modules below K, on small buckets
    eg_rows = 0
    if pre + "ple.layers" in kv:
        layers = kv[pre + "ple.layers"][1][1]
        ng = kv[pre + "ple.ngram_size"][1]
        hpn = kv[pre + "ple.heads_per_ngram"][1]
        cols = (ng - 1) * hpn
        keep = [j for j, il in enumerate(layers) if il in new_of]
        if keep:
            primes = [int(x) for x in a.engram_primes.split(",")] if a.engram_primes else primes_near(1009, cols * len(keep))
            if len(primes) != cols * len(keep):
                raise SystemExit(f"--engram-primes: {cols * len(keep)} wanted")
            mul = kv[pre + "ple.layer_multipliers"]
            offs, base = [], 0
            for p in primes:
                offs.append(base)
                base += p
            eg_rows = base
            out[pre + "ple.layers"] = (ARR, (kv[pre + "ple.layers"][1][0], [new_of[layers[j]] for j in keep]))
            out[pre + "ple.layer_multipliers"] = (ARR, (mul[1][0], [mul[1][1][j * ng + g] for j in keep for g in range(ng)]))
            out[pre + "ple.head_offsets"] = (ARR, (kv[pre + "ple.head_offsets"][1][0], offs))
            out[pre + "ple.head_vocab_sizes"] = (ARR, (kv[pre + "ple.head_vocab_sizes"][1][0], primes))
        else:
            for k in [k for k in out if k.startswith(pre + "ple.") or k.startswith(pre + "engram.")]:
                del out[k]
            del out[pre + "embedding_length_per_layer_input"]
    # the tensors: the trunk's first K layers, everything outside the blocks, the Engram table's first rows
    srcs = []
    for t in t1:
        nm = t["name"]
        if nm.startswith("blk."):
            il = int(nm.split(".")[1])
            if il not in new_of:
                continue
            t = dict(t, name="blk.%d.%s" % (new_of[il], nm.split(".", 2)[2]))
        srcs.append((r1, d1, t, None))
    for t in t2:
        if t["name"] == "per_layer_token_embd.weight":
            if eg_rows:
                srcs.append((r2, d2, t, eg_rows))
        else:
            srcs.append((r2, d2, t, None))
    align = 32
    out["general.alignment"] = (U32, align)
    w = Writer(a.out)
    w.f.write(b"GGUF")
    w.w("<I", ver)
    w.w("<Q", len(srcs))
    w.w("<Q", len(out))
    for k, (t, v) in out.items():
        w.s(k)
        w.w("<I", t)
        w.val(t, v)
    off, plan = 0, []
    for r, d, t, rows in srcs:
        dims = list(t["dims"])
        if rows is not None:
            dims[1] = rows
        nb = nbytes({"name": t["name"], "dims": dims, "type": t["type"]})
        w.s(t["name"])
        w.w("<I", len(dims))
        for x in dims:
            w.w("<Q", x)
        w.w("<I", t["type"])
        w.w("<Q", off)
        plan.append((r, d + t["off"], nb))
        off += (nb + align - 1) // align * align
    pos = w.f.tell()
    w.f.write(b"\0" * ((pos + align - 1) // align * align - pos))
    total = 0
    for r, at, nb in plan:
        r.f.seek(at)
        left = nb
        while left:
            b = r.f.read(min(left, 64 << 20))
            if not b:
                raise SystemExit(f"{r.path}: ends before byte {at + nb} (still downloading?)")
            w.f.write(b)
            left -= len(b)
        total += nb
        pad = (nb + align - 1) // align * align - nb
        w.f.write(b"\0" * pad)
    w.f.close()
    print(f"{a.out}: {len(srcs)} tensors, {total / 1e9:.2f} GB, layers {keep_l}"
          + (f", Engram table {eg_rows} rows" if eg_rows else ", no Engram"))


if __name__ == "__main__":
    sys.exit(main())
