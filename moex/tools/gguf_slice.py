#!/usr/bin/env python3
"""moex/tools/gguf_slice.py - a few layers of a GGUF as a complete, smaller model.

    gguf_slice.py <src.gguf> [<src shard 2> ...] <dst.gguf> --layers 0-3 [--mtp]

A split source is given as all of its shards (metadata from the first); the slice is one file (no split.* keys).

The output keeps every metadata key.  `<arch>.block_count` becomes the slice's length; every per-layer
array (an array whose length is the source's block_count, e.g. `head_count_kv`, `swiglu_clamp_exp`) is
sliced the same way.  With `--mtp`, the source's NextN layers are appended after the slice and renumbered.
Without it, `nextn_predict_layers` becomes 0.  Layers keep their order and are renumbered from 0.

The result loads in llama.cpp as an ordinary (shallow) model.  That gives a CPU oracle for every per-layer
tensor at a few GB instead of 111.  The layer TYPES come from the per-layer arrays, so a slice must keep the
pattern it wants to test, e.g. 0-3 is three KDA + dense layers and one DSA + MoE layer for glm5next.
Leading dense layers are counted from 0, so a slice must start at layer 0 when the model has them.

Needs numpy and llama.cpp's gguf-py:
    ~/AI/llama.cpp/.venv/bin/python -I moex/tools/gguf_slice.py ...   (gguf-py is taken from --gguf-py)
"""
import argparse, os, re, sys


def parse_layers(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="the source shard(s), then the output file")
    ap.add_argument("--layers", required=True, help="e.g. 0-3 or 0-3,7")
    ap.add_argument("--mtp", action="store_true", help="append the NextN layer(s)")
    ap.add_argument("--gguf-py", default=os.path.expanduser("~/AI/llama.cpp-glm5next-mtp/gguf-py"))
    a = ap.parse_args()
    if len(a.paths) < 2:
        raise SystemExit("give the source shard(s) and the output file")
    a.src, a.dst = a.paths[0], a.paths[-1]
    srcs = a.paths[:-1]

    sys.path.insert(0, a.gguf_py)
    import gguf  # noqa: E402

    readers = [gguf.GGUFReader(p) for p in srcs]
    r = next((x for x in readers if "split.no" not in x.fields or int(x.fields["split.no"].contents()) == 0), readers[0])
    arch = r.fields["general.architecture"].contents()
    n_block = int(r.fields[f"{arch}.block_count"].contents())
    nextn_key = f"{arch}.nextn_predict_layers"
    n_nextn = int(r.fields[nextn_key].contents()) if nextn_key in r.fields else 0
    n_trunk = n_block - n_nextn

    layers = parse_layers(a.layers)
    if any(l < 0 or l >= n_trunk for l in layers) or len(set(layers)) != len(layers) or layers != sorted(layers):
        raise SystemExit(f"--layers must be increasing trunk layers in [0, {n_trunk})")
    dense_key = f"{arch}.leading_dense_block_count"
    if dense_key in r.fields:
        n_dense = int(r.fields[dense_key].contents())
        if layers[:n_dense] != list(range(min(n_dense, len(layers)))):
            raise SystemExit(f"the model has {n_dense} leading dense layers: the slice must start with 0..{n_dense - 1}")
    src_of = list(layers) + (list(range(n_trunk, n_block)) if a.mtp else [])
    new_block = len(src_of)
    new_nextn = n_nextn if a.mtp else 0

    w = gguf.GGUFWriter(a.dst, arch=arch, endianess=r.endianess)
    for f in r.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF.") or f.name.startswith("split."):
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        v = f.contents()
        if f.name == f"{arch}.block_count":
            v = new_block
        elif f.name == nextn_key:
            v = new_nextn
        elif vt == gguf.GGUFValueType.ARRAY and isinstance(v, list) and len(v) == n_block and f.name.startswith(arch + "."):
            v = [v[i] for i in src_of]
            print(f"  per-layer {f.name}: {v}")
        elif f.name.endswith(".attention.compress_ratios") and isinstance(v, list) and len(v) > n_block:
            # deepseek4 stores more entries than layers (llama.cpp only needs >= block_count): slice the layers' part
            v = [v[i] for i in src_of]
            print(f"  per-layer {f.name} (longer than block_count): {v}")
        w.add_key_value(f.name, v, vt, sub_type=sub)
    w.add_key_value("moex.slice.source", os.path.basename(a.src), gguf.GGUFValueType.STRING)
    w.add_key_value("moex.slice.layers", ",".join(map(str, src_of)), gguf.GGUFValueType.STRING)

    blk = re.compile(r"^blk\.(\d+)\.(.*)$")
    new_of = {old: new for new, old in enumerate(src_of)}
    picked = []
    for t in (t for x in readers for t in x.tensors):
        m = blk.match(t.name)
        if m:
            old = int(m.group(1))
            if old not in new_of:
                continue
            name = f"blk.{new_of[old]}.{m.group(2)}"
        else:
            name = t.name
        picked.append((name, t))
    total = sum(t.n_bytes for _, t in picked)
    print(f"{len(picked)} tensors, {total / 2**30:.2f} GiB; layers {src_of} -> 0..{new_block - 1}; nextn {new_nextn}")

    for name, t in picked:
        w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, t in picked:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print("wrote", a.dst, os.path.getsize(a.dst), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
