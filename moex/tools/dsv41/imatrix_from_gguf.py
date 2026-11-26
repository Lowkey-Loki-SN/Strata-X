"""moex/tools/dsv41/imatrix_from_gguf.py IMATRIX.gguf OUT.dat - a llama.cpp imatrix (the GGUF form, e.g. Unsloth's
imatrix_unsloth.gguf_file for DeepSeek-V4.1-Flash-1017) as requant_pack's DSV41IMX v1: per layer per routed expert the
gate/up importance (ffn_gate_exps' in_sum2 row: the sum of squared FFN inputs routed to the expert), the down importance
(ffn_down_exps' in_sum2 row: the sum of squared SwiGLU values) and the expert's token count (ffn_gate_exps' counts).

The GGUF is parsed with struct and numpy only (it is downloaded data).  gate and up see the same inputs: their rows must agree,
or the conversion stops.  An expert the imatrix never saw keeps count 0, and requant_pack rounds it without importance, as
llama-quantize does."""
import struct
import sys

import numpy as np

N_LAYER, N_EXPERT, N_EMBD, N_FF = 40, 384, 5120, 2304
DTYPES = {0: np.float32, 1: np.float16, 26: np.int32}


def read_gguf(path):
    b = open(path, "rb").read()
    if b[:4] != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF")
    o = [4]

    def u(f):
        v = struct.unpack_from(f, b, o[0])
        o[0] += struct.calcsize(f)
        return v[0]

    def s():
        n = u("<Q")
        v = b[o[0]:o[0] + n].decode("utf-8", "replace")
        o[0] += n
        return v

    def val(t):
        if t == 8:
            return s()
        if t == 9:
            et, n = u("<I"), u("<Q")
            return [val(et) for _ in range(n)]
        return u({0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}[t])

    u("<I")
    nt, nk = u("<Q"), u("<Q")
    kv = {}
    for _ in range(nk):
        k = s()
        kv[k] = val(u("<I"))
    infos = []
    for _ in range(nt):
        name = s()
        dims = [u("<Q") for _ in range(u("<I"))]
        infos.append((name, dims, u("<I"), u("<Q")))
    align = kv.get("general.alignment", 32)
    data0 = (o[0] + align - 1) // align * align
    tens = {}
    for name, dims, ty, off in infos:
        if ty not in DTYPES:
            raise SystemExit(f"{path}: tensor {name} has type {ty}, not F32/F16/I32")
        tens[name] = np.frombuffer(b, dtype=DTYPES[ty], count=int(np.prod(dims)), offset=data0 + off).reshape(dims[::-1])
    return kv, tens


def main():
    src, out = sys.argv[1], sys.argv[2]
    kv, tens = read_gguf(src)
    print(f"{src}: {kv.get('imatrix.chunk_count')} chunks of {kv.get('imatrix.chunk_size')} tokens, datasets "
          f"{[d.rsplit('/', 1)[-1] for d in kv.get('imatrix.datasets', [])]}")
    worst_gu = 0.0
    with open(out + ".part", "wb") as f:
        f.write(b"DSV41IMX" + struct.pack("<5I", 1, N_LAYER, N_EXPERT, N_EMBD, N_FF))
        zero = 0
        for il in range(N_LAYER):
            g = tens[f"blk.{il}.ffn_gate_exps.weight.in_sum2"].astype(np.float32)
            up = tens[f"blk.{il}.ffn_up_exps.weight.in_sum2"].astype(np.float32)
            d = tens[f"blk.{il}.ffn_down_exps.weight.in_sum2"].astype(np.float32)
            c = np.asarray(tens[f"blk.{il}.ffn_gate_exps.weight.counts"], dtype=np.float64).reshape(-1)
            if g.shape != (N_EXPERT, N_EMBD) or d.shape != (N_EXPERT, N_FF) or c.shape != (N_EXPERT,):
                raise SystemExit(f"layer {il}: shapes {g.shape} {d.shape} {c.shape}")
            rel = float(np.max(np.abs(g - up) / np.maximum(np.abs(g), 1e-30)))
            worst_gu = max(worst_gu, rel)
            if rel > 1e-4:
                raise SystemExit(f"layer {il}: gate and up importance differ (max relative {rel:.3g}): not one input")
            for e in range(N_EXPERT):
                n = int(round(c[e]))
                zero += n == 0
                f.write(g[e].tobytes())
                f.write(d[e].tobytes())
                f.write(struct.pack("<I", n))
    import os
    os.replace(out + ".part", out)
    print(f"{out}: {N_LAYER} layers x {N_EXPERT} experts; gate/up rows agree (max relative {worst_gu:.2g}); {zero} experts "
          f"with no tokens (rounded without importance)")


if __name__ == "__main__":
    main()
