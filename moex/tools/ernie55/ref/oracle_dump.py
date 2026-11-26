"""moex/tools/ernie55/ref/oracle_dump.py - the reference model's op outputs as strata-x's oracle (strata-x --oracle DIR).

The prompt one token at a time - after the first --probe-from tokens read in one call, unrecorded, as the engine's own
--probe-from reads them through its prompt path - each op's output under the name the engine probes it by (src/x/ernie.cpp,
src/x/stage.cpp, src/x/hy4.cpp), in llama_dump's format (moex/ref/recorder.hpp: index.tsv + dump.bin, a record per (step =
position, name, occurrence)). The engine compares each probe with its record, then takes the record's values (teacher
forcing), so every op is checked on the reference's own inputs.

The routed experts come from the GGUF, dequantized by ernie55_expert_dq (ggml's decoders: the weights the engine computes
with); everything else from the FP8 release, exactly, in fp32 (model.py).

    ~/AI/vLLM/.venv/bin/python -I moex/tools/ernie55/ref/oracle_dump.py --gguf M.gguf --ids 1,2,3 --out DIR
        [--probe-from N] [--layers N] [--threads N] [--model DIR] [--tool PATH]
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import collections
import pathlib
import struct
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[4]

# the engine's probe names on ERNIE's path (a decode window's eager ops); model.py's hook name -> the record's
ENGINE = {n: n for n in (
    "hc_mixes", "hc_pre", "hc_post", "hc_comb", "hc_attn_pre", "attn_norm", "hc_attn_post", "hc_ffn_pre", "ffn_norm", "l_out",
    "q_resid", "hca_q", "hca_kv", "hca_o", "hca_z", "attn_out",
    "kv_cmpr", "k_pe", "idx_k", "idx_q", "idx_w", "kqv", "dsa_o",
    "ffn_up", "ffn_gate", "ffn_swiglu", "ffn_out",
    "ffn_moe_logits", "ffn_moe_topk", "ffn_moe_weights_scaled", "ffn_moe_down", "ffn_moe_out", "ffn_shexp",
    "ffn_moe_lat",   # (no engine probe: the experts' latent input, for strata-x --expert-check)
    "hc_head_mixes", "hc_head", "result_norm", "result_output")}
ENGINE["dsa_sel_rec"] = "dsa_sel"


class ExpertDQ:
    """the GGUF's routed experts through ernie55_expert_dq (one process, a pipe), the last `keep` of them cached"""

    def __init__(self, tool: pathlib.Path, gguf: str, keep: int = 384):
        self.p = subprocess.Popen([str(tool), gguf], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.keep = keep
        self.n_read = 0

    def _read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            b = self.p.stdout.read(n - len(out))
            if not b:
                raise RuntimeError("ernie55_expert_dq ended")
            out += b
        return bytes(out)

    def __call__(self, il: int, e: int):
        k = (il, e)
        if k in self.cache:
            self.cache.move_to_end(k)
            return self.cache[k]
        self.p.stdin.write(f"{il} {e}\n".encode())
        self.p.stdin.flush()
        nf, nl, ok, _ = struct.unpack("<4i", self._read(16))
        if not ok:
            raise RuntimeError(f"the GGUF has no expert {e} in layer {il}")
        n = nf * nl * 4
        g = torch.frombuffer(bytearray(self._read(n)), dtype=torch.float32).view(nf, nl)
        u = torch.frombuffer(bytearray(self._read(n)), dtype=torch.float32).view(nf, nl)
        d = torch.frombuffer(bytearray(self._read(n)), dtype=torch.float32).view(nl, nf)
        self.n_read += 1
        self.cache[k] = (g, u, d)
        if len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return g, u, d

    def close(self) -> None:
        try:
            self.p.stdin.write(b"q\n")
            self.p.stdin.close()
        except OSError:
            pass
        self.p.wait()


class Dump:
    """llama_dump's format: dump.bin (the payloads) and index.tsv (step, name, type, ne0..3, offset)"""

    def __init__(self, out: pathlib.Path):
        out.mkdir(parents=True, exist_ok=True)
        self.bin = open(out / "dump.bin", "wb")
        self.idx = open(out / "index.tsv", "w")
        self.off = 0
        self.step = 0
        self.n = 0

    def rec(self, name: str, il: int, t: torch.Tensor) -> None:
        nm = f"{name}-{il}" if il >= 0 else name
        a = t.detach().contiguous()
        if a.dtype in (torch.int32, torch.int64):
            a, ty = a.to(torch.int32), "i32"
        else:
            a, ty = a.to(torch.float32), "f32"
        b = a.numpy().tobytes()
        self.bin.write(b)
        self.idx.write(f"{self.step}\t{nm}\t{ty}\t{a.numel()}\t1\t1\t1\t{self.off}\n")
        self.off += len(b)
        self.n += 1

    def close(self) -> None:
        self.bin.close()
        self.idx.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the ernie55 GGUF the engine runs (its routed experts)")
    ap.add_argument("--ids", required=True, help="comma-separated token ids (or @FILE with them)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-from", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0, help="layers (default: the GGUF's block_count, read from its name 'sliceN')")
    ap.add_argument("--model", default=_os.path.join(_MODELS, 'HF/ERNIE-5.5-Flash-Preview-FP8'))
    ap.add_argument("--tool", default=str(ROOT / "build-dsv41-dev" / "ernie55_expert_dq"))
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    txt = pathlib.Path(a.ids[1:]).read_text() if a.ids.startswith("@") else a.ids
    ids = [int(x) for x in txt.replace("\n", ",").split(",") if x.strip()]
    m = M.Model(a.model)
    layers = a.layers
    if not layers:
        sys.path.insert(0, _os.path.join(_AI, 'llama.cpp/gguf-py'))
        import gguf  # noqa: E402
        r = gguf.GGUFReader(a.gguf)
        f = r.fields["ernie55.block_count"]
        layers = int(f.parts[f.data[0]][0])
    m.cfg.n_layer = layers
    m.keep_dense = True
    dq = ExpertDQ(pathlib.Path(a.tool), a.gguf)
    m.expert_src = dq
    cache = m.new_cache()
    t0 = time.time()
    with torch.inference_mode():
        if a.probe_from > 0:
            m.forward(ids[:a.probe_from], cache, logits="none")
            print(f"read {a.probe_from} tokens in {time.time() - t0:.1f} s", flush=True)
        dump = Dump(pathlib.Path(a.out))
        m.hooks = {"*": lambda n, il, t: dump.rec(ENGINE[n], il, t) if n in ENGINE else None}
        for p in range(max(a.probe_from, 0), len(ids)):
            dump.step = p
            t1 = time.time()
            lg = m.forward([ids[p]], cache)
            dump.rec("logits", -1, lg)
            print(f"  step {p}: token {ids[p]} -> argmax {int(torch.argmax(lg))} ({time.time() - t1:.1f} s, "
                  f"{dq.n_read} experts read)", flush=True)
        dump.close()
    dq.close()
    print(f"{dump.n} records in {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
