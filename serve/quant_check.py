"""moex (moex/docs/QUANTS.md): whether strata-x runs a GGUF set, from its tensor headers alone.

    python3 serve/quant_check.py FIRST_SHARD.gguf               a set on disk (every shard of a split set)
    python3 serve/quant_check.py --hf REPO [FOLDER_OR_FILE]     a Hugging Face set before downloading it: the headers only,
                                                                through HTTP range requests

The server runs the same check on a strata-x run config's -m set before it starts the engine
(STRATA_QUANT_CHECK=0 skips it).  The engine's own checks still run; this one says it in seconds, by tensor name, before
tens of GB load.  A weight the engine has no kernel for stops the start ("errors"); a slow path - experts the GPU cannot
hold, experts whose type switches the batched prompt path off - is said once ("notes").  The rules are the engine's, each
cited to its source: refuse only what the engine refuses, so the check can never stop a file the engine would run.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile  # noqa: E402

# the pinned ggml's enum (build-dsv41/_deps/strata_llamacpp-src/ggml/include/ggml.h), and two ids of llama.cpp branches
# past it that the Engram tables use (src/x/hy4.cpp:720, src/x/dsv41.cpp:2397)
NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
         43: "F8_E4M3/MXFP8", 67: "MXFP8"}
F32, F16, BF16 = 0, 1, 30
# the GPU's decode kernels: kn::native_mmvq_supported (src/kernels/cuda/native_mmvq.cu)
GPU = frozenset({2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 40, 42})
# the batched prompt path's expert tiles in the default build (CMakeLists.txt _strata_mmq_types + STRATA_MOEX_KQUANTS);
# one MoE layer without them switches the path off for the whole model (src/x/tiers.cpp mmq_all)
MMQ = frozenset({42, 16, 17, 22, 18, 21, 20, 23, 8, 10, 11, 12, 13, 14, 39, 40, 29, 19})
# experts of these types have a ggml-cpu dot product (the CPU pool computes them; native_expert.cpp)
CPU = frozenset(GPU | {3, 15, 34, 35, 41, F16, BF16})
ARCHES = {"glm5next": "GLM-5.3-Flash", "glm5-next": "GLM-5.3-Flash", "deepseek4": "DeepSeek-V4-Flash",
          "deepseek41": "DeepSeek-V4.1-Flash", "mimo2": "MiMo-V2.6-Flash", "hy_v4_flash": "Hy4-flash",
          "ernie55": "ERNIE-5.5-Flash"}
# the families whose loaders take BF16 projections (src/x/dsv41.cpp:366, src/x/hy4.cpp:185, src/x/ernie.cpp:157); the others
# take F32 or a GPU-kernel type only (src/x/engine.cpp:769 proj_ok, src/x/dsv4.cpp:161)
BF16_DENSE = frozenset({"deepseek41", "hy_v4_flash", "ernie55"})
# the Engram table's accepted types (per_layer_token_embd.weight): Hy4 F8_E4M3 or Q8_0 (src/x/hy4.cpp:726), DeepSeek-V4.1
# MXFP8 or IQ4_NL (src/x/dsv41.cpp:2473; another file with STRATA_DSV41_ENGRAM_GGUF, or a pack's tables).  The engine's
# load ran both loaders for V4.1 from 2026-11-18 to 2026-11-28, and Hy4's refused the type-67 tables this check accepts
# (DSV41_PLAN §77): a check and an engine that disagree hide a load failure
ENGRAM = {"hy_v4_flash": frozenset({43, 8}), "deepseek41": frozenset({67, 43, 20})}
# DeepSeek-V4's weights the loader takes as F32 only (src/x/dsv4.cpp:180, :237: f32()) - ggml-org's MXFP4 file stores
# indexer.proj as Q8_0, which the engine refuses
DSV4_F32 = (".indexer.proj.weight", ".indexer_compressor_ape.weight", "output_hc_fn.weight")
# DeepSeek-V4.1's weights taken as F32, or as BF16 with STRATA_DSV41_BF16_W=1 (src/x/dsv41.cpp:436-443 f32w): the GGUF
# sets store them as BF16
DSV41_F32W = (".attn_compressor_kv.weight", ".attn_compressor_gate.weight", ".indexer.attn_k.weight",
              ".indexer.attn_q_b.weight", ".indexer.proj.weight")
INDEX_TYPES = frozenset({24, 25, 26, 27, 28})
EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
BLOCK = re.compile(r"^blk\.(\d+)\.")
SPLIT = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


def tname(t: int) -> str:
    return NAMES.get(t, f"type {t}")


@dataclass
class Report:
    model: str = ""
    arch: str = ""
    files: int = 0
    experts: Counter = field(default_factory=Counter)     # type -> elements
    other: Counter = field(default_factory=Counter)       # type -> tensors
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [f"{self.model or 'unknown model'} ({self.arch or 'no architecture'}), {self.files} file(s)"]
        tot = sum(self.experts.values())
        if tot:
            out.append("  routed experts: " + ", ".join(f"{tname(t)} {n / tot:.1%}" for t, n in self.experts.most_common()))
        if self.other:
            out.append("  other weights:  " + ", ".join(f"{tname(t)} {n}" for t, n in self.other.most_common()))
        if self.errors:
            out.append("  does not run:")
            out += [f"    - {e}" for e in self.errors]
        else:
            out.append("  runs" + (", with:" if self.notes else ": every weight has a GPU kernel and the batched prompt path"))
        out += [f"    - {n}" for n in self.notes]
        out.append("  (moex/docs/QUANTS.md: what each model runs from)")
        return out


def _few(names: list[str], k: int = 3) -> str:
    return ", ".join(names[:k]) + (f" and {len(names) - k} more" if len(names) > k else "")


def check(headers: list[tuple[dict, list]], env: dict | None = None, args: list[str] | None = None) -> Report:
    """headers: (metadata, tensors) of each file of one set - tensors as (name, shape, type id).  env / args: a run
    config's, when the server checks it (the switches that change what the engine accepts); args None: no config (the
    command line), so what depends on one is said as a note."""
    server = args is not None
    env, args = env or {}, args or []
    rep = Report(files=len(headers))
    meta = {}
    for kv, _ in headers:
        meta.update({k: v for k, v in kv.items() if not isinstance(v, list)})
    arch = str(meta.get("general.architecture", ""))
    rep.arch = arch
    rep.model = ARCHES.get(arch, "")
    if arch not in ARCHES:
        rep.errors.append(f"architecture {arch or '(none)'!r} is not one strata-x runs (it runs "
                          + ", ".join(sorted(ARCHES)) + "); Qwen3.8 files are upstream Strata's (setup.py)")
        return rep
    blocks = meta.get(f"{arch}.block_count")
    mtp_from = blocks - int(meta.get(f"{arch}.nextn_predict_layers", 0) or 0) if isinstance(blocks, int) else None
    per_layer: dict[int, dict[str, int]] = {}
    bad: dict[str, list[str]] = {}
    bf16_f32w: list[str] = []
    for _, tensors in headers:
        for name, shape, t in tensors:
            m = BLOCK.match(name)
            mtp = "nextn" in name or (m is not None and mtp_from is not None and int(m.group(1)) >= mtp_from)
            e = EXPERT.match(name)
            if e:
                if not mtp:
                    per_layer.setdefault(int(e.group(1)), {})[e.group(2)] = t
                    n = 1
                    for d in shape:
                        n *= d
                    rep.experts[t] += n
                continue
            if name == "per_layer_token_embd.weight" and arch in ENGRAM:
                ok = t in ENGRAM[arch]
                if arch == "deepseek41" and ("--pack" in args or env.get("STRATA_DSV41_ENGRAM_GGUF")):
                    ok = True
                if not ok:
                    want = "F8_E4M3 or Q8_0" if arch == "hy_v4_flash" else \
                        "MXFP8 or IQ4_NL (or STRATA_DSV41_ENGRAM_GGUF=FILE naming such a table, or a pack's)"
                    rep.errors.append(f"the Engram table (per_layer_token_embd.weight) is {tname(t)}; the engine reads it "
                                      f"as {want}")
                continue
            if len(shape) < 2 or t in INDEX_TYPES or name == "token_embd.weight" or mtp:
                continue       # vectors, index tables, the embedding (read on the CPU, any type) and drafter blocks
            rep.other[t] += 1
            if arch == "deepseek41" and name.endswith(DSV41_F32W):
                if t == BF16:
                    bf16_f32w.append(name)
                elif t != F32:
                    bad.setdefault(f"{tname(t)} (F32 or BF16 only there)", []).append(name)
                continue
            if arch == "deepseek4" and name.endswith(DSV4_F32) and t != F32:
                bad.setdefault(f"{tname(t)} (F32 only there)", []).append(name)
                continue
            if t == F32 or t in GPU:
                continue
            if t == BF16 and (arch in BF16_DENSE or (arch == "deepseek4" and name.endswith("ffn_gate_inp.weight"))):
                continue
            bad.setdefault(tname(t), []).append(name)
    for ty, names in bad.items():
        takes = "F32, BF16" if arch in BF16_DENSE else "F32"
        rep.errors.append(f"{len(names)} weight(s) are {ty} ({_few(names)}): {rep.model}'s loader takes {takes} or a "
                          "type with a GPU kernel there")
    if bf16_f32w and env.get("STRATA_DSV41_BF16_W") != "1":
        (rep.errors if server else rep.notes).append(
            f"{len(bf16_f32w)} compressor / indexer weight(s) are BF16 ({_few(bf16_f32w)}): the engine takes them only "
            "with STRATA_DSV41_BF16_W=1 in the run config's env (moex/configs: the GGUF-set configs)")
    if not per_layer and not ("--pack" in args or "--experts-gguf" in args):
        (rep.errors if server else rep.notes).append(
            "no routed experts in these files (blk.N.ffn_*_exps.weight): they hold the dense part only, and the run "
            "config must name a --pack or an --experts-gguf set for the experts")
    mismatched = [l for l, g in sorted(per_layer.items()) if "gate" in g and "up" in g and g["gate"] != g["up"]]
    if mismatched:
        rep.errors.append(f"gate and up experts of different types in {len(mismatched)} layer(s) (blk.{mismatched[0]}, ...): "
                          "the engine needs one type for both (src/x/tiers.cpp)")
    types = {t for g in per_layer.values() for t in g.values()}
    for t in sorted(types):
        if t == F32:
            rep.errors.append("F32 experts: they do not fit the CPU pool's buffers")
        elif t in (F16, BF16) and arch == "deepseek41":
            rep.errors.append(f"{tname(t)} experts: DeepSeek-V4.1's do not fit the CPU pool's buffers")
        elif t not in CPU:
            rep.errors.append(f"{tname(t)} experts: no kernel computes them")
    if not rep.errors:
        cpu_only = sorted(t for t in types if t not in GPU)
        no_mmq = sorted(t for t in types if t not in MMQ)
        if cpu_only:
            rep.notes.append("experts of " + ", ".join(map(tname, cpu_only)) + " have no GPU kernel: they stay in RAM and the "
                             "CPU computes them (slow decode)")
        if no_mmq:
            rep.notes.append("experts of " + ", ".join(map(tname, no_mmq)) + " have no batched prompt kernel: the whole "
                             "model reads prompts in 8-token windows (~25 tok/s against hundreds)")
        if 7 in types:
            rep.notes.append("Q5_1 experts: decode has no grouped Q5_1 kernel, so their VRAM slots are not used")
    return rep


# ------------------------------------------------------------------------------------------------ reading headers
def _header(stream, name: str) -> tuple[dict, list]:
    g = GGUFFile.__new__(GGUFFile)            # tools/gguf_reader.py's parser on any stream (a file, or HTTP ranges)
    g.path, g.metadata, g.tensors, g.version, g.alignment = Path(name), {}, [], 0, 32
    g._parse(stream)
    return g.metadata, [(t.name, list(t.shape), t.type_id) for t in g.tensors]


def set_files(first: Path) -> tuple[list[Path], list[str]]:
    """A split set's shards from any one of them (llama.cpp's -0000N-of-0000M.gguf names), and the missing ones."""
    m = SPLIT.match(first.name)
    if not m:
        return [first], []
    n = int(m.group(3))
    paths = [first.with_name(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]
    return [p for p in paths if p.is_file()], [p.name for p in paths if not p.is_file()]


def check_local(first: str | Path, env: dict | None = None, args: list[str] | None = None) -> Report:
    files, missing = set_files(Path(first))
    headers = []
    for p in files:
        with open(p, "rb") as fh:
            headers.append(_header(fh, p.name))
    rep = check(headers, env, args)
    if missing:
        rep.errors.append(f"{len(missing)} file(s) of the set are not there ({_few(missing)}): -m opens every shard")
    return rep


class _Remote(io.RawIOBase):
    """A seekable reader over an HTTP file that fetches windows on demand - 256 KB first, doubling to 8 MB (a split set's
    later shards have headers of tens of KB; the first carries the vocabulary) - a GGUF's header, never its data."""

    def __init__(self, url):
        self.url, self.pos, self.buf, self.base, self.window = url, 0, b"", 0, 256 << 10

    def _fill(self, need):
        if self.base <= self.pos and self.pos + need <= self.base + len(self.buf):
            return
        start, end = self.pos, self.pos + max(need, self.window) - 1
        self.window = min(self.window * 2, 8 << 20)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "strata-x-quant-check"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.buf, self.base = r.read(), start

    def read(self, n=-1):
        self._fill(n)
        o = self.pos - self.base
        out = self.buf[o:o + n]
        self.pos += len(out)
        return out

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off
        return self.pos

    def tell(self):
        return self.pos


def check_hf(repo: str, path: str = "") -> list[tuple[str, Report]]:
    """A Hugging Face repo's GGUF set(s) under `path` (a folder, a file, or nothing: the whole repo's top level)."""
    folder = path if path and not path.endswith(".gguf") else path.rsplit("/", 1)[0] if "/" in path else ""
    api = f"https://huggingface.co/api/models/{repo}/tree/main/{urllib.parse.quote(folder)}"
    req = urllib.request.Request(api, headers={"User-Agent": "strata-x-quant-check"})
    files = sorted(x["path"] for x in json.load(urllib.request.urlopen(req, timeout=60)) if x["path"].endswith(".gguf"))
    if path.endswith(".gguf"):
        m = SPLIT.match(path.rsplit("/", 1)[-1])
        files = [f for f in files if f == path or (m and SPLIT.match(f.rsplit("/", 1)[-1])
                                                   and SPLIT.match(f.rsplit("/", 1)[-1]).group(1) == m.group(1))]
    sets: dict[str, list[str]] = {}
    for f in files:
        m = SPLIT.match(f.rsplit("/", 1)[-1])
        sets.setdefault(m.group(1) if m else f, []).append(f)
    out = []
    for key, members in sorted(sets.items()):
        headers = [_header(_Remote(f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(f)}"), f)
                   for f in members]
        out.append((key, check(headers)))
    return out


def model_arg(cfg: dict) -> str | None:
    """A run config's -m file, resolved against its cwd."""
    args = cfg.get("args") or []
    if "-m" not in args or args.index("-m") + 1 >= len(args):
        return None
    p = args[args.index("-m") + 1]
    return p if os.path.isabs(p) else os.path.join(cfg.get("cwd") or ".", p)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.split("\n\n")[1])
        return 0
    if argv[0] == "--hf":
        if len(argv) < 2:
            print("--hf REPO [FOLDER_OR_FILE]")
            return 2
        try:
            reports = check_hf(argv[1], argv[2] if len(argv) > 2 else "")
        except (OSError, ValueError) as e:          # urllib's errors are OSErrors: no network, no such repo or file
            print(f"could not read {argv[1]} from Hugging Face: {e}")
            return 2
        if not reports:
            print("no .gguf files there")
            return 1
        for key, rep in reports:
            print(f"{argv[1]}: {key}")
            print("\n".join(rep.lines()))
        return 1 if any(rep.errors for _, rep in reports) else 0
    rep = check_local(argv[0])
    print("\n".join(rep.lines()))
    return 1 if rep.errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
