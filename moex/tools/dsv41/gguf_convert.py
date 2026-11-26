#!/usr/bin/env python3
"""moex/tools/dsv41/gguf_convert.py - DeepSeek-V4.1-Flash-1017 as one lossless GGUF set, by mainline llama.cpp's own
converter (conversion/deepseek41.py), from the copy of the release on this box. Nothing is downloaded.

The release on disk (convert_pack.py's layout, --release):
  native/experts/L00..L39.bin   the routed experts: record e at e * 18,800,640 = [w1 | w3 | w2], each part already
                                ggml's block_mxfp4 (fp4_repack.repack, the same bytes as the converter's
                                repack_mxfp4_blocks: --check-repack), a crc32 per record in native/manifest.json
  native/engram/L01, L14        the Engram tables as released: e4m3 rows of 256 B + one e8m0 per 32 values (8 B a row)
  dense/model-000NN-of-00048.safetensors   every other tensor of shard NN, bytes as released
  hf-small/                     config.json, tokenizer.json, tokenizer_config.json

The set written (--out, <name>-0000k-of-00009.gguf) is the converter's output (names, metadata, tokenizer, the Engram
token map, the FP8 linears as exact BF16, F32 kept F32, the router and Engram gate in F32: --outtype bf16), except:
  - the routed experts are streamed from the records (the converter would repack the same release bytes into the
    same bytes); every record's crc32 is checked against the manifest on the way;
  - the Engram table is MXFP8 instead of the converter's BF16: ggml type 67 (unsloth's id), blocks of [e8m0, 32 x
    e4m3], i.e. the release's codes and scales interleaved - the same values in half the bytes (202.8 GB, not 393).
    Maya and upstream Strata read type 67 as MXFP8. Mainline ggml has no MXFP8 type (its 43 is F8_E4M3, one byte a
    value without scales), so mainline llama.cpp does not load this set.

--shards picks which files of the set a run writes; the plan (which tensors go to which file) is the same every run,
written to plan.json by the first run and checked by every later one. Files not picked are never opened, so files
written by earlier runs stay as they are. A file is written as <file>.part and renamed only when complete.
--verify reads written files back with O_DIRECT: every expert slice against its record, the Engram table against
native/engram (codes and scales, byte for byte), and a few dense tensors against dense/ by value.

Reads and writes avoid the page cache (O_DIRECT reads, written pages dropped after each fdatasync), so the running
server's cached pages are not pushed out. While the server's /metrics says busy, reads go at --busy-mbps (50) in 4 MiB
pieces; idle, at full speed. --max-mbps caps the read rate at all times.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import shutil
import sys
import threading
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HOME = Path.home()
DEF_LLAMA = HOME / "AI/llama.cpp-mainline"
DEF_RELEASE = HOME / "AI/Models/dsv41-1017"
DEF_OUT = DEF_RELEASE / "gguf-mxfp4"
MODEL_DIR_NAME = "DeepSeek-V4.1-Flash-1017"
MXFP8_TYPE = 67            # unsloth's id for MXFP8 (Maya and upstream Strata: MXFP8, 32 values in 33 bytes)
MXFP8_BLOCK = (32, 33)
PAGE = 4096

LOG_FH = None


def log(msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(line + "\n")
        LOG_FH.flush()


# ---------------------------------------------------------------------------------------------------------------
# crc32 of a concatenation from the parts' crc32s (zlib's crc32_combine; Python's zlib does not export it)

def _gf2_times(mat: list[int], vec: int) -> int:
    s, i = 0, 0
    while vec:
        if vec & 1:
            s ^= mat[i]
        vec >>= 1
        i += 1
    return s


def _gf2_square(mat: list[int]) -> list[int]:
    return [_gf2_times(mat, mat[n]) for n in range(32)]


def crc32_shift_op(length: int) -> list[int]:
    """The operator M with crc32(A + B) = M(crc32(A)) ^ crc32(B) for every B of `length` bytes."""
    odd = [0xEDB88320] + [1 << n for n in range(31)]   # one zero bit
    even = _gf2_square(odd)                            # two zero bits
    odd = _gf2_square(even)                            # four zero bits
    op = [1 << n for n in range(32)]                   # identity
    n = length
    while True:
        even = _gf2_square(odd)                        # first pass: one zero byte
        if n & 1:
            op = [_gf2_times(even, c) for c in op]
        n >>= 1
        if not n:
            break
        odd = _gf2_square(even)
        if n & 1:
            op = [_gf2_times(odd, c) for c in op]
        n >>= 1
        if not n:
            break
    return op


def crc32_combine(op: list[int], crc_a: int, crc_b: int) -> int:
    return _gf2_times(op, crc_a) ^ crc_b


# ---------------------------------------------------------------------------------------------------------------
# O_DIRECT reads with read-ahead, and a rate cap

class Rate:
    def __init__(self, mbps: float):
        self.bps = mbps * 1e6 if mbps > 0 else 0.0
        self.t0 = time.monotonic()
        self.n = 0
        self.lock = threading.Lock()

    def take(self, nbytes: int) -> None:
        if not self.bps:
            return
        with self.lock:
            self.n += nbytes
            ahead = self.n / self.bps - (time.monotonic() - self.t0)
        if ahead > 0:
            time.sleep(ahead)


RATE = Rate(0)


class Yield:
    """Reads step aside while the server is busy: its /metrics is polled every 2 s (live.state "reading" or "generating",
    or a queue; unreachable counts as idle). Idle: full speed. Busy: at most `busy_mbps` (0: none at all until idle)."""

    def __init__(self, url: str | None, busy_mbps: float = 0.0):
        self.url = url
        self.busy = False
        self.paused = 0.0
        self.busy_s = 0.0   # seconds the server was seen busy (the poll's samples)
        self.bps = busy_mbps * 1e6
        self.t_free = 0.0
        self.lock = threading.Lock()
        if url:
            threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self) -> None:
        import urllib.request
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=2) as r:
                    live = json.load(r).get("live") or {}
                # (an engine loading - "unloaded" while the server is up - counts as busy: its load reads the drive)
                self.busy = live.get("state") in ("reading", "generating", "unloaded") or int(live.get("queued") or 0) > 0
            except Exception:
                self.busy = False
            self.busy_s += 2 if self.busy else 0
            time.sleep(2)

    def wait(self, nbytes: int) -> None:
        if not self.busy:
            return
        if self.bps <= 0:
            t0 = time.monotonic()
            while self.busy:
                time.sleep(0.5)
            self.paused += time.monotonic() - t0
            return
        with self.lock:
            now = time.monotonic()
            start = max(now, self.t_free)
            self.t_free = start + nbytes / self.bps
        if start > now:
            time.sleep(start - now)
            self.paused += start - now


YIELD = Yield(None)


class Direct:
    """A file opened with O_DIRECT; read(off, n) returns n bytes from off (any alignment) as a numpy view."""

    PIECE = 4 << 20

    def __init__(self, path: Path):
        self.path = Path(path)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
        self.size = os.fstat(self.fd).st_size

    def close(self) -> None:
        os.close(self.fd)

    def read(self, off: int, n: int) -> np.ndarray:
        if off < 0 or off + n > self.size:
            raise ValueError(f"{self.path}: read {off}+{n} past the end ({self.size})")
        a0 = off // PAGE * PAGE
        a1 = -(-(off + n) // PAGE) * PAGE
        buf = mmap.mmap(-1, a1 - a0)
        got, pos = 0, a0
        while pos < min(a1, self.size):   # in pieces of at most 4 MiB (the server's own read size), each one paced
            piece = min(self.PIECE, a1 - pos)
            YIELD.wait(piece)
            RATE.take(piece)
            r = os.preadv(self.fd, [memoryview(buf)[got:got + piece]], pos)
            if r <= 0:
                raise IOError(f"{self.path}: short read at {pos}")
            got += r
            pos += r
            if r < piece:
                break
        if a0 + got < off + n:
            raise IOError(f"{self.path}: read {got} bytes at {a0}, needed {off + n - a0}")
        return np.frombuffer(buf, dtype=np.uint8, count=n, offset=off - a0)


def stream(reader: Direct, spans: list[tuple[int, int]], depth: int = 4):
    """Yield reader.read(off, n) for each span in order, `depth` reads ahead."""
    with ThreadPoolExecutor(depth) as pool:
        q: deque = deque()
        it = iter(spans)
        for off, n in it:
            q.append(pool.submit(reader.read, off, n))
            if len(q) >= depth:
                break
        while q:
            buf = q.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                q.append(pool.submit(reader.read, *nxt))
            yield buf


# ---------------------------------------------------------------------------------------------------------------
# streamed tensors (written chunk by chunk, the written pages dropped from the page cache)

class Streamed:
    SYNC_EVERY = 32 << 20
    LOG_EVERY = 300.0

    def __init__(self, shape: tuple[int, ...], chunks):
        self.shape = shape
        self.dtype = np.dtype(np.uint8)
        self.nbytes = int(np.prod(shape))
        self._chunks = chunks

    def tofile(self, fout) -> None:
        written, since = 0, 0
        fd = fout.fileno()
        t0 = last = time.monotonic()
        b0 = YIELD.busy_s
        for chunk in self._chunks():
            now = time.monotonic()
            if now - last >= self.LOG_EVERY:
                last = now
                log(f"    {written / 1e9:.1f} of {self.nbytes / 1e9:.1f} GB, {now - t0:.0f} s "
                    f"({written / 1e6 / (now - t0):.0f} MB/s), the server busy {100 * (YIELD.busy_s - b0) / (now - t0):.0f}% of it")
            chunk = np.ascontiguousarray(chunk, dtype=np.uint8)
            fout.write(memoryview(chunk).cast("B"))
            written += chunk.nbytes
            since += chunk.nbytes
            if since >= self.SYNC_EVERY:
                fout.flush()
                os.fdatasync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                since = 0
        fout.flush()
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        if written != self.nbytes:
            raise ValueError(f"streamed {written} bytes, expected {self.nbytes}")

    def byteswap(self, inplace: bool = False):
        raise NotImplementedError("streamed tensors are little-endian only")


# ---------------------------------------------------------------------------------------------------------------
# the release on disk

class Release:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.cfg = json.load(open(self.root / "hf-small/config.json"))
        self.tc = {**self.cfg, **self.cfg.get("text_config", {})}
        self.n_main = self.tc["num_hidden_layers"]
        self.n_embd = self.tc["hidden_size"]
        self.n_ff = self.tc["moe_intermediate_size"]
        self.part = self.n_embd * self.n_ff // 32 * 17                       # 6,266,880
        self.stride = 3 * self.part                                         # 18,800,640
        self.manifest = json.load(open(self.root / "native/manifest.json"))
        self.crc: dict[str, list[int]] = {}
        for sh in self.manifest["shards"].values():
            ex = sh.get("experts")
            if ex:
                self.crc[Path(ex["path"]).name] = ex["crc32"]

    def record_file(self, bid: int) -> Path:
        if bid >= self.n_main:
            return self.root / f"native/experts/M{bid - self.n_main}.bin"
        return self.root / f"native/experts/L{bid:02d}.bin"

    def proj_shape(self, proj: str) -> tuple[int, int]:
        """(rows, cols) of one expert's matrix as released ([out, in])."""
        return (self.n_ff, self.n_embd) if proj in ("w1", "w3") else (self.n_embd, self.n_ff)

    def proj_offset(self, proj: str) -> int:
        return {"w1": 0, "w3": self.part, "w2": 2 * self.part}[proj]

    def engram_files(self, bid: int) -> tuple[Path, Path]:
        return (self.root / f"native/engram/L{bid:02d}.weight.bin", self.root / f"native/engram/L{bid:02d}.scale.bin")


REL: Release | None = None


class RecordCheck:
    """Each record's crc32 from its three parts' (streamed in different tensors) against the manifest."""

    def __init__(self, rel: Release):
        self.rel = rel
        self.op = crc32_shift_op(rel.part)
        self.parts: dict[tuple[int, str], list[int]] = {}
        self.checked = 0

    def add(self, bid: int, proj: str, crcs: list[int]) -> None:
        self.parts[(bid, proj)] = crcs
        if all((bid, p) in self.parts for p in ("w1", "w3", "w2")):
            f = self.rel.record_file(bid)
            want = self.rel.crc.get(f.name)
            if want is None:
                raise SystemExit(f"{f.name}: no record crc32s in the manifest")
            c1, c3, c2 = (self.parts.pop((bid, p)) for p in ("w1", "w3", "w2"))
            for e, w in enumerate(want):
                got = crc32_combine(self.op, crc32_combine(self.op, c1[e], c3[e]), c2[e])
                if got != w:
                    raise SystemExit(f"{f.name} record {e}: crc32 {got:08x}, manifest {w:08x} - the record changed")
            self.checked += len(want)
            log(f"  {f.name}: {len(want)} records match the manifest's crc32")


CHECK: RecordCheck | None = None


# ---------------------------------------------------------------------------------------------------------------
# the two converter methods replaced

def write_experts_from_records(self, bid: int, proj: str, key, n_expert: int | None = None) -> None:
    import gguf
    n_expert = self.hparams["n_routed_experts"] if n_expert is None else n_expert
    rows, cols = REL.proj_shape(proj)
    part, off, stride = REL.part, REL.proj_offset(proj), REL.stride
    f = REL.record_file(bid)
    # (a run that leaves the experts' files out plans them from the config alone: the records may be gone by then)
    if f.exists() and f.stat().st_size != n_expert * stride:
        raise SystemExit(f"{f}: {f.stat().st_size} bytes, expected {n_expert} records of {stride}")
    name = self.format_tensor_name(key, bid)

    def chunks():
        log(f"  {name}: {n_expert} x {part:,} B from {f.name} at +{off:,}")
        rd = Direct(f)
        crcs = []
        try:
            for buf in stream(rd, [(e * stride + off, part) for e in range(n_expert)]):
                crcs.append(zlib.crc32(buf))
                yield buf
        finally:
            rd.close()
        if CHECK is not None:
            CHECK.add(bid, proj, crcs)

    data = Streamed((n_expert, rows, cols // 32 * 17), chunks)
    self.gguf_writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)


def write_engram_mxfp8(self, layer_ids: list[int]) -> None:
    import gguf
    dim = self.hparams["engram_head_dim"]
    if dim % 32:
        raise SystemExit(f"engram_head_dim {dim} is not a multiple of 32")
    nb = dim // 32
    parts = []
    for li, bid in enumerate(layer_ids):
        wf, sf = REL.engram_files(bid)
        rows = self.hparams["engram_num_embeddings"][li]
        # (the tables' sizes checked where they are: a run that leaves the Engram's file out plans it from the config alone -
        # native/engram goes once that file is written and verified; writing it again then fails at its first read)
        if (wf.exists() or sf.exists()) and (wf.stat().st_size != rows * dim or sf.stat().st_size != rows * nb):
            raise SystemExit(f"{wf.name}/{sf.name}: sizes {wf.stat().st_size}/{sf.stat().st_size}, expected "
                             f"{rows * dim}/{rows * nb} for {rows} rows of {dim}")
        parts.append((bid, wf, sf, rows))
    name = self.format_tensor_name(gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD)
    n_rows = sum(p[3] for p in parts)
    R = 1 << 18

    def chunks():
        for bid, wf, sf, rows in parts:
            log(f"  {name}: layer {bid}, {rows:,} rows from {wf.name} + {sf.name}")
            w, s = Direct(wf), Direct(sf)
            try:
                spans = [(r0, min(R, rows - r0)) for r0 in range(0, rows, R)]
                ws = stream(w, [(r0 * dim, n * dim) for r0, n in spans], depth=3)
                ss = stream(s, [(r0 * nb, n * nb) for r0, n in spans], depth=3)
                for (r0, n), q, e in zip(spans, ws, ss):
                    if int(e.max()) == 255:
                        raise SystemExit(f"{sf.name}: an E8M0 NaN scale (255) near row {r0}")
                    out = np.empty((n, nb, 33), dtype=np.uint8)
                    out[:, :, 0] = e.reshape(n, nb)
                    out[:, :, 1:] = q.reshape(n, nb, 32)
                    yield out
            finally:
                w.close()
                s.close()

    data = Streamed((n_rows, nb * MXFP8_BLOCK[1]), chunks)
    self.gguf_writer.add_tensor(name, data, raw_dtype=MXFP8_TYPE)


# ---------------------------------------------------------------------------------------------------------------
# the writer: only the picked files, as .part files renamed when complete; the plan checked against plan.json

class NullSink:
    def __init__(self):
        self.pos = 0

    def write(self, b) -> int:
        n = len(memoryview(b).cast("B"))
        self.pos += n
        return n

    def tell(self) -> int:
        return self.pos

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


ARGS = None
PICK: set[int] = set()
DONE: set[int] = set()
PATHS: list[Path] = []


def plan_of(writer, filenames: list[Path]) -> list[dict]:
    out = []
    for fn, tensors in zip(filenames, writer.tensors):
        out.append({"file": fn.name, "bytes": sum(ti.nbytes for ti in tensors.values()),
                    "tensors": [[n, int(ti.dtype), [int(x) for x in ti.shape], int(ti.nbytes)] for n, ti in tensors.items()]})
    return out


def patch_writer() -> None:
    import gguf
    from gguf.gguf_writer import WriterState

    def open_output_file(self, path: Path | None = None) -> None:
        if self.state is WriterState.EMPTY and self.fout is not None and (path is None or path == self.path):
            return
        if self.state is not WriterState.NO_FILE:
            raise ValueError(f"Expected output file to be not yet opened, got {self.state}")
        if path is not None:
            self.path = path
        filenames = self.format_shard_names(self.path)
        plan = plan_of(self, filenames)
        pf = Path(ARGS.out) / "plan.json"
        log("plan: " + ", ".join(f"{i + 1}: {len(p['tensors'])} tensors {p['bytes'] / 1e9:.2f} GB" for i, p in enumerate(plan)))
        if pf.exists():
            old = json.load(open(pf))
            if old != plan:
                raise SystemExit(f"{pf}: the plan changed since the first run - refusing to mix files of two plans")
        else:
            tmp = pf.with_suffix(".json.part")
            json.dump(plan, open(tmp, "w"), indent=1)
            os.replace(tmp, pf)
            log(f"wrote {pf}")
        if ARGS.plan:
            for i, p in enumerate(plan):
                types = {}
                for _, t, _, nb in p["tensors"]:
                    types[t] = types.get(t, 0) + nb
                log(f"  {p['file']}: {len(p['tensors'])} tensors, {p['bytes']:,} B, by ggml type: " +
                    ", ".join(f"{t}: {b / 1e9:.3f} GB" for t, b in sorted(types.items())))
            raise SystemExit(0)
        bad = [i + 1 for i in PICK if i >= len(filenames)]
        if bad:
            raise SystemExit(f"--shards {bad}: the set has {len(filenames)} files")
        need = sum(plan[i]["bytes"] for i in PICK)
        free = shutil.disk_usage(Path(ARGS.out)).free
        if free < need + ARGS.reserve_gb * 1e9:
            raise SystemExit(f"{need / 1e9:.1f} GB to write, {free / 1e9:.1f} GB free (keeping {ARGS.reserve_gb} GB)")
        for i in sorted(PICK):
            if filenames[i].exists() and not ARGS.force:
                raise SystemExit(f"{filenames[i]} exists (--force to write it again)")
        PATHS[:] = filenames
        self.fout = [open(str(fn) + ".part", "wb") if i in PICK else NullSink() for i, fn in enumerate(filenames)]
        self.state = WriterState.EMPTY
        log(f"writing files {sorted(i + 1 for i in PICK)} of {len(filenames)}: {need / 1e9:.2f} GB, {free / 1e9:.1f} GB free")

    def write_tensors_to_file(self, *, progress: bool = False) -> None:
        self.write_ti_data_to_file()
        for fout in self.fout:
            self.write_padding(fout, fout.tell())
        for i, (fout, tensors) in enumerate(zip(self.fout, self.tensors)):
            if i not in PICK:
                for ti in tensors.values():
                    ti.tensor = None
                continue
            t0 = time.monotonic()
            total = sum(ti.nbytes for ti in tensors.values())
            log(f"file {i + 1}: {len(tensors)} tensors, {total / 1e9:.2f} GB")
            for name, ti in tensors.items():
                assert ti.tensor is not None and ti.tensor.nbytes == ti.nbytes
                start = fout.tell()
                ti.tensor.tofile(fout)
                fout.flush()
                if fout.tell() - start != ti.nbytes:
                    raise ValueError(f"tensor {name!r} wrote {fout.tell() - start} bytes, expected {ti.nbytes}")
                self.write_padding(fout, ti.nbytes)
                ti.tensor = None
            fout.flush()
            os.fsync(fout.fileno())
            os.posix_fadvise(fout.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            dt = time.monotonic() - t0
            log(f"file {i + 1} written: {total / 1e9:.2f} GB in {dt:.0f} s ({total / 1e6 / max(dt, 1e-9):.0f} MB/s)")
            DONE.add(i)
        self.state = WriterState.WEIGHTS

    orig_close = gguf.GGUFWriter.close

    def close(self) -> None:
        orig_close(self)
        for i in sorted(DONE):
            part = Path(str(PATHS[i]) + ".part")
            os.replace(part, PATHS[i])
            log(f"complete: {PATHS[i].name} ({PATHS[i].stat().st_size:,} B)")
        if DONE:
            dfd = os.open(PATHS[0].parent, os.O_RDONLY)
            os.fsync(dfd)
            os.close(dfd)

    gguf.GGUFWriter.open_output_file = open_output_file
    gguf.GGUFWriter.write_tensors_to_file = write_tensors_to_file
    gguf.GGUFWriter.close = close


# ---------------------------------------------------------------------------------------------------------------

def stage_model_dir(rel: Release, out: Path) -> Path:
    """config + tokenizer + the dense safetensors, linked into a folder the converter reads as the model."""
    src = out / ".src" / MODEL_DIR_NAME
    src.mkdir(parents=True, exist_ok=True)
    want = {}
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        want[f] = rel.root / "hf-small" / f
    for f in sorted((rel.root / "dense").glob("model-*.safetensors")):
        want[f.name] = f
    for name in os.listdir(src):
        if name not in want:
            os.unlink(src / name)
    for name, target in want.items():
        p = src / name
        if p.is_symlink() and os.readlink(p) == str(target):
            continue
        if p.exists() or p.is_symlink():
            p.unlink()
        p.symlink_to(target)
    return src


def parse_shards(s: str) -> set[int]:
    out: set[int] = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a) - 1, int(b)))
        elif part:
            out.add(int(part) - 1)
    return out


def check_repack(rel: Release, llama: Path) -> None:
    """The records hold what the converter would write from the release's expert tensors."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    import fp4_repack
    from conversion.base import ModelBase

    def ours_vs_theirs(w: np.ndarray, s: np.ndarray) -> bool:
        a = fp4_repack.repack(w, s)
        b = ModelBase.repack_mxfp4_blocks(torch.from_numpy(w.view(np.int8).copy()), torch.from_numpy(s.copy()))
        return a.shape == b.shape and np.array_equal(a, b)

    rng = np.random.default_rng(1017)
    for rows, k in ((7, 64), (64, 5120), (5120, 2304)):
        w = rng.integers(0, 256, size=(rows, k // 2), dtype=np.uint8)
        s = rng.integers(0, 255, size=(rows, k // 32), dtype=np.uint8)
        if not ours_vs_theirs(w, s):
            raise SystemExit(f"repack differs from the converter's on random [{rows}, {k}]")
    log("check: fp4_repack.repack == the converter's repack_mxfp4_blocks on random data (3 shapes)")

    # real records: back to the release's layout (fp4_repack.unrepack), through the converter's repack, the same bytes
    for bid, e in ((0, 0), (17, 191), (39, 383)):
        f = rel.record_file(bid)
        rd = Direct(f)
        rec = rd.read(e * rel.stride, rel.stride).copy()
        rd.close()
        if zlib.crc32(rec) != rel.crc[f.name][e]:
            raise SystemExit(f"{f.name} record {e}: crc32 differs from the manifest")
        for proj in ("w1", "w3", "w2"):
            rows, cols = rel.proj_shape(proj)
            o = rel.proj_offset(proj)
            blocks = rec[o:o + rel.part].reshape(rows, cols // 32 * 17)
            w, s = fp4_repack.unrepack(blocks, rows, cols)
            b = ModelBase.repack_mxfp4_blocks(torch.from_numpy(w.view(np.int8).copy()), torch.from_numpy(s.copy()))
            if not np.array_equal(b, blocks):
                raise SystemExit(f"{f.name} record {e} {proj}: the converter's repack gives other bytes")
    log("check: 3 real records (layers 0/17/39): the converter's repack of their release form gives their bytes")

    # the crc32 combination used for the manifest check
    for la, lb in ((0, 5), (13, 1), (1000, 4096), (rel.part, rel.part)):
        a, b = rng.integers(0, 256, la, dtype=np.uint8).tobytes(), rng.integers(0, 256, lb, dtype=np.uint8).tobytes()
        if crc32_combine(crc32_shift_op(lb), zlib.crc32(a), zlib.crc32(b)) != zlib.crc32(a + b):
            raise SystemExit(f"crc32_combine wrong for lengths {la}, {lb}")
    log("check: crc32_combine == zlib.crc32 of the concatenation (4 length pairs)")


# ---------------------------------------------------------------------------------------------------------------
# --verify: read written files back

def gguf_header(path: Path) -> tuple[list[tuple[str, int, int]], int, int]:
    """A GGUF file's tensors (name, ggml type, offset in the data section), its data start and its alignment - read
    here because gguf-py's reader refuses a type it does not know (67, MXFP8)."""
    import struct
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize("<" + fmt)
            return struct.unpack("<" + fmt, f.read(n))
        def string():
            (n,) = rd("Q")
            return f.read(n).decode("utf-8", "replace")
        def skip(t):
            if t in sizes:
                f.seek(sizes[t], 1)
            elif t == 8:
                (n,) = rd("Q")
                f.seek(n, 1)
            elif t == 9:
                et, cnt = rd("IQ")
                if et in sizes:
                    f.seek(sizes[et] * cnt, 1)
                else:
                    for _ in range(cnt):
                        skip(et)
            else:
                raise ValueError(f"{path}: metadata value type {t}")
        magic, version, n_t, n_kv = rd("IIQQ")
        if magic != 0x46554747 or version != 3:
            raise ValueError(f"{path}: not a GGUF v3 file")
        align = 32
        for _ in range(n_kv):
            key = string()
            (t,) = rd("I")
            if key == "general.alignment" and t == 4:
                (align,) = rd("I")
            else:
                skip(t)
        tensors = []
        for _ in range(n_t):
            name = string()
            (nd,) = rd("I")
            rd("%dQ" % nd)
            ty, off = rd("IQ")
            tensors.append((name, ty, off))
        data = -(-f.tell() // align) * align
    return tensors, data, align


def verify(rel: Release, out: Path, name: str, picks: set[int]) -> None:
    plan = json.load(open(out / "plan.json"))
    ok = True
    for i in sorted(picks):
        fn = out / plan[i]["file"]
        tens, data, align = gguf_header(fn)
        want = plan[i]["tensors"]
        if [[t[0], t[1]] for t in tens] != [[w[0], w[1]] for w in want]:
            log(f"VERIFY {fn.name}: tensors differ from the plan")
            ok = False
            continue
        offs = {t[0]: data + t[2] for t in tens}
        pos = data   # the tensors one after another (each padded to the alignment), as the plan's sizes say
        for (tname, _, _, tb), t in zip(want, tens):
            if data + t[2] != pos:
                log(f"VERIFY {fn.name}: {tname} at {t[2]}, expected {pos - data}")
                ok = False
            pos += -(-tb // align) * align
        if pos > fn.stat().st_size or not ok:
            log(f"VERIFY {fn.name}: the header's offsets do not match the plan's sizes ({pos} vs {fn.stat().st_size} B)")
            ok = False
            continue
        g = Direct(fn)
        t0 = time.monotonic()
        nbytes = 0
        # the file's sha256 on the way (the tensors are read in file order): header, then every chunk compared
        import hashlib
        sha = hashlib.sha256()
        first = min(offs.values()) if offs else g.size
        with open(fn, "rb") as fh:
            sha.update(fh.read(first))
        hashed = first

        def hashing(chunks):
            nonlocal hashed
            for c in chunks:
                sha.update(c)
                hashed += c.nbytes
                yield c
        for tname, ttype, shape, tbytes in want:
            m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", tname)
            if m:
                bid, proj = int(m.group(1)), {"gate": "w1", "up": "w3", "down": "w2"}[m.group(2)]
                f = rel.record_file(bid)
                r = Direct(f)
                n_exp = tbytes // rel.part
                a = hashing(stream(g, [(offs[tname] + e * rel.part, rel.part) for e in range(n_exp)]))
                b = stream(r, [(e * rel.stride + rel.proj_offset(proj), rel.part) for e in range(n_exp)])
                bad = [e for e, (x, y) in enumerate(zip(a, b)) if not np.array_equal(x, y)]
                r.close()
                nbytes += tbytes
                if bad:
                    log(f"VERIFY {tname}: experts {bad[:8]} differ from {f.name}")
                    ok = False
                continue
            if tname == "per_layer_token_embd.weight":
                dim = rel.tc["engram_head_dim"]
                nb = dim // 32
                row0 = 0
                for li, bid in enumerate(rel.tc["engram_layer_ids"]):
                    rows = rel.tc["engram_num_embeddings"][li]
                    wf, sf = rel.engram_files(bid)
                    w, s = Direct(wf), Direct(sf)
                    R = 1 << 18
                    spans = [(r0, min(R, rows - r0)) for r0 in range(0, rows, R)]
                    gs = hashing(stream(g, [(offs[tname] + (row0 + r0) * nb * 33, n * nb * 33) for r0, n in spans], depth=3))
                    ws = stream(w, [(r0 * dim, n * dim) for r0, n in spans], depth=3)
                    ss = stream(s, [(r0 * nb, n * nb) for r0, n in spans], depth=3)
                    for (r0, n), x, q, e in zip(spans, gs, ws, ss):
                        x = x.reshape(n, nb, 33)
                        if not (np.array_equal(x[:, :, 0], e.reshape(n, nb)) and np.array_equal(x[:, :, 1:], q.reshape(n, nb, 32))):
                            log(f"VERIFY engram layer {bid}: rows {r0}..{r0 + n} differ from {wf.name}/{sf.name}")
                            ok = False
                            break
                    w.close()
                    s.close()
                    row0 += rows
                nbytes += tbytes
                log(f"  engram: {row0:,} rows ({tbytes / 1e9:.1f} GB) compared with native/engram")
                continue
        g.close()
        dt = time.monotonic() - t0
        if ok and hashed < fn.stat().st_size and fn.stat().st_size - hashed < 4096:   # the last tensor's padding
            with open(fn, "rb") as fh:
                fh.seek(hashed)
                tail = fh.read()
            sha.update(tail)
            hashed += len(tail)
        if hashed == fn.stat().st_size:
            digest = sha.hexdigest()
            log(f"  {fn.name}: sha256 {digest}")
            sums = out / "SHA256SUMS"
            lines = [l for l in (open(sums).read().splitlines() if sums.exists() else []) if not l.endswith("  " + fn.name)]
            lines.append(f"{digest}  {fn.name}")
            open(sums, "w").write("\n".join(sorted(lines, key=lambda l: l.split("  ")[1])) + "\n")
        if nbytes:
            log(f"VERIFY {fn.name}: {nbytes / 1e9:.1f} GB compared in {dt:.0f} s" + ("" if ok else " - DIFFERENCES"))
        else:
            log(f"VERIFY {fn.name}: {len(want)} tensors, the header matches the plan (no data compared)")
    if not ok:
        raise SystemExit("verify: differences found")
    log("verify: all compared bytes are identical")


# ---------------------------------------------------------------------------------------------------------------

def main() -> None:
    global ARGS, REL, CHECK, RATE, LOG_FH, YIELD
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama", type=Path, default=DEF_LLAMA, help="mainline llama.cpp checkout (default %(default)s)")
    ap.add_argument("--release", type=Path, default=DEF_RELEASE, help="the release on disk (default %(default)s)")
    ap.add_argument("--out", type=Path, default=DEF_OUT, help="output folder (default %(default)s)")
    ap.add_argument("--name", default="DeepSeek-V4.1-Flash-1017-MXFP4", help="file name stem")
    ap.add_argument("--split-max-size", default="51G", help="largest file of the set (default %(default)s: 7 layers)")
    ap.add_argument("--shards", default="", help="files of the set to write, 1-based (e.g. 1,8 or 2-7,9)")
    ap.add_argument("--plan", action="store_true", help="print the set's plan and write plan.json, nothing else")
    ap.add_argument("--check-repack", action="store_true", help="check the records against the converter's repack")
    ap.add_argument("--verify", default="", help="read these written files back and compare (1-based)")
    ap.add_argument("--force", action="store_true", help="write picked files even if they exist")
    ap.add_argument("--chat-template", type=Path, default=None,
                    help="a jinja chat template for the set's metadata (tokenizer.chat_template: mainline's converter writes "
                         "none for V4.1; e.g. upstream Strata's serve/fixtures/deepseek41/chat_template.jinja, unsloth's port)")
    ap.add_argument("--max-mbps", type=float, default=0, help="cap the read rate (MB/s, 0: none)")
    ap.add_argument("--reserve-gb", type=float, default=20, help="free space to keep (default %(default)s GB)")
    ap.add_argument("--yield-to", default="http://127.0.0.1:1234/metrics",
                    help="the server whose /metrics says busy or idle ('': none; default %(default)s)")
    ap.add_argument("--busy-mbps", type=float, default=50,
                    help="read rate while that server is busy (MB/s, default %(default)s; 0: wait for idle)")
    args = ARGS = ap.parse_args()

    sys.path.insert(0, str(args.llama / "gguf-py"))
    sys.path.insert(0, str(args.llama))
    args.out.mkdir(parents=True, exist_ok=True)
    LOG_FH = open(args.out / "convert.log", "a")
    RATE = Rate(args.max_mbps)
    YIELD = Yield(args.yield_to or None, args.busy_mbps)
    REL = Release(args.release)
    log(f"=== gguf_convert {' '.join(sys.argv[1:])} (llama.cpp {args.llama})")

    if args.check_repack:
        check_repack(REL, args.llama)
        return
    if args.verify:
        verify(REL, args.out, args.name, parse_shards(args.verify))
        return
    PICK.update(parse_shards(args.shards))
    if not PICK and not args.plan:
        raise SystemExit("nothing to do: --shards, --plan, --check-repack or --verify")

    import gguf
    gguf.GGML_QUANT_SIZES[MXFP8_TYPE] = MXFP8_BLOCK          # (gguf.quants reads the same dict)
    patch_writer()
    import convert_hf_to_gguf
    from conversion import deepseek41
    deepseek41.DeepseekV41Model._write_mxfp4_experts = write_experts_from_records
    deepseek41.DeepseekV41Model._write_engram_tables = write_engram_mxfp8
    if args.chat_template:   # the set's chat template (V4.1's converter removes the V4 one and writes none)
        template = args.chat_template.read_text(encoding="utf-8")
        orig_set_vocab = deepseek41.DeepseekV41Model.set_vocab

        def set_vocab_with_template(self):
            orig_set_vocab(self)
            self.gguf_writer.add_chat_template(template)
            log(f"chat template: {args.chat_template} ({len(template)} chars)")
        deepseek41.DeepseekV41Model.set_vocab = set_vocab_with_template
    CHECK = RecordCheck(REL)

    src = stage_model_dir(REL, args.out)
    sys.argv = ["convert_hf_to_gguf.py", str(src), "--outfile", str(args.out / f"{args.name}.gguf"),
                "--outtype", "bf16", "--split-max-size", args.split_max_size, "--no-tensor-first-split"]
    convert_hf_to_gguf.main()
    if CHECK.checked:
        log(f"records checked against the manifest: {CHECK.checked}")
    log("done")


if __name__ == "__main__":
    main()
