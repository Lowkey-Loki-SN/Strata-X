"""moex/tools/dsv41/gguf_pack.py - a DeepSeek-V4.1 GGUF's routed experts straight from Hugging Face into Strata-X's pack.

    python gguf_pack.py --repo ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF --quant IQ2_XS --out PACK_DIR
        [--probe | --verify N | --sidecar | --rebuild DIR]

The experts are taken as published: each layer's three tensors (blk.N.ffn_{gate,up,down}_exps.weight, every expert's rows
consecutive) are copied byte for byte into the pack's records - per layer one file of 384 records, each a NativeFmt blob
[gate rows | up rows | down rows] zero-padded to a 4 KiB multiple (the engine reads records with O_DIRECT) - and
experts.tsv names each layer's types, stride and file. Nothing is requantized; the dense tensors in the shards are skipped
(the engine's dense GGUF serves them), and a shard that holds no expert tensor (a metadata-only first shard, an Engram
table's) is not downloaded at all. The shards stream as convert_pack.py's do: each is read once over HTTP, hashed as
it goes and checked against its SHA-256 on Hugging Face (a dropped connection resumes with Range inside the run); a shard
whose hash matches is recorded in manifest.json, so a stopped run starts again at the first shard not recorded. The
records are written as *.part files and renamed when all three of their layer's tensors are in.

--probe reads only each shard's header (a range request) and prints the expert tensors it holds, their types and the
layout checks, with the download speed. --verify N reads N experts of each finished record file back from the GGUF (range
requests) and compares them byte for byte, the records' padding too. Neither writes anything. --sidecar keeps every shard
byte the pack does not hold - the headers, the alignment gaps, the dense tensors (~6 GB for IQ2_XS) - in PACK/gguf-sidecar;
--rebuild DIR then writes the shards again from it and the records, each checked against its SHA-256 on Hugging Face.

The GGUF is parsed here (no gguf library): the header's key-value pairs are walked to find general.alignment and the
tensor table, and only the tensor table and the expert tensors' bytes are used.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import struct
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from convert_pack import IntegrityError, Out, Stream, fsync_dir  # noqa: E402

N_EMBD, N_FF, N_EXPERT, N_LAYER = 5120, 2304, 384, 40
# ggml types the engine's expert kernels take (native_fmt, the GPU MMVQ / MMQ, the CPU pool): bytes per 256-value block.
# MXFP4 (39) is 17 bytes per 32 values, ggml's block_mxfp4 - the native pack's own record format (fp4_repack.py), so a
# GGUF's MXFP4 expert tensor is copied as it is (Unsloth's UD quants keep the release's MXFP4 downs in some layers)
BLOCK = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 16: 66, 17: 74, 18: 98, 21: 110, 22: 82, 23: 136, 29: 56, 39: 136}
TYPE_NAME = {10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
             21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 39: "MXFP4"}
EXP_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def row_bytes(ty: int, n: int) -> int:
    if ty not in BLOCK:
        raise IntegrityError(f"expert type {TYPE_NAME.get(ty, ty)} has no kernel in the engine")
    if n % 256:
        raise IntegrityError(f"a row of {n} values is not whole blocks")
    return n // 256 * BLOCK[ty]


class Header:
    """A GGUF header parsed from its first bytes: the key-value pairs walked, the tensor table kept."""

    VAL = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def __init__(self, buf: bytes):
        self.buf, self.off = buf, 0
        if buf[:4] != b"GGUF":
            raise IntegrityError("not a GGUF file")
        self.off = 4
        self.version, = self._rd("<I")
        n_tensors, n_kv = self._rd("<QQ")
        self.kv = {}
        for _ in range(n_kv):
            k = self._str()
            t, = self._rd("<I")
            self.kv[k] = self._val(t)
        self.tensors = []   # (name, dims, type, offset from the data start)
        for _ in range(n_tensors):
            name = self._str()
            nd, = self._rd("<I")
            dims = self._rd("<" + "Q" * nd)
            ty, off = self._rd("<IQ")
            self.tensors.append((name, tuple(dims), ty, off))
        align = int(self.kv.get("general.alignment", 32))
        self.data_start = (self.off + align - 1) // align * align

    def _need(self, n: int):
        if self.off + n > len(self.buf):
            raise EOFError(self.off + n)

    def _rd(self, fmt: str):
        n = struct.calcsize(fmt)
        self._need(n)
        v = struct.unpack_from(fmt, self.buf, self.off)
        self.off += n
        return v

    def _str(self) -> str:
        n, = self._rd("<Q")
        self._need(n)
        s = self.buf[self.off:self.off + n].decode("utf-8", "replace")
        self.off += n
        return s

    def _val(self, t: int):
        if t == 8:
            return self._str()
        if t == 9:
            at, = self._rd("<I")
            n, = self._rd("<Q")
            if at == 8:
                for _ in range(n):
                    self._str()
                return f"<{n} strings>"
            if at not in self.VAL:
                raise IntegrityError(f"an array of type {at}")
            size = struct.calcsize(self.VAL[at])
            self._need(size * n)
            self.off += size * n
            return f"<{n} values>"
        if t not in self.VAL:
            raise IntegrityError(f"a value of type {t}")
        return self._rd(self.VAL[t])[0]


def fetch_header(url: str) -> tuple[Header, float]:
    """The header from range requests, growing until it parses; -> (header, MB/s of the reads)."""
    n, got, t0 = 4 << 20, b"", time.time()
    while True:
        req = urllib.request.Request(url, headers={"Range": f"bytes={len(got)}-{n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            got += r.read()
        try:
            h = Header(got)
            return h, len(got) / 1e6 / max(1e-6, time.time() - t0)
        except EOFError as e:
            n = max(n * 2, int(e.args[0]) + (1 << 20))


def shard_list(repo: str, quant: str) -> list[dict]:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=120) as r:
        api = json.load(r)
    out = []
    for s in api["siblings"]:
        if re.search(rf"-{re.escape(quant)}-\d{{5}}-of-\d{{5}}\.gguf$", s["rfilename"]):
            out.append({"name": s["rfilename"], "size": int(s["size"]), "sha256": (s.get("lfs") or {}).get("sha256")})
    out.sort(key=lambda s: s["name"])
    if not out or any(not s["sha256"] for s in out):
        raise SystemExit(f"{repo}: no {quant} shards with SHA-256 sums")
    return out


def expert_layout(tensors_by_shard: list[list[tuple]]) -> dict[int, dict]:
    """Per layer: the gate/up and down types and the three tensors' places, after the shape and type checks."""
    lay: dict[int, dict] = {}
    for si, tensors in enumerate(tensors_by_shard):
        for name, dims, ty, off in tensors:
            m = EXP_RE.match(name)
            if not m:
                continue
            il, part = int(m.group(1)), m.group(2)
            want = (N_EMBD, N_FF, N_EXPERT) if part in ("gate", "up") else (N_FF, N_EMBD, N_EXPERT)
            if dims != want:
                raise IntegrityError(f"{name}: dims {dims}, expected {want}")
            lay.setdefault(il, {})[part] = (si, ty, off)
    if sorted(lay) != list(range(N_LAYER)) or any(len(v) != 3 for v in lay.values()):
        raise IntegrityError("the shards do not hold every layer's three expert tensors")
    for il, v in lay.items():
        if v["gate"][1] != v["up"][1]:
            raise IntegrityError(f"layer {il}: gate and up of different types (the record format has one)")
        gu_row, d_row = row_bytes(v["gate"][1], N_EMBD), row_bytes(v["down"][1], N_FF)
        blob = 2 * N_FF * gu_row + N_EMBD * d_row
        v["gu_row"], v["d_row"], v["blob"] = gu_row, d_row, blob
        v["stride"] = (blob + 4095) // 4096 * 4096
    return lay


def expert_spans(h: Header) -> list[tuple[int, int, str]]:
    """The file's byte ranges [a, b) that hold expert tensor data, with the tensor's name, in file order."""
    out = []
    for name, dims, ty, off in h.tensors:
        if EXP_RE.match(name):
            a = h.data_start + off
            out.append((a, a + tensor_bytes(dims, ty), name))
    return sorted(out)


def other_spans(h: Header, size: int) -> list[tuple[int, int]]:
    """The rest of the file: the header, the alignment gaps and every non-expert tensor - what the pack does not hold."""
    out, pos = [], 0
    for a, b, _ in expert_spans(h):
        if a > pos:
            out.append((pos, a))
        pos = b
    if pos < size:
        out.append((pos, size))
    return out


def sidecar(out: pathlib.Path, headers: list, shards: list, base: str, log) -> int:
    """Every byte of each shard the pack does not hold, from range requests, into out/gguf-sidecar: <shard>.bin (the
    spans in file order) and <shard>.json (the spans, the size, the SHA-256) - with the pack, the shards again (--rebuild)."""
    d = out / "gguf-sidecar"
    d.mkdir(exist_ok=True)
    for h, s in zip(headers, shards):
        spans = other_spans(h, s["size"])
        meta = d / (s["name"] + ".json")
        if meta.exists() and json.loads(meta.read_text()).get("sha256") == s["sha256"]:
            log(f"{s['name']}: sidecar kept from an earlier run")
            continue
        total = sum(b - a for a, b in spans)
        t0 = time.time()
        with open(d / (s["name"] + ".bin.part"), "wb") as f:
            for a, b in spans:
                st = Stream(base + s["name"], s["size"], log)
                st.pos = a
                rest = b - a
                while rest:
                    chunk = st.read(min(rest, 64 << 20))
                    f.write(chunk)
                    rest -= len(chunk)
                st.close()
            f.flush()
            os.fsync(f.fileno())
        os.replace(d / (s["name"] + ".bin.part"), d / (s["name"] + ".bin"))
        meta.write_text(json.dumps({"size": s["size"], "sha256": s["sha256"], "spans": spans}, indent=1))
        fsync_dir(d)
        log(f"{s['name']}: sidecar {total / 1e6:,.1f} MB in {len(spans)} spans ({time.time() - t0:.0f} s)")
    return 0


def rebuild(out: pathlib.Path, dest: pathlib.Path, headers: list, shards: list, lay: dict, log) -> int:
    """Each shard written again from the sidecar and the pack's records, its SHA-256 checked against Hugging Face's."""
    import hashlib
    d = out / "gguf-sidecar"
    dest.mkdir(parents=True, exist_ok=True)
    bad = 0
    for h, s in zip(headers, shards):
        meta = json.loads((d / (s["name"] + ".json")).read_text())
        pieces = [(a, b, None) for a, b in meta["spans"]] + expert_spans(h)
        pieces.sort()
        sha, t0 = hashlib.sha256(), time.time()
        target = dest / (s["name"] + ".part")
        with open(d / (s["name"] + ".bin"), "rb") as side, open(target, "wb") as f:
            for a, b, name in pieces:
                if name is None:                                 # the sidecar's bytes, in file order
                    rest = b - a
                    while rest:
                        chunk = side.read(min(rest, 64 << 20))
                        if not chunk:
                            raise IntegrityError(f"{s['name']}: the sidecar is short")
                        f.write(chunk)
                        sha.update(chunk)
                        rest -= len(chunk)
                    continue
                m = EXP_RE.match(name)
                il, part = int(m.group(1)), m.group(2)
                v = lay[il]
                rows = N_FF if part in ("gate", "up") else N_EMBD
                per = rows * (v["gu_row"] if part != "down" else v["d_row"])
                at = {"gate": 0, "up": N_FF * v["gu_row"], "down": 2 * N_FF * v["gu_row"]}[part]
                with open(out / "experts" / f"L{il:02d}.bin", "rb") as rec:
                    for e in range(N_EXPERT):
                        rec.seek(e * v["stride"] + at)
                        chunk = rec.read(per)
                        f.write(chunk)
                        sha.update(chunk)
        ok = sha.hexdigest() == meta["sha256"]
        if ok:
            os.replace(target, dest / s["name"])
        bad += not ok
        log(f"{s['name']}: rebuilt in {time.time() - t0:.0f} s, SHA-256 {'matches Hugging Face' if ok else 'DIFFERS (kept as .part)'}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--layers", default="", help="--verify: only these layers (comma-separated)")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="check N experts of each finished record file against the GGUF (range requests); nothing written")
    ap.add_argument("--sidecar", action="store_true",
                    help="keep every shard byte the pack does not hold (headers, gaps, dense tensors) in PACK/gguf-sidecar")
    ap.add_argument("--rebuild", type=pathlib.Path, metavar="DIR",
                    help="write the shards again into DIR from the sidecar and the records, each checked by its SHA-256")
    a = ap.parse_args()
    base = f"https://huggingface.co/{a.repo}/resolve/main/"
    shards = shard_list(a.repo, a.quant)
    log = lambda s: print(time.strftime("%H:%M:%S ") + s, flush=True)
    headers = []
    for s in shards:
        h, mbs = fetch_header(base + s["name"])
        headers.append(h)
        n_exp = sum(1 for t in h.tensors if EXP_RE.match(t[0]))
        log(f"{s['name']}: GGUF v{h.version}, {len(h.tensors)} tensors ({n_exp} expert tensors), data at {h.data_start:,}, "
            f"arch {h.kv.get('general.architecture')}; header read at {mbs:.1f} MB/s")
    lay = expert_layout([h.tensors for h in headers])
    total = sum(v["stride"] for v in lay.values()) * N_EXPERT
    for il in range(N_LAYER):
        v = lay[il]
        log(f"  L{il:02d}: gate/up {TYPE_NAME[v['gate'][1]]}, down {TYPE_NAME[v['down'][1]]}; blob {v['blob']:,} B, stride "
            f"{v['stride']:,} B; shards {sorted({v[p][0] + 1 for p in ('gate', 'up', 'down')})}")
    log(f"the pack: {total / 1e9:.1f} GB of records ({total / 2**30:.1f} GiB) from {sum(s['size'] for s in shards) / 1e9:.1f} GB "
        f"of shards")
    # a shard with no expert tensor (the metadata-only first shard, the dense tail, Unsloth's Engram table) is not streamed:
    # the engine's dense GGUF and the release's FP8 Engram serve what it holds
    has_exp = [any(EXP_RE.match(t[0]) for t in h.tensors) for h in headers]
    log(f"streamed: {sum(s['size'] for s, e in zip(shards, has_exp) if e) / 1e9:.1f} GB in {sum(has_exp)} shards; skipped "
        + (", ".join(f"{s['name']} ({s['size'] / 1e9:.1f} GB)" for s, e in zip(shards, has_exp) if not e) or "none"))
    if a.probe:
        return 0
    if a.verify:
        only = {int(x) for x in a.layers.split(",") if x.strip()}
        return verify(a.out, lay, headers, shards, base, a.verify, log, only)
    if a.sidecar:
        return sidecar(a.out, headers, shards, base, log)
    if a.rebuild:
        return rebuild(a.out, a.rebuild, headers, shards, lay, log)

    out = a.out
    (out / "experts").mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"repo": a.repo, "quant": a.quant, "shards": {},
                                                                         "tensors": []}
    done_t = set(man["tensors"])
    files: dict[int, Out] = {}

    def layer_file(il: int) -> Out:
        if il not in files:
            size = lay[il]["stride"] * N_EXPERT
            path = out / "experts" / f"L{il:02d}.bin"
            part = path.with_name(path.name + ".part")
            if path.exists() and not part.exists():   # finished in a run stopped before its shard was recorded: keep its data
                os.replace(path, part)
            files[il] = Out(path, None, resume_at=size) if part.exists() and part.stat().st_size == size else Out(path, size)
        return files[il]

    for si, s in enumerate(shards):
        if man["shards"].get(s["name"]) == s["sha256"]:
            log(f"{s['name']}: done in an earlier run")
            continue
        if not has_exp[si]:
            log(f"{s['name']}: no expert tensors - not streamed")
            continue
        h = headers[si]
        mine = sorted((t for t in h.tensors), key=lambda t: t[3])
        st = Stream(base + s["name"], s["size"], log)
        t0 = time.time()
        st.read(h.data_start)                                   # the header and its padding, hashed
        pos = 0                                                 # within the data section
        for name, dims, ty, off in mine:
            m = EXP_RE.match(name)
            if off < pos:
                raise IntegrityError(f"{name}: tensors overlap")
            if off > pos:
                st.read(off - pos)                              # the padding before it
                pos = off
            nbytes = (dims[0] // 256 * BLOCK[ty] if m else None)
            if not m:                                           # a dense tensor: skipped (hashed)
                size = tensor_bytes(dims, ty)
                skip = size
                while skip:
                    skip -= len(st.read(min(skip, 64 << 20)))
                pos += size
                continue
            il, part = int(m.group(1)), m.group(2)
            v = lay[il]
            rows = N_FF if part in ("gate", "up") else N_EMBD
            per = rows * nbytes                                 # one expert's rows of this tensor
            at = {"gate": 0, "up": N_FF * v["gu_row"], "down": 2 * N_FF * v["gu_row"]}[part]
            f = layer_file(il)
            batch = max(1, (64 << 20) // per)
            for e0 in range(0, N_EXPERT, batch):
                n = min(batch, N_EXPERT - e0)
                data = st.read(n * per)
                for j in range(n):
                    f.pwrite(data[j * per:(j + 1) * per], (e0 + j) * v["stride"] + at)
            pos += per * N_EXPERT
            done_t.add(name)
            mb = st.pos / 1e6
            log(f"  {name}: {TYPE_NAME[ty]}, {per * N_EXPERT / 1e9:.2f} GB ({mb / max(1e-6, time.time() - t0):.1f} MB/s)")
        rest = s["size"] - st.pos
        while rest:
            rest -= len(st.read(min(rest, 64 << 20)))
        st.close()
        got = st.sha.hexdigest()
        if got != s["sha256"]:
            raise IntegrityError(f"{s['name']}: SHA-256 {got}, Hugging Face has {s['sha256']}")
        for il in list(files):
            if all(f"blk.{il}.ffn_{p}_exps.weight" in done_t for p in ("gate", "up", "down")):
                files.pop(il).finish()
        man["shards"][s["name"]] = s["sha256"]
        man["tensors"] = sorted(done_t)
        tmp = man_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(man, indent=1))
        os.replace(tmp, man_path)
        fsync_dir(out)
        log(f"{s['name']}: SHA-256 matches; {s['size'] / 1e9:.1f} GB in {(time.time() - t0) / 60:.1f} min")
    if files:
        raise IntegrityError(f"layers {sorted(files)} are missing tensors after every shard")
    with open(out / "experts.tsv.tmp", "w") as f:
        f.write(f"# {a.repo} {a.quant}: the routed experts as published (gguf_pack.py)\n")
        for il in range(N_LAYER):
            v = lay[il]
            ty = f"{v['gate'][1]},{v['down'][1]}" if v["gate"][1] != v["down"][1] else f"{v['gate'][1]}"
            f.write(f"{il}\t{ty}\t{v['stride']}\texperts/L{il:02d}.bin\n")
    os.replace(out / "experts.tsv.tmp", out / "experts.tsv")
    fsync_dir(out)
    log(f"done: {out}/experts.tsv and 40 record files")
    return 0


def verify(out: pathlib.Path, lay: dict, headers: list, shards: list, base: str, n: int, log, only: set | None = None) -> int:
    """Each finished record file against the GGUF: n experts a layer (the first, the last, the rest spread), their gate, up
    and down bytes read back from Hugging Face with range requests and compared, and each record's padding zero."""
    bad = checked = 0
    for il in range(N_LAYER):
        path = out / "experts" / f"L{il:02d}.bin"
        if not path.exists() or (only and il not in only):
            continue
        v = lay[il]
        if path.stat().st_size != v["stride"] * N_EXPERT:
            log(f"  L{il:02d}: {path.stat().st_size:,} bytes, not {N_EXPERT} records of {v['stride']:,}")
            bad += 1
            continue
        spread = list(range(N_EXPERT // max(1, n), N_EXPERT - 1, N_EXPERT // max(1, n)))[:max(0, n - 2)]
        experts = sorted({0, N_EXPERT - 1, *spread})
        with open(path, "rb") as f:
            for e in experts:
                for part in ("gate", "up", "down"):
                    si, ty, off = v[part]
                    rows = N_FF if part in ("gate", "up") else N_EMBD
                    per = rows * (v["gu_row"] if part != "down" else v["d_row"])
                    at = {"gate": 0, "up": N_FF * v["gu_row"], "down": 2 * N_FF * v["gu_row"]}[part]
                    g0 = headers[si].data_start + off + e * per
                    req = urllib.request.Request(base + shards[si]["name"], headers={"Range": f"bytes={g0}-{g0 + per - 1}"})
                    for attempt in range(5):   # (a range read beside a running download can time out: tried again)
                        try:
                            with urllib.request.urlopen(req, timeout=120) as r:
                                want = r.read()
                            break
                        except OSError as ex:
                            if attempt == 4:
                                raise
                            log(f"  L{il:02d} expert {e} {part}: {ex}; again in 10 s")
                            time.sleep(10)
                    f.seek(e * v["stride"] + at)
                    if f.read(per) != want or len(want) != per:
                        log(f"  L{il:02d} expert {e} {part}: the record differs from the GGUF")
                        bad += 1
                f.seek(e * v["stride"] + v["blob"])
                if any(f.read(v["stride"] - v["blob"])):
                    log(f"  L{il:02d} expert {e}: padding not zero")
                    bad += 1
                checked += 1
        log(f"  L{il:02d}: {len(experts)} experts checked against the GGUF")
    log(f"verify: {checked} records checked, {bad} problem(s)")
    return 1 if bad else 0


def tensor_bytes(dims: tuple, ty: int) -> int:
    """A tensor's bytes from its ggml type (the dense tensors this tool skips)."""
    n = 1
    for d in dims:
        n *= d
    blk = {0: (1, 4), 1: (1, 2), 30: (1, 2), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 8: (32, 34), 2: (32, 18),
           3: (32, 20), 6: (32, 22), 7: (32, 24), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
           14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
           21: (256, 110), 22: (256, 82), 23: (256, 136), 29: (256, 56), 39: (32, 17)}
    if ty not in blk:
        raise IntegrityError(f"a tensor of ggml type {ty} (size unknown here)")
    be, bb = blk[ty]
    if n % be:
        raise IntegrityError(f"a tensor of {n} values in blocks of {be}")
    return n // be * bb


if __name__ == "__main__":
    sys.exit(main())
