"""moex/tools/dsv41/convert_pack.py - DeepSeek-V4.1-Flash-1017 from Hugging Face straight into Strata-X's native pack.

    python convert_pack.py --fast DIR_ON_EXT4 --dense DIR --tree TREE.json [--shards 3,4,43] [--jobs 2]

No raw copy of the release is kept: each shard is read as one HTTP stream and its tensors go straight to their place
in the pack, while the whole stream is hashed and checked against the file's SHA-256 on Hugging Face. A shard's
outputs are written as *.part and renamed only after the hash matches, and the shard is then recorded in
manifest.json (fsynced, with its directory), so a stopped run resumes at the first shard not recorded. Dropped
connections are resumed with a Range request inside the same process (the hash state is kept).

The two engram shards are ~101 GB each (~3 h here), so they also resume across runs: a kept engram *.part is cut
back to the last point its data was flushed (a multiple of 256 MiB), those local bytes are hashed again, and the
download continues from there with a Range request. The shard's SHA-256 still covers every byte, so a damaged local
part fails the check; the shard's *.part files are then deleted and it starts from zero on the next run. Engram parts
are kept when a run stops for any other reason (network, kill, shutdown).

The pack (lossless: the release's bits, see fp4_repack.py; DSV41_PLAN.md §5A):
  FAST/experts/L00.bin .. L39.bin   384 records each, FAST/experts/M0.bin .. M2.bin (DSpark) 128 records each:
                                    record e at e * 18,800,640 = [w1 | w3 | w2] in ggml block_mxfp4, crc32 per record
  FAST/engram/L01.weight.bin ...    the FP8 rows as released (256 B each, 16 per 4 KiB page, starting at 0), and
  FAST/engram/L01.scale.bin ...     their E8M0 scales (8 B per row)
  DENSE/model-000NN.safetensors     every other tensor of shard NN, bytes as released (loaded once, at start)
  FAST/manifest.json                per shard: source sha256, outputs, sizes, record crc32s
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fp4_repack  # noqa: E402

RECORD = 18_800_640
EXPERT_RE = re.compile(r"^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
ENGRAM_RE = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")
DROP_EVERY = 256 << 20     # bytes between fdatasync + page-cache drops; also the engram resume granularity
COPY_CHUNK = 64 << 20      # divides DROP_EVERY, so the flush points fall on its multiples


class IntegrityError(Exception):
    """The shard's bytes are wrong (hash, sizes, layout): its partial outputs must not be resumed."""


def fsync_dir(path: pathlib.Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # FUSE (ntfs-3g) may not support fsync on directories
        pass


class Stream:
    """One shard as a byte stream, hashed as it is read, resumed with Range on errors."""

    def __init__(self, url: str, size: int, log):
        self.url, self.size, self.log = url, size, log
        self.pos, self.sha = 0, hashlib.sha256()
        self.resp = None
        self.retries = 0

    def _open(self):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-"} if self.pos else {})
        self.resp = urllib.request.urlopen(req, timeout=120)
        if self.pos and self.resp.status != 206:
            raise IOError(f"server ignored Range (status {self.resp.status})")

    def skip_to(self, pos: int, local_hash) -> None:
        """Continue the stream at `pos`; the bytes in between are hashed from the caller (local data)."""
        local_hash(self.sha)
        if self.resp is not None:
            self.resp.close()
            self.resp = None
        self.pos = pos

    def read(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            try:
                if self.resp is None:
                    self._open()
                chunk = self.resp.read(min(n - len(out), 8 << 20))
                if not chunk:
                    raise IOError("connection closed early")
            except (OSError, urllib.error.URLError) as e:
                self.retries += 1
                if self.retries > 50:
                    raise
                self.log(f"  stream error at {self.pos:,}: {e}; resuming in 10 s (retry {self.retries})")
                try:
                    if self.resp is not None:
                        self.resp.close()
                except Exception:
                    pass
                self.resp = None
                time.sleep(10)
                continue
            out += chunk
            self.pos += len(chunk)
            self.sha.update(chunk)
        return bytes(out)

    def close(self):
        if self.resp is not None:
            self.resp.close()


class Out:
    """A file written with pwrite, its page cache dropped as it goes (500 GB must not squeeze the desktop).
    resume_at: keep the existing *.part, cut back to that many bytes, instead of starting it empty."""

    def __init__(self, path: pathlib.Path, size: int | None, resume_at: int | None = None):
        self.final = path
        self.part = path.with_name(path.name + ".part")
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume_at is None:
            self.fd = os.open(self.part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            if size:
                os.posix_fallocate(self.fd, 0, size)
        else:
            self.fd = os.open(self.part, os.O_WRONLY)
            os.ftruncate(self.fd, resume_at)
        self.dirty = 0
        self.size = resume_at or 0

    def pwrite(self, data: bytes, at: int):
        mv = memoryview(data)
        while mv:
            n = os.pwrite(self.fd, mv, at)
            mv, at = mv[n:], at + n
        self.size = max(self.size, at)
        self.dirty += len(data)
        if self.dirty >= DROP_EVERY:
            self.drop()

    def drop(self):
        os.fdatasync(self.fd)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        self.dirty = 0

    def finish(self):
        self.drop()
        os.close(self.fd)
        os.replace(self.part, self.final)
        fsync_dir(self.final.parent)

    def abandon(self, keep: bool = False):
        """keep: flush and leave the *.part for the next run to resume (engram only)."""
        try:
            if keep:
                self.drop()
            os.close(self.fd)
        finally:
            if not keep:
                self.part.unlink(missing_ok=True)


def engram_path(fast: pathlib.Path, k: str) -> pathlib.Path | None:
    g = ENGRAM_RE.match(k)
    return None if g is None else fast / "engram" / f"L{int(g.group(1)):02d}.{g.group(2)}.bin"


def resume_plan(hdr: dict, order: list[str], fast: pathlib.Path) -> dict[str, int]:
    """Bytes of each leading engram tensor that a kept *.part can supply: tensors in file order from data offset 0,
    each cut back to a multiple of DROP_EVERY (the points where its data was fdatasync'ed); stops at the first
    tensor that is not complete locally."""
    plan, pos = {}, 0
    for k in order:
        a, b = hdr[k]["data_offsets"]
        p = engram_path(fast, k)
        if p is None or a != pos:
            break
        part = p.with_name(p.name + ".part")
        if not part.exists():
            break
        have = min(part.stat().st_size, b - a)
        if have < b - a:
            have = have // DROP_EVERY * DROP_EVERY
        if have == 0:
            break
        plan[k] = have
        pos = a + have
        if have < b - a:
            break
    return plan


def hash_local(paths_and_lengths: list[tuple[pathlib.Path, int]]):
    def run(sha):
        for path, n in paths_and_lengths:
            with open(path, "rb") as f:
                done = 0
                while done < n:
                    chunk = f.read(min(COPY_CHUNK, n - done))
                    if not chunk:
                        raise IntegrityError(f"{path}: shorter than {n:,} bytes")
                    sha.update(chunk)
                    done += len(chunk)
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return run


def convert_shard(name: str, size: int, sha256: str, url_base: str, fast: pathlib.Path, dense: pathlib.Path,
                  log) -> dict:
    url = url_base + name
    s = Stream(url, size, log)
    t0 = time.time()
    n_hdr = struct.unpack("<Q", s.read(8))[0]
    header_bytes = s.read(n_hdr)
    hdr = json.loads(header_bytes)
    meta = hdr.pop("__metadata__", None)
    base = 8 + n_hdr
    order = sorted(hdr, key=lambda k: hdr[k]["data_offsets"][0])

    seeded = resume_plan(hdr, order, fast)
    outs: dict[str, Out] = {}
    expert_file = None
    n_records = 0
    crcs: dict[int, int] = {}
    pending: dict[int, dict] = {}
    dense_tensors: dict[str, tuple[dict, bytes]] = {}
    engram_files = {}
    try:
        if seeded:
            local = [(engram_path(fast, k).with_name(engram_path(fast, k).name + ".part"), n)
                     for k, n in seeded.items()]
            resume_at = base + max(hdr[k]["data_offsets"][0] + n for k, n in seeded.items())
            t_hash = time.time()
            s.skip_to(resume_at, hash_local(local))
            log(f"  {name}: resuming at byte {resume_at:,} of {size:,} ({(resume_at - base) / 1e9:.1f} GB from the "
                f"kept engram part, hashed again in {time.time() - t_hash:.0f} s)")
        for k in order:
            t = hdr[k]
            a, b = t["data_offsets"]
            n = b - a
            r0 = seeded.get(k, 0)
            if r0 < n:  # bytes of this tensor still to come from the stream
                start = base + a + r0
                if s.pos > start:
                    raise IntegrityError(f"{name}: overlapping tensor {k}")
                if s.pos < start:
                    s.read(start - s.pos)
            m = EXPERT_RE.match(k)
            g = ENGRAM_RE.match(k)
            if m:
                kind, li, e, w, part = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
                if expert_file is None:
                    fname = f"L{li:02d}.bin" if kind == "layers" else f"M{li}.bin"
                    n_records = 384 if kind == "layers" else 128
                    expert_file = fast / "experts" / fname
                    outs["experts"] = Out(expert_file, n_records * RECORD)
                dt = np.int8 if part == "weight" else np.uint8
                arr = np.frombuffer(s.read(n), dtype=dt).reshape(t["shape"])
                pending.setdefault(e, {})[(w, part)] = arr
                p = pending[e]
                if len(p) == 6:
                    rec = fp4_repack.expert_record(p[("w1", "weight")], p[("w1", "scale")], p[("w3", "weight")],
                                                   p[("w3", "scale")], p[("w2", "weight")], p[("w2", "scale")])
                    if len(rec) != RECORD:
                        raise IntegrityError(f"{name}: expert {e} record is {len(rec)} B")
                    outs["experts"].pwrite(rec, e * RECORD)
                    crcs[e] = zlib.crc32(rec)
                    del pending[e]
            elif g:
                li, part = int(g.group(1)), g.group(2)
                key = f"engram.{part}"
                outs[key] = Out(engram_path(fast, k), None, resume_at=r0 if k in seeded else None)
                engram_files[part] = dict(path=str(outs[key].final), bytes=n, dtype=t["dtype"], shape=t["shape"])
                at = r0
                while at < n:
                    chunk = s.read(min(COPY_CHUNK - at % COPY_CHUNK, n - at))
                    outs[key].pwrite(chunk, at)
                    at += len(chunk)
            else:
                dense_tensors[k] = (t, s.read(n))
        rest = size - s.pos
        if rest < 0:
            raise IntegrityError(f"{name}: read past the end")
        if rest:
            s.read(rest)
        got = s.sha.hexdigest()
        if got != sha256:
            raise IntegrityError(f"{name}: sha256 {got} != {sha256}" + (" (after a resume)" if seeded else ""))
        if pending:
            raise IntegrityError(f"{name}: incomplete experts {sorted(pending)[:5]}")
        if expert_file is not None and len(crcs) != n_records:
            raise IntegrityError(f"{name}: {len(crcs)} expert records of {n_records}")

        dense_path = None
        if dense_tensors:
            dense_path = dense / name
            hd, off = {}, 0
            for k in dense_tensors:
                t, data = dense_tensors[k]
                hd[k] = dict(dtype=t["dtype"], shape=t["shape"], data_offsets=[off, off + len(data)])
                off += len(data)
            if meta:
                hd["__metadata__"] = meta
            hb = json.dumps(hd, separators=(",", ":")).encode()
            hb += b" " * (-len(hb) % 8)
            o = Out(dense_path, None)
            outs["dense"] = o
            o.pwrite(struct.pack("<Q", len(hb)) + hb, 0)
            at = 8 + len(hb)
            for k in dense_tensors:
                data = dense_tensors[k][1]
                o.pwrite(data, at)
                at += len(data)
        for o in outs.values():
            o.finish()
    except BaseException as e:
        keep = not isinstance(e, IntegrityError)
        for key, o in outs.items():
            o.abandon(keep=keep and key.startswith("engram."))
        if not keep:  # parts this run never opened (a resume that failed its hash) must go too
            for k in order:
                p = engram_path(fast, k)
                if p is not None:
                    p.with_name(p.name + ".part").unlink(missing_ok=True)
        raise
    finally:
        s.close()
    dt = time.time() - t0
    return dict(shard=name, sha256=sha256, bytes=size, seconds=round(dt, 1), retries=s.retries,
                resumed_from=(base + max(hdr[k]["data_offsets"][0] + n for k, n in seeded.items())) if seeded else 0,
                experts=None if expert_file is None else dict(path=str(expert_file), records=n_records,
                                                              record_bytes=RECORD,
                                                              crc32=[crcs[e] for e in range(n_records)]),
                engram=engram_files or None, dense=None if dense_path is None else dict(
                    path=str(dense_path), tensors=len(dense_tensors)))


def write_manifest(path: pathlib.Path, manifest: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as f:
        f.write(json.dumps(manifest, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", required=True, type=pathlib.Path, help="ext4 dir for experts and engram (O_DIRECT)")
    ap.add_argument("--dense", required=True, type=pathlib.Path, help="dir for the dense safetensors")
    ap.add_argument("--tree", required=True, type=pathlib.Path, help="HF tree listing (api/models/REPO/tree/main)")
    ap.add_argument("--repo", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--url-base", default=None, help="where the shards are served (default: the repo on HF)")
    ap.add_argument("--shards", default="", help="comma list of shard numbers (default: all)")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    url_base = a.url_base or f"https://huggingface.co/{a.repo}/resolve/main/"

    files = {f["path"]: f for f in json.loads(a.tree.read_text()) if f["path"].endswith(".safetensors")}
    names = sorted(files)
    if a.shards:
        want = {int(x) for x in a.shards.split(",")}
        names = [n for n in names if int(re.search(r"model-(\d+)-of", n).group(1)) in want]
    a.fast.mkdir(parents=True, exist_ok=True)
    a.dense.mkdir(parents=True, exist_ok=True)
    man_path = a.fast / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else dict(repo=a.repo, shards={})
    todo = [n for n in names if n not in manifest["shards"]]
    total = sum(files[n]["size"] for n in todo)
    print(f"{len(todo)} shards to convert ({total / 1e9:.1f} GB), {len(names) - len(todo)} already done", flush=True)

    def log(msg):
        print(msg, flush=True)

    done_bytes, t0 = 0, time.time()
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(convert_shard, n, files[n]["size"], files[n]["lfs"]["oid"], url_base, a.fast, a.dense,
                          log): n for n in todo}
        for f in cf.as_completed(futs):
            n = futs[f]
            try:
                r = f.result()
            except Exception as e:  # keep going with the other shards; this one is retried on the next run
                log(f"FAILED {n}: {type(e).__name__}: {e}")
                continue
            manifest["shards"][n] = r
            write_manifest(man_path, manifest)
            done_bytes += r["bytes"] - r.get("resumed_from", 0)
            el = time.time() - t0
            log(f"ok {n}: {r['bytes'] / 1e9:.2f} GB in {r['seconds']:.0f} s, retries {r['retries']}"
                + (f", resumed at {r['resumed_from'] / 1e9:.1f} GB" if r.get("resumed_from") else "")
                + f"; {done_bytes / 1e9:.1f} of {total / 1e9:.1f} GB, {done_bytes / el / 1e6:.1f} MB/s overall")
    left = [n for n in names if n not in manifest["shards"]]
    print("ALL DONE" if not left else f"{len(left)} shards left: {left}", flush=True)
    return 0 if not left else 1


if __name__ == "__main__":
    raise SystemExit(main())
