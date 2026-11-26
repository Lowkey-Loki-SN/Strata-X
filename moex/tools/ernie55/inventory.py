#!/usr/bin/env python3
"""moex/tools/ernie55/inventory.py - the tensors of an ERNIE 5.5 Flash checkpoint, read from its shard headers.

    python3 -I moex/tools/ernie55/inventory.py [--model DIR] [--repo REPO] [--rev SHA] [--cache DIR] [--json OUT]
                                              [--same-as REPO] [--same-rev SHA] [--sample N] [--quiet]

Every tensor's shard, dtype, shape and byte range come from the safetensors headers. A shard that is on disk whole (its
size the one its own header implies, no .part beside it) is read there; any other shard's header is fetched from the
hub with two HTTP range requests (its 8-byte length, then the JSON) and kept in --cache, so a second run asks nothing.
No tensor data is read, except with --same-as: then dense tensors (not the routed experts) that both repositories hold
are read from each - the whole tensor up to 4 MiB, else its first 4 MiB - and compared byte for byte.

The summary groups the tensors by name with the layer, expert and draft-stage numbers taken out, and checks the layer
kinds against the config's compress_ratios (128: HCA, -2: MLA with the DSA indexer).
"""
from __future__ import annotations
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")

import argparse
import collections
import hashlib
import json
import os
import pathlib
import re
import struct
import sys
import time
import urllib.request

FP8_REPO = "baidu/ERNIE-5.5-Flash-Preview-FP8"
FP8_REV = "be26a1d08b6153aed72a047e8690e1df2d02843b"
MODEL = pathlib.Path(_os.path.join(_MODELS, 'HF/ERNIE-5.5-Flash-Preview-FP8'))
CACHE = pathlib.Path(_os.path.join(_DATA, 'ernie55-meta'))
INDEX = "model.safetensors.index.json"
SAMPLE_BYTES = 4 << 20
UA = {"User-Agent": "strata-x-moex-ernie55-inventory"}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def url_of(repo: str, rev: str, fname: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"


def http_get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    """GET url, or bytes [start, end] of it. A range answered with anything but 206 is refused before its body is read
    (a server that ignored the range would send the whole 2.4 GB shard)."""
    last = None
    for attempt in range(5):
        req = urllib.request.Request(url, headers=dict(UA))
        if start is not None:
            req.add_header("Range", f"bytes={start}-{end}")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if start is not None and r.status != 206:
                    raise IOError(f"{url}: range request answered {r.status}")
                data = r.read()
            if start is not None and len(data) != end - start + 1:
                raise IOError(f"{url}: {len(data)} bytes for a range of {end - start + 1}")
            return data
        except Exception as e:  # noqa: BLE001 - retried, then raised
            last = e
            time.sleep(2 * (attempt + 1))
    raise IOError(f"{url}: {last}")


def main_revision(repo: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/models/{repo}", headers=UA),
                                timeout=120) as r:
        return json.load(r)["sha"]


class Repo:
    """One checkpoint: its index, and each shard's header from disk (when whole) or from the hub (cached)."""

    def __init__(self, repo: str, rev: str, model: pathlib.Path | None, cache: pathlib.Path):
        self.repo, self.rev, self.model = repo, rev, model
        self.cache = cache / repo.replace("/", "__") / rev
        self.cache.mkdir(parents=True, exist_ok=True)
        self.weight_map = json.loads(self._small(INDEX))["weight_map"]
        self.shards = sorted(set(self.weight_map.values()))
        self.headers: dict[str, tuple[dict, int, bool]] = {}

    def _small(self, fname: str) -> bytes:
        for d in (self.model, self.cache):
            if d is not None and (d / fname).is_file():
                return (d / fname).read_bytes()
        data = http_get(url_of(self.repo, self.rev, fname))
        (self.cache / fname).write_bytes(data)
        return data

    def local_whole(self, shard: str) -> pathlib.Path | None:
        if self.model is None:
            return None
        p = self.model / shard
        if not p.is_file() or (self.model / (shard + ".part")).exists():
            return None
        return p

    def header(self, shard: str) -> tuple[dict, int, bool]:
        """(header without __metadata__, data base offset, read from a whole local file)"""
        if shard in self.headers:
            return self.headers[shard]
        p = self.local_whole(shard)
        if p is not None:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            end = 8 + n + max((t["data_offsets"][1] for t in hdr.values()), default=0)
            if os.path.getsize(p) == end:
                self.headers[shard] = (hdr, 8 + n, True)
                return self.headers[shard]
            log(f"{shard}: {os.path.getsize(p)} bytes on disk, its header implies {end} - reading the hub's header")
        cp = self.cache / f"{shard}.header.json"
        if cp.is_file():
            d = json.loads(cp.read_text())
        else:
            url = url_of(self.repo, self.rev, shard)
            n = struct.unpack("<Q", http_get(url, 0, 7))[0]
            if n > 64 << 20:
                raise ValueError(f"{shard}: a header of {n} bytes")
            hdr = json.loads(http_get(url, 8, 8 + n - 1))
            hdr.pop("__metadata__", None)
            d = {"header": hdr, "base": 8 + n}
            cp.write_text(json.dumps(d))
        self.headers[shard] = (d["header"], d["base"], False)
        return self.headers[shard]

    def tensors(self) -> dict[str, tuple]:
        out = {}
        for i, shard in enumerate(self.shards):
            hdr, base, _ = self.header(shard)
            for name, t in hdr.items():
                a, b = t["data_offsets"]
                out[name] = (shard, t["dtype"], tuple(t["shape"]), base + a, base + b)
            if (i + 1) % 16 == 0:
                log(f"{self.repo}: {i + 1}/{len(self.shards)} shard headers")
        missing = set(self.weight_map) - set(out)
        if missing:
            raise ValueError(f"{self.repo}: {len(missing)} indexed tensors in no shard header, e.g. {sorted(missing)[:3]}")
        return out

    def read(self, rec: tuple, nbytes: int) -> bytes:
        shard, _, _, a, b = rec
        n = min(nbytes, b - a)
        p = self.local_whole(shard)
        if p is not None and self.header(shard)[2]:
            with open(p, "rb") as f:
                f.seek(a)
                data = f.read(n)
                try:   # a few MB at most - but keep them out of the page cache the live server shares
                    os.posix_fadvise(f.fileno(), a, n, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
                return data
        return http_get(url_of(self.repo, self.rev, shard), a, a + n - 1)


def klass(name: str) -> str:
    k = re.sub(r"\.layers\.\d+\.", ".layers.L.", name)
    k = re.sub(r"\.experts\.\d+\.", ".experts.E.", k)
    return re.sub(r"^model\.mtp\.\d+\.", "model.mtp.S.", k)


def ranges(nums) -> str:
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(out)


def summarize(tensors: dict, config: dict | None) -> dict:
    groups = collections.OrderedDict()
    for name in sorted(tensors):
        shard, dt, shape, a, b = tensors[name]
        g = groups.setdefault(klass(name), {"count": 0, "dtypes": set(), "shapes": set(), "bytes": 0, "layers": set()})
        g["count"] += 1
        g["dtypes"].add(dt)
        g["shapes"].add(shape)
        g["bytes"] += b - a
        m = re.search(r"\.layers\.(\d+)\.", name) or re.match(r"model\.mtp\.(\d+)\.", name)
        if m:
            g["layers"].add(int(m.group(1)))
    total = sum(g["bytes"] for g in groups.values())
    experts = sum(g["bytes"] for k, g in groups.items() if ".experts.E." in k)
    per_layer = collections.Counter()
    for name, (_, _, _, a, b) in tensors.items():
        m = re.match(r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.weight$", name)
        if m:
            per_layer[int(m.group(1))] += 1
    report = {"tensors": len(tensors), "bytes": total, "expert_bytes": experts, "experts_per_layer": dict(per_layer),
              "groups": {k: {"count": g["count"], "dtypes": sorted(g["dtypes"]), "shapes": sorted(g["shapes"]),
                             "bytes": g["bytes"], "layers": ranges(g["layers"])} for k, g in groups.items()}}
    if config:
        ratios = config.get("compress_ratios", [])
        n_layer = config.get("num_hidden_layers", len(ratios))
        mla = {int(m.group(1)) for n in tensors for m in [re.match(r"model\.layers\.(\d+)\.self_attn\.kv_b_proj\.", n)] if m}
        hca = {int(m.group(1)) for n in tensors for m in [re.match(r"model\.layers\.(\d+)\.self_attn\.linear_kv_proj\.", n)]
               if m}
        want_mla = {i for i in range(n_layer) if i < len(ratios) and ratios[i] == -2}
        want_hca = {i for i in range(n_layer) if i < len(ratios) and ratios[i] > 0}
        report["layer_kinds"] = {"mla": ranges(mla), "hca": ranges(hca),
                                 "agree_with_config": mla == want_mla and hca == want_hca}
    return report


def print_report(repo: Repo, rep: dict) -> None:
    print(f"{repo.repo} @ {repo.rev}: {rep['tensors']} tensors, {rep['bytes'] / 1e9:.3f} GB "
          f"({rep['expert_bytes'] / 1e9:.3f} GB routed experts), {len(repo.shards)} shards")
    if "layer_kinds" in rep:
        lk = rep["layer_kinds"]
        print(f"  layer kinds: MLA {lk['mla']} | HCA {lk['hca']} | agree with compress_ratios: {lk['agree_with_config']}")
    epl = rep["experts_per_layer"]
    if epl:
        counts = collections.Counter(epl.values())
        print(f"  routed experts: layers {ranges(epl)}; per layer {dict(counts)}")
    for k, g in rep["groups"].items():
        shapes = g["shapes"] if len(g["shapes"]) <= 2 else g["shapes"][:2] + ["..."]
        print(f"  {g['count']:6d} {','.join(g['dtypes']):9s} {str(shapes):34s} {g['bytes'] / 1e9:8.3f} GB  "
              f"[{g['layers'] or '-'}] {k}")


def same_as(a: Repo, ta: dict, b: Repo, tb: dict, sample: int) -> None:
    common = [n for n in ta if n in tb and ".experts." not in n]
    only_a = sorted(klass(n) for n in ta if n not in tb and ".experts." not in n)
    only_b = sorted(klass(n) for n in tb if n not in ta and ".experts." not in n)
    print(f"\ndense tensors: {len(common)} in both; {len(set(only_a))} classes only in {a.repo}, "
          f"{len(set(only_b))} only in {b.repo}")
    for k in sorted(set(only_a))[:12]:
        print(f"  only in {a.repo}: {k}")
    for k in sorted(set(only_b))[:12]:
        print(f"  only in {b.repo}: {k}")
    meta_diff = [n for n in common if ta[n][1:3] != tb[n][1:3]]
    print(f"  dtype or shape differs: {len(meta_diff)}" + (f", e.g. {meta_diff[:3]}" if meta_diff else ""))
    # one tensor of each class, from the layers spread over the stack (and the top-level ones)
    pick, seen = [], set()
    for want in (1, 7, 30, 42, 0, 2, 40, None):
        for n in sorted(common):
            k = klass(n)
            m = re.search(r"\.layers\.(\d+)\.", n) or re.match(r"model\.mtp\.(\d+)\.", n)
            lay = int(m.group(1)) if m else None
            if k in seen or (want is not None and lay != want) or (want is None and lay is not None and k in seen):
                continue
            seen.add(k)
            pick.append(n)
    pick = pick[:sample]
    same = diff = 0
    for n in pick:
        da, db = a.read(ta[n], SAMPLE_BYTES), b.read(tb[n], SAMPLE_BYTES)
        eq = hashlib.sha256(da).digest() == hashlib.sha256(db).digest()
        same += eq
        diff += not eq
        if not eq:
            print(f"  DIFFERENT: {n} ({len(da)} bytes compared)")
    print(f"  bytes compared on {len(pick)} tensors (one per class, whole up to 4 MiB): {same} identical, {diff} differ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=pathlib.Path, default=MODEL, help="the local download (shards read where whole)")
    ap.add_argument("--repo", default=FP8_REPO)
    ap.add_argument("--rev", default=FP8_REV)
    ap.add_argument("--cache", type=pathlib.Path, default=CACHE)
    ap.add_argument("--json", type=pathlib.Path, help="write every tensor's record here")
    ap.add_argument("--same-as", help="a second repository to compare the dense tensors with (headers from the hub)")
    ap.add_argument("--same-rev", help="its revision (default: main's current one, printed)")
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--quiet", action="store_true", help="no per-class table")
    a = ap.parse_args()

    model = a.model if a.model and a.model.is_dir() else None
    repo = Repo(a.repo, a.rev, model, a.cache)
    tensors = repo.tensors()
    config = None
    try:
        config = json.loads(repo._small("config.json"))
    except Exception as e:  # noqa: BLE001
        log(f"no config.json: {e}")
    rep = summarize(tensors, config)
    local = sum(1 for s in repo.shards if repo.header(s)[2])
    print(f"(shard headers: {local} from whole local files, {len(repo.shards) - local} from the hub)")
    if not a.quiet:
        print_report(repo, rep)
    if a.json:
        a.json.write_text(json.dumps({"repo": repo.repo, "rev": repo.rev, "summary": rep,
                                      "tensors": {n: list(t) for n, t in tensors.items()}}, default=list))
        print(f"wrote {a.json}")
    if a.same_as:
        rev = a.same_rev or main_revision(a.same_as)
        other = Repo(a.same_as, rev, None, a.cache)
        to = other.tensors()
        try:
            oc = json.loads(other._small("config.json"))
        except Exception:  # noqa: BLE001
            oc = None
        orep = summarize(to, oc)
        print()
        print_report(other, orep)
        if oc and config:
            keys = sorted(set(oc) | set(config))
            cd = [k for k in keys if oc.get(k) != config.get(k)]
            print(f"\nconfig keys that differ: {cd}")
            for k in cd:
                print(f"  {k}: {config.get(k)!r} | {oc.get(k)!r}")
        same_as(repo, tensors, other, to, a.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
