"""hf_fetch.py - a GGUF set from Hugging Face, file by file as published (nothing converted), each checked against its SHA-256.

    python3 -I moex/tools/dsv41/hf_fetch.py --repo unsloth/DeepSeek-V4.1-Flash-1017-GGUF --quant UD-IQ1_M \
        --out ~/AI/Models/dsv41-1017/ud-iq1m [--rev COMMIT] [--workers 2]

The file list, sizes and SHA-256s come from the repo's API at one commit (--rev, default: the repo's current one, printed
and then used for every request), so a file cannot change under the download.  Each file streams into NAME.part while its
SHA-256 is computed; a dropped connection resumes with a Range request inside the run, and a restarted run hashes the
.part it finds and resumes after it.  A finished file is renamed to its own name only when its SHA-256 matches; a mismatch
leaves NAME.bad and stops that file (no endless retries).  Files already there with the right size and a NAME.sha256 that
matches are skipped.  The files are taken in the set's order (file 1 first), --workers at a time.  The last log line is
"hf_fetch exit N" (0: every file there and checked).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

CHUNK = 4 << 20
LOCK = threading.Lock()


def log(msg: str) -> None:
    with LOCK:
        print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def api(repo: str, rev: str | None) -> tuple[str, list[dict]]:
    url = f"https://huggingface.co/api/models/{repo}" + (f"/revision/{rev}" if rev else "") + "?blobs=true"
    with urllib.request.urlopen(url, timeout=120) as r:
        m = json.load(r)
    return m["sha"], m.get("siblings", [])


def fetch(base: str, f: dict, out: pathlib.Path) -> bool:
    name = f["name"].split("/")[-1]
    dst, part, ok = out / name, out / (name + ".part"), out / (name + ".sha256")
    if dst.exists() and dst.stat().st_size == f["size"] and ok.exists() and ok.read_text().split()[0] == f["sha256"]:
        log(f"{name}: there, SHA-256 checked before - skipped")
        return True
    sha, have = hashlib.sha256(), 0
    if part.exists():   # a restarted run: hash what is there, then resume after it
        with open(part, "rb") as fh:
            while b := fh.read(CHUNK):
                sha.update(b)
                have += len(b)
        if have > f["size"]:
            part.unlink()
            sha, have = hashlib.sha256(), 0
        log(f"{name}: resuming after {have / 1e9:.2f} of {f['size'] / 1e9:.2f} GB")
    t0, last, n0, tries = time.time(), time.time(), have, 0
    with open(part, "ab") as fh:
        while have < f["size"]:
            try:
                req = urllib.request.Request(base + f["name"], headers={"Range": f"bytes={have}-"} if have else {})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if have and r.status != 206:
                        raise IOError(f"asked for bytes {have}- and got HTTP {r.status}")
                    while b := r.read(CHUNK):
                        fh.write(b)
                        sha.update(b)
                        have += len(b)
                        tries = 0
                        if time.time() - last >= 60:
                            last = time.time()
                            rate = (have - n0) / max(1e-9, last - t0)
                            log(f"{name}: {have / 1e9:7.2f} of {f['size'] / 1e9:.2f} GB, {rate / 1e6:5.1f} MB/s, "
                                f"{(f['size'] - have) / max(rate, 1) / 60:5.0f} min left")
            except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
                tries += 1
                if tries > 30:
                    log(f"{name}: giving up after 30 failed tries in a row at {have / 1e9:.2f} GB ({e})")
                    return False
                fh.flush()
                log(f"{name}: connection lost at {have / 1e9:.2f} GB ({e}); resuming in {min(60, 5 * tries)} s")
                time.sleep(min(60, 5 * tries))
    got = sha.hexdigest()
    if have != f["size"] or got != f["sha256"]:
        part.rename(out / (name + ".bad"))
        log(f"{name}: SHA-256 {got} ({have} bytes), Hugging Face has {f['sha256']} ({f['size']} bytes) - left as {name}.bad")
        return False
    part.rename(dst)
    ok.write_text(f"{got}  {name}\n")
    log(f"{name}: done, SHA-256 matches ({f['size'] / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--quant", required=True, help="the folder (or name part) of the set's files, e.g. UD-IQ1_M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rev", default=None, help="the repo commit to take (default: the current one)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", action="append", default=[],
                    help="fetch only the files whose name holds this text (repeatable; SOURCE.txt still lists the whole set)")
    ap.add_argument("--file", action="append", default=[],
                    help="(moex) a file of the repo by its exact name, any type - e.g. one safetensors shard (repeatable; "
                         "--quant is then only the label in the log)")
    a = ap.parse_args()
    out = pathlib.Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    rev, sib = api(a.repo, a.rev)
    def wanted(name: str) -> bool:
        if a.file:   # (moex) exact names, any type
            return name in a.file
        return name.endswith(".gguf") and (name.startswith(a.quant + "/") or f"-{a.quant}-" in name.split("/")[-1])

    files = sorted(({"name": s["rfilename"], "size": int(s["size"]), "sha256": (s.get("lfs") or {}).get("sha256")}
                    for s in sib if wanted(s["rfilename"])),
                   key=lambda f: f["name"])
    if not files or any(not f["sha256"] for f in files):
        log(f"no files for {a.quant} in {a.repo} at {rev} (or one has no SHA-256)")
        return 2
    (out / "SOURCE.txt").write_text(f"{a.repo} @ {rev}\n" + "".join(f"{f['name']}  {f['size']}  {f['sha256']}\n" for f in files))
    log(f"{a.repo} @ {rev}: {len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB, into {out}")
    todo = [f for f in files if not a.only or any(o in f["name"] for o in a.only)]
    if not todo:
        log(f"--only {' '.join(a.only)}: no file of the set matches")
        return 2
    if len(todo) < len(files):
        log(f"--only {' '.join(a.only)}: {len(todo)} of the {len(files)} files")
    base = f"https://huggingface.co/{a.repo}/resolve/{rev}/"
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        res = list(ex.map(lambda f: fetch(base, f, out), todo))
    bad = [f["name"] for f, r in zip(todo, res) if not r]
    log("every file there and checked" if not bad else f"not done: {', '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    rc = main()
    log(f"hf_fetch exit {rc}")
    sys.exit(rc)
