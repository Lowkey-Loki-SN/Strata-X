"""moex/tools/dsv41/test_convert_resume.py - convert_pack.py's resume paths, against a local server, no network.

    python test_convert_resume.py [--converter convert_pack.py] [--work DIR]

A synthetic engram shard (the real layout: embed.weight [N, 256] F8_E4M3 from data offset 0, embed.scale [N, 8]
F8_E8M0 right after it, then small dense tensors) is served over HTTP with Range support, throttled, and with an
optional dropped connection. Checks:
  1. a clean run: the engram files are the released bytes, the dense safetensors holds the rest, sha256 checked;
  2. SIGKILL in the middle (a power-off), then a rerun: it resumes at the last 256 MiB flush point of the kept part
     and ends with the same files;
  3. a kept part damaged below the resume point: the rerun fails the sha256 check and deletes the parts, and the
     run after that converts the shard from zero;
  4. the server drops the connection once in the middle: the same run resumes with Range and finishes.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import pathlib
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
NAME = "model-00047-of-00048.safetensors"
MiB = 1 << 20


def make_shard(path: pathlib.Path, rows: int) -> dict:
    rng = __import__("numpy").random.default_rng(5)
    tensors = [("layers.1.engram.embed.weight", "F8_E4M3", [rows, 256]),
               ("layers.1.engram.embed.scale", "F8_E8M0", [rows, 8]),
               ("layers.1.engram.q_weight", "BF16", [4, 64]),
               ("layers.1.engram.k_weight", "BF16", [4, 64]),
               ("layers.1.engram.wkv.weight", "F8_E4M3", [256, 192]),
               ("layers.1.engram.wkv.scale", "F8_E8M0", [8, 6])]
    item = {"F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2}
    hdr, off = {}, 0
    for k, dt, shape in tensors:
        n = item[dt]
        for d in shape:
            n *= d
        hdr[k] = dict(dtype=dt, shape=shape, data_offsets=[off, off + n])
        off += n
    hb = json.dumps(hdr).encode()
    hb += b" " * (-len(hb) % 8)
    sha = hashlib.sha256()
    with open(path, "wb") as f:
        head = struct.pack("<Q", len(hb)) + hb
        f.write(head)
        sha.update(head)
        left = off
        while left:
            chunk = rng.integers(0, 256, size=min(left, 32 * MiB), dtype="uint8").tobytes()
            f.write(chunk)
            sha.update(chunk)
            left -= len(chunk)
    return dict(path=NAME, size=path.stat().st_size, lfs=dict(oid=sha.hexdigest()), header=hdr, base=len(head))


class Server:
    """Serves one directory with Range support; `rate` bytes/s; `drop_at`: close the connection once at that
    absolute offset."""

    def __init__(self, root: pathlib.Path):
        self.root, self.rate, self.drop_at, self.dropped = root, 0, None, False
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                p = outer.root / self.path.rsplit("/", 1)[-1]
                size = p.stat().st_size
                start = 0
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(size - start))
                self.end_headers()
                with open(p, "rb") as f:
                    f.seek(start)
                    pos, t0, sent = start, time.time(), 0
                    while pos < size:
                        chunk = f.read(min(MiB, size - pos))
                        if outer.drop_at is not None and not outer.dropped and pos <= outer.drop_at < pos + len(chunk):
                            outer.dropped = True
                            self.wfile.write(chunk[: outer.drop_at - pos])
                            self.wfile.flush()
                            self.connection.shutdown(2)
                            return
                        try:
                            self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            return
                        pos += len(chunk)
                        sent += len(chunk)
                        if outer.rate:
                            ahead = sent / outer.rate - (time.time() - t0)
                            if ahead > 0:
                                time.sleep(ahead)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--converter", default=str(HERE / "convert_pack.py"))
    ap.add_argument("--work", type=pathlib.Path, default=pathlib.Path("/tmp/dsv41-convert-resume-test"))
    ap.add_argument("--rows", type=int, default=2_800_000)  # ~717 MB of rows: three 256 MiB flush points
    a = ap.parse_args()
    if a.work.exists():
        shutil.rmtree(a.work)
    src = a.work / "src"
    src.mkdir(parents=True)
    info = make_shard(src / NAME, a.rows)
    tree = a.work / "tree.json"
    tree.write_text(json.dumps([dict(path=info["path"], size=info["size"], lfs=info["lfs"])]))
    hdr, base = info["header"], info["base"]
    srv = Server(src)
    fast, dense = a.work / "fast", a.work / "dense"
    wpart = fast / "engram" / "L01.weight.bin.part"
    fails = []

    def converter(log_name):
        log = open(a.work / log_name, "w")
        return subprocess.Popen([sys.executable, a.converter, "--fast", str(fast), "--dense", str(dense), "--tree",
                                 str(tree), "--url-base", srv.url, "--jobs", "1"], stdout=log, stderr=subprocess.STDOUT)

    def run(log_name):
        p = converter(log_name)
        p.wait(timeout=600)
        return p.returncode, (a.work / log_name).read_text()

    def outputs_ok(tag):
        with open(src / NAME, "rb") as f:
            for k, fname in (("layers.1.engram.embed.weight", "L01.weight.bin"),
                             ("layers.1.engram.embed.scale", "L01.scale.bin")):
                s0, s1 = hdr[k]["data_offsets"]
                f.seek(base + s0)
                want = hashlib.sha256(f.read(s1 - s0)).hexdigest()
                p = fast / "engram" / fname
                got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
                if got != want:
                    fails.append(f"{tag}: {fname}")
        d = dense / NAME
        if not d.exists():
            fails.append(f"{tag}: dense missing")
        else:
            n = struct.unpack("<Q", d.read_bytes()[:8])[0]
            dh = json.loads(d.read_bytes()[8:8 + n])
            if sorted(dh) != sorted(k for k in hdr if ".embed." not in k):
                fails.append(f"{tag}: dense tensors {sorted(dh)}")
        left = list((fast / "engram").glob("*.part"))
        if left:
            fails.append(f"{tag}: parts left {left}")
        print(f"  {tag}: {'outputs equal the source' if not [x for x in fails if x.startswith(tag)] else 'WRONG'}")

    def reset():
        for d in (fast, dense):
            if d.exists():
                shutil.rmtree(d)

    def kill_after(log_name, at_least):
        srv.rate = 120 * MiB
        p = converter(log_name)
        t0 = time.time()
        while not (wpart.exists() and wpart.stat().st_size >= at_least):
            if p.poll() is not None or time.time() - t0 > 120:
                raise SystemExit(f"converter ended or stalled before {at_least:,} bytes")
            time.sleep(0.05)
        p.send_signal(signal.SIGKILL)
        p.wait()
        srv.rate = 0
        return wpart.stat().st_size

    print("1. clean run")
    rc, log = run("run1.log")
    if rc != 0:
        fails.append("1: exit code")
    outputs_ok("1")

    print("2. SIGKILL in the middle, then a rerun")
    reset()
    size = kill_after("run2a.log", 600 * MiB)
    expect = base + (size // (256 * MiB)) * 256 * MiB
    rc, log = run("run2b.log")
    resumed = f"resuming at byte {expect:,}" in log
    print(f"  part was {size:,} B when killed; rerun {'resumed at ' + format(expect, ',') if resumed else 'did NOT resume as expected'}")
    if rc != 0 or not resumed:
        fails.append("2: resume")
        print(log[-2000:])
    outputs_ok("2")

    print("3. a damaged kept part fails the hash, and the next run starts over")
    reset()
    size = kill_after("run3a.log", 600 * MiB)
    with open(wpart, "r+b") as f:  # flip one byte well below the resume point
        f.seek(12345)
        b = f.read(1)
        f.seek(12345)
        f.write(bytes([b[0] ^ 0xFF]))
    rc, log = run("run3b.log")
    failed = rc != 0 and "IntegrityError" in log and "sha256" in log and not wpart.exists()
    print(f"  rerun with the damaged part: {'failed the sha256 check and deleted the part' if failed else 'UNEXPECTED'}")
    if not failed:
        fails.append("3: damaged part not caught")
        print(log[-2000:])
    rc, log = run("run3c.log")
    if rc != 0 or "resuming" in log:
        fails.append("3: fresh run")
    outputs_ok("3")

    print("4. the server drops the connection once")
    reset()
    srv.drop_at, srv.dropped = base + 300 * MiB + 777, False
    rc, log = run("run4.log")
    if rc != 0 or "stream error" not in log:
        fails.append("4: in-run resume")
        print(log[-2000:])
    else:
        print("  the run resumed with Range after the drop")
    srv.drop_at = None
    outputs_ok("4")

    print("PASS" if not fails else f"FAIL: {fails}")
    if not fails:
        shutil.rmtree(a.work)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
