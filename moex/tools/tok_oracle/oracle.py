"""moex/tools/tok_oracle/oracle.py - a Python client for tok_oracle (llama.cpp's tokenizer behind a pipe).

    from oracle import Oracle
    o = Oracle("/path/model.gguf")             # starts moex/build/tok_oracle (vocabulary only: no weights, no GPU)
    o.tokenize("Hello <|user|>", add_special=False, parse_special=True) -> [ids]
    o.info() -> {"bos": .., "eog": [..], ...}  o.piece(i) -> bytes  o.detokenize([ids]) -> bytes

Text is sent as UTF-8 bytes with a length prefix, so any string (NUL, newlines, odd characters) is passed exactly:
no argv, no shell quoting.  The loader's own log (it prints the EOG list) goes to <gguf-stem>.oracle.log under
the scratch directory given, or is kept in `.log`.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.abspath(os.path.join(HERE, "..", "..", "build", "tok_oracle"))


class Oracle:
    def __init__(self, gguf: str, binary: str = BINARY):
        if not os.path.exists(binary):
            raise FileNotFoundError(binary + " - build it with moex/tools/tok_oracle/build.sh")
        self._errfile = tempfile.TemporaryFile()
        self.proc = subprocess.Popen([binary, "-m", gguf], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._errfile, bufsize=1 << 16)
        self.gguf = gguf
        self._info = None

    # ------------------------------------------------------------------ plumbing
    def _send(self, header: str, payload: bytes = b""):
        self.proc.stdin.write(header.encode("ascii") + b"\n" + payload)
        self.proc.stdin.flush()

    def _readline(self) -> bytes:
        line = self.proc.stdout.readline()
        if not line.endswith(b"\n"):
            raise EOFError("tok_oracle ended: " + self.log()[-2000:])
        return line[:-1]

    def _readn(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = self.proc.stdout.read(n - len(out))
            if not chunk:
                raise EOFError("tok_oracle ended: " + self.log()[-2000:])
            out += chunk
        return bytes(out)

    @property
    def log(self):
        def read():
            self._errfile.flush()
            self._errfile.seek(0)
            data = self._errfile.read().decode("utf-8", "replace")
            self._errfile.seek(0, 2)
            return data
        return read

    # ------------------------------------------------------------------ API
    def tokenize(self, text, add_special: bool = False, parse_special: bool = True) -> list[int]:
        data = text if isinstance(text, bytes) else text.encode("utf-8", "surrogatepass")
        self._send("TOK %d %d %d" % (add_special, parse_special, len(data)), data)
        line = self._readline()
        return [int(x) for x in line.split(b",")] if line else []

    def tokenize_many(self, texts, add_special: bool = False, parse_special: bool = True) -> list[list[int]]:
        """Many strings in one go: the requests are written by a thread while the answers are read, so the pipes never
        fill up on both sides (several times the speed of one round trip per string)."""
        import threading
        payloads = [t if isinstance(t, bytes) else t.encode("utf-8", "surrogatepass") for t in texts]

        def feed():
            try:
                for i in range(0, len(payloads), 256):
                    blob = b"".join(b"TOK %d %d %d\n" % (add_special, parse_special, len(p)) + p for p in payloads[i:i + 256])
                    self.proc.stdin.write(blob)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        th = threading.Thread(target=feed, daemon=True)
        th.start()
        out = []
        try:
            for _ in payloads:
                line = self._readline()
                out.append([int(x) for x in line.split(b",")] if line else [])
        finally:
            th.join(timeout=5)
        return out

    def piece(self, token_id: int, special: bool = True) -> bytes:
        self._send("PIECE %d %d" % (token_id, special))
        n = int(self._readline())
        return self._readn(n)

    def detokenize(self, ids, remove_special: bool = False, unparse_special: bool = True) -> bytes:
        payload = ",".join(map(str, ids)).encode("ascii")
        self._send("DETOK %d %d %d" % (remove_special, unparse_special, len(payload)), payload)
        n = int(self._readline())
        return self._readn(n)

    def info(self) -> dict:
        if self._info is None:
            self._send("INFO")
            d = {}
            while True:
                line = self._readline().decode()
                if line == "END":
                    break
                k, v = line.split("=", 1)
                d[k] = [int(x) for x in v.split(",")] if k == "eog" and v else ([] if k == "eog" else int(v))
            self._info = d
        return self._info

    def attrs(self) -> list[tuple[int, int, bool, bytes]]:
        """(id, llama_token_attr bitmask, is_eog, text bytes) for every token."""
        self._send("ATTRS")
        out = []
        while True:
            line = self._readline().decode()
            if line == "END":
                break
            i, a, e, hx = line.split(" ")
            out.append((int(i), int(a), e == "1", bytes.fromhex(hx)))
        return out

    def close(self):
        try:
            self._send("QUIT")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
