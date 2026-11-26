"""serve_smoke.py OUT_DIR - strata-dsv41.json as it stands, through the whole server (port 1235, its own engine log in OUT_DIR:
the live server's log is untouched): the start, one short chat request streamed (a question about this repo's AGENTS.md), one
with a picture (a generated gradient, as the encoder's warm-up draws), the stop. Pass: the server gets ready, both replies
stream, and the engine log names the drive guard, the scales' pre-read and no error."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
import base64, io, json, os, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
R = _REPO
cfg = json.loads(Path(os.environ.get("CONFIG") or f"{R}/strata-dsv41.json").read_text())   # CONFIG=FILE: another config
cfg.update(port=1235, log=str(out / "engine.log"), open_browser=False, diagnostics=True)   # (its own log, whatever the config says)
(out / "config.json").write_text(json.dumps(cfg, indent=1))
t0 = time.monotonic()
srv = subprocess.Popen([_os.path.join(_AI, 'Strata/.venv/bin/python'), f"{R}/serve/server.py", "--engine", "strata", "--config",
                        str(out / "config.json"), "--port", "1235"], cwd=R, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       bufsize=1)
console, ready = [], threading.Event()


def read():
    for line in srv.stdout:
        console.append(line.rstrip("\n"))
        if line.startswith("ready:"):
            ready.set()


threading.Thread(target=read, daemon=True).start()
ok_ready = ready.wait(600)
t_ready = time.monotonic() - t0


def ask(content, max_tokens):   # -> (the streamed text, seconds to its first piece)
    req = urllib.request.Request("http://127.0.0.1:1235/v1/chat/completions", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": cfg["model_name"], "stream": True, "max_tokens": max_tokens,
                                                  "messages": [{"role": "user", "content": content}]}).encode())
    t1, got, first = time.monotonic(), "", None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if line.startswith("data: {"):
                d = json.loads(line[6:])
                ch = (d.get("choices") or [{}])[0].get("delta") or {}
                piece = (ch.get("content") or "") + (ch.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.monotonic() - t1
                got += piece
    return got, first


text, first, ptext, pfirst = "", None, "", None
if ok_ready:
    text, first = ask("In two sentences, what is this repo?\n\n" + (Path(R) / "AGENTS.md").read_text(), 200)
    from PIL import Image
    import numpy as np
    g = np.add.outer(np.arange(544, dtype=np.uint16), np.arange(544, dtype=np.uint16)) % 256
    buf = io.BytesIO()
    Image.fromarray(np.stack([g, g[::-1], g.T], axis=-1).astype(np.uint8), "RGB").save(buf, "PNG")
    ptext, pfirst = ask([{"type": "text", "text": "What colours does this picture have?"},
                         {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}], 64)
srv.send_signal(signal.SIGINT)
try:
    srv.wait(timeout=120)
except subprocess.TimeoutExpired:
    srv.kill()
log = (out / "engine.log").read_text(errors="replace") if (out / "engine.log").exists() else ""
(out / "console.txt").write_text("\n".join(console) + "\n")
checks = {
    "the server got ready": ok_ready,
    "the reply streamed": bool(text.strip()),
    "the picture's reply streamed": bool(ptext.strip()),
    "the drive guard is on": "the drive guard on (" in log,
    # (moex: with the engram from a GGUF file - STRATA_DSV41_ENGRAM_GGUF - each row carries its scales: nothing to pre-read)
    "the scale tables were pre-read": "scale tables read into the page cache" in log or
                                      "the scales are inside the GGUF's rows" in log,
    "no engine error": not any(w in log for w in ("ERR ", "failed", "error:", "CUDA error")),
}
print(f"  ready {t_ready:.0f} s after the start; the first reply text {first if first is None else round(first, 1)} s after the request; "
      f"{len(text)} characters; the picture's {pfirst if pfirst is None else round(pfirst, 1)} s, {len(ptext)} characters", flush=True)
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
print("ALL PASS" if all(checks.values()) else "SOME FAILED", flush=True)
