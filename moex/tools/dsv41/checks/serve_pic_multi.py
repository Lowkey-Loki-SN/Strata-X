"""serve_pic_multi.py LOG_PREFIX EXE PIC_DIR [MANIFEST=1] - pictures read with their text (DSV41_BACKLOG #113) past one picture a request:
PIC_DIR holds serve_pic_merge.py's encoded pictures (num.sve: "7319", 206 rows; shot.sve: a 1920x1080 screenshot, 968 rows).
One engine (strata-dsv41.json's arguments and switches through the serve protocol), greedy: (a) a turn with the screenshot
and then the number picture, asked for the number; (b) a follow-up that reuses (a) - its two pictures in the prefix - and
brings the number picture again; (c) a text-only follow-up after (b).  Each request carries every picture of its prompt (the
records in order).  Prints each read's tokens (reused / read), its time and the answer."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families  # noqa: E402

prefix_log, exe, pics = sys.argv[1], sys.argv[2], Path(sys.argv[3])
R = _REPO
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
sp = lambda s: tok.encode(s, parse_special=True)
MANIFEST = __import__("os").environ.get("MANIFEST") == "1"   # moex (§56 A): the pictures by manifest, not one combined file
rows = {k: int((pics / f"{k}.rows").read_text()) for k in ("num", "shot")}
work = Path(f"{prefix_log}.d")
work.mkdir(parents=True, exist_ok=True)


def sve(*names):   # the pictures' records in the prompt's order
    if MANIFEST:   # moex (DSV41_PLAN §56 A): a GENI manifest naming the records' own files, as the server writes it
        from serve.server import write_manifest
        out = work / ("_".join(names) + ".svm")
        write_manifest(out, [(pics / f"{n}.sve", rows[n], 0) for n in names])
        return str(out.resolve())
    out = work / ("_".join(names) + ".sve")
    out.write_bytes(b"".join((pics / f"{n}.sve").read_bytes() for n in names))
    return str(out.resolve())


import os  # noqa: E402
ENV = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_DSV4", "STRATA_ARENA", "STRATA_X_STAGE", "STRATA_POOL"))}
ENV.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_QUIET="0")
ENV.update(cfg["env"])
ARGS = ["--serve"] + cfg["args"] + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--image-pad", "129264"]
log = f"{prefix_log}.log"
err = open(log, "w")
p = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", exe] + ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                     text=True, bufsize=1, env=ENV)
for line in p.stdout:
    if line.startswith("READY"):
        break


def gen(ids, emb, n=8):
    mark = len(Path(log).read_text(errors="replace"))
    p.stdin.write(f"GENI {n} temperature=0 {emb} " + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    toks, done = [], ""
    for line in p.stdout:
        if line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith("DONE") or line.startswith("ERR"):
            done = line.strip()
            break
    err.flush()
    s = Path(log).read_text(errors="replace")[mark:]
    head = (re.findall(r"serve: (\d+ prompt tokens \(\d+ reused, \d+ read\) in [0-9.]+ s)", s) or ["-"])[-1]
    return toks, head, done if done.startswith("ERR") else ""


PAD = 129264
ask = "\n\nWhat number is shown in the {}? Answer with the number only.<｜Assistant｜></think>"
a = sp("<｜User｜>") + [PAD] * rows["shot"] + sp("\n\n") + [PAD] * rows["num"] + sp(ask.format("second picture"))
ta, ha, ea = gen(a, sve("shot", "num"))
print(f"(a) two pictures in one turn: {ha}; the answer {tok.decode(ta)!r} {ea}", flush=True)
b = a + ta + sp("<｜User｜>") + [PAD] * rows["num"] + sp(ask.format("picture above"))
tb, hb, eb = gen(b, sve("shot", "num", "num"))
print(f"(b) a follow-up with both in its prefix and a third picture: {hb}; the answer {tok.decode(tb)!r} {eb}", flush=True)
c = b + tb + sp("<｜User｜>Which number did the last picture show? Answer with the number only.<｜Assistant｜></think>")
tc, hc, ec = gen(c, sve("shot", "num", "num"))
print(f"(c) a text-only follow-up: {hc}; the answer {tok.decode(tc)!r} {ec}", flush=True)
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=180)
err.close()
