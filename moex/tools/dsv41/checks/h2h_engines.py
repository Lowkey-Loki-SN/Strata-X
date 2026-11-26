"""h2h_engines.py ENGINE PREFIX - one engine's run of the head-to-head on one GGUF set: DeepSeek-V4.1-Flash-1017 as llama.cpp's
lossless conversion of the release (gguf_convert.py: the experts MXFP4 and the Engram MXFP8 as released, the dense BF16;
files 2-7 and 9 byte-identical to unsloth's UD-Q8_K_XL), the same token ids to each engine through the --serve protocol
they all speak (GEN / T / DONE), greedy, after the page cache of every file of the set is dropped (each starts cold), each
engine in a systemd-run --user scope.  ab_engines.py's turns and readouts (October), on the set instead of our pack.

  ENGINE  strata        strata-x (STRATA_X_EXE, the dev build by default) from the set alone (-m <file 1>): the dense BF16
                        as stored (exact), our native config's switches without the route bias (the routing as the model's)
          strata-q8     the same with STRATA_DSV41_DENSE_Q8=1 (the BF16 projections quantized to Q8_0 as they load: the bits
                        of our daily native config)
          strata-bias   strata-q8 with the config's route bias (lossy: our daily setting, reported apart)
          upstream      upstream Strata's strata-dsv41 (its own build of its latest commit) on its pack_gguf.py index of the set,
                        as its setup plans two cards: --tp 2; no drafter, no vision
          maya          Project Maya's maya-ds4 (its latest commit) on its pack_gguf.py index, as the October run: --tp 2, no
                        drafter, no vision
CTX (262144): every engine's --max-context (upstream tested to 256K; Maya's advice with experts on the SSD is <= 262K).
Every engine starts from the same learned expert profile (ours as is; the others' binary counts converted from it).
H2H_RESERVE_MIB (1536): the VRAM every engine leaves free on each card (theirs: --reserve-mib, which is per card; ours:
--vram-reserve-mib and --vram-reserve-later-mib) - the display card's room.  H2H_RAM_GIB (unset: each engine sizes its own):
the RAM every engine gives the experts - theirs as the RAM store (--ssd-cache, GB), ours as the SSD cache with no RAM copy
(--ssd-cache-gib, the headroom set so nothing is left for a copy).  H2H_EXTRA: more flags for the engine (any engine),
H2H_ENV: its switches (K=V ...).  READY_S (600): a start that takes longer is stopped.

The turns: FIRST_N (16000) tokens of this repo's docs and a question, then TURNS (4) follow-ups, each the conversation so
far and SHORT_N (400) new tokens, DECODE_N (128) generated a turn.  Printed a turn: the read (GEN to the first token), the
decode's rate (the first token to DONE), the drive meanwhile (GB read, ms a read, seconds in the slow mode's state - >= 80
ms a read and < 2,000 reads a second, the drive guard's test), the first tokens.  PREFIX.json keeps it all; PREFIX.log the
engine's own messages."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

ENGINE, PREFIX = sys.argv[1], sys.argv[2]
R = _REPO
SET = pathlib.Path(os.environ.get("SET", _os.path.join(_MODELS, 'dsv41-1017/gguf-mxfp4')))
H2H = pathlib.Path(_os.path.join(_MODELS, 'dsv41-1017/h2h'))
FILE1 = sorted(SET.glob("*-00001-of-*.gguf"))[0]
sys.path[:0] = [R, R + "/tools"]
from serve import families  # noqa: E402

FIRST = int(os.environ.get("FIRST_N", "16000"))
TURNS = int(os.environ.get("TURNS", "4"))
SHORT = int(os.environ.get("SHORT_N", "400"))
N = int(os.environ.get("DECODE_N", "128"))
CTX = os.environ.get("CTX", "262144")
cfg = json.loads(pathlib.Path(R + "/strata-dsv41-native.json").read_text())
tok = families.from_config(cfg).load_tokenizer(pathlib.Path(cfg["tokenizer"]))
git = lambda f: subprocess.run(["git", "-C", R, "show", "645844e9:" + f], capture_output=True, text=True, check=True).stdout
sp = lambda s: tok.encode(s, parse_special=True)
DOCS = ("docs/DETAILS.md", "docs/AI_SETUP.md", "moex/docs/DSV41_PORT.md", "docs/MULTI_GPU.md", "docs/AMD_HIP.md",
        "docs/MCP_SERVER.md", "moex/docs/DSV41_PLAN.md")
docs = [tok.encode(git(f)) for f in DOCS]
long_text = [t for d in docs[:3] + docs[6:] for t in d]
if len(long_text) < FIRST:
    sys.exit("the docs hold %d tokens, fewer than FIRST_N %d" % (len(long_text), FIRST))
first = sp("<｜User｜>") + long_text[:FIRST] + sp("\n\nWhat does the text above describe?<｜Assistant｜></think>")

ENV = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_", "DS4_", "MAYA_"))}
ENV.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1")
for kv in os.environ.get("H2H_ENV", "").split():   # (an engine's own diagnostic switches for a breakdown run, e.g. "DS4_PROF=1")
    k, _, v = kv.partition("=")
    ENV[k] = v
profile_txt = pathlib.Path(cfg["args"][cfg["args"].index("--expert-profile") + 1])


def binary_profile() -> str:
    """Our learned profile as Maya's / upstream's binary counts (uint32 a layer and expert), a fresh copy a run."""
    prof = pathlib.Path(PREFIX + ".profile.bin")
    counts = [0] * (40 * 384)
    for line in profile_txt.read_text().splitlines():
        if line and not line.startswith("#"):
            il, e, c = line.split()
            counts[int(il) * 384 + int(e)] = min(0xFFFFFFFF, int(round(float(c))))
    prof.write_bytes(b"".join(x.to_bytes(4, "little") for x in counts))
    return str(prof)


RESERVE = int(os.environ.get("H2H_RESERVE_MIB", "1536"))
RAM_GIB = float(os.environ["H2H_RAM_GIB"]) if os.environ.get("H2H_RAM_GIB") else None
THEIRS = ["--reserve-mib", str(RESERVE)] + (["--ssd-cache", "%.1f" % (RAM_GIB * 1.073741824)] if RAM_GIB is not None else [])

if ENGINE in ("strata", "strata-q8", "strata-bias"):
    exe = os.environ.get("STRATA_X_EXE", R + "/build-dsv41-dev/strata-x")
    a, args, i = list(cfg["args"]), [], 0
    while i < len(a):   # the set alone: -m its first file; no pack, no expert set of its own
        if a[i] in ("--pack", "--experts-gguf"):
            i += 2
        elif a[i] == "-m":
            args += ["-m", str(FILE1)]
            i += 2
        else:
            args.append(a[i])
            i += 1
    args[args.index("--max-context") + 1] = CTX
    if RAM_GIB is not None:   # the experts' RAM as the SSD cache only: the headroom leaves nothing for a RAM copy
        args[args.index("--ssd-cache-gib") + 1] = "%g" % RAM_GIB
        avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2**20
        args += ["--resident-headroom-gib", "%g" % max(8.0, avail - RAM_GIB)]
    cmd = [exe, "--serve"] + args + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804",
                                     "--image-pad", "129264", "--vram-reserve-mib", str(RESERVE), "--vram-reserve-later-mib",
                                     str(RESERVE)]
    # (H2H_EXTRA: engine flags after the others, e.g. "--vram-reserve-later-mib 512" - the display card left as the others
    #  leave it, ~0.5 GB, where our daily config keeps 2 GB for the desktop)
    env = dict(cfg["env"])
    env.pop("STRATA_DSV41_ENGRAM_GGUF", None)   # (the set's own table)
    if ENGINE != "strata-bias":
        env.pop("STRATA_DSV41_ROUTE_BIAS", None)
    if ENGINE in ("strata-q8", "strata-bias"):
        env["STRATA_DSV41_DENSE_Q8"] = "1"
    ENV.update(env)
elif ENGINE in ("upstream", "upstream-tp1"):
    # as upstream's setup writes it for discrete cards (setup.py ~9494: --eos 1, --vision, --dspark auto - none here: the set
    # has no drafter - and dsv41_fit's --tp 2 for two cards); -tp1: the layer split, Maya's calibration's other candidate
    cmd = [os.environ.get("UPSTREAM", _os.path.join(_AI, 'strata-upstream-h2h/build-dsv41/strata-dsv41')), "--serve",
           "--pack", str(H2H / "upstream-pack"), "--max-context", CTX, "--eos", "1", "--tp", "1" if ENGINE.endswith("tp1") else "2",
           "--vision", "off", "--dspark", "none", "--expert-profile", binary_profile()] + THEIRS
elif ENGINE in ("maya", "maya-tp1"):
    # as Maya's setup writes it (maya.py ~5770: --tp min(2, cards), --vision off, --dspark none); -tp1: the layer split its
    # calibration (tools/calibrate_ds4_tp.py) measures beside --tp 2 on up to four cards (the measured decode decides)
    cmd = [os.environ.get("MAYA", _os.path.join(_AI, 'maya-h2h/build/maya-ds4')), "--serve", "--pack", str(H2H / "maya-pack"),
           "--max-context", CTX, "--tp", "1" if ENGINE.endswith("tp1") else "2", "--vision", "off", "--dspark", "none",
           "--expert-profile", binary_profile()] + THEIRS
elif ENGINE == "dry":
    sys.exit("first turn %d tokens; follow-up docs %s tokens; the set %s" % (len(first), [len(d) for d in docs[3:6]], FILE1))
else:
    sys.exit("ENGINE: strata, strata-q8, strata-bias, upstream, upstream-tp1, maya or maya-tp1")

cmd += os.environ.get("H2H_EXTRA", "").split()   # (any engine's extra flags, last)

# every file of the set from the drive: its page cache dropped
for p in sorted(SET.glob("*.gguf")):
    fd = os.open(os.path.realpath(p), os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def disk():
    v = [int(x) for x in open("/sys/block/nvme0n1/stat").read().split()]
    return v[0], v[2], v[3]   # reads completed, sectors read, the reads' milliseconds


slow_s = [0]
stop = threading.Event()


def watch():   # (the drive guard's own test, a second at a time)
    a = disk()
    while not stop.wait(1.0):
        b = disk()
        n, ms = b[0] - a[0], b[2] - a[2]
        if n > 0 and ms / n >= 80 and n < 2000:
            slow_s[0] += 1
        a = b


threading.Thread(target=watch, daemon=True).start()
log = open(PREFIX + ".log", "w")
t_start = time.time()
p = subprocess.Popen(["systemd-run", "--user", "--scope", "-q"] + cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=log, text=True, bufsize=1, env=ENV)
t_ready = threading.Timer(float(os.environ.get("READY_S", "600")), p.kill)   # (a start that stalls is stopped)
t_ready.start()
for line in p.stdout:
    if line.startswith("READY"):
        break
else:
    sys.exit("%s: the engine ended before READY (%s.log)" % (ENGINE, PREFIX))
t_ready.cancel()
ready = time.time() - t_start
print("%s: ready in %.1f s" % (ENGINE, ready), flush=True)
out = {"engine": ENGINE, "cmd": cmd, "env": {k: v for k, v in ENV.items() if k.startswith(("STRATA_", "DS4_", "MAYA_"))},
       "ready_s": ready, "turns": []}
conv = first
for t in range(TURNS + 1):
    if t > 0:
        d, at = docs[3 + (t - 1) % 3], (t - 1) // 3 * SHORT
        conv = conv + sp("<｜User｜>") + d[at:at + SHORT] + sp("\n\nAnd this part: what does it add?<｜Assistant｜></think>")
    d0, s0 = disk(), slow_s[0]
    t0 = time.time()
    p.stdin.write("GEN %d temperature=0 %s\n" % (N, ",".join(map(str, conv))))
    p.stdin.flush()
    toks, t1, done, stat = [], None, "", ""
    for line in p.stdout:
        if line.startswith("T "):
            if t1 is None:
                t1 = time.time()
            toks.append(int(line.split()[1]))
        elif line.startswith("STAT"):   # (upstream's / Maya's per-answer counters, every 8 tokens: the last one is the turn's)
            stat = line.strip()
        elif line.startswith("DONE") or line.startswith("ERR"):
            done = line.strip()
            break
    t2 = time.time()
    d1 = disk()
    n_r = d1[0] - d0[0]
    turn = {"prompt": len(conv), "read_s": (t1 or t2) - t0, "decode_tok_s": (len(toks) - 1) / (t2 - t1) if t1 and len(toks) > 1
            else 0.0, "generated": len(toks), "drive_gb": (d1[1] - d0[1]) * 512 / 1e9, "drive_reads": n_r,
            "ms_a_read": (d1[2] - d0[2]) / n_r if n_r else 0.0, "slow_s": slow_s[0] - s0, "done": done, "tokens": toks,
            "stat": stat}
    out["turns"].append(turn)
    print("  turn %d: %6d tokens, read %6.1f s; decode %5.2f tok/s (%d); drive %6.2f GB, %.2f ms a read, slow %d s; %s" % (
        t, len(conv), turn["read_s"], turn["decode_tok_s"], len(toks), turn["drive_gb"], turn["ms_a_read"], turn["slow_s"],
        ",".join(map(str, toks[:12]))), flush=True)
    if stat:
        print("          " + stat, flush=True)
    conv = conv + toks
p.stdin.write("QUIT\n")
p.stdin.flush()
try:
    p.wait(timeout=300)
except subprocess.TimeoutExpired:
    p.kill()
stop.set()
log.close()
pathlib.Path(PREFIX + ".json").write_text(json.dumps(out))
