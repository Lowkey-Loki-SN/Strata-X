"""eg_reader.py EXE TAG ARM - one arm of the engram-fetch A/B (DSV41_BACKLOG #59) through the whole server.

ARM "reader": STRATA_DSV41_EG_READER=1 (Strata's n-gram reader: a chunk's pages deduplicated and sorted, a bounded number in
flight, a row cache); ARM "mapped": the mapped reads (the default).  Both print each request's first distribution's top 8
(STRATA_X_FIRST_TOP=8): the arms must agree bit for bit.  Extra variables in the environment (STRATA_DSV41_EG_INFLIGHT,
STRATA_IO_THREADS, STRATA_DSV41_EG_CACHE_MIB, STRATA_DSV41_EG_AHEAD, STRATA_DSV41_EG_VERIFY - every row compared with the
mapped tables, slow) reach the engine.  The engram tables are dropped from the page cache first; then a ~12K-token generated text read cold, and
the same text again behind another system prompt (no prefix to reuse: the rows are the ones read before - the reader's row
cache, the mapped reads' page cache), then a follow-up of ~400 new tokens (rows mostly new: the common case).  Prints each request's prompt-path line and, for the reader, the checks' verdict."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

X = Path(_REPO)
S = Path(os.environ.get("STRATA_CHECK_DIR", _os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks')))
exe, tag, arm = sys.argv[1], sys.argv[2], sys.argv[3]
assert arm in ("reader", "mapped")
PORT = 1242
cfg = json.loads((X / "strata-dsv41.json").read_text())
cfg["exe"] = exe
cfg["log"] = str(S / f"egr-{tag}-{arm}-server.log")
cfg["port"] = PORT
cfg.pop("expert_profile_save", None)
a = cfg["args"]
a[a.index("--ssd-cache-gib") + 1] = "30"   # test runs: 30 GiB
env = cfg.setdefault("env", {})
env["STRATA_X_FIRST_TOP"] = "8"
if arm == "reader":
    env["STRATA_DSV41_EG_READER"] = "1"
for k in ("STRATA_DSV41_EG_INFLIGHT", "STRATA_IO_THREADS", "STRATA_DSV41_EG_CACHE_MIB", "STRATA_DSV41_EG_VERIFY", "STRATA_DSV41_EG_AHEAD"):
    if k in os.environ:
        env[k] = os.environ[k]
(S / f"strata-dsv41-egr-{tag}-{arm}.json").write_text(json.dumps(cfg, indent=1))
WORDS = "amber birch cobalt delta ember fjord granite harbor iris juniper kelp lumen marble nectar onyx pine quartz reef".split()


def records(n, seed):
    r = random.Random(seed)
    return "\n".join(f"Entry {i:05d}: crate {r.randint(1, 999)} from {r.choice(WORDS)} dock, {r.randint(1, 80)} units of "
                     f"{r.choice(WORDS)}, inspected by {r.choice(WORDS)} team at shift {r.randint(1, 3)}." for i in range(n))


def chat(messages):
    body = {"model": "deepseek-v4.1-flash", "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"thinking": False},
            "messages": messages}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return (r["choices"][0]["message"].get("content") or ""), time.time() - t0


eg = Path(_os.path.join(_MODELS, 'dsv41-1017/native/engram'))
tables = [str(eg / f) for f in ("L01.weight.bin", "L01.scale.bin", "L14.weight.bin", "L14.scale.bin")]
subprocess.run([_os.path.join(_AI, 'Strata/.venv/bin/python'), str(Path(__file__).with_name("pagecache.py")), "--evict", *tables], check=True)
srv = subprocess.Popen([_os.path.join(_AI, 'Strata/.venv/bin/python'), str(X / "serve/server.py"), "--engine", "strata", "--config",
                        str(S / f"strata-dsv41-egr-{tag}-{arm}.json"), "--port", str(PORT)], cwd=str(X), env=dict(os.environ),
                       stdout=open(S / f"egr-{tag}-{arm}-stdout.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
try:
    for _ in range(600):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read()
            break
        except Exception:
            if srv.poll() is not None:
                raise SystemExit("the server exited: see " + str(S / f"egr-{tag}-{arm}-stdout.log"))
    print(f"server up ({arm})", flush=True)
    text = records(460, 3) + "\nReply with OK only."
    reply, sec = chat([{"role": "system", "content": "You check a shipping ledger."}, {"role": "user", "content": text}])
    print(f"cold: {sec:.1f} s, reply {reply.strip()[:40]!r}", flush=True)
    again = [{"role": "system", "content": "You audit a warehouse ledger for errors."}, {"role": "user", "content": text}]
    reply, sec = chat(again)
    print(f"again: {sec:.1f} s, reply {reply.strip()[:40]!r}", flush=True)
    # a follow-up of ~400 new tokens behind it: the common case, its rows mostly new (the reader's misses)
    follow = again + [{"role": "assistant", "content": reply},
                      {"role": "user", "content": records(15, 11) + "\nAnd these? Reply with OK only."}]
    reply, sec = chat(follow)
    print(f"follow-up: {sec:.1f} s, reply {reply.strip()[:40]!r}", flush=True)
    log = open(S / f"egr-{tag}-{arm}-server.log", errors="replace").read().split("\n")
    for l in log:
        l = re.sub(r"^strata-glm( serve)?:", r"strata-x\1:", l)   # a log of a build before 2026-11-26 (strata-glm)
        if l.startswith("strata-x serve: prompt path:") or re.match(r"^strata-x serve: \d+ prompt tokens", l) or \
                l.startswith("strata-x: prompt path:") or l.startswith("strata-x: engram"):
            print("  " + l[:400], flush=True)
        if l.startswith("strata-x: first-token top"):
            print("  " + l[:400], flush=True)
    checks = [l for l in log if "engram reader check" in l]
    if checks:
        bad = sum(int(re.search(r", (\d+) of \d+ rows unlike", l).group(1)) for l in checks)
        print(f"reader checks: {len(checks)} chunks compared with the mapped tables, {bad} rows unlike them: "
              f"{'PASS' if bad == 0 else 'FAIL'}", flush=True)
finally:
    os.killpg(srv.pid, signal.SIGTERM)
    try:
        srv.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(srv.pid, signal.SIGKILL)
