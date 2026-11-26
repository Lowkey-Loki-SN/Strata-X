"""ck_deep.py EXE TAG - the deep checkpoints (src/x/session.cpp plan_cuts, --prompt-cache-deep) through the whole server.

A conversation with a 4K-token system prompt (the root lands at its end) and a first message that brings it to ~65,200 tokens,
then twelve short turns that carry it past 65,536 one turn at a time.  The engine's periodic checkpoints used to be counted
from each read's start, so this conversation got none; now the turn whose read crosses 65,536 cuts there.  Twelve turns also
push the early turn checkpoints out of the LRU (--prompt-cache 6).  Last, the conversation is sent with the message right
after the crossing changed: it must resume at 65,536 - the old policy resumed at the root and read ~61K tokens again.

The texts are generated (numbered records, tool names, notes): nobody's data.  Thinking off, 8 tokens a reply.  Prints each
request's engine line and the verdict."""
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
exe, tag = sys.argv[1], sys.argv[2]
PORT = 1241
cfg = json.loads((X / "strata-dsv41.json").read_text())
cfg["exe"] = exe
cfg["log"] = str(S / f"ckdeep-{tag}-server.log")
cfg["port"] = PORT
cfg.pop("expert_profile_save", None)
a = cfg["args"]
a[a.index("--ssd-cache-gib") + 1] = "30"   # test runs: 30 GiB
(S / f"strata-dsv41-ckdeep-{tag}.json").write_text(json.dumps(cfg, indent=1))
WORDS = "amber birch cobalt delta ember fjord granite harbor iris juniper kelp lumen marble nectar onyx pine quartz reef".split()


def records(n, seed):
    r = random.Random(seed)
    return "\n".join(f"Record {i:05d}: unit {r.randint(1, 99)} at {r.randint(0, 23):02d}:{r.randint(0, 59):02d} read "
                     f"{r.uniform(0, 10):.3f} V, state {r.choice(WORDS)}, next check {r.choice(WORDS)}-{r.randint(100, 999)}."
                     for i in range(n))


def tools_text(n):
    r = random.Random(7)
    return "\n".join(f"- tool_{i:03d}({r.choice(WORDS)}: str, {r.choice(WORDS)}: int): returns the {r.choice(WORDS)} of a "
                     f"{r.choice(WORDS)} record, or an error when the {r.choice(WORDS)} is missing." for i in range(n))


def note(k, revised=False):
    r = random.Random(1000 + k)
    body = " ".join(r.choice(WORDS) for _ in range(90))
    return f"Note {k}{' (revised)' if revised else ''}: {body}. Reply with OK only."


def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def chat(messages):
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": "deepseek-v4.1-flash", "max_tokens": 8, "temperature": 0,
                                      "chat_template_kwargs": {"thinking": False}, "messages": messages})
    return (r["choices"][0]["message"].get("content") or ""), time.time() - t0


def count(system, user):
    return post("/v1/messages/count_tokens", {"model": "deepseek-v4.1-flash", "system": system,
                                              "messages": [{"role": "user", "content": user}]})["input_tokens"]


def lines():
    pat = re.compile(r"^strata-(?:x|glm) serve: (\d+) prompt tokens \((\d+) reused, (\d+) read\) in ([\d.]+) s")
    return [m.groups() for m in (pat.match(l) for l in open(S / f"ckdeep-{tag}-server.log", errors="replace")) if m]


srv = subprocess.Popen([_os.path.join(_AI, 'Strata/.venv/bin/python'), str(X / "serve/server.py"), "--engine", "strata", "--config",
                        str(S / f"strata-dsv41-ckdeep-{tag}.json"), "--port", str(PORT)], cwd=str(X), env=dict(os.environ),
                       stdout=open(S / f"ckdeep-{tag}-stdout.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
try:
    for _ in range(600):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read()
            break
        except Exception:
            if srv.poll() is not None:
                raise SystemExit("the server exited: see " + str(S / f"ckdeep-{tag}-stdout.log"))
    print("server up", flush=True)
    system = "You keep a maintenance log. The tools you may call:\n" + tools_text(150)
    per = (count(system, records(1000, 1)) - count(system, "x")) / 1000.0
    nrec = int((65200 - count(system, "x")) / per)
    for _ in range(3):   # land the first prompt at 65,100-65,300 tokens
        got = count(system, records(nrec, 1))
        if 65100 <= got <= 65300:
            break
        nrec += int((65200 - got) / per)
    print(f"system prompt {count(system, 'x')} tokens; first message {nrec} records, {count(system, records(nrec, 1))} tokens",
          flush=True)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": records(nrec, 1) + "\nReply with OK only."}]
    for k in range(1, 14):
        reply, sec = chat(msgs)
        print(f"turn {k}: {sec:.1f} s, reply {reply.strip()[:40]!r}", flush=True)
        msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": note(k + 1)}]
    msgs.pop()   # the conversation ends with turn 13's reply
    got = lines()
    for i, (n, ru, rd, sec) in enumerate(got, 1):
        print(f"  request {i}: {n} prompt tokens, {ru} reused, {rd} read in {sec} s", flush=True)
    c = next(i for i, g in enumerate(got, 1) if int(g[0]) - 1 > 65536)   # the request whose read crossed 65,536
    if int(got[c - 1][1]) >= 65536 or c > 7:
        raise SystemExit(f"the crossing fell in request {c} (reused {got[c - 1][1]}): resize the first message")
    # the message right after the crossing changed: request c's own turn checkpoint is out of the LRU, the later ones lie
    # past the change - only the deep checkpoint at 65,536 (or the root) is a prefix
    # msgs: [system, user 1, assistant 1, user 2, ...]: user c+1 sits at index 2c+1
    assert msgs[2 * c + 1]["role"] == "user" and msgs[2 * c + 1]["content"].startswith(f"Note {c + 1}:")
    final = msgs[: 2 * c + 1] + [{"role": "user", "content": note(c + 1, revised=True)}]
    reply, sec = chat(final)
    n, ru, rd, ps = lines()[-1]
    ok = int(ru) == 65536
    print(f"final: user {c + 1} changed - {n} prompt tokens, {ru} reused, {rd} read in {ps} s ({sec:.1f} s, reply "
          f"{reply.strip()[:40]!r}): {'PASS - resumed at the deep checkpoint' if ok else 'FAIL - expected 65536 reused'}", flush=True)
finally:
    os.killpg(srv.pid, signal.SIGTERM)
    try:
        srv.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(srv.pid, signal.SIGKILL)
