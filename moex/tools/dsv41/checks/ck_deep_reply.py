"""ck_deep_reply.py EXE TAG - DSV41_BACKLOG #62 through the whole server: a mark of --prompt-cache-every that a REPLY
crosses gets a deep checkpoint at the next request (a snapshot of the live state there); before, only marks inside a
batched prompt read got one.  A 4K-token system prompt and a first message to ~65,000 tokens; the second turn asks for a
reply long enough to carry the conversation past 65,536; eight short turns push the early turn checkpoints out of the
LRU; then the conversation is sent again with the third user message changed.  It must resume at the third request's
reuse point (the snapshot, past 65,536) - without #62 it resumed at the root.  The texts are generated: nobody's data."""
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
PORT = 1243
cfg = json.loads((X / "strata-dsv41.json").read_text())
cfg["exe"] = exe
cfg["log"] = str(S / f"ckreply-{tag}-server.log")
cfg["port"] = PORT
cfg.pop("expert_profile_save", None)
a = cfg["args"]
a[a.index("--ssd-cache-gib") + 1] = "30"   # test runs: 30 GiB
(S / f"strata-dsv41-ckreply-{tag}.json").write_text(json.dumps(cfg, indent=1))
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
    r = random.Random(2000 + k)
    return f"Note {k}{' (revised)' if revised else ''}: {' '.join(r.choice(WORDS) for _ in range(40))}. Reply with OK only."


def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def chat(messages, max_tokens=8):
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": "deepseek-v4.1-flash", "max_tokens": max_tokens, "temperature": 0,
                                      "chat_template_kwargs": {"thinking": False}, "messages": messages})
    return (r["choices"][0]["message"].get("content") or ""), time.time() - t0


def count(system, user):
    return post("/v1/messages/count_tokens", {"model": "deepseek-v4.1-flash", "system": system,
                                              "messages": [{"role": "user", "content": user}]})["input_tokens"]


def lines():
    pat = re.compile(r"^strata-(?:x|glm) serve: (\d+) prompt tokens \((\d+) reused, (\d+) read\) in ([\d.]+) s, (\d+) generated")
    return [m.groups() for m in (pat.match(l) for l in open(S / f"ckreply-{tag}-server.log", errors="replace")) if m]


srv = subprocess.Popen([_os.path.join(_AI, 'Strata/.venv/bin/python'), str(X / "serve/server.py"), "--engine", "strata", "--config",
                        str(S / f"strata-dsv41-ckreply-{tag}.json"), "--port", str(PORT)], cwd=str(X), env=dict(os.environ),
                       stdout=open(S / f"ckreply-{tag}-stdout.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
try:
    for _ in range(600):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read()
            break
        except Exception:
            if srv.poll() is not None:
                raise SystemExit("the server exited: see " + str(S / f"ckreply-{tag}-stdout.log"))
    print("server up", flush=True)
    system = "You keep a maintenance log. The tools you may call:\n" + tools_text(150)
    per = (count(system, records(1000, 1)) - count(system, "x")) / 1000.0
    nrec = int((64900 - count(system, "x")) / per)
    for _ in range(3):   # the first prompt at 64,800-65,000 tokens: the second turn's reply crosses 65,536
        got = count(system, records(nrec, 1))
        if 64800 <= got <= 65000:
            break
        nrec += int((64900 - got) / per)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": records(nrec, 1) + "\nReply with OK only."}]
    reply, sec = chat(msgs)
    print(f"turn 1: {sec:.1f} s, reply {reply.strip()[:30]!r}", flush=True)
    msgs += [{"role": "assistant", "content": reply},
             {"role": "user", "content": "Count from 1 to 400, one number on each line, nothing else."}]
    reply, sec = chat(msgs, max_tokens=1200)
    print(f"turn 2 (the long reply): {sec:.1f} s, {len(reply)} chars", flush=True)
    msgs += [{"role": "assistant", "content": reply}]
    for k in range(3, 12):
        msgs.append({"role": "user", "content": note(k)})
        reply, sec = chat(msgs)
        print(f"turn {k}: {sec:.1f} s, reply {reply.strip()[:30]!r}", flush=True)
        msgs.append({"role": "assistant", "content": reply})
    got = lines()
    for i, (n, ru, rd, sec, gen) in enumerate(got, 1):
        print(f"  request {i}: {n} prompt tokens, {ru} reused, {rd} read in {sec} s, {gen} generated", flush=True)
    live2 = int(got[1][0]) + int(got[1][4])   # the conversation's end after turn 2's reply (roughly: its prompt + reply)
    snap = int(got[2][1])                     # request 3 continued the live tokens: its reuse point is the snapshot
    if live2 <= 65536 or int(got[1][0]) > 65536:
        raise SystemExit(f"turn 2's reply did not cross 65,536 (its prompt {got[1][0]}, ~{live2} after it): resize")
    # msgs: [system, user 1, assistant 1, user 2, assistant 2, user 3, ...]: user 3 sits at index 5
    assert msgs[5]["role"] == "user" and msgs[5]["content"].startswith("Note 3:")
    final = msgs[:5] + [{"role": "user", "content": note(3, revised=True)}]
    reply, sec = chat(final)
    n, ru, rd, ps, gen = lines()[-1]
    ok = int(ru) == snap and int(ru) > 65536
    print(f"final: user 3 changed - {n} prompt tokens, {ru} reused, {rd} read in {ps} s; request 3 reused {snap}: "
          f"{'PASS - resumed at the snapshot past 65,536' if ok else 'FAIL'}", flush=True)
finally:
    os.killpg(srv.pid, signal.SIGTERM)
    try:
        srv.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(srv.pid, signal.SIGKILL)
