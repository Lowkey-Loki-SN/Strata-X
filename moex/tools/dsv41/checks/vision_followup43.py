"""vision_followup43.py EXE TAG - DSV41_BACKLOG #43 through the whole server: a picture question with thinking on (no tools: the
template drops a past turn's reasoning, so the live tokens stop being a prefix of the follow-up), then a follow-up turn.  Before
the fix a request with a picture made no checkpoint, so the follow-up read the whole conversation again from 0; with it the
follow-up resumes at the first request's last-turn checkpoint.  Prints each request's engine line and the answers.
STRATA_CHECK_TEXT_FIRST=1 puts the question before the picture: the turn then ends with the picture, whose span end
carries the last-turn cut (DSV41_BACKLOG #63).  The follow-up must reuse up to it there too."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
from serve_pic import picture_url

X = Path(_REPO)
S = Path(os.environ.get("STRATA_CHECK_DIR", _os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks')))
exe, tag = sys.argv[1], sys.argv[2]
cfg = json.loads((X / "strata-dsv41.json").read_text())
cfg["exe"] = exe
cfg["log"] = str(S / f"v43-{tag}-server.log")
cfg["port"] = 1240
cfg.pop("expert_profile_save", None)
(S / f"strata-dsv41-v43-{tag}.json").write_text(json.dumps(cfg, indent=1))
url = picture_url(S / "v43-number.png")

srv = subprocess.Popen([_os.path.join(_AI, 'Strata/.venv/bin/python'), str(X / "serve/server.py"), "--engine", "strata", "--config",
                        str(S / f"strata-dsv41-v43-{tag}.json"), "--port", "1240"], cwd=str(X), env=dict(os.environ),
                       stdout=open(S / f"v43-{tag}-stdout.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)


def ask(messages, max_tokens):
    body = {"model": "deepseek-v4.1-flash", "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "low",
            "messages": messages}
    req = urllib.request.Request("http://127.0.0.1:1240/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    msg = r["choices"][0]["message"]
    return msg, time.time() - t0


def engine_lines():
    return [l.strip() for l in open(S / f"v43-{tag}-server.log", errors="replace") if "prompt tokens (" in l]


try:
    for _ in range(600):
        time.sleep(2)
        try:
            urllib.request.urlopen("http://127.0.0.1:1240/v1/models", timeout=5).read()
            break
        except Exception:
            if srv.poll() is not None:
                raise SystemExit("the server exited: see " + str(S / f"v43-{tag}-stdout.log"))
    print("server up", flush=True)
    pic, q = {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "What number is written in this picture?"}
    turn1 = [{"role": "user", "content": [q, pic] if os.environ.get("STRATA_CHECK_TEXT_FIRST") == "1" else [pic, q]}]
    m1, t1 = ask(turn1, 600)
    print(f"turn 1: {t1:.1f} s, answer {(m1.get('content') or '').strip()[:160]!r}", flush=True)
    turn2 = turn1 + [{"role": "assistant", "content": m1.get("content") or "", "reasoning_content": m1.get("reasoning_content") or ""},
                     {"role": "user", "content": "And what is that number plus one? Answer with the number only."}]
    m2, t2 = ask(turn2, 600)
    print(f"turn 2: {t2:.1f} s, answer {(m2.get('content') or '').strip()[:160]!r}", flush=True)
    for line in engine_lines():
        print("  " + line[:220], flush=True)
finally:
    os.killpg(srv.pid, signal.SIGTERM)
    try:
        srv.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(srv.pid, signal.SIGKILL)
