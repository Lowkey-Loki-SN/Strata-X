"""serve_pic_lm.py EXE TAG - DSV41_BACKLOG #76 with a picture, through the whole server: a request whose text part spans
several prompt chunks (a synthetic ~20K-token list), then the picture of "7319" and a question - read with the layer-major
switch off and then on (each a fresh server from a copy of strata-dsv41.json with diagnostics on, one tier: --no-adaptive
--expert-slots 1800).  The text part goes through the layer-major windows, the picture's span as before (it is read whole);
the first token's logits (the probe's hash) and the answer must be the same."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path
from serve_pic import picture_url

X = Path(_REPO)
S = Path(os.environ.get("STRATA_CHECK_DIR", _os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks')))
exe, tag = sys.argv[1], sys.argv[2]
url = picture_url(S / "v43-number.png")
words = ["amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kestrel", "lagoon"]
LONG = "\n".join(f"Entry {i:05d}: {words[i % 12]} {words[(i * 7) % 12]} {words[(i * 5 + 3) % 12]} - "
                 f"code {(i * 7919) % 100000:05d}, weight {(i * 37) % 997}.{i % 10}" for i in range(1400))
res = {}
for arm, lm in (("off", False), ("on", True)):
    cfg = json.loads((X / "strata-dsv41.json").read_text())
    cfg["exe"] = exe
    cfg["log"] = str(S / f"piclm-{tag}-{arm}-server.log")
    cfg["port"] = 1241
    cfg["diagnostics"] = True
    cfg.pop("expert_profile_save", None)
    cfg["args"] = cfg["args"] + ["--no-adaptive", "--expert-slots", "1800"]
    env = dict(cfg.get("env", {}), STRATA_X_FIRST_TOP="8")
    env.pop("STRATA_DSV41_LAYER_MAJOR", None)
    if lm:
        env.update(STRATA_DSV41_LAYER_MAJOR="1", STRATA_DSV41_BOUNCE="96,32")
    cfg["env"] = env
    conf = S / f"strata-dsv41-piclm-{tag}-{arm}.json"
    conf.write_text(json.dumps(cfg, indent=1))
    srv = subprocess.Popen([_os.path.join(_AI, 'Strata/.venv/bin/python'), str(X / "serve/server.py"), "--engine", "strata", "--config",
                            str(conf), "--port", "1241"], cwd=str(X), env=dict(os.environ),
                           stdout=open(S / f"piclm-{tag}-{arm}-stdout.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(900):
            time.sleep(2)
            try:
                urllib.request.urlopen("http://127.0.0.1:1241/v1/models", timeout=5).read()
                break
            except Exception:
                if srv.poll() is not None:
                    raise SystemExit("the server exited: see " + str(S / f"piclm-{tag}-{arm}-stdout.log"))
        body = {"model": "deepseek-v4.1-flash", "max_tokens": 48, "temperature": 0, "reasoning_effort": "low",
                "messages": [{"role": "user", "content": [{"type": "text", "text": LONG}, {"type": "image_url", "image_url": {"url": url}},
                                                          {"type": "text", "text": "What number is written in this picture?"}]}]}
        req = urllib.request.Request("http://127.0.0.1:1241/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        dt = time.time() - t0
        msg = r["choices"][0]["message"]
        log = Path(cfg["log"]).read_text(errors="replace")
        h = re.findall(r"first-token top at position (\d+):[^|\n]*\| logits ([0-9a-f]{16})", log)
        lmline = [l.strip() for l in log.splitlines() if "layer-major:" in l]
        res[arm] = (h[-1] if h else None, (msg.get("content") or "").strip())
        print(f"{arm}: {dt:.1f} s, usage {r.get('usage')}, first token at/logits {res[arm][0]}, answer {res[arm][1][:120]!r}", flush=True)
        for l in lmline:
            print("   ", l[:300], flush=True)
    finally:
        os.killpg(srv.pid, signal.SIGTERM)
        srv.wait(timeout=120)
same = res["off"] == res["on"] and res["on"][0] is not None
print(f"{'PASS' if same else 'FAIL'}: a picture request whose text spans several chunks - the first token's logits and the answer "
      f"the same with the layer-major windows", flush=True)
