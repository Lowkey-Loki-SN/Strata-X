"""serve_restart_e2e.py OUTDIR - DSV41_BACKLOG #53's restart half through the real server (DSV41_PLAN §66, check 2).

A copy of strata-dsv41.json (diagnostics on, the log, the conversation folder and the learned profile in OUTDIR, port 1299;
the installed engine) runs a ~70K-token conversation over the OpenAI endpoint (greedy, a short reply, a tool declared as
dsh declares them, so the template keeps the reply's reasoning), then SIGINT to the server's process group - what Ctrl+C in
its terminal sends - then the server again and the follow-up.  Passes when the engine saved the conversation on the stop,
read it back for the follow-up and reused all of it (the identity held across two starts of the real server), the
follow-up taking seconds.  Nothing outside OUTDIR is written (the learned profile is a copy).
"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, re, shutil, signal, subprocess, sys, time, urllib.request
from pathlib import Path

R = Path(_REPO)
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
PORT = 1299
cfg = json.loads((R / "strata-dsv41.json").read_text())
prof = out / "profile-learned.txt"
shutil.copy(cfg["expert_profile_save"], prof)   # exists at both starts, as the server's own does
cfg.update(diagnostics=True, log=str(out / "engine.log"), conversation_dir=str(out / "conversations"), port=PORT,
           expert_profile_save=str(prof), open_browser=False)
(out / "config.json").write_text(json.dumps(cfg, indent=1))
docs = "".join(subprocess.run(["git", "-C", str(R), "show", f"645844e9:{f}"], capture_output=True, text=True, check=True).stdout
               for f in ("moex/docs/DSV41_PLAN.md", "docs/DETAILS.md", "docs/AI_SETUP.md", "moex/docs/DSV41_PORT.md"))
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]


def start(tag):
    srv = open(out / f"server-{tag}.out", "w")
    p = subprocess.Popen([sys.executable, str(R / "serve/server.py"), "--engine", "strata", "--config", str(out / "config.json"),
                          "--port", str(PORT)], cwd=str(R), stdout=srv, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            sys.exit(f"server {tag} ended at start (exit {p.returncode}); see {srv.name}")
        try:   # GET /health says "loaded": true once the engine is up
            if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).read()).get("loaded"):
                break
        except Exception:
            pass
        time.sleep(3)
    print(f"server {tag}: up in {time.time() - t0:.0f} s", flush=True)
    return p


def chat(messages, max_tokens):
    req = {"model": cfg["model_name"], "messages": messages, "tools": TOOLS, "max_tokens": max_tokens, "temperature": 0,
           "stream": False}
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                                      data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}),
                               timeout=3600)
    return json.loads(r.read()), time.time() - t0


def log_since(mark):
    return (out / "engine.log").read_text(errors="replace")[mark:]


def stop(p, tag):
    t0 = time.time()
    os.killpg(p.pid, signal.SIGINT)   # Ctrl+C in the server's terminal: the server and the engine both get it
    p.wait(timeout=300)
    print(f"server {tag}: ended {time.time() - t0:.1f} s after SIGINT (exit {p.returncode})", flush=True)


fails = []
user1 = docs[:240000] + "\n\nIn one sentence: what is this plan about?"
msgs = [{"role": "system", "content": "You are a careful assistant."}, {"role": "user", "content": user1}]
p = start("1")
mark = len((out / "engine.log").read_text(errors="replace"))
r1, s1 = chat(msgs, 32)
u1 = r1.get("usage", {})
m1 = r1["choices"][0]["message"]
print(f"request 1: {u1.get('prompt_tokens')} prompt tokens in {s1:.0f} s; finish {r1['choices'][0].get('finish_reason')}", flush=True)
stop(p, "1")
saved = re.findall(r"conversation cache: saved (\d+) tokens \(([0-9]+) MiB, (\d+) checkpoints\) in (\d+) ms", log_since(mark))
print(f"  saved on the stop: {saved}", flush=True)
if not saved:
    fails.append("nothing saved on SIGINT")

follow = msgs + [{"role": "assistant", "content": m1.get("content") or "", "reasoning_content": m1.get("reasoning_content") or ""},
                 {"role": "user", "content": "And which part of it reads the experts?"}]
p = start("2")
mark = len((out / "engine.log").read_text(errors="replace"))
r2, s2 = chat(follow, 16)
new = log_since(mark)
back = re.findall(r"conversation cache: read (\d+) tokens back from disk[^\n]*", new)
reused = re.findall(r"strata-(?:x|glm) serve: (\d+) prompt tokens \((\d+) reused, (\d+) read\) in ([0-9.]+) s", new)
dropped = re.findall(r"conversation cache: [^\n]* dropped \([^)]*\)", new)
print(f"request 2: {r2.get('usage', {}).get('prompt_tokens')} prompt tokens in {s2:.1f} s; read back from disk: {back}; "
      f"the engine's line: {reused}; dropped: {dropped}", flush=True)
stop(p, "2")
if not back:
    fails.append("the follow-up did not read the saved conversation back" + (f" ({dropped})" if dropped else ""))
if reused and saved and int(reused[-1][1]) < int(saved[-1][0]):
    fails.append(f"reused {reused[-1][1]} of the {saved[-1][0]} saved tokens")
print("ALL PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails), flush=True)
sys.exit(1 if fails else 0)
