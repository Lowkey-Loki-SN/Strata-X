"""moex/tools/ernie55/long_session.py PORT | --log ENGINE_LOG (PORT_AUDIT_ALL.md §6) - a session shaped like the one that drained the
RAM copy's exchange blobs: a long first read, many follow-ups that each add a tool-output-sized block of this checkout's docs and
sources and ask a short question, past 65,536 and 131,072 positions (two cache growths), then a new conversation (the caches'
release) and more follow-ups.  Greedy, thinking off, 256 tokens a reply.  A JSON line a turn: the prompt's tokens, the cached ones,
the reply's tokens and seconds.

--log ENGINE_LOG: one line per request from the engine's serve lines - context, reply tokens, tok/s, the CPU pool's GB/s, the
routings in VRAM, the VRAM tier's whole upkeep in ms a token (the swap rounds' share alone moves between its parts as the rules
change), and the RAM copy's line (free blobs, experts only in the file, doubles, the events) plus the tier check's verdict when
STRATA_X_TIER_CHECK=1 (both printed after the request's other lines)."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))
import glob, json, re, subprocess, sys, time, urllib.request

# conversation 1: (characters of new text, question); then a new conversation (None); then conversation 2
PLAN = [(120000, "In one sentence: what is the first section about?")] + \
       [(24000, "In one sentence: what does the last block you were given do?")] * 18 + [None] + \
       [(4000, "In one sentence: what is this about?")] + [(40000, "In one sentence: what does the last block do?")] * 6


def corpus():
    # the checkout's tracked files only (the same text on every checkout of a commit; untracked notes left out)
    pats = ["moex/docs/*.md", "docs/*.md", "src/x/*.cpp", "serve/*.py"]
    try:
        rel = subprocess.run(["git", "-C", _REPO, "ls-files", "--", *pats], capture_output=True, text=True, check=True).stdout.split()
        dirs = {_os.path.dirname(x) for x in pats}   # (a git pathspec's * crosses '/': the folders' own files only, as glob)
        files = sorted(_os.path.join(_REPO, r) for r in rel if _os.path.dirname(r) in dirs)
    except (OSError, subprocess.CalledProcessError):
        files = sorted(sum((glob.glob(_os.path.join(_REPO, x)) for x in pats), []))
    text = ""
    for p in files:
        text += f"\n\n===== {_os.path.relpath(p, _REPO)} =====\n" + open(p, errors="replace").read()
    return text


def run(port):
    text, at = corpus(), 0
    need = sum(c for c, _ in (x for x in PLAN if x))
    if len(text) < need:
        sys.exit(f"the corpus holds {len(text)} characters, the plan wants {need}")
    msgs, conv, turn = [], 1, 0
    for step in PLAN:
        if step is None:   # a new conversation: the caches' release (STRATA_HY4_KV_GROW)
            msgs, conv, turn = [], conv + 1, 0
            continue
        chars, q = step
        block, at = text[at:at + chars], at + chars
        turn += 1
        msgs.append({"role": "user", "content": f"<tool_output>\n{block}\n</tool_output>\n\n{q}"})
        body = {"model": "ernie-5.5-flash", "temperature": 0, "max_tokens": 256, "messages": msgs,
                "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                       data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=7200).read())
        m, u = r["choices"][0]["message"], r.get("usage") or {}
        print(json.dumps({"conv": conv, "turn": turn, "prompt": u.get("prompt_tokens"),
                          "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                          "reply": u.get("completion_tokens"), "s": round(time.time() - t, 1)}), flush=True)
        msgs.append({"role": "assistant", "content": m.get("content") or ""})


def report(path):
    lines = open(path, errors="replace").read().splitlines()
    s = max((i for i, l in enumerate(lines) if "engine started" in l), default=0)
    rows, cur = [], {}
    for l in lines[s:]:
        if m := re.search(r"serve: (\d+) prompt tokens \((\d+) reused, (\d+) read\) in ([\d.]+) s, (\d+) generated in ([\d.]+) s "
                          r"\(([\d.]+) T/s\)", l):
            cur = {"ctx": int(m[1]), "read": int(m[3]), "read_s": float(m[4]), "gen": int(m[5]), "tps": float(m[7])}
            rows.append(cur)
        elif m := re.search(r"decode: \d+ tokens in .*?(\d+)% of the routings in VRAM", l):
            cur["vram"] = int(m[1])
        elif m := re.search(r"a token's other -?[\d.]+ ms: the head -?[\d.]+, the VRAM tier's upkeep (-?[\d.]+)", l):
            cur["upkeep_ms"] = float(m[1])
        elif m := re.search(r"decode CPU pool: .*?\(([\d.]+) GB/s\)", l):
            cur["pool"] = float(m[1])
        elif m := re.search(r"the RAM copy: (\d+) exchange blobs free, (\d+) experts only in the file, (\d+) doubled.*?taken (\d+) by "
                            r"swaps and (\d+) by growths, (\d+) given back, (\d+) released from doubles, (\d+) swap attempts waited for an owed "
                            r"blob; for want of a blob (\d+) victims and (\d+) growths' experts to the file; (\d+) victims to the file in "
                            r"place", l):
            cur.update(free=int(m[1]), file_only=int(m[2]), doubled=int(m[3]), take_swap=int(m[4]), take_grow=int(m[5]),
                       give=int(m[6]), recycled=int(m[7]), waited=int(m[8]), fallback=int(m[9]), grow_file=int(m[10]),
                       in_place=int(m[11]))
        elif m := re.search(r"the tier check: (OK|FAILED)(?: in (\d+) ms)?", l):
            cur["check"] = m[1] + (f" {m[2]} ms" if m[2] else "")
    print(f"{'ctx':>7} {'read':>6} {'rd s':>5} {'gen':>4} {'T/s':>5} {'GB/s':>5} {'VRAM':>4} {'upk':>5} | {'free':>4} {'file':>5} {'dbl':>4} | "
          f"{'tkS':>4} {'tkG':>3} {'give':>4} {'rcy':>4} {'wait':>4} {'fbk':>4} {'g>F':>3} {'inpl':>4} | check")
    for r in rows:
        g = lambda k, f="{}": f.format(r[k]) if k in r else "-"
        print(f"{r['ctx']:>7} {r['read']:>6} {r['read_s']:>5.1f} {r['gen']:>4} {r['tps']:>5.1f} {g('pool', '{:.1f}'):>5} {g('vram'):>4} "
              f"{g('upkeep_ms', '{:.1f}'):>5} | {g('free'):>4} {g('file_only'):>5} {g('doubled'):>4} | {g('take_swap'):>4} "
              f"{g('take_grow'):>3} {g('give'):>4} {g('recycled'):>4} {g('waited'):>4} {g('fallback'):>4} {g('grow_file'):>3} "
              f"{g('in_place'):>4} | {g('check')}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--log":
        report(sys.argv[2])
    elif len(sys.argv) == 2:
        run(int(sys.argv[1]))
    else:
        sys.exit(__doc__)
