"""speed_ab.py OUT_PREFIX [S] [X] [K] [R] [L] [F]  |  speed_ab.py --cmp A.json B.json - one engine binary's decode speed and its
bits on a config, through the engine's serve protocol (no HTTP server, no vision encoder, no conversation folder, no profile
save: the user's saved state is not touched).  DSV41_PLAN §75, §76 (2026-11-10).

  S  CONFIG as deployed: three prompts (the plan's text, code, prose - through the family's template, thinking on, effort
     max), greedy, DECODE_N tokens each (384), after a 64-token settle; prints tok/s and the engine's own decode lines.
  X  bits: the route bias off, --no-adaptive, --expert-slots SLOTS_X (2000) with STRATA_X_STAGE_SLOTS=STAGE_SLOTS_X
     (1200,800), the cache CACHE_GIB_X (30) and no warm fill: per request the first-token logits hash (STRATA_X_FIRST_TOP)
     and DECODE_N_X (96) greedy tokens - a read (the prompt path), its reply (decode; the code prompt's reply copies text, so
     prompt-lookup windows run), a short follow-up (<= 96 new tokens: decode windows) and a long one (a gated chunk).
  K  bits and time of a short follow-up that crosses a periodic checkpoint mark: X's setup with --prompt-cache-every 4096,
     a ~3,800-token read, a 64-token reply, then ~400 new tokens across position 4,096.
  R  X's first conversation REPS times (12) in one engine: what repeats from one repetition to the next.
  L  X's setup, one cold read of ~16K tokens (LONG_CHARS characters of the plan, 50,000): several chunks, layer-major.
  F  X's setup with one engram fetch made to fail (STRATA_X_EG_FAIL=EG_FAIL, 40): the request fails, the conversation sent
     again gives X's bits.

Two binaries are compared with X_LOOKUP=1 (--lookup-window 1: no prompt-lookup drafts): the lookup's policy is timed, so a
draft window can start a token earlier or later from run to run or build to build, and on a set with IQ2_S / IQ2_XS rows a
token's bits depend on whether it ran alone or in a window (DSV41_BACKLOG #135; DSV41_PLAN §76).  X_PROMPTS: X's
conversations (plan,code; prose: 66 tokens, decode windows alone); X_LONG=0: no long follow-up.

EXE: the engine (default: CONFIG's "exe"); CONFIG: default strata-dsv41-ud-q2kxl.json; SWITCHES=K=V,...: more environment;
EXTRA_ARGS: more engine arguments.  One load at a time; the user's server must be stopped.  Writes OUT_PREFIX-<load>.log
(the engine's stderr) and OUT_PREFIX.json; --cmp compares two runs' X / K entries (hashes and tokens)."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, re, subprocess, sys, time
from pathlib import Path

R = _REPO
if len(sys.argv) >= 4 and sys.argv[1] == "--cmp":
    a, b = (json.loads(Path(p).read_text()) for p in sys.argv[2:4])
    ka, kb = {r["key"]: r for r in a["bits"]}, {r["key"]: r for r in b["bits"]}
    bad = 0
    for k in sorted(set(ka) | set(kb)):
        if k not in ka or k not in kb:
            print(f"  {k:<22} only in {'A' if k in ka else 'B'}")
            bad += 1   # (a missing request is a failed comparison)
            continue
        x, y = ka[k], kb[k]
        same = x["hash"] == y["hash"] and x["toks"] == y["toks"]
        first = next((i for i, (p, q) in enumerate(zip(x["toks"], y["toks"])) if p != q), min(len(x["toks"]), len(y["toks"])))
        hx, hy = x.get("hashes") or [], y.get("hashes") or []
        if hx and hy:   # (every token's hash: where the two runs part)
            d = next((i for i, (p, q) in enumerate(zip(hx, hy)) if p != q), None)
            if d is not None:
                same = False
                print(f"      per-token hashes part at sampled row {d} of {min(len(hx), len(hy))} (position {hx[d][0]})")
        print(f"  {k:<22} {x['hash']} {y['hash']}  {'same' if same else 'DIFFERENT (tokens from %d)' % first}  "
              f"prompt {x['prompt_ms'] / 1000:.1f} / {y['prompt_ms'] / 1000:.1f} s, decode {x['tok_s']:.1f} / {y['tok_s']:.1f} tok/s")
        bad += not same
    print("bits:", "the same in every request" if not bad else f"{bad} requests DIFFER")
    sys.exit(1 if bad else 0)

sys.path[:0] = [R, f"{R}/tools"]
from serve import families                       # noqa: E402
from serve.family_template import FamilyChatTemplate   # noqa: E402

prefix = sys.argv[1]
loads = sys.argv[2:] or ["S", "X"]
CFG = json.loads(Path(os.environ.get("CONFIG", f"{R}/strata-dsv41-ud-q2kxl.json")).read_text())
EXE = os.environ.get("EXE", CFG["exe"])
N, NX = int(os.environ.get("DECODE_N", "384")), int(os.environ.get("DECODE_N_X", "96"))
spec = families.from_config(CFG)
tok = spec.load_tokenizer(Path(CFG["tokenizer"]))
tmpl = FamilyChatTemplate(Path(R) / "serve" / "family_templates" / spec.template_file, spec)
sp = lambda s: list(tok.encode(s, parse_special=True))   # noqa: E731
show = lambda f: subprocess.run(["git", "-C", R, "show", "645844e9:" + f], capture_output=True, text=True, check=True).stdout   # noqa: E731


def ids(content: str) -> list:
    return sp(tmpl.render([{"role": "user", "content": content}], enable_thinking=True, reasoning_effort="max"))


plan, code = show("moex/docs/DSV41_PLAN.md"), show("serve/family_parsers.py")
PROMPTS = {
    "plan": ids(plan[:8000] + "\n\nExplain in detail how the plan above reads the experts from the drive."),
    "code": ids(f"```python\n{code[:7000]}\n```\n\nRewrite the first function above so that it is easier to read. Give the "
                "complete new function, then explain the changes briefly."),
    "prose": ids("Write a long, detailed essay on the history of the printing press in Europe: who built the first presses, "
                 "how printing spread, and how it changed religion, science and politics."),
}
plan_ids = list(tok.encode(plan))


def with_arg(args: list, flag: str, value: str) -> list:
    a = list(args)
    if flag in a:
        a[a.index(flag) + 1] = value
    else:
        a += [flag, value]
    return a


def start(name: str, args: list, env_over=None, env_drop=()):
    argv = [EXE, "--serve"] + list(args) + os.environ.get("EXTRA_ARGS", "").split()
    argv += ["--layer-split", str(CFG.get("layer_split", "20")), "--eos-ids", ",".join(map(str, spec.eos_ids)),
             "--turn-tokens", ",".join(map(str, spec.turn_token_ids))]
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(map(str, CFG.get("gpu", [0, 1]))))
    env.update(CFG["env"])
    for k in env_drop:
        env.pop(k, None)
    env.update(env_over or {})
    for kv in filter(None, os.environ.get("SWITCHES", "").split(",")):
        k, v = kv.split("=", 1)
        env[k] = v
    logp = f"{prefix}-{name}.log"
    log = open(logp, "w")
    log.write(" ".join(argv) + "\n" + json.dumps({k: v for k, v in env.items() if k.startswith("STRATA_")}) + "\n")
    log.flush()
    t0 = time.time()
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=env,
                         cwd=CFG.get("cwd", R))
    for line in p.stdout:
        if line.startswith("READY"):
            print(f"[{name}] ready in {time.time() - t0:.0f} s", flush=True)
            return p, logp
    raise RuntimeError(f"[{name}] the engine ended before READY - see {logp}")


def gen(p, logp: str, prompt: list, n: int) -> dict:
    mark = os.path.getsize(logp)
    p.stdin.write(f"GEN {n} temperature=0 " + ",".join(map(str, prompt)) + "\n")
    p.stdin.flush()
    toks = []
    for line in p.stdout:
        if line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            f = line.split()
            r = dict(gen=int(f[1]), prompt_tokens=int(f[2]), prompt_ms=float(f[3]), decode_ms=float(f[4]), finish=f[5],
                     acc=int(f[6]), off=int(f[7]), reused=int(f[8]), toks=toks)
            r["tok_s"] = r["gen"] / max(r["decode_ms"], 1e-9) * 1000.0
            # (DSV41_BACKLOG #148: the drive's slow episodes begun in the request, the seconds its guard held the reads back)
            r["episodes"], r["slow_s"] = (int(f[16]), float(f[17])) if len(f) > 17 else (0, 0.0)
            time.sleep(0.2)   # (the engine's lines of this request reach the log)
            with open(logp, errors="replace") as fh:
                fh.seek(mark)
                text = fh.read()
            h = re.findall(r"\| logits ([0-9a-f]{16})", text)
            r["hash"] = h[0] if h else None
            r["hashes"] = re.findall(r"position (\d+) \| tok-logits ([0-9a-f]{16})", text)   # (STRATA_X_EVERY_TOP=1)
            r["lines"] = [ln for ln in text.splitlines() if "serve: decode" in ln or "prompt path:" in ln or "layer-major" in ln]
            return r
        elif line.startswith("ERR"):
            raise RuntimeError(line.strip())
    try:
        rc = p.wait(timeout=30)
    except Exception:   # noqa: BLE001
        rc = None
    raise RuntimeError(f"the engine ended during a request (exit status {rc})")


def stop(p):
    try:
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=180)
    except Exception:   # noqa: BLE001
        p.terminate()
        p.wait(timeout=60)


def dec(r: dict) -> str:   # the engine's own decode figures of a request
    t = " ".join(r["lines"])
    g = lambda pat: (re.search(pat, t) or [None, "?"])[1]   # noqa: E731
    return (f"wait {g(r'a token: ([0-9.]+) ms waiting')} ms, pool {g(r'([0-9.]+) ms in the CPU pool')} ms, other "
            f"{g(r'other ([0-9.]+) ms')} ms, VRAM {g(r'([0-9]+)% of the routings in VRAM')}%, SSD {g(r'; ([0-9.]+) experts read from the SSD')}/tok")


out = {"exe": EXE, "config": os.environ.get("CONFIG", "strata-dsv41-ud-q2kxl.json"), "speed": [], "bits": []}
X_ENV = {"STRATA_X_FIRST_TOP": "8", "STRATA_X_STAGE_SLOTS": os.environ.get("STAGE_SLOTS_X", "1200,800")}
X_DROP = ("STRATA_DSV41_ROUTE_BIAS", "STRATA_DSV41_CACHE_WARM")


def x_args() -> list:
    a = with_arg(CFG["args"], "--ssd-cache-gib", os.environ.get("CACHE_GIB_X", "30"))
    # X_LOOKUP=1: no prompt-lookup drafts (--lookup-window 1; 0 is the engine's default size). The lookup's policy is timed, so where a draft window starts can move by a token
    # from run to run or build to build, and on a set with IQ2_S / IQ2_XS rows a token's bits depend on whether it ran alone
    # or in a window (DSV41_BACKLOG #135) - two binaries are compared without it
    if os.environ.get("X_LOOKUP") is not None:
        a = with_arg(a, "--lookup-window", os.environ["X_LOOKUP"])
    return with_arg(a, "--expert-slots", os.environ.get("SLOTS_X", "2000")) + ["--no-adaptive"]


def bit(key: str, r: dict):
    print(f"  {key:<22} logits {r['hash']}  {r['prompt_tokens']:5d} prompt tokens ({r['reused']} reused) in "
          f"{r['prompt_ms'] / 1000:6.1f} s, {r['gen']:3d} tokens at {r['tok_s']:5.1f} tok/s, drafts {r['acc']}/{r['off']}", flush=True)
    for ln in r["lines"]:
        if "layer-major" in ln:
            print("      " + ln.strip()[:150], flush=True)
    out["bits"].append(dict(key=key, hash=r["hash"], toks=r["toks"], prompt_ms=r["prompt_ms"], tok_s=r["tok_s"], hashes=r.get("hashes", [])))


if "S" in loads:
    p, logp = start("S", CFG["args"])
    gen(p, logp, PROMPTS["plan"], 64)   # (after the load: settle)
    tot_g = tot_ms = 0.0
    for name, P in PROMPTS.items():
        if os.environ.get("PROMPTS") and name not in os.environ["PROMPTS"].split(","):
            continue
        r = gen(p, logp, P, N)
        tot_g, tot_ms = tot_g + r["gen"], tot_ms + r["decode_ms"]
        ep = f"  ** {r['episodes']} drive episode(s), reads held back {r['slow_s']:.0f} s **" if r["episodes"] or r["slow_s"] >= 1.0 else ""
        print(f"  S {name:<6} {r['tok_s']:6.2f} tok/s ({r['gen']} tokens, drafts {r['acc']}/{r['off']}; prompt {r['prompt_tokens']} "
              f"in {r['prompt_ms'] / 1000:.1f} s)  {dec(r)}{ep}", flush=True)
        out["speed"].append(dict(prompt=name, **{k: v for k, v in r.items() if k not in ("toks", "lines")}, lines=r["lines"]))
    print(f"  S all    {1000.0 * tot_g / tot_ms:6.2f} tok/s over {int(tot_g)} tokens", flush=True)
    out["tok_s"] = 1000.0 * tot_g / tot_ms
    stop(p)
if "X" in loads:
    p, logp = start("X", x_args(), env_over=X_ENV, env_drop=X_DROP)
    # X_PROMPTS: the conversations (plan,code; "prose" is 66 tokens - under --short-read, so decode windows alone: a set
    # whose first read would stream most of its records, the full MXFP4 one, is checked with it); X_LONG=0: no long follow-up
    for name in [n for n in ("plan", "code", "prose") if n in os.environ.get("X_PROMPTS", "plan,code").split(",")]:
        P = PROMPTS[name]
        r = gen(p, logp, P, NX)
        bit(f"{name} read+reply", r)
        live = P + r["toks"]
        short = live + sp("<｜User｜>") + list(tok.encode("And in one sentence?")) + sp("<｜Assistant｜></think>")
        r2 = gen(p, logp, short, 32)
        bit(f"{name} short follow-up", r2)
        if os.environ.get("X_LONG", "1") == "0":
            continue
        long_ = short + r2["toks"] + sp("<｜User｜>") + plan_ids[9000:9600] + list(tok.encode("\n\nWhat does this part add?")) + \
            sp("<｜Assistant｜></think>")
        bit(f"{name} long follow-up", gen(p, logp, long_, 32))
    stop(p)
if "R" in loads:
    # the plan conversation's three requests REPS times in ONE engine (12): from the second time on a request resumes at its
    # checkpoint and decodes the same tokens through the same windows, so every repetition's hashes must be the second's - a
    # run-to-run difference in the decode or its windows shows without a start a sample
    p, logp = start("R", x_args(), env_over=X_ENV, env_drop=X_DROP)
    P = PROMPTS["plan"]
    table = []
    for rep in range(int(os.environ.get("REPS", "12"))):
        r1 = gen(p, logp, P, NX)
        live = P + r1["toks"]
        short = live + sp("<｜User｜>") + list(tok.encode("And in one sentence?")) + sp("<｜Assistant｜></think>")
        r2 = gen(p, logp, short, 32)
        long_ = short + r2["toks"] + sp("<｜User｜>") + plan_ids[9000:9600] + list(tok.encode("\n\nWhat does this part add?")) + \
            sp("<｜Assistant｜></think>")
        r3 = gen(p, logp, long_, 32)
        row = [r1["hash"], r2["hash"], r3["hash"]]
        toks = [r1["toks"], r2["toks"], r3["toks"]]
        table.append((row, toks))
        ref = table[min(1, len(table) - 1)]
        mark = "" if rep < 2 or (row == ref[0] and toks == ref[1]) else "   <-- PARTS from repetition 2"
        print(f"  rep {rep + 1:2d}  {row[0]} {row[1]} {row[2]}  reused {r1['reused']}/{r2['reused']}/{r3['reused']}, "
              f"drafts {r1['acc']}/{r1['off']}{mark}", flush=True)
    bad = sum(1 for row, toks in table[2:] if row != table[1][0] or toks != table[1][1]) if len(table) > 2 else 0
    print(f"repetitions: {'every one the same as the second' if not bad else str(bad) + ' PART from the second'}", flush=True)
    out["reps"] = [row for row, _ in table]
    out["reps_bad"] = bad
    stop(p)
if "L" in loads:
    # a cold read of several chunks - layer-major, a window's engram rows fetched chunk by chunk: LONG_CHARS characters of the
    # plan's text (50,000: ~16K tokens) and a question, 16 tokens; its bits and the engine's lines of the read
    p, logp = start("L", x_args(), env_over=X_ENV, env_drop=X_DROP)
    LP = ids(plan[:int(os.environ.get("LONG_CHARS", "50000"))] + "\n\nWhich of the plan's sections above measured the most?")
    r = gen(p, logp, LP, 16)
    bit("long read+reply", r)
    for ln in r["lines"]:
        if "prompt path:" in ln:
            print("      " + ln.strip()[:330], flush=True)
    stop(p)
if "F" in loads:
    # a decode window's engram fetch made to fail (STRATA_X_EG_FAIL=N: the helper's N-th job, EG_FAIL here, 40): the request
    # must fail with the fetch's error and the engine end (a failed request ends it: the server starts it again) - not hang
    # at a GPU's wait for rows that never come
    p, logp = start("F", x_args(), env_over=dict(X_ENV, STRATA_X_EG_FAIL=os.environ.get("EG_FAIL", "40")), env_drop=X_DROP)
    t1 = time.time()
    try:
        gen(p, logp, PROMPTS["plan"], NX)
        print("  F: the request did NOT fail", flush=True)
        stop(p)
    except RuntimeError as e:
        try:
            rc = p.wait(timeout=120)
        except Exception:   # noqa: BLE001
            rc = None
            p.kill()
        print(f"  F: the request failed as asked ({str(e)[:90]}), the engine ended with exit status {rc} "
              f"{time.time() - t1:.0f} s after the request was sent", flush=True)
if "K" in loads:
    a = with_arg(x_args(), "--prompt-cache-every", os.environ.get("K_EVERY", "4096"))   # (K_EVERY=65536: no mark there - the reference)
    p, logp = start("K", a, env_over=X_ENV, env_drop=X_DROP)
    P = sp("<｜User｜>") + plan_ids[:3780] + list(tok.encode("\n\nSummarise the plan above in one sentence.")) + sp("<｜Assistant｜></think>")
    r = gen(p, logp, P, 64)
    bit("mark: read", r)
    F = P + r["toks"] + sp("<｜User｜>") + plan_ids[9000:9400] + list(tok.encode("\n\nWhat does this part add?")) + sp("<｜Assistant｜></think>")
    print(f"  (the follow-up: positions {len(P) + len(r['toks'])} to {len(F)}, the mark at 4,096)")
    bit("mark: crossing", gen(p, logp, F, 32))
    stop(p)
Path(f"{prefix}.json").write_text(json.dumps(out, indent=1))
