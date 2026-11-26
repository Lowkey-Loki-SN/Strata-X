#!/usr/bin/env python3
"""moex/experiments/glm_prompt_ab.py - prompt-read A/B inside ONE loaded strata-x (the per-request keys), on a family config.

    . moex/env.sh && moex_gpus_free && ~/AI/Strata/.venv/bin/python moex/experiments/glm_prompt_ab.py \\
        [--config strata-glm.json] [--prefix 32768] [--follow 300] [--reps 4] [--fresh 8192] [--fresh-reps 4] \\
        [--plan "300:base,ring144:4;2000:base,ring144:3"] [--decode 32] [--engine-extra "..."]

The engine is started with the config's own arguments (serve/server.py engine_args + family_engine_args) but WITHOUT
--expert-profile-save (an A/B must not overwrite the learned profile) and with its log in moex/out/.  Then:
  1. a prefix of --prefix tokens is read (code: the repo's C++ sources), one decode token;
  2. follow-ups: each request is the previous one's prompt + its reply + N new tokens (prose: docs/*.md), so the engine
     reads ~N tokens at depth (its live sequence is reused) and decodes --decode greedy tokens; per --plan entry
     "N:arm,arm:rounds" the arms rotate (a Latin square over the rounds): base (the engine's own rules), ringK
     (read_ring=K: the prompt ring lent for the read);
  3. --fresh-reps fresh reads of --fresh tokens (resume 0).
Per arm: the follow-ups' read ms (DONE's prompt ms), decode tok/s, and the fresh reads' tok/s - medians and every value.
"""
import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time

X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, X)
sys.path.insert(0, os.path.join(X, "tools"))
from serve import families  # noqa: E402
from serve import server as S  # noqa: E402

# the per-request keys of each arm: read_ring=N (the prompt ring lent for the read; 144 / 160 are the full rings of the
# 256- / 288-expert models)
ARM_KEYS = {"base": ""}
ARM_KEYS.update({f"ring{r}": f"read_ring={r}" for r in range(32, 513, 16)})


class Engine:
    def __init__(self, exe, args, log, env=None):
        self.log = open(log, "w")
        self.p = subprocess.Popen([exe, "--serve"] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                  text=True, bufsize=1, env=env)
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        while True:
            line = self.q.get(timeout=1800)
            if line is None:
                raise RuntimeError("the engine ended before READY (see the log)")
            if line.startswith("READY"):
                break

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def gen(self, ids, max_new, keys):
        self.p.stdin.write(f"GEN {max_new} {keys} {','.join(map(str, ids))}\n".replace("  ", " "))
        self.p.stdin.flush()
        r = {"tokens": [], "done": None, "err": None}
        while True:
            line = self.q.get(timeout=3600)
            if line is None:
                r["err"] = "engine ended"
                return r
            if line.startswith("T "):
                r["tokens"].append(int(line.split()[1]))
            elif line.startswith("DONE"):
                r["done"] = line.split()[1:]
                return r
            elif line.startswith("ERR"):
                r["err"] = line
                return r

    def quit(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=300)
        except Exception:
            self.p.kill()


def text_tokens(tok, paths, want):
    out = []
    for p in paths:
        try:
            out += tok.encode(open(p, encoding="utf-8", errors="replace").read(), parse_special=False)
        except OSError:
            continue
        if len(out) >= want:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(X, "strata-glm.json"))
    ap.add_argument("--prefix", type=int, default=32768)
    ap.add_argument("--follow", type=int, default=300)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--fresh", type=int, default=8192)
    ap.add_argument("--fresh-reps", type=int, default=4)
    ap.add_argument("--arms", default="base,ring144")
    ap.add_argument("--plan", default="", help='follow-up sizes, arms and rounds: "300:base,ring32:4;2000:base,ring32:3" '
                                                '(overrides --follow / --arms / --reps)')
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--engine-extra", default="")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    fam = families.from_config(cfg)
    args = S.engine_args(cfg) + S.family_engine_args(fam, cfg)
    for flag in ("--expert-profile-save", "--expert-profile-save-every", "--image-pad"):   # never overwrite the learned profile
        while flag in args:
            i = args.index(flag)
            del args[i:i + 2]
    args += a.engine_extra.split()
    tok = fam.load_tokenizer(fam.tokenizer_path() if not cfg.get("tokenizer") else __import__("pathlib").Path(cfg["tokenizer"]))
    code = text_tokens(tok, [os.path.join(X, p) for p in ("src/program/generate.cpp", "src/core/verify.cpp",
                                                         "src/core/expert_source.cpp", "src/prefill/prefill.cpp")],
                       a.prefix + a.fresh * a.fresh_reps + 1)
    prose = text_tokens(tok, [os.path.join(X, "docs", f) for f in sorted(os.listdir(os.path.join(X, "docs"))) if f.endswith(".md")],
                        a.follow * a.reps * 8 + 1)
    plan = [(int(sz), arms_.split(","), int(reps)) for sz, arms_, reps in (p.split(":") for p in a.plan.split(";") if p)] \
        if a.plan else [(a.follow, a.arms.split(","), a.reps)]
    need = sum(sz * len(arms_) * reps for sz, arms_, reps in plan)
    prose = prose if len(prose) >= need else text_tokens(tok, [os.path.join(X, "docs", f) for f in sorted(os.listdir(os.path.join(X, "docs")))
                                                                if f.endswith(".md")] + [os.path.join(X, "serve/server.py")], need + 1)
    if len(code) < a.prefix + a.fresh * a.fresh_reps or len(prose) < need:
        sys.exit(f"not enough text: {len(code)} code tokens, {len(prose)} prose tokens (need {need})")
    os.makedirs(os.path.join(X, "moex/out"), exist_ok=True)
    log = os.path.join(X, "moex/out", f"prompt-ab-{fam.key}-{time.strftime('%m%d-%H%M%S')}.log")
    print(f"[ab] {fam.key}: {' '.join(args)}\n[ab] engine log {log}", flush=True)
    t0 = time.time()
    e = Engine(os.path.join(X, "build/strata-x") if not cfg.get("exe") else cfg["exe"], args, log)
    print(f"[ab] loaded in {time.time() - t0:.0f} s", flush=True)
    try:
        ids = code[:a.prefix]
        r = e.gen(ids, 1, "temperature=0")
        print(f"[ab] prefix {len(ids)} tokens read in {float(r['done'][2]) / 1000:.1f} s", flush=True)
        ids = ids + r["tokens"][:-1]
        res = {}
        k, at = 0, 0
        for sz, arms, reps in plan:
            for rep in range(reps):
                order = arms[rep % len(arms):] + arms[:rep % len(arms)]
                for arm in order:
                    new = prose[at:at + sz]
                    at += sz
                    k += 1
                    ids = ids + new
                    r = e.gen(ids, a.decode, ("temperature=0 " + ARM_KEYS[arm]).strip())
                    if r["err"] or not r["done"]:
                        sys.exit(f"[ab] {arm}: {r['err']}")
                    d = r["done"]
                    read_ms, dec_ms, gen, read_n = float(d[2]), float(d[3]), int(d[0]), int(d[13])
                    cell = res.setdefault((sz, arm), {"read_ms": [], "tok_s": []})
                    cell["read_ms"].append(read_ms)
                    cell["tok_s"].append(gen / (dec_ms / 1000) if dec_ms > 0 else 0.0)
                    print(f"[ab] follow-up {k:2d} {sz:5d} {arm:10s} at {len(ids):6d}: read {read_n} in {read_ms / 1000:6.2f} s, "
                          f"decode {gen} at {cell['tok_s'][-1]:5.1f} tok/s", flush=True)
                    ids = ids + r["tokens"][:-1]
        fresh = {"base": []}
        for i in range(a.fresh_reps):
            arm = "base"
            seg = code[a.prefix + i * a.fresh:a.prefix + (i + 1) * a.fresh]
            r = e.gen(seg, 1, ("temperature=0 " + ARM_KEYS[arm]).strip())
            d = r["done"]
            fresh[arm].append(int(d[13]) / (float(d[2]) / 1000))
            print(f"[ab] fresh {i + 1} {arm:4s}: {d[13]} tokens in {float(d[2]) / 1000:.2f} s = {fresh[arm][-1]:.0f} tok/s", flush=True)
        print("\n[ab]  size arm         follow-up read s (median)   decode tok/s (median)   reads")
        for (sz, arm), cell in res.items():
            rm = statistics.median(cell["read_ms"]) / 1000
            ts = statistics.median(cell["tok_s"])
            print(f"[ab] {sz:5d} {arm:10s}  {rm:8.2f}                    {ts:6.1f}                 "
                  f"{', '.join(f'{x / 1000:.2f}' for x in cell['read_ms'])}")
        for arm, v in fresh.items():
            if v:
                print(f"[ab] fresh {arm:4s}: {', '.join(f'{x:.0f}' for x in v)} tok/s (median {statistics.median(v):.0f})")
    finally:
        e.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
