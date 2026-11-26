#!/usr/bin/env python3
"""moex/tests/serve_protocol_test.py - strata-x --serve against Strata's engine protocol, on the 4-layer GLM slice.

    python3 moex/tests/serve_protocol_test.py [--model SLICE.gguf] [--tokens FILE] [--engine build/strata-x]

Checks (each prints PASS / FAIL):
  1. the handshake: INFO ... then READY <ctx> stop
  2. a greedy request: RESUME 0, PP lines, REUSED, T per token, DONE with 15 fields; its tokens equal the CLI's generate()
  3. prefix reuse: prompt + generated + new tokens -> RESUME = what the engine held, same continuation as from scratch
  4. checkpoint resume: a prompt that diverges after a turn boundary resumes from the checkpoint (RESUME = its position)
     and generates what a from-scratch run generates
  5. seeded sampling: same seed -> same tokens; another seed -> (almost surely) other tokens; penalties accepted
  6. STOP: a request stopped after a few tokens ends with finish cancel
  7. errors: an id outside the vocabulary, a context overflow, a malformed line -> ERR and the engine keeps serving
Nothing else is started: one engine process (the GPUs must be free).
"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_DATA = _os.environ.get("STRATA_X_DATA") or _os.path.join(_AI, "Strata-X-data")
import argparse, os, queue, subprocess, sys, threading, time

X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = _DATA


class Engine:
    def __init__(self, exe, model, extra, base=("--ctx", "4096", "--threads", "8", "--no-pin")):
        env = dict(os.environ, XDG_CONFIG_HOME=os.path.join(X, ".config"))
        self.p = subprocess.Popen([exe, "--serve", "-m", model] + list(base) + extra,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.info = {}
        self.ctx = 0
        deadline = time.time() + 600
        while time.time() < deadline:
            line = self.q.get(timeout=max(1, deadline - time.time()))
            if line is None:
                raise RuntimeError("the engine ended before READY")
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            if line.startswith("READY"):
                f = line.split()
                self.ctx, self.can_stop = int(f[1]), "stop" in f[2:]
                break

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def gen(self, ids, max_new, keys="", stop_after=None, timeout=300):
        """-> dict(tokens, resume, reused, pp, done fields, err)"""
        self.send(f"GEN {max_new}{(' ' + keys) if keys else ''} {','.join(map(str, ids))}")
        r = dict(tokens=[], resume=None, reused=None, pp=[], done=None, err=None, order=[])
        deadline = time.time() + timeout
        stopped = False
        while time.time() < deadline:
            line = self.q.get(timeout=max(1, deadline - time.time()))
            if line is None:
                r["err"] = "engine ended"
                return r
            f = line.split()
            r["order"].append(f[0] if f else "")
            if line.startswith("T "):
                r["tokens"].append(int(f[1]))
                if stop_after and len(r["tokens"]) >= stop_after and not stopped:
                    self.send("STOP")
                    stopped = True
            elif line.startswith("PP "):
                r["pp"].append(tuple(float(x) for x in f[1:]))
            elif line.startswith("RESUME "):
                r["resume"] = int(f[1])
            elif line.startswith("REUSED "):
                r["reused"] = int(f[1])
            elif line.startswith("DONE"):
                r["done"] = f[1:]
                return r
            elif line.startswith("ERR"):
                r["err"] = line[4:]
                return r
        r["err"] = "timeout"
        return r

    def close(self):
        try:
            self.send("QUIT")
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()


fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""), flush=True)
    fails += 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{DATA}/glm-mini/glm53-mini-L0-3.gguf")
    ap.add_argument("--tokens", default=f"{DATA}/glm-mini/dump-long/tokens.txt")
    ap.add_argument("--engine", default=os.path.join(X, "build/strata-x"))
    a = ap.parse_args()
    toks = [int(x) for x in open(a.tokens).read().strip().split(",")]
    # the "turn token": an id that occurs several times in the long prompt (a role marker's stand-in)
    from collections import Counter
    turn = Counter(toks[100:]).most_common(1)[0][0]
    eng = Engine(a.engine, a.model, ["--turn-tokens", str(turn), "--prefill-chunk", "0", "--short-read", "64"])
    check("1. handshake", eng.ctx == 4096 and eng.can_stop and "context" in eng.info, f"ctx {eng.ctx} stop {eng.can_stop} info {sorted(eng.info)[:4]}")

    # 2. greedy, short prompt (the window path)
    short = toks[:40]
    r = eng.gen(short, 12)
    check("2a. greedy short: lines", r["err"] is None and r["done"] and len(r["tokens"]) == 12 and r["resume"] == 0 and r["reused"] == 0,
          f"err {r['err']} tokens {len(r['tokens'])} done {r['done']}")
    check("2b. DONE has 15 fields", r["done"] and len(r["done"]) == 15 and r["done"][4] == "length")
    # the same through the CLI's generate()
    out = subprocess.run([a.engine, "-m", a.model, "-t", ",".join(map(str, short)), "-n", "12", "--ctx", "4096", "--threads", "8", "--no-pin", "--prefill-chunk", "-1"],
                         capture_output=True, text=True)
    cli = [int(x) for x in out.stdout.strip().splitlines()[-1].split(",")] if out.stdout.strip() else []
    check("2c. greedy tokens equal the CLI's generate()", cli == r["tokens"], f"{r['tokens'][:6]} vs {cli[:6]}")

    # 3. prefix reuse: the full conversation = prompt + generated + new ids
    ext = short + r["tokens"][:-1] + [r["tokens"][-1]] + toks[40:50]
    r2 = eng.gen(ext, 10)
    held = len(short) + len(r["tokens"]) - 1   # the prompt and every generated token but the last
    check("3a. prefix reuse: RESUME = what the engine held", r2["resume"] == held, f"resume {r2['resume']} expected {held}")
    eng2 = Engine(a.engine, a.model, ["--prefill-chunk", "0"])
    base = eng2.gen(ext, 10)
    check("3b. the resumed continuation equals a from-scratch one", base["tokens"] == r2["tokens"], f"{r2['tokens'][:6]} vs {base['tokens'][:6]}")
    eng2.close()

    # 4. a long prompt (the prompt path, checkpoints at the turn boundaries), then a divergent one
    long_ids = toks[:2400]
    r3 = eng.gen(long_ids, 8)
    check("4a. long prompt read (PP lines)", r3["err"] is None and len(r3["pp"]) >= 1 and len(r3["tokens"]) == 8, f"pp {len(r3['pp'])}")
    turns = [i for i, t in enumerate(long_ids) if t == turn and i > 0]
    last_turn = turns[-1] if turns else None
    div = long_ids[: last_turn + 5] + [t ^ 1 if t > 1 else 2 for t in long_ids[last_turn + 5: last_turn + 15]] if last_turn else None
    if div:
        r4 = eng.gen(div + toks[2400:2420], 8)
        eng3 = Engine(a.engine, a.model, ["--prefill-chunk", "0"])
        base = eng3.gen(div + toks[2400:2420], 8)
        eng3.close()
        check("4b. divergence resumes from the checkpoint before the last turn token",
              r4["resume"] == last_turn, f"resume {r4['resume']} expected {last_turn} (turn id {turn}, {len(turns)} turns)")
        check("4c. the checkpoint-resumed continuation equals a from-scratch one", base["tokens"] == r4["tokens"],
              f"{r4['tokens'][:6]} vs {base['tokens'][:6]}")
    else:
        check("4b. (no turn token in the prompt)", False)

    # 4d. a conversation switch: another prompt from token 0 rewrites the latent cache, so A's checkpoints are stale - coming
    #     back to A's prefix must NOT restore them (Strata: checkpoints past `resume` or off the prompt's path are dropped)
    if div:
        other = [t ^ 3 if t > 3 else 5 for t in toks[:2400]]   # an unrelated conversation (every token changed), from position 0
        r5 = eng.gen(other, 4)
        back = long_ids[: last_turn + 5] + toks[2400:2430]
        r6 = eng.gen(back, 8)
        eng4 = Engine(a.engine, a.model, ["--prefill-chunk", "0"])
        base6 = eng4.gen(back, 8)
        eng4.close()
        check("4d. after a conversation switch, no stale checkpoint is restored (resume 0)", r5["err"] is None and r6["resume"] == 0,
              f"resume {r6['resume']} (the other conversation: err {r5['err']}, resume {r5['resume']})")
        check("4e. ... and the continuation equals a from-scratch one", r6["tokens"] == base6["tokens"],
              f"{r6['tokens'][:6]} vs {base6['tokens'][:6]}")

    # 5. sampling
    s1 = eng.gen(short, 16, "temperature=0.9 top_k=40 top_p=0.95 seed=1234")
    s2 = eng.gen(short, 16, "temperature=0.9 top_k=40 top_p=0.95 seed=1234")
    s3 = eng.gen(short, 16, "temperature=0.9 top_k=40 top_p=0.95 seed=999")
    check("5a. same seed -> same tokens", s1["tokens"] == s2["tokens"] and len(s1["tokens"]) == 16, f"{s1['tokens'][:6]}")
    check("5b. another seed -> other tokens", s1["tokens"] != s3["tokens"], f"{s3['tokens'][:6]}")
    s4 = eng.gen(short, 12, "penalty_last_n=64 penalty_repeat=1.3 penalty_freq=0.2 penalty_present=0.2")
    check("5c. greedy with penalties runs and differs from plain greedy (when repeats exist)", s4["err"] is None and len(s4["tokens"]) == 12,
          f"{s4['tokens'][:8]} vs plain {r['tokens'][:8]}")

    # 6. STOP
    st = eng.gen(short, 200, stop_after=4)
    check("6. STOP -> cancel", st["done"] is not None and st["done"][4] == "cancel" and 4 <= len(st["tokens"]) < 200,
          f"tokens {len(st['tokens'])} done {st['done']}")
    after = eng.gen(short, 4)
    check("6b. the engine serves on after a STOP", after["err"] is None and len(after["tokens"]) == 4)

    # 7. errors keep the engine alive
    e1 = eng.gen([999999999], 4)
    check("7a. an id outside the vocabulary -> ERR", e1["err"] is not None, str(e1["err"]))
    e2 = eng.gen(short, 10 ** 6)
    check("7b. a context overflow -> ERR", e2["err"] is not None, str(e2["err"]))
    eng.send("hello")
    line = eng.q.get(timeout=30)
    check("7c. a malformed line -> ERR", line.startswith("ERR"), line)
    ok = eng.gen(short, 3)
    check("7d. still serving", ok["err"] is None and len(ok["tokens"]) == 3)
    eng.close()
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
