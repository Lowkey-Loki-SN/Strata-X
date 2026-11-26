"""serve_restart.py EXE OUTDIR - DSV41_BACKLOG #53's restart half (--conversation-dir) and the clean end on SIGINT, through the
--serve protocol, set up as first_bits.py sets an engine (strata-dsv41.json's arguments and switches, the route bias off,
--no-adaptive, slots pinned, the 30 GiB test cache; --conversation-cache-min-tokens 2048 so its 4,013-token prompt counts).

  A: a fresh engine reads first_bits.py's prompt (greedy, 4 tokens): its first-token logits must be first_bits.py's hash
     (REF, the installed engine's: the same bits); then SIGINT - the engine must save the conversation and end by itself.
  B: an engine started the same way gets the follow-up (A's tokens + a question, greedy, 8 tokens): it must read the saved
     conversation back from disk and reuse all of it.
  C: a fresh engine without the directory reads the prompt and then the follow-up in one run: B's follow-up must give C's
     first-token logits and tokens - the restored state is the uninterrupted one.
"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
import json, os, re, signal, subprocess, sys, time
from pathlib import Path
sys.path[:0] = [_REPO, _os.path.join(_REPO, 'tools')]
from serve import families  # noqa: E402

exe, out = sys.argv[1], Path(sys.argv[2])
REF = os.environ.get("REF", "3278fb7b32159f1e")
out.mkdir(parents=True, exist_ok=True)
conv = out / "conversations"
R = _REPO
cfg = json.loads(Path(f"{R}/strata-dsv41.json").read_text())
tok = families.from_config(cfg).load_tokenizer(Path(cfg["tokenizer"]))
sp = lambda s: tok.encode(s, parse_special=True)
text = tok.encode(subprocess.run(["git", "-C", R, "show", "645844e9:moex/docs/DSV41_PLAN.md"], capture_output=True, text=True,
                                 check=True).stdout)
prompt = sp("<｜User｜>") + text[:4000] + sp("\n\nSummarise the plan above in one sentence.<｜Assistant｜></think>")
args = [a for a in cfg["args"]]
args[args.index("--ssd-cache-gib") + 1] = os.environ.get("CACHE_GIB", "30")
for flag in ("--conversation-dir", "--conversation-dir-mib"):   # the config's own, if it has them: this check's instead
    if flag in args:
        i = args.index(flag)
        del args[i:i + 2]
base = [exe, "--serve"] + args + ["--layer-split", "20", "--eos-ids", "1", "--turn-tokens", "128799,128803,128804", "--image-pad",
                                  "129264", "--no-adaptive", "--expert-slots", "1700", "--conversation-cache-min-tokens", "2048"]
env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_DSV4", "STRATA_MMVQ"))}
env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1", STRATA_X_FIRST_TOP="8")
# the slots per stage pinned as first_bits.py's reference was taken (DSV41_PLAN §65): else each start splits them by the free
# VRAM it finds, other experts sit in VRAM, and the engines' bits differ for that alone
env.setdefault("STRATA_X_STAGE_SLOTS", "1142,558")
env.update({k: v for k, v in cfg["env"].items() if k != "STRATA_DSV41_ROUTE_BIAS"})


class Engine:
    def __init__(self, tag, with_dir):
        self.tag, self.log = tag, out / f"{tag}.log"
        cmd = base + (["--conversation-dir", str(conv)] if with_dir else [])
        self.err = open(self.log, "w")
        t0 = time.time()
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, env=env)
        for line in self.p.stdout:
            if line.startswith("READY"):
                break
        print(f"{tag}: ready in {time.time() - t0:.0f} s", flush=True)

    def gen(self, ids, n, what):
        mark = len(self.log.read_text(errors="replace"))
        t0 = time.time()
        self.p.stdin.write(f"GEN {n} temperature=0 " + ",".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        toks, reused, done = [], None, ""
        for line in self.p.stdout:
            if line.startswith("REUSED "):
                reused = int(line.split()[1])
            elif line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith(("DONE", "ERR")):
                done = line.strip()
                break
        self.err.flush()
        new = self.log.read_text(errors="replace")[mark:]
        h = re.findall(r"\| logits ([0-9a-f]{16})", new)
        back = re.findall(r"conversation cache: read (\d+) tokens back from disk[^\n]*", new)
        r = dict(tokens=toks, reused=reused, hash=h[-1] if h else None, from_disk=back[-1] if back else None, s=time.time() - t0)
        print(f"{self.tag} {what}: {len(ids)} tokens, reused {reused}, {r['s']:.1f} s, logits {r['hash']}, tokens {toks}"
              f"{', read back from disk: ' + back[-1] + ' tokens' if back else ''}  [{done[:40]}]", flush=True)
        return r


fails = []
a = Engine("A", True)
r1 = a.gen(prompt, 4, "prompt")
if r1["hash"] != REF:
    fails.append(f"A's first-token logits {r1['hash']} are not the reference {REF}")
t0 = time.time()
a.p.send_signal(signal.SIGINT)   # what Ctrl+C in the server's terminal sends it too
code = a.p.wait(timeout=300)
a.err.close()
alog = a.log.read_text(errors="replace")
saved = re.findall(r"conversation cache: saved (\d+) tokens[^\n]*", alog)
print(f"A: SIGINT -> exit code {code} after {time.time() - t0:.1f} s; {saved[-1] + ' tokens saved' if saved else 'nothing saved'}",
      flush=True)
if code != 0 or not saved:
    fails.append(f"A did not end cleanly on SIGINT (exit {code}, saved {saved})")
follow = prompt + r1["tokens"] + sp("\n\nWhich parts of it read the experts?<｜Assistant｜></think>")

b = Engine("B", True)
r2 = b.gen(follow, 8, "follow-up")
b.p.stdin.write("QUIT\n"); b.p.stdin.flush(); b.p.wait(timeout=300); b.err.close()
if not r2["from_disk"] or (r2["reused"] or 0) < len(prompt):
    fails.append(f"B did not put the saved conversation back (reused {r2['reused']}, from disk {r2['from_disk']})")

c = Engine("C", False)
c.gen(prompt, 4, "prompt")
r3 = c.gen(follow, 8, "follow-up")
c.p.stdin.write("QUIT\n"); c.p.stdin.flush(); c.p.wait(timeout=300); c.err.close()
if (r2["hash"], r2["tokens"]) != (r3["hash"], r3["tokens"]):
    fails.append(f"B's follow-up {r2['hash']} {r2['tokens']} is not C's {r3['hash']} {r3['tokens']}")
print("ALL PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails), flush=True)
sys.exit(1 if fails else 0)
