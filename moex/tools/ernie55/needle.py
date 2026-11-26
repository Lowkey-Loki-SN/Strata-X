"""moex/tools/ernie55/needle.py PORT CHARS DEPTH (ERNIE55_PLAN.md P6) - a fact placed at DEPTH (0..1) of CHARS characters of Strata's docs, asked for; then a follow-up
on the same text (the prompt cache's reuse)."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))
import glob, json, sys, time, urllib.request
port, chars, depth = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
docs = sorted(glob.glob(_os.path.join(_REPO, 'moex/docs/*.md')), key=lambda p: -len(open(p, errors="replace").read()))
text = ""
for p in docs:
    text += open(p, errors="replace").read() + "\n\n"
    if len(text) >= chars:
        break
text = text[:chars]
needle = "\n\nNote for the reader: the secret passphrase for the lighthouse is 'amber-falcon-42'. Remember it.\n\n"
k = int(len(text) * depth)
k = text.rfind("\n", 0, k) + 1
doc = text[:k] + needle + text[k:]
def ask(q, max_tokens=4096):
    body = {"model": "ernie-5.5-flash", "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "Here is a document:\n\n<document>\n" + doc + "\n</document>\n\n" + q}]}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                   data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=7200).read())
    m = r["choices"][0]["message"]
    print(json.dumps({"q": q[:60], "content": m.get("content"), "reasoning": (m.get("reasoning_content") or "")[-400:],
                      "finish": r["choices"][0]["finish_reason"], "usage": r.get("usage"), "s": round(time.time() - t, 1)},
                     ensure_ascii=False), flush=True)
    return m.get("content") or ""
a = ask("What is the secret passphrase for the lighthouse? Answer with the passphrase only.")
print("PASS" if "amber-falcon-42" in a else "FAIL", "needle", flush=True)
b = ask("In one sentence: what is this document mostly about?")
print("PASS" if len(b.strip()) > 20 else "FAIL", "follow-up", flush=True)
