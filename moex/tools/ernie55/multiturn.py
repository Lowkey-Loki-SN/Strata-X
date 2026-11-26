"""moex/tools/ernie55/multiturn.py PORT CHARS (ERNIE55_PLAN.md P6) - a long document in turn 1, then turn 2 (the reply sent back, a new question) and turn 3: each
follow-up's cached_tokens, its time and its answer."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))
import glob, json, sys, time, urllib.request
port, chars = int(sys.argv[1]), int(sys.argv[2])
docs = sorted(glob.glob(_os.path.join(_REPO, 'moex/docs/*.md')), key=lambda p: -len(open(p, errors="replace").read()))
text = ""
for p in docs:
    text += open(p, errors="replace").read() + "\n\n"
    if len(text) >= chars:
        break
doc = text[:chars]
k = doc.rfind("\n", 0, int(len(doc) * 0.7)) + 1
doc = doc[:k] + "\n\nNote for the reader: the lighthouse keeper's cat is called 'Pistachio'.\n\n" + doc[k:]
msgs = [{"role": "user", "content": "Here is a document:\n\n<document>\n" + doc + "\n</document>\n\nWhat engine does this document describe? One sentence."}]
def ask(stream=False):
    body = {"model": "ernie-5.5-flash", "temperature": 0, "max_tokens": 4096, "messages": msgs}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                   data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=7200).read())
    m = r["choices"][0]["message"]
    u = r.get("usage") or {}
    print(json.dumps({"content": m.get("content"), "prompt": u.get("prompt_tokens"),
                      "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "s": round(time.time() - t, 1)},
                     ensure_ascii=False), flush=True)
    msgs.append({"role": "assistant", "content": m.get("content") or "", "reasoning_content": m.get("reasoning_content")})
    return m.get("content") or "", u
a, u1 = ask()
msgs.append({"role": "user", "content": "What is the lighthouse keeper's cat called? Just the name."})
b, u2 = ask()
ok2 = "pistachio" in b.lower() and (u2.get("prompt_tokens_details") or {}).get("cached_tokens", 0) >= u1["prompt_tokens"] - 64
print("PASS" if ok2 else "FAIL", "turn 2 (reuse and answer)", flush=True)
msgs.append({"role": "user", "content": "And what did I ask you first? Short answer."})
c, u3 = ask()
ok3 = (u3.get("prompt_tokens_details") or {}).get("cached_tokens", 0) >= u2["prompt_tokens"] - 64
print("PASS" if ok3 else "FAIL", "turn 3 (reuse)", flush=True)
