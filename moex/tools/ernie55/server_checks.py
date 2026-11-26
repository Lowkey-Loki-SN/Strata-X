"""moex/tools/ernie55/server_checks.py [PORT] (ERNIE55_PLAN.md P6) - the ERNIE-5.5 server end to end: OpenAI chat (thinking on / off), a stream, a tool call and its
follow-up (plain and streamed), the Anthropic API (plain, streamed, a tool).  Prints each reply and a PASS / FAIL line."""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18234
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "ernie-5.5-flash"
results = []


def post(path, body, stream=False, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        return json.loads(r.read())
    return r


def sse(r):
    """-> the data payloads of an SSE stream (OpenAI's [DONE] dropped), with Anthropic's event names"""
    ev = None
    for raw in r:
        line = raw.decode("utf-8").rstrip("\n")
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            d = line[5:].strip()
            if d == "[DONE]":
                return
            yield ev, json.loads(d)


def check(name, ok, info=""):
    results.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "The current weather in a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "The city's name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}

# 1. OpenAI, thinking on (the family's default), greedy
t = time.time()
r = post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "What is 17 * 23? Give just the number."}],
                                   "max_tokens": 2048, "temperature": 0})
m = r["choices"][0]["message"]
print("1:", json.dumps({"reasoning": (m.get("reasoning_content") or "")[:300], "content": m.get("content"),
                        "finish": r["choices"][0]["finish_reason"], "usage": r.get("usage")}, ensure_ascii=False), f"{time.time() - t:.1f} s")
check("openai chat, thinking on", "391" in (m.get("content") or "") and bool(m.get("reasoning_content")))

# 2. OpenAI, thinking off
r = post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "What is the capital of France? One word."}],
                                   "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
m = r["choices"][0]["message"]
print("2:", json.dumps({"reasoning": m.get("reasoning_content"), "content": m.get("content"), "finish": r["choices"][0]["finish_reason"]},
                       ensure_ascii=False))
check("openai chat, thinking off", "paris" in (m.get("content") or "").lower() and not m.get("reasoning_content"))

# 3. OpenAI, reasoning_effort low, streamed, default sampling
r = post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "Write one sentence about the sea."}],
                                   "max_tokens": 1024, "stream": True, "reasoning_effort": "low"}, stream=True)
reason, content, finish, n = "", "", None, 0
for _, d in sse(r):
    for c in d.get("choices", []):
        n += 1
        reason += c.get("delta", {}).get("reasoning_content") or ""
        content += c.get("delta", {}).get("content") or ""
        finish = c.get("finish_reason") or finish
print("3:", json.dumps({"chunks": n, "reasoning": reason[:300], "content": content, "finish": finish}, ensure_ascii=False))
check("openai stream", n > 3 and len(content.strip()) > 10 and finish == "stop" and "<ernie:" not in content)

# 4. a tool call and its follow-up
msgs = [{"role": "user", "content": "What's the weather like in Paris right now? Use the tool."}]
r = post("/v1/chat/completions", {"model": MODEL, "messages": msgs, "tools": [WEATHER], "max_tokens": 2048, "temperature": 0})
c0 = r["choices"][0]
m = c0["message"]
print("4:", json.dumps({"content": m.get("content"), "tool_calls": m.get("tool_calls"), "finish": c0["finish_reason"]}, ensure_ascii=False))
tc = (m.get("tool_calls") or [None])[0]
ok = tc is not None and tc["function"]["name"] == "get_weather" and "paris" in json.loads(tc["function"]["arguments"]).get("city", "").lower()
check("openai tool call", ok and c0["finish_reason"] == "tool_calls")
if tc:
    msgs += [{"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"],
              "reasoning_content": m.get("reasoning_content")},
             {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"temp_c": 18, "sky": "cloudy"})}]
    r = post("/v1/chat/completions", {"model": MODEL, "messages": msgs, "tools": [WEATHER], "max_tokens": 2048, "temperature": 0})
    m = r["choices"][0]["message"]
    print("4b:", json.dumps({"content": m.get("content"), "finish": r["choices"][0]["finish_reason"]}, ensure_ascii=False))
    check("openai tool follow-up", "18" in (m.get("content") or "") and r["choices"][0]["finish_reason"] == "stop")

# 5. a streamed tool call
r = post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "Check the weather in Tokyo in fahrenheit."}],
                                   "tools": [WEATHER], "max_tokens": 2048, "temperature": 0, "stream": True}, stream=True)
calls, finish, content = {}, None, ""
for _, d in sse(r):
    for c in d.get("choices", []):
        content += c.get("delta", {}).get("content") or ""
        for t_ in c.get("delta", {}).get("tool_calls") or []:
            e = calls.setdefault(t_["index"], {"name": "", "args": ""})
            e["name"] += (t_.get("function") or {}).get("name") or ""
            e["args"] += (t_.get("function") or {}).get("arguments") or ""
        finish = c.get("finish_reason") or finish
print("5:", json.dumps({"content": content, "calls": calls, "finish": finish}, ensure_ascii=False))
ok = bool(calls) and calls[0]["name"] == "get_weather"
if ok:
    a = json.loads(calls[0]["args"])
    ok = "tokyo" in a.get("city", "").lower() and a.get("unit") == "fahrenheit"
check("openai streamed tool call", ok and finish == "tool_calls")

# 6. Anthropic messages, plain
r = post("/v1/messages", {"model": MODEL, "max_tokens": 2048, "temperature": 0,
                          "messages": [{"role": "user", "content": "Name the three primary colors of light, comma-separated."}]})
kinds = [b["type"] for b in r["content"]]
text = "".join(b.get("text", "") for b in r["content"] if b["type"] == "text")
print("6:", json.dumps({"kinds": kinds, "text": text, "stop": r.get("stop_reason"), "usage": r.get("usage")}, ensure_ascii=False))
check("anthropic messages", all(w in text.lower() for w in ("red", "green", "blue")) and r.get("stop_reason") == "end_turn")

# 7. Anthropic, streamed, with a tool
tool_a = {"name": "get_weather", "description": "The current weather in a city.", "input_schema": WEATHER["function"]["parameters"]}
r = post("/v1/messages", {"model": MODEL, "max_tokens": 2048, "temperature": 0, "stream": True, "tools": [tool_a],
                          "messages": [{"role": "user", "content": "Use the tool: weather in Berlin, celsius."}]}, stream=True)
blocks, stop, parts = {}, None, {}
for ev, d in sse(r):
    if ev == "content_block_start":
        blocks[d["index"]] = d["content_block"]
        parts[d["index"]] = ""
    elif ev == "content_block_delta":
        dl = d["delta"]
        parts[d["index"]] += dl.get("text") or dl.get("thinking") or dl.get("partial_json") or ""
    elif ev == "message_delta":
        stop = d.get("delta", {}).get("stop_reason") or stop
summary = [(blocks[i]["type"], blocks[i].get("name"), parts[i][:200]) for i in sorted(blocks)]
print("7:", json.dumps({"blocks": summary, "stop": stop}, ensure_ascii=False))
tu = [i for i in blocks if blocks[i]["type"] == "tool_use"]
ok = bool(tu) and blocks[tu[0]]["name"] == "get_weather" and "berlin" in json.loads(parts[tu[0]] or "{}").get("city", "").lower()
check("anthropic streamed tool_use", ok and stop == "tool_use")

print(f"\n{sum(ok for _, ok in results)} of {len(results)} passed")
