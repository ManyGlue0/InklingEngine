"""End-to-end check of the OpenAI-compatible server: a tool-call round trip plus a follow-up turn (prompt cache).
Usage: python tools/server_smoke.py [base_url]   (default http://127.0.0.1:8080)"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"


def chat(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req) as r:
        out = json.loads(r.read())
    return out, time.time() - t


tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]},
    },
}]
messages = [{"role": "user", "content": "What's the weather like in Paris right now? Use the tool."}]
r, dt = chat({"messages": messages, "tools": tools, "reasoning_effort": "low", "temperature": 0})
msg = r["choices"][0]["message"]
print(f"turn 1 ({dt:.2f}s) finish={r['choices'][0]['finish_reason']} usage={r['usage']}")
print("  reasoning:", (msg.get("reasoning_content") or "")[:160].replace("\n", " "))
print("  tool_calls:", json.dumps(msg.get("tool_calls")))
assert msg.get("tool_calls"), "expected a tool call"
call = msg["tool_calls"][0]
messages.append(msg)
messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"city": "Paris", "temperature_c": 17, "sky": "light rain"})})
r, dt = chat({"messages": messages, "tools": tools, "reasoning_effort": "low", "temperature": 0})
msg = r["choices"][0]["message"]
print(f"turn 2 ({dt:.2f}s) finish={r['choices'][0]['finish_reason']} usage={r['usage']}")
print("  content:", msg.get("content"))
messages.append(msg)
messages.append({"role": "user", "content": "Should I take an umbrella? One short sentence."})
r, dt = chat({"messages": messages, "tools": tools, "reasoning_effort": "low", "temperature": 0})
msg = r["choices"][0]["message"]
print(f"turn 3 ({dt:.2f}s) finish={r['choices'][0]['finish_reason']} usage={r['usage']}")
print("  content:", msg.get("content"))
