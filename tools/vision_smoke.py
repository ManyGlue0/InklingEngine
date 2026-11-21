"""Image input through the OpenAI-compatible server: describe tests/data/vision_test.png, then a follow-up turn
(the image is resent, so the prompt cache and the server's embedding cache should both hit).
Usage: python tools/vision_smoke.py [base_url] [image]   (default http://127.0.0.1:8080)"""
import base64
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
IMG = sys.argv[2] if len(sys.argv) > 2 else "tests/data/vision_test.png"
sys.stdout.reconfigure(encoding="utf-8")


def chat(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req) as r:
        out = json.loads(r.read())
    return out, time.time() - t


url = "data:image/png;base64," + base64.b64encode(open(IMG, "rb").read()).decode()
messages = [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": url}},
    {"type": "text", "text": "Describe this image: list the shapes with their colors, and read any text."}]}]
r, dt = chat({"messages": messages, "reasoning_effort": "low", "temperature": 0, "max_tokens": 400})
msg = r["choices"][0]["message"]
print(f"turn 1 ({dt:.2f}s) usage={r['usage']}")
print("  reasoning:", (msg.get("reasoning_content") or "")[:300].replace("\n", " "))
print("  content:", msg.get("content"))
messages.append({"role": "assistant", "content": msg.get("content") or ""})
messages.append({"role": "user", "content": "Which shape is on the far right? One word."})
r, dt = chat({"messages": messages, "reasoning_effort": "low", "temperature": 0, "max_tokens": 200})
msg = r["choices"][0]["message"]
print(f"turn 2 ({dt:.2f}s) usage={r['usage']}")
print("  content:", msg.get("content"))
