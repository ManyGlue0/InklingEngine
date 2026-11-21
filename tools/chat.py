"""Minimal streaming chat client for `inkling serve` (standard library only).
Usage: python tools/chat.py "your message" [--image file.png] [--effort low|medium|high|0..0.99] [--base URL]
Prints the reasoning (dimmed), then the answer, then token counts and speed."""
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("message")
ap.add_argument("--image", help="attach an image file")
ap.add_argument("--effort", default="low", help="reasoning_effort (default low)")
ap.add_argument("--base", default="http://127.0.0.1:8080", help="server URL")
ap.add_argument("--max-tokens", type=int, default=1024)
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
if os.name == "nt":
    os.system("")  # enables ANSI colors in the Windows console

content = args.message
if args.image:
    mime = mimetypes.guess_type(args.image)[0] or "image/png"
    url = f"data:{mime};base64," + base64.b64encode(open(args.image, "rb").read()).decode()
    content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": args.message}]
try:
    effort = float(args.effort)
except ValueError:
    effort = args.effort
body = {"messages": [{"role": "user", "content": content}], "reasoning_effort": effort, "stream": True,
        "max_tokens": args.max_tokens, "stream_options": {"include_usage": True}}
req = urllib.request.Request(args.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
DIM, RESET = "\x1b[2m", "\x1b[0m"
t0 = time.time()
first = None
mode = None
usage = None
with urllib.request.urlopen(req) as r:
    for raw in r:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            for key, style in (("reasoning_content", DIM), ("content", "")):
                text = d.get(key)
                if not text:
                    continue
                if first is None:
                    first = time.time()
                if mode != key:
                    print(RESET + ("\n\n" if mode else "") + style, end="")
                    mode = key
                print(text, end="", flush=True)
            for tc in d.get("tool_calls") or []:
                print(f"{RESET}\n[tool call] {tc['function']['name']}({tc['function']['arguments']})", end="")
print(RESET)
if usage and first:
    dt = time.time() - first
    n = usage["completion_tokens"]
    print(f"{DIM}[prompt {usage['prompt_tokens']} tok, first token {first - t0:.2f}s | {n} tok at {max(n - 1, 1) / max(dt, 1e-6):.0f} tok/s]{RESET}")
