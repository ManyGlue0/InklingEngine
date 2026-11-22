"""Minimal streaming chat client for `inkling serve` (standard library only).
Usage: python tools/chat.py "your message" [--image file.png] [--effort low|medium|high|0..0.99] [--base URL]
Prints the reasoning (dimmed), then the answer, then token counts and speed."""
import argparse
import base64
import json
import mimetypes
import os
import shutil
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("message")
ap.add_argument("--image", help="attach an image file")
ap.add_argument("--effort", default="low", help="reasoning_effort (default low)")
ap.add_argument("--base", default="http://127.0.0.1:8080", help="server URL")
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--temperature", type=float, help="sampling temperature (default: the server's)")
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
if args.temperature is not None:
    body["temperature"] = args.temperature
req = urllib.request.Request(args.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
DIM, RESET = "\x1b[2m", "\x1b[0m"


class Wrap:
    """Streams text with word wrapping at the terminal width."""

    def __init__(self):
        self.width = max(20, shutil.get_terminal_size().columns - 1)
        self.col = 0
        self.word = ""

    def _flush_word(self):
        if self.word:
            if self.col > 0 and self.col + len(self.word) > self.width:
                print()
                self.col = 0
            print(self.word, end="")
            self.col += len(self.word)
            self.word = ""

    def write(self, text):
        for c in text:
            if c == "\n":
                self._flush_word()
                print()
                self.col = 0
            elif c.isspace():
                self._flush_word()
                if 0 < self.col < self.width:
                    print(" ", end="")
                    self.col += 1
            else:
                self.word += c
        sys.stdout.flush()

    def raw(self, text):
        self._flush_word()
        print(text, end="")
        if "\n" in text:
            self.col = 0


out = Wrap()
t0 = time.time()
first = None
mode = None
usage = None
try:
    resp = urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    sys.exit(f"server error {e.code}: {e.read().decode('utf-8', 'replace')}")
except urllib.error.URLError as e:
    sys.exit(f"cannot reach {args.base} ({e.reason}). Is `inkling serve` running and done loading?")
with resp as r:
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
                    out.raw(RESET + ("\n\n" if mode else "") + style)
                    mode = key
                out.write(text)
            for tc in d.get("tool_calls") or []:
                out.raw(f"{RESET}\n[tool call] {tc['function']['name']}({tc['function']['arguments']})")
out.raw("")
print(RESET)
if usage and first:
    dt = time.time() - first
    n = usage["completion_tokens"]
    print(f"{DIM}[prompt {usage['prompt_tokens']} tok, first token {first - t0:.2f}s | {n} tok at {max(n - 1, 1) / max(dt, 1e-6):.0f} tok/s]{RESET}")
