"""Render tests/data/chat_cases.json with the model's own Jinja chat template (HF semantics) as the oracle for the
native renderer (inkling-core chat.rs). Usage: python tools/chat_oracle.py <template.jinja> <cases.json>
Prints one JSON-encoded rendered string per line."""
import json
import sys

import jinja2


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)


def main():
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    template = open(sys.argv[1], encoding="utf-8").read()
    # HF's {% generation %} only marks assistant spans; it renders its body unchanged
    template = template.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    t = env.from_string(template)
    for case in json.load(open(sys.argv[2], encoding="utf-8")):
        msgs = json.loads(json.dumps(case["messages"]))
        # HF clients pass tool-call arguments as objects (the template rejects strings): canonicalize like the server
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                a = tc["function"].get("arguments")
                if isinstance(a, str):
                    tc["function"]["arguments"] = json.loads(a) if a.strip() else {}
        kw = {"messages": msgs}
        if "tools" in case:
            kw["tools"] = case["tools"]
        if "reasoning_effort" in case:
            kw["reasoning_effort"] = case["reasoning_effort"]
        print(json.dumps(t.render(**kw), ensure_ascii=False))


if __name__ == "__main__":
    main()
