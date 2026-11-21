//! Inkling 2 chat format: a native rendering of the GGUF's Jinja chat template (OpenAI-style messages, tools,
//! reasoning effort) and a streaming parser of the model's output (reasoning, text, tool calls).
//!
//! Prompt (no generation prompt: the model emits its own role token):
//!   [<|message_system|>tool_declare<|content_xml|>{tool specs, sorted keys, compact}<|end_message|>]
//!   messages, with "<|message_system|><|content_text|>Thinking effort level: E<|end_message|>" before the first
//!   non-system message (or at the end when there is none);
//!   each message: ROLE[name]<|content_thinking|>…<|end_message|>, ROLE<|content_text|>…<|end_message|>,
//!   <|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>, and assistant turns end
//!   with <|content_model_end_sampling|>.

use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value};

/// Python `json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=sort)`.
fn dumps(v: &Value, sort: bool) -> String {
    if sort {
        serde_json::to_string(&sorted(v)).unwrap()
    } else {
        serde_json::to_string(v).unwrap()
    }
}

fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for k in keys {
                out.insert(k.clone(), sorted(&m[k]));
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        _ => v.clone(),
    }
}

fn get<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    v.get(k).filter(|x| !x.is_null())
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
    }
}

fn effort_value(effort: Option<&Value>) -> Result<String> {
    let num = match effort {
        None | Some(Value::Null) => 0.9,
        Some(Value::String(s)) => match s.trim() {
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "xhigh" | "max" => 0.99,
            other => bail!("Unknown reasoning_effort: {other}"),
        },
        Some(Value::Number(n)) => n.as_f64().ok_or_else(|| anyhow!("bad reasoning_effort"))?,
        Some(v) => bail!("bad reasoning_effort {v}"),
    };
    if !(0.0..=0.99).contains(&num) {
        bail!("reasoning_effort must be in [0.0, 0.99]");
    }
    // Jinja prints the float with Python's repr; 0 is written as "0"
    Ok(if num == 0.0 { "0".into() } else { format!("{num}") })
}

/// Render an OpenAI-style conversation to the prompt text (special tokens as text: tokenize with specials on).
/// Tool-call `arguments` may be a JSON string (OpenAI wire format) or an object.
pub fn render(messages: &[Value], tools: Option<&[Value]>, reasoning_effort: Option<&Value>) -> Result<String> {
    let role_token = |r: &str| -> Option<&'static str> {
        Some(match r {
            "user" => "<|message_user|>",
            "assistant" => "<|message_model|>",
            "system" => "<|message_system|>",
            "tool" => "<|message_tool|>",
            _ => return None,
        })
    };
    let mut out = String::new();
    if let Some(tools) = tools.filter(|t| !t.is_empty()) {
        let mut specs = Vec::new();
        for tool in tools {
            let f = get(tool, "function").unwrap_or(tool);
            let mut spec = Map::new();
            spec.insert("description".into(), get(f, "description").filter(|d| truthy(Some(d))).cloned().unwrap_or(Value::String(String::new())));
            spec.insert("name".into(), get(f, "name").cloned().ok_or_else(|| anyhow!("tool without a name"))?);
            spec.insert("parameters".into(), get(f, "parameters").filter(|p| truthy(Some(p))).cloned().unwrap_or(Value::Object(Map::new())));
            spec.insert("type".into(), get(tool, "type").filter(|t| truthy(Some(t))).cloned().unwrap_or(Value::String("function".into())));
            specs.push(Value::Object(spec));
        }
        out.push_str("<|message_system|>tool_declare<|content_xml|>");
        out.push_str(&dumps(&Value::Array(specs), true));
        out.push_str("<|end_message|>");
    }
    let effort = effort_value(reasoning_effort)?;
    let emit_effort = |out: &mut String| {
        out.push_str("<|message_system|><|content_text|>Thinking effort level: ");
        out.push_str(&effort);
        out.push_str("<|end_message|>");
    };
    let mut effort_emitted = false;
    for msg in messages {
        let role = get(msg, "role").and_then(Value::as_str).ok_or_else(|| anyhow!("message without a role"))?;
        let rt = role_token(role).ok_or_else(|| anyhow!("Unknown message role: {role}"))?;
        if !effort_emitted && role != "system" {
            emit_effort(&mut out);
            effort_emitted = true;
        }
        let mut header = rt.to_string();
        if role == "tool" {
            if let Some(n) = get(msg, "name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                header.push_str(n);
            } else if let Some(id) = get(msg, "tool_call_id").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                // the template appends every matching name (normally exactly one)
                for prev in messages {
                    if get(prev, "role").and_then(Value::as_str) == Some("assistant") {
                        for tc in get(prev, "tool_calls").and_then(Value::as_array).into_iter().flatten() {
                            if get(tc, "id").and_then(Value::as_str) == Some(id) {
                                if let Some(n) = get(tc, "function").and_then(|f| get(f, "name")).and_then(Value::as_str) {
                                    header.push_str(n);
                                }
                            }
                        }
                    }
                }
            }
        } else if role == "assistant" {
            if let Some(n) = get(msg, "name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                header.push_str(n);
            }
        }
        let tool_calls = get(msg, "tool_calls").and_then(Value::as_array).filter(|a| !a.is_empty());
        let reasoning = get(msg, "reasoning_content").and_then(Value::as_str).filter(|s| !s.is_empty());
        if role == "assistant" {
            if let Some(r) = reasoning {
                out.push_str(&format!("{header}<|content_thinking|>{r}<|end_message|>"));
            }
        }
        match get(msg, "content") {
            Some(Value::String(c)) if !c.is_empty() || !(role == "assistant" && (tool_calls.is_some() || reasoning.is_some())) => {
                out.push_str(&format!("{header}<|content_text|>{c}<|end_message|>"));
            }
            Some(Value::Array(parts)) if !parts.is_empty() => {
                for part in parts {
                    match part {
                        Value::String(t) => out.push_str(&format!("{header}<|content_text|>{t}<|end_message|>")),
                        _ => {
                            let ty = get(part, "type").and_then(Value::as_str);
                            match ty {
                                None | Some("text") | Some("input_text") => {
                                    let t = get(part, "text").and_then(Value::as_str).unwrap_or("");
                                    out.push_str(&format!("{header}<|content_text|>{t}<|end_message|>"));
                                }
                                Some("image") | Some("input_image") | Some("image_url") => {
                                    if role != "user" && role != "tool" {
                                        bail!("Image content is only supported for user and tool messages");
                                    }
                                    out.push_str(&format!("{header}<|content_image|><|unused_200054|><|end_message|>"));
                                }
                                Some("audio") | Some("input_audio") | Some("audio_url") => bail!("Inkling 2 does not accept audio input"),
                                Some(t) => bail!("Unsupported content part type: {t}"),
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        if role == "assistant" {
            for tc in tool_calls.into_iter().flatten() {
                let f = get(tc, "function").ok_or_else(|| anyhow!("tool call without a function"))?;
                let name = get(f, "name").and_then(Value::as_str).ok_or_else(|| anyhow!("tool call function name must be a string"))?;
                if let Some(n) = get(msg, "name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                    if n != name {
                        bail!("assistant name {n} does not match its tool call {name}");
                    }
                }
                let args = match get(f, "arguments") {
                    None => Value::Object(Map::new()),
                    Some(Value::String(s)) if s.trim().is_empty() => Value::Object(Map::new()),
                    Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| anyhow!("tool call arguments are not valid JSON: {e}"))?,
                    Some(v) => v.clone(),
                };
                let args = args.as_object().ok_or_else(|| anyhow!("tool call arguments must be an object"))?;
                out.push_str(&format!("<|message_model|>{name}<|content_invoke_tool_json|>"));
                out.push_str(&format!("{{\"name\":{},\"args\":{{", dumps(&Value::String(name.into()), true)));
                let mut keys: Vec<&String> = args.keys().collect();
                keys.sort();
                for (i, k) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&dumps(&Value::String((*k).clone()), false));
                    out.push(':');
                    out.push_str(&dumps(&args[*k], false));
                }
                out.push_str("}}<|end_message|>");
            }
            out.push_str("<|content_model_end_sampling|>");
        }
    }
    if !effort_emitted {
        emit_effort(&mut out);
    }
    Ok(out)
}

/// One parsed piece of model output.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Reasoning(String),
    Content(String),
    /// a complete tool call: name and its arguments as a JSON object string
    ToolCall { name: String, arguments: String },
    /// end of the turn (<|content_model_end_sampling|>)
    End,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Seg {
    /// between segments (expects <|message_model|>)
    Idle,
    /// after <|message_model|>: collecting the optional name until a content token
    Header,
    Thinking,
    Text,
    Tool,
}

/// Streaming parser of generated tokens.
pub struct Parser {
    ids: Ids,
    seg: Seg,
    name: Vec<u8>,
    tool: Vec<u8>,
    pending: Vec<u8>, // bytes of an incomplete UTF-8 sequence
}

#[derive(Clone, Copy)]
struct Ids {
    model: u32,
    thinking: u32,
    text: u32,
    tool: u32,
    end: u32,
    stop: u32,
}

impl Parser {
    pub fn new(tok: &crate::Tokenizer) -> Result<Self> {
        let id = |s: &str| tok.token_id(s).ok_or_else(|| anyhow!("special token {s} missing"));
        Ok(Self {
            ids: Ids {
                model: id("<|message_model|>")?,
                thinking: id("<|content_thinking|>")?,
                text: id("<|content_text|>")?,
                tool: id("<|content_invoke_tool_json|>")?,
                end: id("<|end_message|>")?,
                stop: id("<|content_model_end_sampling|>")?,
            },
            seg: Seg::Idle,
            name: Vec::new(),
            tool: Vec::new(),
            pending: Vec::new(),
        })
    }

    fn flush_text(&mut self, bytes: &[u8], out: &mut Vec<Event>) {
        self.pending.extend_from_slice(bytes);
        // emit the longest valid UTF-8 prefix, keep an incomplete tail
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(e) => e.valid_up_to() + e.error_len().unwrap(),
        };
        if valid == 0 {
            return;
        }
        let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        match self.seg {
            Seg::Thinking => out.push(Event::Reasoning(s)),
            _ => out.push(Event::Content(s)),
        }
    }

    /// Feed one generated token; returns the events it completes.
    pub fn push(&mut self, tok: &crate::Tokenizer, id: u32) -> Vec<Event> {
        let mut out = Vec::new();
        let ids = self.ids;
        if id == ids.stop {
            self.finish_segment(&mut out);
            out.push(Event::End);
            return out;
        }
        if id == ids.model {
            self.finish_segment(&mut out);
            self.seg = Seg::Header;
            self.name.clear();
            return out;
        }
        if id == ids.end {
            self.finish_segment(&mut out);
            self.seg = Seg::Idle;
            return out;
        }
        if id == ids.thinking || id == ids.text || id == ids.tool {
            if self.seg == Seg::Text || self.seg == Seg::Thinking || self.seg == Seg::Tool {
                self.finish_segment(&mut out);
            }
            self.seg = if id == ids.thinking { Seg::Thinking } else if id == ids.text { Seg::Text } else { Seg::Tool };
            self.tool.clear();
            return out;
        }
        let bytes = tok.token_bytes(id, false);
        match self.seg {
            Seg::Header => self.name.extend_from_slice(&bytes),
            Seg::Tool => self.tool.extend_from_slice(&bytes),
            Seg::Thinking | Seg::Text => self.flush_text(&bytes, &mut out),
            Seg::Idle => {
                // text outside a segment (should not happen): treat it as content
                self.seg = Seg::Text;
                self.flush_text(&bytes, &mut out);
            }
        }
        out
    }

    fn finish_segment(&mut self, out: &mut Vec<Event>) {
        match self.seg {
            Seg::Thinking | Seg::Text => {
                if !self.pending.is_empty() {
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    out.push(if self.seg == Seg::Thinking { Event::Reasoning(s) } else { Event::Content(s) });
                }
            }
            Seg::Tool => {
                let raw = String::from_utf8_lossy(&self.tool).into_owned();
                let header_name = String::from_utf8_lossy(&self.name).trim().to_string();
                let (name, arguments) = match serde_json::from_str::<Value>(&raw) {
                    Ok(v) => {
                        let name = get(&v, "name").and_then(Value::as_str).map(str::to_string).unwrap_or(header_name);
                        let args = get(&v, "args").or_else(|| get(&v, "arguments")).cloned().unwrap_or(Value::Object(Map::new()));
                        (name, serde_json::to_string(&args).unwrap())
                    }
                    Err(_) => (header_name, raw), // malformed: pass the raw text through
                };
                out.push(Event::ToolCall { name, arguments });
                self.tool.clear();
            }
            _ => {}
        }
        self.seg = Seg::Idle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn effort_formatting() {
        assert_eq!(effort_value(None).unwrap(), "0.9");
        assert_eq!(effort_value(Some(&json!("none"))).unwrap(), "0");
        assert_eq!(effort_value(Some(&json!("medium"))).unwrap(), "0.7");
        assert_eq!(effort_value(Some(&json!(0.5))).unwrap(), "0.5");
        assert!(effort_value(Some(&json!(1.0))).is_err());
    }

    #[test]
    fn renders_basic_conversation() {
        let m = vec![json!({"role": "system", "content": "Be brief."}), json!({"role": "user", "content": "Hi"})];
        assert_eq!(
            render(&m, None, None).unwrap(),
            "<|message_system|><|content_text|>Be brief.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"
        );
    }
}
