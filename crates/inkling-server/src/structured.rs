//! Structured output (`response_format` json_schema / json_object) with llguidance.
//!
//! The model's turn is a series of segments (`<|message_model|>[name]<|content_thinking|>…<|end_message|>`,
//! `…<|content_text|>…<|end_message|>`, then `<|content_model_end_sampling|>`). The constraint leaves the reasoning
//! free and applies to the answer only:
//! - Free: anything but a tool call or the end of the turn (the answer must come);
//! - Answer (after `<|content_text|>`): the tokens the JSON schema allows, and `<|end_message|>` once the JSON is
//!   complete;
//! - Close (after `<|end_message|>`): only `<|content_model_end_sampling|>`.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use inkling_core::tokenizer::Tokenizer;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};
use serde_json::{Value, json};

/// The schema a request asks for, or None for free text. Errors are the client's (400).
pub fn requested_schema(body: &Value) -> Result<Option<Value>> {
    let Some(rf) = body.get("response_format") else { return Ok(None) };
    match rf.get("type").and_then(Value::as_str) {
        None | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(json!({ "type": "object" }))),
        Some("json_schema") => {
            let schema = rf
                .get("json_schema")
                .and_then(|j| j.get("schema"))
                .ok_or_else(|| anyhow!("response_format json_schema needs json_schema.schema"))?;
            Ok(Some(schema.clone()))
        }
        Some(other) => Err(anyhow!("response_format type `{other}` is not supported (text, json_object, json_schema)")),
    }
}

/// Built once per server: the vocabulary as llguidance sees it.
pub struct Guidance {
    factory: ParserFactory,
    n_vocab: usize,
    text: u32,
    end: u32,
    end_sampling: u32,
    tool: u32,
}

impl Guidance {
    pub fn new(tok: &Tokenizer) -> Result<Self> {
        let id = |s: &str| tok.token_id(s).with_context(|| format!("token {s} missing"));
        let (text, end, end_sampling, tool) =
            (id("<|content_text|>")?, id("<|end_message|>")?, id("<|content_model_end_sampling|>")?, id("<|content_invoke_tool_json|>")?);
        let n_vocab = tok.n_vocab();
        let words: Vec<Vec<u8>> = (0..n_vocab as u32)
            .map(|t| {
                if tok.is_special(t) {
                    let mut w = vec![TokTrie::SPECIAL_TOKEN_MARKER];
                    w.extend(tok.token_bytes(t, true));
                    w
                } else {
                    tok.token_bytes(t, false)
                }
            })
            .collect();
        // `<|end_message|>` closes the answer: it is the end of sequence for the grammar
        let trie = TokTrie::from(&TokRxInfo::new(n_vocab as u32, end), &words);
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&env)?;
        factory.quiet();
        Ok(Self { factory, n_vocab, text, end, end_sampling, tool })
    }

    pub fn start(&self, schema: Value) -> Result<Constrained> {
        let parser = self.factory.create_parser(TopLevelGrammar::from_json_schema(schema)).context("JSON schema not usable")?;
        let matcher = Matcher::new(Ok(parser));
        if let Some(e) = matcher_error(&matcher) {
            return Err(anyhow!("JSON schema not usable: {e}"));
        }
        Ok(Constrained { matcher, phase: Phase::Free, words: self.n_vocab.div_ceil(32), text: self.text, end: self.end, end_sampling: self.end_sampling, tool: self.tool })
    }
}

fn matcher_error(m: &Matcher) -> Option<String> {
    m.is_error().then(|| m.get_error().unwrap_or_default())
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    Free,
    Answer,
    Close,
}

/// The constraint of one request.
pub struct Constrained {
    matcher: Matcher,
    phase: Phase,
    words: usize,
    text: u32,
    end: u32,
    end_sampling: u32,
    tool: u32,
}

impl Constrained {
    fn only(&self, t: u32) -> Vec<u32> {
        let mut bits = vec![0u32; self.words];
        bits[(t / 32) as usize] |= 1 << (t % 32);
        bits
    }

    /// The allowed tokens for the next step (one bit per token).
    pub fn mask(&mut self) -> Result<Option<Vec<u32>>> {
        Ok(Some(match self.phase {
            Phase::Free => {
                let mut bits = vec![u32::MAX; self.words];
                for t in [self.tool, self.end_sampling] {
                    bits[(t / 32) as usize] &= !(1 << (t % 32));
                }
                bits
            }
            Phase::Answer => {
                if self.matcher.is_stopped() {
                    return Ok(Some(self.only(self.end)));
                }
                let m = self.matcher.compute_mask_or_eos().map_err(|e| anyhow!("structured output: {e}"))?;
                let mut bits = m.as_slice().to_vec();
                bits.resize(self.words, 0);
                bits
            }
            Phase::Close => self.only(self.end_sampling),
        }))
    }

    /// Follows the generated token.
    pub fn push(&mut self, t: u32) -> Result<()> {
        match self.phase {
            Phase::Free if t == self.text => self.phase = Phase::Answer,
            Phase::Free => {}
            Phase::Answer if t == self.end => self.phase = Phase::Close,
            Phase::Answer => self.matcher.consume_token(t).map_err(|e| anyhow!("structured output: {e}"))?,
            Phase::Close => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::requested_schema;
    use serde_json::json;

    #[test]
    fn no_response_format_or_text_means_free_text() {
        assert!(requested_schema(&json!({})).unwrap().is_none());
        assert!(requested_schema(&json!({"response_format": {"type": "text"}})).unwrap().is_none());
    }

    #[test]
    fn json_object_is_any_object() {
        assert_eq!(requested_schema(&json!({"response_format": {"type": "json_object"}})).unwrap(), Some(json!({"type": "object"})));
    }

    #[test]
    fn json_schema_takes_the_schema() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]});
        let body = json!({"response_format": {"type": "json_schema", "json_schema": {"name": "r", "strict": true, "schema": schema}}});
        assert_eq!(requested_schema(&body).unwrap(), Some(schema));
    }

    #[test]
    fn json_schema_without_schema_and_unknown_types_are_errors() {
        assert!(requested_schema(&json!({"response_format": {"type": "json_schema", "json_schema": {"name": "r"}}})).is_err());
        assert!(requested_schema(&json!({"response_format": {"type": "grammar"}})).is_err());
    }
}
