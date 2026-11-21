//! Byte-level BPE tokenizer (GPT-2 byte mapping, o200k / GPT-4o pre-tokenizer) loaded from GGUF metadata.
//!
//! Matches llama.cpp's `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = inkling2` behaviour:
//! no BOS/EOS added, control tokens are only recognised in the text when `parse_special` is set.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use fancy_regex::Regex;

use crate::gguf::{Gguf, Value};

/// o200k pre-tokenizer pattern (from the model's tokenizer.json).
const PRE_TOKENIZER: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
);

const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;

pub struct Tokenizer {
    tokens: Vec<String>,
    is_special: Vec<bool>,
    token_to_id: HashMap<String, u32>,
    /// (left id, right id) -> (merge rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// byte -> token id of its mapped single-character token
    byte_token: [u32; 256],
    /// mapped char -> byte (decoding)
    char_byte: HashMap<char, u8>,
    /// special tokens sorted by length, longest first
    specials: Vec<(String, u32)>,
    pre: Regex,
    pub eos: u32,
}

/// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100.. in order.
fn byte_to_char_table() -> [char; 256] {
    let mut t = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        t[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    t
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let model = g.str("tokenizer.ggml.model")?;
        ensure!(model == "gpt2", "unsupported tokenizer model `{model}`");
        let tokens: Vec<String> = g
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_arr)
            .context("missing tokenizer.ggml.tokens")?
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_owned())
            .collect();
        let types: Vec<i64> = g
            .get("tokenizer.ggml.token_type")
            .and_then(Value::as_arr)
            .map(|a| a.iter().map(|v| v.as_u64().map(|x| x as i64).unwrap_or(1)).collect())
            .unwrap_or_else(|| vec![1; tokens.len()]);
        let token_to_id: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();

        let mut merges = HashMap::new();
        if let Some(arr) = g.get("tokenizer.ggml.merges").and_then(Value::as_arr) {
            for (rank, m) in arr.iter().enumerate() {
                let Some(s) = m.as_str() else { continue };
                let Some((a, b)) = s.split_once(' ') else { continue };
                if let (Some(&ia), Some(&ib), Some(&im)) =
                    (token_to_id.get(a), token_to_id.get(b), token_to_id.get(&format!("{a}{b}")))
                {
                    merges.entry((ia, ib)).or_insert((rank as u32, im));
                }
            }
        }

        let table = byte_to_char_table();
        let mut byte_token = [0u32; 256];
        let mut char_byte = HashMap::with_capacity(256);
        for b in 0..256 {
            let c = table[b];
            char_byte.insert(c, b as u8);
            byte_token[b] = *token_to_id
                .get(&c.to_string())
                .with_context(|| format!("vocab lacks the single-byte token for byte {b:#04x}"))?;
        }

        let is_special: Vec<bool> =
            types.iter().map(|&t| t == TOKEN_TYPE_CONTROL || t == TOKEN_TYPE_USER_DEFINED).collect();
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| is_special[*i] && !t.is_empty())
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Ok(Self {
            eos: g.u64("tokenizer.ggml.eos_token_id")? as u32,
            tokens,
            is_special,
            token_to_id,
            merges,
            byte_token,
            char_byte,
            specials,
            pre: Regex::new(PRE_TOKENIZER)?,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.is_special.get(id as usize).copied().unwrap_or(false)
    }

    /// Encode text. With `parse_special`, control-token strings such as `<|message_user|>` become single tokens.
    pub fn encode(&self, text: &str, parse_special: bool) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        if !parse_special {
            self.encode_plain(text, &mut out)?;
            return Ok(out);
        }
        let mut rest = text;
        while !rest.is_empty() {
            // find the earliest special-token occurrence (longest match at that position)
            let mut best: Option<(usize, usize, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(p) = rest.find(s.as_str()) {
                    if best.is_none_or(|(bp, bl, _)| p < bp || (p == bp && s.len() > bl)) {
                        best = Some((p, s.len(), *id));
                    }
                }
            }
            match best {
                Some((p, l, id)) => {
                    self.encode_plain(&rest[..p], &mut out)?;
                    out.push(id);
                    rest = &rest[p + l..];
                }
                None => {
                    self.encode_plain(rest, &mut out)?;
                    break;
                }
            }
        }
        Ok(out)
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for m in self.pre.find_iter(text) {
            self.bpe(m?.as_str().as_bytes(), out);
        }
        Ok(())
    }

    /// Classic rank-ordered BPE over byte tokens; ties merge the leftmost pair first.
    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) {
        let mut ids: Vec<u32> = piece.iter().map(|&b| self.byte_token[b as usize]).collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(ids[i], ids[i + 1])) {
                    if best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, i, merged));
                    }
                }
            }
            let Some((rank, _, _)) = best else { break };
            // merge every occurrence of this pair, left to right
            let mut merged_ids = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() {
                    if let Some(&(r, m)) = self.merges.get(&(ids[i], ids[i + 1])) {
                        if r == rank {
                            merged_ids.push(m);
                            i += 2;
                            continue;
                        }
                    }
                }
                merged_ids.push(ids[i]);
                i += 1;
            }
            ids = merged_ids;
        }
        out.extend_from_slice(&ids);
    }

    /// Raw bytes of one token (special tokens render as their text when `render_special`).
    pub fn token_bytes(&self, id: u32, render_special: bool) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else { return Vec::new() };
        if self.is_special(id) {
            return if render_special { t.as_bytes().to_vec() } else { Vec::new() };
        }
        t.chars().map(|c| self.char_byte.get(&c).copied().unwrap_or(b'?')).collect()
    }

    pub fn decode(&self, ids: &[u32], render_special: bool) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.token_bytes(id, render_special)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
