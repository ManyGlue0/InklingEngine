//! Exact-match tokenization against reference ids produced by llama.cpp's `llama-tokenize --no-escape`.
//! Needs the model: set INKLING_MODEL=path/to/Inkling-2-Mini-*.gguf (skipped otherwise).

use std::path::PathBuf;

use inkling_core::{Gguf, Tokenizer};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data").join(name)
}

fn load() -> Option<Tokenizer> {
    let path = std::env::var("INKLING_MODEL").ok()?;
    Some(Tokenizer::from_gguf(&Gguf::open(path).expect("open model")).expect("tokenizer"))
}

fn expected(name: &str) -> Vec<u32> {
    let s = std::fs::read_to_string(data(name)).unwrap();
    s.trim().trim_start_matches('[').trim_end_matches(']').split(',').map(|x| x.trim().parse().unwrap()).collect()
}

#[test]
fn corpus_matches_reference() {
    let Some(tok) = load() else { return eprintln!("INKLING_MODEL not set; skipping") };
    let text = std::fs::read_to_string(data("tokenizer_corpus.txt")).unwrap();
    let ids = tok.encode(&text, false).unwrap();
    assert_eq!(ids, expected("tokenizer_corpus.ids.json"));
    assert_eq!(tok.decode(&ids, true), text);
}

#[test]
fn special_tokens_match_reference() {
    let Some(tok) = load() else { return eprintln!("INKLING_MODEL not set; skipping") };
    let text = std::fs::read_to_string(data("tokenizer_special.txt")).unwrap();
    let ids = tok.encode(&text, true).unwrap();
    assert_eq!(ids, expected("tokenizer_special.ids.json"));
    assert_eq!(tok.decode(&ids, true), text);
}
