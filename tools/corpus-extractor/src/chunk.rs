//! Token-accurate chunking via GPT-2's tokenizer (tiktoken's `r50k_base` — the same
//! encoding `tiktoken.get_encoding("gpt2")` gives you in Python, and the exact tokenizer
//! custom-gpt-10m trains against). Chunking by token count rather than character count
//! means a `--chunk-tokens 512` chunk is actually ~512 tokens once it reaches that
//! project's own training pipeline, not an estimate.

use anyhow::{Context, Result};
use tiktoken_rs::CoreBPE;

pub struct Chunk {
    pub text: String,
    pub token_count: usize,
}

/// Load the GPT-2 tokenizer once; callers reuse the same `CoreBPE` across every file
/// rather than reconstructing it per file (construction re-parses the whole merge table).
pub fn load_tokenizer() -> Result<CoreBPE> {
    tiktoken_rs::r50k_base().context("loading GPT-2 (r50k_base) tokenizer")
}

/// Split `text` into overlapping windows of `chunk_tokens` GPT-2 tokens, sliding forward
/// by `chunk_tokens - overlap` each step. The tail window may be shorter than
/// `chunk_tokens` — it isn't padded or dropped, since a short final chunk is still real,
/// usable training text.
pub fn chunk_text(bpe: &CoreBPE, text: &str, chunk_tokens: usize, overlap: usize) -> Vec<Chunk> {
    if text.trim().is_empty() || chunk_tokens == 0 {
        return Vec::new();
    }

    let ids = bpe.encode_ordinary(text);
    if ids.is_empty() {
        return Vec::new();
    }

    let overlap = overlap.min(chunk_tokens.saturating_sub(1));
    let step = (chunk_tokens - overlap).max(1);

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < ids.len() {
        let end = (start + chunk_tokens).min(ids.len());
        let window = &ids[start..end];
        if let Ok(text) = bpe.decode(window.to_vec()) {
            chunks.push(Chunk {
                text,
                token_count: window.len(),
            });
        }
        if end == ids.len() {
            break;
        }
        start += step;
    }
    chunks
}

/// Wrap the whole (already-cleaned) file text as a single unchunked `Chunk` — used by
/// `--raw-text-only`. Still tokenized against the same GPT-2 tokenizer so the resulting
/// record's `token_count` stays comparable to a chunked run's, even though no windowing
/// happens.
pub fn whole_file(bpe: &CoreBPE, text: &str) -> Vec<Chunk> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let token_count = bpe.encode_ordinary(text).len();
    vec![Chunk {
        text: text.to_string(),
        token_count,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_short_text_into_one_piece() {
        let bpe = load_tokenizer().unwrap();
        let chunks = chunk_text(&bpe, "hello world, this is a short test.", 512, 50);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].token_count > 0);
    }

    #[test]
    fn empty_text_produces_no_chunks() {
        let bpe = load_tokenizer().unwrap();
        assert!(chunk_text(&bpe, "", 512, 50).is_empty());
        assert!(chunk_text(&bpe, "   \n  ", 512, 50).is_empty());
    }

    #[test]
    fn long_text_produces_overlapping_chunks() {
        let bpe = load_tokenizer().unwrap();
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(200);
        let chunks = chunk_text(&bpe, &text, 50, 10);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.token_count <= 50);
        }
    }

    #[test]
    fn whole_file_produces_exactly_one_unwindowed_chunk() {
        let bpe = load_tokenizer().unwrap();
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(200);
        let chunks = whole_file(&bpe, &text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
        assert!(chunks[0].token_count > 50); // proves it wasn't windowed like chunk_text would
    }

    #[test]
    fn whole_file_empty_text_produces_no_chunks() {
        let bpe = load_tokenizer().unwrap();
        assert!(whole_file(&bpe, "").is_empty());
        assert!(whole_file(&bpe, "   \n  ").is_empty());
    }
}
