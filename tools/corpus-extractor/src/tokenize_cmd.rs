//! `corpus-extractor tokenize` — the Rust counterpart to `custom-gpt-153m`'s
//! `gpt-tokenize` (`dataset.py::build_token_bin()`). Stream-tokenizes a text corpus
//! (GPT-2 `r50k_base`, via `chunk::load_tokenizer()` — the same tokenizer `extract`
//! chunks against) into a flat `uint16` `.bin`, plus a `.bin.json` fingerprint file
//! Python's `load_token_array()` can validate against, whichever implementation wrote
//! it.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use tiktoken_rs::CoreBPE;

use crate::chunk;
use crate::cli::TokenizeArgs;
use crate::dataset::with_commas;

/// GPT-2 (`r50k_base`)'s fixed vocabulary size — `tiktoken.get_encoding("gpt2").n_vocab`
/// in Python. Not exposed by `tiktoken-rs`'s public API, so hardcoded here as the same
/// well-known constant `dataset.py`'s module docstring cites.
const N_VOCAB: usize = 50257;

/// Text read per streaming-tokenize chunk — matches `dataset.py`'s `CHUNK_CHARS`.
const CHUNK_CHARS: usize = 8 * 1024 * 1024;

/// Deliberately mixes ASCII words, a digit run, the document separator, and a
/// non-ASCII character — the same probe `dataset.py::FINGERPRINT_PROBE` uses, so a
/// `.bin.json` written by either implementation stays comparable.
const FINGERPRINT_PROBE: &str = "The quick brown fox 12345 \n\n caf\u{e9}";

const DOCUMENT_SEPARATOR: &str = "\n\n";

fn token_bin_path(text_path: &Path) -> PathBuf {
    text_path.with_extension("bin")
}

fn bin_meta_path(bin_path: &Path) -> PathBuf {
    let mut s = bin_path.as_os_str().to_os_string();
    s.push(".json");
    PathBuf::from(s)
}

fn tmp_path_for(bin_path: &Path) -> PathBuf {
    let mut s = bin_path.as_os_str().to_os_string();
    s.push(".tmp");
    PathBuf::from(s)
}

fn is_stale(text_path: &Path, bin_path: &Path) -> Result<bool> {
    if !bin_path.exists() {
        return Ok(true);
    }
    let bin_mtime = fs::metadata(bin_path)?.modified()?;
    let text_mtime = fs::metadata(text_path)?.modified()?;
    Ok(bin_mtime < text_mtime)
}

fn tokenizer_fingerprint(bpe: &CoreBPE) -> serde_json::Value {
    let ids = bpe.encode_ordinary(FINGERPRINT_PROBE);
    let joined = ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
    let digest = Sha256::digest(joined.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    serde_json::json!({
        "n_vocab": N_VOCAB,
        "probe_ids": ids.len(),
        "probe_sha256": &hex[..16],
    })
}

/// Return `(encodable_now, carry_over)`. Cutting at a document separator (a plain
/// whitespace run) can never land mid-word; `max_buffer` bounds the carry for a single
/// document longer than the cap, where cutting immediately before a space is the same
/// safety property without waiting for a separator to appear — direct port of
/// `dataset.py::_split_at_document_boundary()`.
fn split_at_document_boundary(buffer: &str, is_final: bool, max_buffer: usize) -> (String, String) {
    if is_final {
        return (buffer.to_string(), String::new());
    }
    if let Some(byte_idx) = buffer.rfind(DOCUMENT_SEPARATOR) {
        let cut = byte_idx + DOCUMENT_SEPARATOR.len();
        return (buffer[..cut].to_string(), buffer[cut..].to_string());
    }
    if buffer.chars().count() < max_buffer {
        return (String::new(), buffer.to_string());
    }
    match buffer.rfind(' ') {
        Some(space) if space > 0 => (buffer[..space].to_string(), buffer[space..].to_string()),
        _ => (buffer.to_string(), String::new()), // no safe whitespace at all; accept the split rather than hang
    }
}

/// Stream-tokenize `text_path` into a flat uint16 `.bin`, atomically. Returns the
/// token count. Reads raw bytes (not Python's character-mode `read()`), so a chunk
/// boundary can land mid-UTF-8-sequence; incomplete trailing bytes are carried forward
/// as raw bytes rather than converted, keeping every `str::from_utf8` call on a
/// complete, valid slice.
pub fn build_token_bin(bpe: &CoreBPE, text_path: &Path, bin_path: &Path, mut progress: impl FnMut(usize)) -> Result<usize> {
    if !text_path.exists() {
        anyhow::bail!("{} not found — nothing to tokenize.", text_path.display());
    }
    if let Some(parent) = bin_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let max_buffer = (CHUNK_CHARS * 4).max(1 << 20);

    let tmp_path = tmp_path_for(bin_path);
    let mut reader = BufReader::new(File::open(text_path).with_context(|| format!("opening {}", text_path.display()))?);
    let mut writer = BufWriter::new(File::create(&tmp_path).with_context(|| format!("creating {}", tmp_path.display()))?);

    let mut total = 0usize;
    let mut max_id = 0usize;
    let mut carry = String::new();
    let mut pending_bytes: Vec<u8> = Vec::new();
    let mut read_buf = vec![0u8; CHUNK_CHARS];

    loop {
        let read_len = reader.read(&mut read_buf).context("reading source text")?;
        let is_final = read_len == 0;

        pending_bytes.extend_from_slice(&read_buf[..read_len]);
        let (decoded, tail) = match std::str::from_utf8(&pending_bytes) {
            Ok(s) => (s.to_string(), Vec::new()),
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                let s = std::str::from_utf8(&pending_bytes[..valid_up_to]).unwrap().to_string();
                (s, pending_bytes[valid_up_to..].to_vec())
            }
        };
        pending_bytes = tail;

        let buffer = carry + &decoded;
        if buffer.is_empty() && is_final {
            break;
        }
        let (encodable, next_carry) = split_at_document_boundary(&buffer, is_final, max_buffer);
        carry = next_carry;

        if !encodable.is_empty() {
            let ids = bpe.encode_ordinary(&encodable);
            if !ids.is_empty() {
                let mut out = Vec::with_capacity(ids.len() * 2);
                for &id in &ids {
                    max_id = max_id.max(id);
                    out.extend_from_slice(&(id as u16).to_le_bytes());
                }
                writer.write_all(&out)?;
                total += ids.len();
                progress(total);
            }
        }
        if is_final {
            break;
        }
    }
    writer.flush()?;

    if max_id > u16::MAX as usize {
        drop(writer);
        fs::remove_file(&tmp_path).ok();
        anyhow::bail!("Token id {max_id} exceeds uint16's range — widen the on-disk token dtype and rebuild every .bin.");
    }

    let meta = serde_json::json!({
        "tokenizer": tokenizer_fingerprint(bpe),
        "tokens": total,
        "source": text_path.display().to_string(),
    });
    fs::write(bin_meta_path(bin_path), serde_json::to_string_pretty(&meta)?)
        .with_context(|| format!("writing {}", bin_meta_path(bin_path).display()))?;

    drop(writer);
    fs::rename(&tmp_path, bin_path) // atomic: a killed run never leaves a half-written .bin
        .with_context(|| format!("renaming {} -> {}", tmp_path.display(), bin_path.display()))?;
    Ok(total)
}

fn human(mut n: f64) -> String {
    for unit in ["", "K", "M", "B"] {
        if n.abs() < 1000.0 {
            return format!("{n:.0}{unit}");
        }
        n /= 1000.0;
    }
    format!("{n:.1}T")
}

pub fn run(args: TokenizeArgs) -> Result<()> {
    let bpe = chunk::load_tokenizer()?;

    let targets: Vec<PathBuf> = if !args.file.is_empty() {
        args.file.clone()
    } else {
        vec![args.data_dir.join("train.txt"), args.data_dir.join("test.txt")]
    };

    let mut total_tokens = 0usize;
    for text_path in &targets {
        let bin_path = token_bin_path(text_path);
        if !text_path.exists() {
            println!("skip   {} (not found)", text_path.display());
            continue;
        }
        if !args.force && !is_stale(text_path, &bin_path)? {
            println!("ok     {} is up to date", bin_path.display());
            continue;
        }

        let src_mb = fs::metadata(text_path)?.len() as f64 / (1024.0 * 1024.0);
        println!("build  {} ({src_mb:.0} MB) -> {}", text_path.display(), bin_path.display());
        let started = Instant::now();
        let mut last_report = started;

        let count = build_token_bin(&bpe, text_path, &bin_path, |count| {
            let now = Instant::now();
            if now.duration_since(last_report).as_secs_f64() < 5.0 {
                return;
            }
            last_report = now;
            let elapsed = now.duration_since(started).as_secs_f64().max(1e-6);
            let rate = count as f64 / elapsed;
            println!("       {} tokens  ({}/s)", human(count as f64), human(rate));
        })?;

        let elapsed = started.elapsed().as_secs_f64();
        let size_mb = fs::metadata(&bin_path)?.len() as f64 / (1024.0 * 1024.0);
        total_tokens += count;
        println!("done   {} tokens in {elapsed:.0}s -> {size_mb:.0} MB (uint16)", human(count as f64));
    }

    if total_tokens > 0 {
        println!("\nTotal tokenized this run: {}", with_commas(total_tokens));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_chunk_is_encoded_whole_with_no_carry() {
        let (encodable, carry) = split_at_document_boundary("tail text", true, 100);
        assert_eq!(encodable, "tail text");
        assert_eq!(carry, "");
    }

    #[test]
    fn cuts_at_the_last_document_separator() {
        let (encodable, carry) = split_at_document_boundary("doc one\n\ndoc two star", false, 1000);
        assert_eq!(encodable, "doc one\n\n");
        assert_eq!(carry, "doc two star");
    }

    #[test]
    fn holds_everything_below_max_buffer_when_no_separator_seen_yet() {
        let (encodable, carry) = split_at_document_boundary("no separator here", false, 1000);
        assert_eq!(encodable, "");
        assert_eq!(carry, "no separator here");
    }

    #[test]
    fn falls_back_to_last_space_once_max_buffer_is_exceeded() {
        let buffer = "aaaa bbbb cccc";
        let (encodable, carry) = split_at_document_boundary(buffer, false, 5);
        assert_eq!(encodable, "aaaa bbbb");
        assert_eq!(carry, " cccc");
    }

    #[test]
    fn accepts_the_whole_buffer_when_max_buffer_exceeded_with_no_whitespace() {
        let buffer = "nowhitespacehere";
        let (encodable, carry) = split_at_document_boundary(buffer, false, 5);
        assert_eq!(encodable, "nowhitespacehere");
        assert_eq!(carry, "");
    }
}
