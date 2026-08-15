//! The extracted-chunk record type, plus dedupe / shuffle-split / write — the same
//! shape of pipeline stage custom-gpt-10m's own `build_corpus()` runs (shuffle once
//! across everything before splitting, so no single source/file dominates a stretch
//! of either split), reimplemented here rather than shared, since this is a standalone
//! Rust tool with no dependency on that project's Python package.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;

#[derive(Serialize, Clone)]
pub struct Record {
    pub text: String,
    pub source_path: String,
    pub file_type: String,
    pub chunk_index: usize,
    pub char_count: usize,
    pub token_count: usize,
}

#[derive(Default)]
pub struct BuildStats {
    pub files_scanned: usize,
    pub files_extracted: usize,
    pub files_failed: usize,
    pub chunks_before_filter: usize,
    pub chunks_dropped_quality: usize,
    pub chunks_dropped_duplicate: usize,
    pub per_extension: BTreeMap<String, usize>,
}

impl BuildStats {
    pub fn chunks_kept(&self) -> usize {
        self.per_extension.values().sum()
    }

    pub fn print_summary(&self) {
        println!("--- corpus-extractor summary ---");
        println!("files scanned:            {}", self.files_scanned);
        println!("files extracted OK:       {}", self.files_extracted);
        println!("files failed to extract:  {}", self.files_failed);
        println!("chunks before filtering:  {}", self.chunks_before_filter);
        println!("chunks dropped (quality): {}", self.chunks_dropped_quality);
        println!("chunks dropped (dupe):    {}", self.chunks_dropped_duplicate);
        println!("chunks kept:              {}", self.chunks_kept());
        println!("kept chunks by type:");
        for (ext, count) in &self.per_extension {
            println!("  {:<6} {}", ext, count);
        }
    }
}

/// Exact-duplicate-chunk removal, keyed on the chunk's text. A hash-based `HashSet`
/// (not a full string comparison) keeps this O(n) rather than O(n^2) — worth it once a
/// folder has more than a few thousand chunks, which is the common case this tool
/// targets, not an edge case.
pub fn dedupe(records: Vec<Record>) -> (Vec<Record>, usize) {
    let mut seen = HashSet::new();
    let mut kept = Vec::with_capacity(records.len());
    let mut dropped = 0usize;
    for record in records {
        let mut hasher = DefaultHasher::new();
        record.text.hash(&mut hasher);
        let digest = hasher.finish();
        if seen.insert(digest) {
            kept.push(record);
        } else {
            dropped += 1;
        }
    }
    (kept, dropped)
}

/// Shuffle with a fixed seed (reproducible across re-runs on unchanged input, matching
/// custom-gpt-10m's own `random.seed(seed)` convention) and split at `train_ratio`.
pub fn shuffle_and_split(mut records: Vec<Record>, train_ratio: f64, seed: u64) -> (Vec<Record>, Vec<Record>) {
    let mut rng = StdRng::seed_from_u64(seed);
    records.shuffle(&mut rng);
    let split_at = ((records.len() as f64) * train_ratio).round() as usize;
    let split_at = split_at.min(records.len());
    let test = records.split_off(split_at);
    (records, test)
}

pub fn write_jsonl(path: &Path, records: &[Record]) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

/// Chunks joined by a blank line — matches custom-gpt-10m's `train.txt` convention
/// directly, so this file can be dropped straight into that project's `data/` folder.
pub fn write_plain_text(path: &Path, records: &[Record]) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for (i, record) in records.iter().enumerate() {
        if i > 0 {
            writer.write_all(b"\n\n")?;
        }
        writer.write_all(record.text.as_bytes())?;
    }
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
