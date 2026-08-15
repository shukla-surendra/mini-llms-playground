mod chunk;
mod cli;
mod clean;
mod dataset;
mod extract;
mod walk;

use std::fs;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use cli::Args;
use dataset::{BuildStats, Record};

fn main() -> Result<()> {
    // extract.rs's catch_panic() turns a panic during one file's extraction into a normal
    // Result::Err, reported cleanly through the [skip] path below. Without silencing the
    // default hook here, the raw "thread 'main' panicked at ..." message would still print
    // to stderr for every caught panic too, ahead of and independent of our own message —
    // noisy and redundant now that it's handled, not a crash.
    std::panic::set_hook(Box::new(|_| {}));

    let args = Args::parse();

    if !args.input.is_dir() {
        anyhow::bail!("--input {} is not a directory", args.input.display());
    }
    fs::create_dir_all(&args.output)
        .with_context(|| format!("creating output directory {}", args.output.display()))?;

    let extensions = walk::parse_extensions(&args.extensions);
    let files = walk::find_files(&args.input, &extensions);
    println!("found {} file(s) matching [{}]", files.len(), args.extensions);

    let bpe = chunk::load_tokenizer()?;

    let mut stats = BuildStats {
        files_scanned: files.len(),
        ..Default::default()
    };
    let mut records: Vec<Record> = Vec::new();

    // Extraction (esp. PDF) is the slow stage — one file can take real seconds. At a few
    // hundred files with no feedback, a silent terminal is indistinguishable from a hang;
    // this bar is the fix, not decoration.
    let progress = ProgressBar::new(files.len() as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} [{bar:40.cyan/blue}] {pos}/{len} files ({eta} left) {msg}",
        )
        .unwrap()
        .progress_chars("=> "),
    );

    for path in &files {
        progress.set_message(path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string());
        let file_type = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_lowercase();

        let raw_text = match extract::extract(path) {
            Ok(Some(text)) => text,
            Ok(None) => {
                progress.inc(1);
                continue; // extension not handled — shouldn't happen, walk.rs already filtered
            }
            Err(err) => {
                progress.suspend(|| eprintln!("[skip] {}: {err:#}", path.display()));
                stats.files_failed += 1;
                progress.inc(1);
                continue;
            }
        };
        stats.files_extracted += 1;

        let cleaned = clean::normalize_whitespace(&raw_text);
        let chunks = chunk::chunk_text(&bpe, &cleaned, args.chunk_tokens, args.chunk_overlap);
        stats.chunks_before_filter += chunks.len();

        for (i, c) in chunks.into_iter().enumerate() {
            if !clean::is_quality_text(&c.text, args.min_chars, args.min_ascii_ratio) {
                stats.chunks_dropped_quality += 1;
                continue;
            }
            *stats.per_extension.entry(file_type.clone()).or_insert(0) += 1;
            records.push(Record {
                char_count: c.text.chars().count(),
                token_count: c.token_count,
                text: c.text,
                source_path: path.display().to_string(),
                file_type: file_type.clone(),
                chunk_index: i,
            });
        }
        progress.inc(1);
    }
    progress.finish_and_clear();

    if !args.no_dedupe {
        let (deduped, dropped) = dataset::dedupe(records);
        records = deduped;
        stats.chunks_dropped_duplicate = dropped;
        // Re-derive per-extension counts post-dedupe — the running counts above were
        // taken before dedupe removed anything.
        stats.per_extension.clear();
        for r in &records {
            *stats.per_extension.entry(r.file_type.clone()).or_insert(0) += 1;
        }
    }

    if records.is_empty() {
        stats.print_summary();
        anyhow::bail!("no chunks survived extraction/filtering — nothing to write");
    }

    if args.no_split {
        dataset::write_jsonl(&args.output.join("dataset.jsonl"), &records)?;
        if !args.no_emit_text {
            dataset::write_plain_text(&args.output.join("dataset.txt"), &records)?;
        }
        println!("wrote {} record(s) to {}", records.len(), args.output.join("dataset.jsonl").display());
    } else {
        let (train, test) = dataset::shuffle_and_split(records, args.train_ratio, args.seed);
        dataset::write_jsonl(&args.output.join("train.jsonl"), &train)?;
        dataset::write_jsonl(&args.output.join("test.jsonl"), &test)?;
        if !args.no_emit_text {
            dataset::write_plain_text(&args.output.join("train.txt"), &train)?;
            dataset::write_plain_text(&args.output.join("test.txt"), &test)?;
        }
        println!(
            "wrote {} train / {} test record(s) to {}",
            train.len(),
            test.len(),
            args.output.display()
        );
    }

    stats.print_summary();
    Ok(())
}
