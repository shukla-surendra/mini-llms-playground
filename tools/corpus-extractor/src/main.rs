mod build_corpus;
mod chunk;
mod cli;
mod clean;
mod corpus_text;
mod dataset;
mod extract;
mod hf;
mod parquet_rows;
mod sources;
mod tokenize_cmd;
mod walk;

use std::fs;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use cli::{Cli, Command, ExtractArgs};
use dataset::{BuildStats, Record};

/// One file's worth of extraction output, computed independently of every other file —
/// this independence is exactly what makes the extraction stage embarrassingly
/// parallel (see run_extract()'s par_iter() below). Merged into the shared BuildStats/
/// records sequentially, after every file has finished, rather than updating shared
/// mutable state from inside the parallel closure itself.
#[derive(Default)]
struct FileOutcome {
    extracted: bool,
    failed: bool,
    chunks_before_filter: usize,
    chunks_dropped_quality: usize,
    records: Vec<Record>,
}

fn main() -> Result<()> {
    // extract.rs's catch_panic() turns a panic during one file's extraction into a normal
    // Result::Err, reported cleanly through the [skip] path below. Without silencing the
    // default hook here, the raw "thread 'main' panicked at ..." message would still print
    // to stderr for every caught panic too, ahead of and independent of our own message —
    // noisy and redundant now that it's handled, not a crash.
    std::panic::set_hook(Box::new(|_| {}));

    let cli = Cli::parse();
    match cli.command {
        Command::Extract(args) => run_extract(args),
        Command::BuildCorpus(args) => build_corpus::run(args),
        Command::Tokenize(args) => tokenize_cmd::run(args),
    }
}

fn run_extract(args: ExtractArgs) -> Result<()> {
    if !args.input.is_dir() {
        anyhow::bail!("--input {} is not a directory", args.input.display());
    }
    fs::create_dir_all(&args.output)
        .with_context(|| format!("creating output directory {}", args.output.display()))?;

    let extensions = walk::parse_extensions(&args.extensions);
    let files = walk::find_files(&args.input, &extensions);
    println!("found {} file(s) matching [{}]", files.len(), args.extensions);

    // 0 (the default) leaves sizing to rayon's own default (one thread per logical
    // CPU) — this only exists to let a run be throttled deliberately, e.g. to leave
    // headroom on a machine doing something else at the same time.
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("configuring the extraction thread pool")?;
    }

    let bpe = chunk::load_tokenizer()?;

    let mut stats = BuildStats {
        files_scanned: files.len(),
        ..Default::default()
    };
    let mut records: Vec<Record> = Vec::new();

    // Extraction (esp. PDF) is the slow stage — one file can take real seconds. At a few
    // hundred files with no feedback, a silent terminal is indistinguishable from a hang;
    // this bar is the fix, not decoration. ProgressBar's methods take &self and use
    // interior mutability (it's Clone + Sync by design, meant to be shared across
    // threads exactly like this) — calling .inc()/.suspend() concurrently from the
    // par_iter() closure below is the intended, supported usage, not a data race.
    let progress = ProgressBar::new(files.len() as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} [{bar:40.cyan/blue}] {pos}/{len} files ({eta} left) {msg}",
        )
        .unwrap()
        .progress_chars("=> "),
    );

    // Each file's extraction -> clean -> chunk -> quality-filter is fully independent
    // of every other file's — nothing here reads or writes another file's output. That
    // independence is what makes this embarrassingly parallel: par_iter() hands files
    // out across rayon's thread pool and collects each one's FileOutcome, with no
    // shared mutable state touched from inside the closure itself (the progress bar
    // aside, which is explicitly designed for concurrent access — see above). Stats and
    // records are only merged afterward, sequentially, once every file is done.
    let outcomes: Vec<FileOutcome> = files
        .par_iter()
        .map(|path| {
            progress.set_message(path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string());
            let file_type = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("unknown")
                .to_lowercase();

            let mut outcome = FileOutcome::default();

            let raw_text = match extract::extract(path) {
                Ok(Some(text)) => text,
                Ok(None) => {
                    progress.inc(1);
                    return outcome; // extension not handled — shouldn't happen, walk.rs already filtered
                }
                Err(err) => {
                    progress.suspend(|| eprintln!("[skip] {}: {err:#}", path.display()));
                    outcome.failed = true;
                    progress.inc(1);
                    return outcome;
                }
            };
            outcome.extracted = true;

            let cleaned = clean::normalize_whitespace(&raw_text);
            let chunks = if args.raw_text_only {
                chunk::whole_file(&bpe, &cleaned)
            } else {
                chunk::chunk_text(&bpe, &cleaned, args.chunk_tokens, args.chunk_overlap)
            };
            outcome.chunks_before_filter = chunks.len();

            for (i, c) in chunks.into_iter().enumerate() {
                if !clean::is_quality_text(&c.text, args.min_chars, args.min_ascii_ratio) {
                    outcome.chunks_dropped_quality += 1;
                    continue;
                }
                outcome.records.push(Record {
                    char_count: c.text.chars().count(),
                    token_count: c.token_count,
                    text: c.text,
                    source_path: path.display().to_string(),
                    file_type: file_type.clone(),
                    chunk_index: i,
                });
            }
            progress.inc(1);
            outcome
        })
        .collect();
    progress.finish_and_clear();

    for outcome in outcomes {
        if outcome.extracted {
            stats.files_extracted += 1;
        }
        if outcome.failed {
            stats.files_failed += 1;
        }
        stats.chunks_before_filter += outcome.chunks_before_filter;
        stats.chunks_dropped_quality += outcome.chunks_dropped_quality;
        for record in &outcome.records {
            *stats.per_extension.entry(record.file_type.clone()).or_insert(0) += 1;
        }
        records.extend(outcome.records);
    }

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
