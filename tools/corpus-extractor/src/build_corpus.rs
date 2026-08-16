//! `corpus-extractor build-corpus` — the Rust counterpart to `custom-gpt-153m`'s
//! `gpt-data` (`prepare.py::build_corpus()` + `sources.py`). Downloads the registered
//! Hugging Face datasets, parses every row into `(Role, text)` turns, quality-filters
//! them, pools them with any `--extra-jsonl` documents, and writes
//! train.txt/test.txt/test_prompts.txt.
//!
//! Parallelized with plain `rayon` across sources rather than Python's
//! `ProcessPoolExecutor` workaround — there is no GIL to route around in Rust, so each
//! source's (download, parse) pipeline just runs on an ordinary thread. `rayon`'s
//! `par_iter().map().collect::<Vec<_>>()` preserves the input order in its result
//! automatically, so the reproducible-shuffle property `prepare.py`'s docstring notes
//! (`all_conversations` must be built in the same order every run) holds for free.

use std::fs;

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::cli::BuildCorpusArgs;
use crate::corpus_text::{self, Role};
use crate::dataset::{read_extra_documents, with_commas};
use crate::hf;
use crate::parquet_rows::{load_oasst_conversations, load_rows};
use crate::sources::{self, DatasetSource, Schema};

const DOCUMENT_SEPARATOR: &str = "\n\n";
const PROMPT_FILE_SEPARATOR: &str = "\n\n<END_PROMPT>\n\n";

struct SourceResult {
    hf_id: &'static str,
    conversations: Vec<Vec<(Role, String)>>,
    log_lines: Vec<String>,
}

/// One source's entire (download, parse) pipeline — mirrors
/// `prepare.py::_download_and_parse_source()`, minus the multiprocess plumbing that
/// function needs only because Python's row-parsing loop runs under the GIL.
fn process_source(client: &HFClientSync, source: &DatasetSource, args: &BuildCorpusArgs) -> SourceResult {
    let mut log_lines = Vec::new();
    let raw_root = args.data_dir.join("raw");

    let files = if args.skip_download {
        hf::parquet_files(&raw_root.join(source.slug()))
    } else {
        match hf::download_source(client, source, &raw_root, |line| log_lines.push(line)) {
            Ok(files) => files,
            Err(err) => {
                let note = if source.gated { " (gated — accept terms and set HF_TOKEN)" } else { "" };
                log_lines.push(format!("[warn] could not download {}{note}: {err:#}", source.hf_id));
                Vec::new()
            }
        }
    };

    if files.is_empty() {
        log_lines.push(format!("[warn] no parquet files for {}, skipping", source.hf_id));
        return SourceResult { hf_id: source.hf_id, conversations: Vec::new(), log_lines };
    }

    let conversations = if source.schema == Schema::OasstTree {
        load_oasst_conversations(&files, args.max_per_dataset, args.min_turns, args.min_turn_chars, args.min_ascii_ratio)
    } else {
        load_rows(&files, source.schema, args.max_per_dataset, args.min_turns, args.min_turn_chars, args.min_ascii_ratio)
    };
    log_lines.push(format!("[parsed] {}: {} conversations kept", source.hf_id, conversations.len()));
    SourceResult { hf_id: source.hf_id, conversations, log_lines }
}

pub fn run(args: BuildCorpusArgs) -> Result<()> {
    if args.list {
        for source in sources::DATASETS {
            let flag = if source.gated { "gated " } else { "public" };
            println!("[{flag}] {:<40} {}", source.hf_id, source.name);
            println!("           {}", source.summary);
        }
        return Ok(());
    }

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("configuring the build-corpus thread pool")?;
    }

    let chosen = sources::selected(!args.no_gated);
    if args.no_gated {
        let skipped: Vec<&str> = sources::DATASETS.iter().filter(|d| d.gated).map(|d| d.hf_id).collect();
        println!("[info] skipping gated dataset(s): {}", skipped.join(", "));
    }

    let raw_root = args.data_dir.join("raw");
    fs::create_dir_all(&raw_root).with_context(|| format!("creating {}", raw_root.display()))?;

    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;

    println!(
        "[info] processing {} source(s) across up to {} worker thread(s)",
        chosen.len(),
        rayon::current_num_threads()
    );

    let results: Vec<SourceResult> = chosen.par_iter().map(|source| process_source(&client, source, &args)).collect();

    let mut all_conversations: Vec<Vec<(Role, String)>> = Vec::new();
    let mut per_source_counts: Vec<(&'static str, usize)> = Vec::new();
    for result in results {
        for line in &result.log_lines {
            println!("{line}");
        }
        per_source_counts.push((result.hf_id, result.conversations.len()));
        all_conversations.extend(result.conversations);
    }

    if all_conversations.len() < 10 {
        anyhow::bail!(
            "Too few valid conversations parsed. Check dataset access/schema — for the gated LMSYS \
             set you must accept its terms and provide HF_TOKEN, or pass --no-gated to build from \
             the public datasets only."
        );
    }

    let mut rng = StdRng::seed_from_u64(args.seed);
    all_conversations.shuffle(&mut rng);
    let split_idx = ((all_conversations.len() as f64) * args.train_ratio) as usize;
    let test_rows = all_conversations.split_off(split_idx);
    let train_rows = all_conversations;
    let total_conversations = train_rows.len() + test_rows.len();

    let chat_train_docs: Vec<String> = train_rows.iter().map(|t| corpus_text::turns_to_text(t)).collect();
    let chat_test_docs: Vec<String> = test_rows.iter().map(|t| corpus_text::turns_to_text(t)).collect();

    // extra_documents get the same shuffle-then-split treatment, independently of the
    // chat conversations, then the two train pools (and the two test pools) are pooled
    // and reshuffled together — same 4-shuffle sequence as prepare.py::build_corpus().
    let mut extra_documents: Vec<String> = Vec::new();
    for path in &args.extra_jsonl {
        let docs = read_extra_documents(path).with_context(|| format!("loading {}", path.display()))?;
        println!("[info] loaded {} extra document(s) from {}", docs.len(), path.display());
        extra_documents.extend(docs);
    }
    extra_documents.shuffle(&mut rng);
    let extra_split_idx = ((extra_documents.len() as f64) * args.train_ratio) as usize;
    let book_test_docs = extra_documents.split_off(extra_split_idx);
    let book_train_docs = extra_documents;
    let book_train_len = book_train_docs.len();
    let book_test_len = book_test_docs.len();
    let extra_total = book_train_len + book_test_len;

    let mut train_docs: Vec<String> = chat_train_docs;
    train_docs.extend(book_train_docs);
    let mut test_docs: Vec<String> = chat_test_docs;
    test_docs.extend(book_test_docs);
    train_docs.shuffle(&mut rng);
    test_docs.shuffle(&mut rng);

    fs::create_dir_all(&args.data_dir).with_context(|| format!("creating {}", args.data_dir.display()))?;
    let train_path = args.data_dir.join("train.txt");
    let test_path = args.data_dir.join("test.txt");
    let prompts_path = args.data_dir.join("test_prompts.txt");

    fs::write(&train_path, format!("{}\n", train_docs.join(DOCUMENT_SEPARATOR)))
        .with_context(|| format!("writing {}", train_path.display()))?;
    fs::write(&test_path, format!("{}\n", test_docs.join(DOCUMENT_SEPARATOR)))
        .with_context(|| format!("writing {}", test_path.display()))?;

    let prompts: Vec<String> = test_rows
        .iter()
        .take(args.num_prompts)
        .filter_map(|t| corpus_text::prompt_from_turns(t))
        .collect();
    let prompts_content = if prompts.is_empty() {
        String::new()
    } else {
        format!("{}{}", prompts.join(PROMPT_FILE_SEPARATOR), PROMPT_FILE_SEPARATOR)
    };
    fs::write(&prompts_path, prompts_content).with_context(|| format!("writing {}", prompts_path.display()))?;

    println!("\n=== Corpus built ===");
    for (hf_id, count) in &per_source_counts {
        println!("  {:<40} {:>8} conversations", hf_id, with_commas(*count));
    }
    println!("  {:<40} {:>8}", "TOTAL", with_commas(total_conversations));
    if extra_total > 0 {
        println!("  {:<40} {:>8}", "extra_documents (books/repos)", with_commas(extra_total));
    }
    println!(
        "\n  train: {} ({} conversations + {} extra docs)",
        train_path.display(),
        with_commas(train_rows.len()),
        with_commas(book_train_len)
    );
    println!(
        "  test:  {} ({} conversations + {} extra docs)",
        test_path.display(),
        with_commas(test_rows.len()),
        with_commas(book_test_len)
    );
    println!("  prompts: {} ({} prompts)", prompts_path.display(), prompts.len());

    Ok(())
}
