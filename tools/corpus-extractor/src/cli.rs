//! Command-line arguments. One struct, one place every knob is documented —
//! `main.rs` never re-parses or re-documents anything defined here.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "corpus-extractor",
    version,
    about = "Extract pdf/epub/txt/md/rs/html/js/py files from a folder into a token-chunked LLM training dataset (JSONL, optional plain text, train/test split)."
)]
pub struct Args {
    /// Folder to scan recursively for source files.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output directory — JSONL (and, unless --no-emit-text, plain-text) files are written here.
    #[arg(short, long, default_value = "dataset_out")]
    pub output: PathBuf,

    /// Comma-separated extensions to include (no dots).
    #[arg(long, default_value = "pdf,epub,txt,md,rs,html,js,py")]
    pub extensions: String,

    /// Target chunk size, in GPT-2 (r50k_base) tokens — the same tokenizer
    /// custom-gpt-10m trains against, so token counts here match what that
    /// project will actually see.
    #[arg(long, default_value_t = 512)]
    pub chunk_tokens: usize,

    /// Token overlap between consecutive chunks from the same file — keeps
    /// context from being hard-cut exactly at a chunk boundary. Must be
    /// smaller than --chunk-tokens.
    #[arg(long, default_value_t = 50)]
    pub chunk_overlap: usize,

    /// Drop any chunk shorter than this many characters after cleaning —
    /// cheap filter against near-empty trailing fragments.
    #[arg(long, default_value_t = 40)]
    pub min_chars: usize,

    /// Drop any chunk whose ASCII-character ratio falls below this — cheap
    /// filter against binary/garbled extraction output (mostly relevant to
    /// PDF text extraction, which can produce mojibake on malformed PDFs).
    #[arg(long, default_value_t = 0.5)]
    pub min_ascii_ratio: f64,

    /// Fraction of chunks written to the train split; the rest go to test.
    #[arg(long, default_value_t = 0.9)]
    pub train_ratio: f64,

    /// Shuffle seed — fixed by default so re-running on unchanged input
    /// reproduces the same train/test split.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Skip the train/test split entirely; write a single dataset.jsonl
    /// (and dataset.txt) instead of train/test pairs.
    #[arg(long)]
    pub no_split: bool,

    /// Skip exact-duplicate chunk removal.
    #[arg(long)]
    pub no_dedupe: bool,

    /// Skip writing a plain-text corpus alongside the JSONL. By default a
    /// plain-text file (train.txt/test.txt, or dataset.txt with
    /// --no-split) is written too — chunks joined by a blank line,
    /// matching custom-gpt-10m's train.txt convention directly.
    #[arg(long)]
    pub no_emit_text: bool,
}
