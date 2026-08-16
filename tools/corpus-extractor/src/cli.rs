//! Command-line arguments. One struct per subcommand, one place every knob is
//! documented — the subcommand modules never re-parse or re-document anything defined
//! here.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "corpus-extractor",
    version,
    about = "Build an LLM training corpus in Rust: extract text from a local folder, \
             build a corpus from Hugging Face datasets, or tokenize a corpus into \
             flat uint16 .bin files."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Extract pdf/epub/txt/md/rs/html/js/py files from a local folder into a
    /// token-chunked JSONL (+ optional plain-text) dataset.
    Extract(ExtractArgs),
    /// Download the registered Hugging Face datasets, parse them into
    /// (role, text) turns, quality-filter, and write train.txt/test.txt/test_prompts.txt.
    BuildCorpus(BuildCorpusArgs),
    /// Stream-tokenize train.txt/test.txt (GPT-2 r50k_base) into flat uint16 .bin files.
    Tokenize(TokenizeArgs),
}

#[derive(Parser, Debug)]
pub struct ExtractArgs {
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

    /// Extract each file's cleaned text as a single unchunked record instead of
    /// GPT-2-token windows — --chunk-tokens/--chunk-overlap are ignored when this is
    /// set. Still passes through the same quality filter, dedupe, and train/test split
    /// as chunked runs; only the chunking stage itself is skipped.
    #[arg(long)]
    pub raw_text_only: bool,

    /// Worker threads for file extraction (the slow, embarrassingly-parallel stage —
    /// each file's PDF/EPUB/HTML parsing and chunking is fully independent of every
    /// other file's). 0 (the default) hands sizing to rayon, which uses one thread per
    /// logical CPU. Extraction is CPU-bound text/parsing work, not matrix math, so this
    /// is real, ordinary thread parallelism — not a GPU-eligible workload.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
}

#[derive(Parser, Debug)]
pub struct BuildCorpusArgs {
    /// Directory to write train.txt/test.txt/test_prompts.txt into, and (unless
    /// --skip-download) to download parquet shards under <data-dir>/raw/<slug>/.
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Skip datasets that require accepting terms + an HF token (currently only
    /// lmsys/lmsys-chat-1m).
    #[arg(long)]
    pub no_gated: bool,

    /// Reuse whatever parquet files already exist under <data-dir>/raw/<slug>/
    /// instead of re-downloading from Hugging Face.
    #[arg(long)]
    pub skip_download: bool,

    /// Cap on conversations kept per dataset.
    #[arg(long, default_value_t = 100_000)]
    pub max_per_dataset: usize,

    /// Minimum turns to keep a conversation. 2 is the real floor (one exchange) —
    /// instruction-schema sources are always exactly 2 turns, so anything higher
    /// silently drops 100% of single-turn sources, not just short multi-turn ones.
    #[arg(long, default_value_t = 2)]
    pub min_turns: usize,

    /// Minimum characters for a single turn to pass the quality filter.
    #[arg(long, default_value_t = 24)]
    pub min_turn_chars: usize,

    /// Minimum ASCII-character ratio for a single turn to pass the quality filter.
    #[arg(long, default_value_t = 0.995)]
    pub min_ascii_ratio: f64,

    /// Held-out prompts to derive from test conversations (test_prompts.txt).
    #[arg(long, default_value_t = 50)]
    pub num_prompts: usize,

    /// Fraction of conversations/documents written to the train split.
    #[arg(long, default_value_t = 0.9)]
    pub train_ratio: f64,

    /// Shuffle seed — fixed by default for a reproducible split across re-runs.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Path to a corpus-extractor `extract` JSONL output (e.g.
    /// data/books_staging/dataset.jsonl) — pooled and shuffled in alongside the chat
    /// conversations. Repeatable: pass multiple times to combine several extraction
    /// runs (books, multiple repos, ...) in one build.
    #[arg(long = "extra-jsonl")]
    pub extra_jsonl: Vec<PathBuf>,

    /// Worker threads for downloading+parsing sources in parallel (each registered
    /// source is fully independent of every other one). 0 (the default) hands sizing
    /// to rayon, which uses one thread per logical CPU.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// List the registered datasets and exit.
    #[arg(long)]
    pub list: bool,
}

#[derive(Parser, Debug)]
pub struct TokenizeArgs {
    /// Directory holding train.txt/test.txt (the default tokenize targets when --file
    /// isn't given).
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,

    /// Tokenize this file instead of <data-dir>/train.txt + <data-dir>/test.txt.
    /// Repeatable.
    #[arg(long = "file")]
    pub file: Vec<PathBuf>,

    /// Rebuild even if the .bin is already newer than its source .txt.
    #[arg(long)]
    pub force: bool,
}
