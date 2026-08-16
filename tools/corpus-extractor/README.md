# corpus-extractor

A Rust CLI for building an LLM training corpus at scale, in three subcommands:

- **`extract`** — point it at a local folder, get back a token-chunked JSONL (+
  plain-text) dataset. Walks recursively, extracts `.pdf`, `.epub`, `.txt`, `.md`,
  `.rs`, `.html`, `.js`, `.py`, cleans and chunks by GPT-2 token count, filters
  low-quality chunks, deduplicates, writes a shuffled train/test split.
- **`build-corpus`** — download the registered Hugging Face chat/instruction
  datasets, parse every row into `(role, text)` turns, quality-filter, and write
  `train.txt`/`test.txt`/`test_prompts.txt`.
- **`tokenize`** — stream-tokenize a text corpus (GPT-2 `r50k_base`) into flat
  `uint16` `.bin` files, the format `gpt-train` reads directly.

## Why this exists

The [`from_scratch/custom-gpt-153m`](../../from_scratch/custom-gpt-153m/) project
originally built its training corpus with a Python pipeline
(`src/gpt/data/prepare.py` + `dataset.py`) — download from Hugging Face, parse every
row in a Python loop, tokenize with `tiktoken`. That pipeline's own docstrings call
out the real bottleneck: per-row parsing runs under the GIL, forced into a
`ProcessPoolExecutor` workaround to get any parallelism at all. This tool moves that
work to Rust — ordinary thread parallelism (no GIL to route around), columnar parquet
reads via `arrow`/`parquet` instead of Python's row-by-row `datasets` library, and a
memory-safe, crash-proof extraction path for messy real-world input (PDFs in
particular). The Python pipeline stays in the tree as a fallback/reference
(`make data-legacy`, `make tokenize-legacy` in that project's `Makefile`) — both
implementations write the same `train.txt`/`test.txt`/`.bin` shapes, so either one's
output is a drop-in for the other's.

It deliberately chunks/tokenizes by **GPT-2 (`r50k_base`) token count**, not character
count — the exact tokenizer `custom-gpt-10m`/`custom-gpt-153m` train against
(`tiktoken.get_encoding("gpt2")` in Python; `tiktoken_rs::r50k_base()` here) — so a
`--chunk-tokens 512` chunk, or a `.bin` file's token stream, is exactly what that
project's training loop will see, not an estimate that drifts once real tokenization
happens.

## Build

```bash
cargo build --release
```

Binary at `target/release/corpus-extractor`. No system dependencies (PDF extraction is
pure-Rust via `pdf-extract`; no `poppler`/`libpdf` install required).

## `extract` — local folder -> JSONL/plain-text dataset

```bash
corpus-extractor extract --input /path/to/folder --output dataset_out
```

```bash
# only source code, larger chunks, no train/test split
corpus-extractor extract --input ./my-project --output out \
  --extensions rs,py,js --chunk-tokens 1024 --no-split

# feed straight into custom-gpt-10m
corpus-extractor extract --input ~/notes --output out
cp out/train.txt out/test.txt ../../from_scratch/custom-gpt-10m/data/

# one record per file, no token-windowing
corpus-extractor extract --input ./my-notes --output out --raw-text-only
```

### Features

- **Parallel extraction** — each file's extract/clean/chunk/filter pipeline is fully
  independent of every other file's, so `rayon` fans the file list out across every
  logical CPU by default (`--threads N` to cap it). Deterministic regardless of thread
  count: results are merged back in the same order the files were found in, so a
  `--seed`-reproducible run stays byte-identical whether it ran on 1 thread or 16.
- **7 input formats**: `.pdf`, `.epub`, `.txt`, `.md`, `.rs`, `.html`/`.htm`, `.js`, `.py`
  — pure-text formats read directly (lossy UTF-8 fallback on bad bytes), HTML/EPUB
  converted via `html2text`, PDF via pure-Rust `pdf-extract` (no `poppler`/system deps).
- **Recursive, `.gitignore`-aware directory walk** — same walker `ripgrep` uses, so
  `target/`, `node_modules/`, `.venv/`, and similar generated/vendored trees are skipped
  by default even outside a git repo.
- **Token-accurate chunking with overlap** — sliding GPT-2 (`r50k_base`) token windows
  (`--chunk-tokens`/`--chunk-overlap`), or, with `--raw-text-only`, each file kept as a
  single unchunked record instead.
- **Quality filtering** — drops chunks that are too short (`--min-chars`) or too
  non-ASCII (`--min-ascii-ratio`), a cheap proxy for garbled/binary extraction output.
- **Exact-duplicate removal** (hash-based, O(n)) — disable with `--no-dedupe`.
- **Seeded shuffle + train/test split** (`--train-ratio`, `--seed`), or skip it entirely
  with `--no-split` for a single combined dataset.
- **Crash-proof per-file extraction** — a panic inside a format crate (real-world PDFs
  are known to trigger this) is caught and reported as a normal skipped file instead of
  aborting the whole batch.
- **Progress bar with ETA**, plus an end-of-run summary (files scanned/extracted/failed,
  chunks before/after filtering and dedupe, per-extension counts kept).

### `extract` flag reference

| Flag | Default | What it does |
|---|---|---|
| `-i, --input <DIR>` | *(required)* | Folder to scan recursively for source files. |
| `-o, --output <DIR>` | `dataset_out` | Output directory for JSONL (and, unless `--no-emit-text`, plain-text) files. |
| `--extensions <LIST>` | `pdf,epub,txt,md,rs,html,js,py` | Comma-separated extensions to include (no dots). |
| `--chunk-tokens <N>` | `512` | Target chunk size in GPT-2 (`r50k_base`) tokens. Ignored with `--raw-text-only`. |
| `--chunk-overlap <N>` | `50` | Token overlap between consecutive chunks from the same file. Must be smaller than `--chunk-tokens`; ignored with `--raw-text-only`. |
| `--min-chars <N>` | `40` | Drop any chunk shorter than this many characters after cleaning. |
| `--min-ascii-ratio <F>` | `0.5` | Drop any chunk whose ASCII-character ratio falls below this. |
| `--train-ratio <F>` | `0.9` | Fraction of chunks written to the train split; the rest go to test. |
| `--seed <N>` | `42` | Shuffle seed — fixed by default for a reproducible split across re-runs. |
| `--no-split` | off | Skip the train/test split; write a single `dataset.jsonl`/`dataset.txt` instead. |
| `--no-dedupe` | off | Skip exact-duplicate chunk removal. |
| `--no-emit-text` | off | Skip writing the plain-text corpus alongside the JSONL. |
| `--raw-text-only` | off | One unchunked record per file (whole cleaned text) instead of token-windowed chunks. Still passes through the quality filter, dedupe, and split stages. |
| `--threads <N>` | `0` (rayon default: one per logical CPU) | Cap worker threads for extraction. |

**Pipeline stages:** `walk` (`src/walk.rs`) -> `extract` (`src/extract.rs`) -> `clean`
(`src/clean.rs`) -> `chunk` (`src/chunk.rs`) -> `filter` (`src/clean.rs`) -> `dedupe`
(`src/dataset.rs`) -> `split` (`src/dataset.rs`) -> `write` (`src/dataset.rs`).

**JSONL output** — one record per chunk (or, with `--raw-text-only`, one per file,
always `"chunk_index": 0`):

```json
{"text": "...", "source_path": "/abs/path/file.py", "file_type": "py", "chunk_index": 0, "char_count": 812, "token_count": 512}
```

**Plain-text output** (`train.txt`/`test.txt`, on by default — disable with
`--no-emit-text`) — chunk text joined by a blank line, matching `custom-gpt-10m`'s own
`train.txt` convention exactly.

## `build-corpus` — Hugging Face datasets -> train.txt/test.txt/test_prompts.txt

```bash
corpus-extractor build-corpus --list                     # show the registered datasets and exit
corpus-extractor build-corpus --data-dir data             # download + build the full corpus
corpus-extractor build-corpus --data-dir data --no-gated  # skip the gated LMSYS set (no HF_TOKEN needed)

# pool in a corpus-extractor `extract` JSONL output (books, extracted repo source, ...)
corpus-extractor build-corpus --data-dir data \
  --extra-jsonl data/books_staging/dataset.jsonl
```

Reads `HF_TOKEN` from the environment for gated datasets, same as Python's
`huggingface_hub`. Each registered source's (download, parse) pipeline is fully
independent of every other's, so `rayon` runs them across every logical CPU by default
(`--threads N` to cap it) — this is where the Rust path's real speedup over the
Python pipeline comes from: no GIL to route a `ProcessPoolExecutor` around, and
columnar parquet reads (`parquet`/`arrow`) instead of row-by-row Python dict
construction.

Three row schemas, matching `custom-gpt-153m/src/gpt/data/sources.py`'s registry
exactly (kept manually in sync — see `src/sources.rs`'s doc comment):

- **conversation** — a `List<Struct{role/from, content/value}>` column (`conversation`/
  `conversations`/`messages`) — UltraChat, SmolTalk, No Robots, LMSYS.
- **instruction** — flat `instruction`/`prompt`/`question` + optional `input` +
  `output`/`response`/`answer`/`completion` columns — Dolly, GSM8K.
- **oasst_tree** — OASST1's raw schema: a flat table of messages forming reply trees
  (`message_id`/`parent_id`/`rank`), reconstructed into linear conversations by
  walking from each root and taking the best-ranked (`rank` 0) reply at every branch.

### `build-corpus` flag reference

| Flag | Default | What it does |
|---|---|---|
| `--data-dir <DIR>` | `data` | Where to write `train.txt`/`test.txt`/`test_prompts.txt`, and (unless `--skip-download`) download parquet shards under `<data-dir>/raw/<slug>/`. |
| `--no-gated` | off | Skip datasets that require accepting terms + an HF token (currently only `lmsys/lmsys-chat-1m`). |
| `--skip-download` | off | Reuse whatever parquet files already exist under `<data-dir>/raw/<slug>/`. |
| `--max-per-dataset <N>` | `100000` | Cap on conversations kept per dataset. |
| `--min-turns <N>` | `2` | Minimum turns to keep a conversation (2 = one exchange; instruction-schema sources are always exactly 2). |
| `--min-turn-chars <N>` | `24` | Minimum characters for a single turn to pass the quality filter. |
| `--min-ascii-ratio <F>` | `0.995` | Minimum ASCII-character ratio for a single turn. |
| `--num-prompts <N>` | `50` | Held-out prompts to derive from test conversations (`test_prompts.txt`). |
| `--train-ratio <F>` | `0.9` | Fraction of conversations/documents written to the train split. |
| `--seed <N>` | `42` | Shuffle seed — fixed by default for a reproducible split. |
| `--extra-jsonl <PATH>` | — | An `extract`-produced JSONL file, pooled and shuffled in alongside the chat conversations. Repeatable. |
| `--threads <N>` | `0` (rayon default) | Cap worker threads for downloading+parsing sources in parallel. |
| `--list` | off | List the registered datasets and exit. |

## `tokenize` — text corpus -> flat uint16 `.bin`

```bash
corpus-extractor tokenize --data-dir data           # tokenizes <data-dir>/train.txt + test.txt if stale
corpus-extractor tokenize --data-dir data --force    # rebuild unconditionally
corpus-extractor tokenize --file data/extra.txt      # tokenize an arbitrary file instead
```

Streams the source file in 8 MiB chunks, cutting only at document-separator (`"\n\n"`)
or whitespace boundaries so the token stream comes out identical to tokenizing the
whole file at once — a naive fixed-size cut can split mid-word and stops BPE from
forming merges across it, inflating the token count for no reason. Writes
`<name>.bin` (little-endian `uint16`, 2 bytes/token — GPT-2's 50,257-token vocabulary
fits comfortably) plus `<name>.bin.json`, a fingerprint (`n_vocab`, a probe string's
token count, and a truncated SHA-256 of its token ids) that Python's
`gpt/data/dataset.py::load_token_array()` validates against before training — this is
the one place the two implementations' output has to be **byte-identical**, not just
equivalent, since either one may write a `.bin` the other later reads. Verified
directly: a `.bin`/`.bin.json` pair written by this command loads through Python's
`load_token_array()` with no fingerprint mismatch.

### `tokenize` flag reference

| Flag | Default | What it does |
|---|---|---|
| `--data-dir <DIR>` | `data` | Directory holding `train.txt`/`test.txt` (the default targets when `--file` isn't given). |
| `--file <PATH>` | — | Tokenize this file instead of `<data-dir>/train.txt` + `<data-dir>/test.txt`. Repeatable. |
| `--force` | off | Rebuild even if the `.bin` is already newer than its source `.txt`. |

## Known limitations

- **PDF extraction has no OCR.** `pdf-extract` reads a PDF's existing text layer; a
  scanned/image-only PDF correctly extracts to empty or near-empty text, not an error —
  there's no text there to extract without OCR, which this tool doesn't do.
- **A single malformed file can't crash an `extract` batch, but it can still take a
  real toll on output quality.** `pdf-extract` (and, defensively, the other format
  crates too) can panic internally on unusual real-world input rather than returning
  an error — hit for real on a PDF with an unusual `DeviceN` colorspace. `extract.rs`'s
  `catch_panic()` catches this and reports it as a normal `[skip]`/`files failed to
  extract` entry instead of ending the run, but a crate that panics on a case like this
  may also produce subtly wrong (not just missing) text on other malformed input it
  doesn't panic on — worth spot-checking output on a large, uncurated real-world
  folder, not just trusting a clean `files failed to extract: 0`.
- **EPUB reading order follows the book's spine**, not filename order.
- **`extract`'s chunking is not code-aware.** A 512-token window can land in the middle
  of a function — simple and fast, a known trade-off rather than a hidden one.
- **`extract`'s quality filter is intentionally minimal** (length + ASCII ratio + some
  alphabetic content) — a first pass, not a substitute for manually reviewing a sample
  of `train.jsonl`.
- **`build-corpus`'s dataset registry (`src/sources.rs`) is a manually-synced mirror**
  of `custom-gpt-153m/src/gpt/data/sources.py`'s `DATASETS`, not derived from it — that
  Python file still owns `docs/DATASETS.md` generation. Adding, removing, or
  reclassifying a dataset needs updating both.
- **`build-corpus` parquet reads require Snappy/zstd/lz4/brotli codec support** (all
  enabled by default in `Cargo.toml`) — HF-hosted parquet shards are commonly Snappy
  or zstd-compressed, and a codec feature that isn't compiled in fails that shard with
  a clear "Disabled feature at compile time" error rather than silently skipping it.
