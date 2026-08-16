# corpus-extractor

A small Rust CLI: point it at a folder, get back a token-chunked LLM training dataset.
Walks a directory recursively, extracts text from `.pdf`, `.epub`, `.txt`, `.md`, `.rs`,
`.html`, `.js`, and `.py` files, cleans and chunks it by GPT-2 token count, filters out
low-quality chunks, deduplicates, and writes a shuffled train/test split — both as JSONL
(with metadata) and as a plain-text corpus.

## Why this exists

The [`from_scratch/`](../../from_scratch/) projects in this repo train on Hugging
Face-hosted conversational datasets via a Python pipeline
(`custom-gpt-10m/src/gpt/data/prepare.py`). This tool is the complementary path: turning
an arbitrary **local folder** — notes, docs, a codebase, scraped HTML, PDFs — into that
same shape of training data, for anyone who wants to train on their own material instead
of (or alongside) the public chat corpora.

It deliberately chunks by **GPT-2 (`r50k_base`) token count**, not character count — the
exact tokenizer `custom-gpt-10m`/`custom-gpt-153m` train against
(`tiktoken.get_encoding("gpt2")` in that project's Python code; `tiktoken_rs::r50k_base()`
here) — so a `--chunk-tokens 512` chunk really is ~512 tokens once it reaches that
project's training loop, not an estimate that drifts once real tokenization happens.

## Features

- **Parallel extraction** — each file's extract/clean/chunk/filter pipeline is fully
  independent of every other file's, so `rayon` fans the file list out across every
  logical CPU by default (`--threads N` to cap it). Deterministic regardless of thread
  count: results are merged back in the same order the files were found in, so a
  `--seed`-reproducible run stays byte-identical whether it ran on 1 thread or 16
  (verified directly — `--threads 1` and the default parallel run produce identical
  `dataset.jsonl` output). This is CPU-bound parsing/string work, not matrix math —
  there's no GPU-eligible part of this pipeline to offload.
- **7 input formats**: `.pdf`, `.epub`, `.txt`, `.md`, `.rs`, `.html`/`.htm`, `.js`, `.py`
  — pure-text formats read directly (lossy UTF-8 fallback on bad bytes), HTML/EPUB
  converted via `html2text`, PDF via pure-Rust `pdf-extract` (no `poppler`/system deps).
- **Recursive, `.gitignore`-aware directory walk** — same walker `ripgrep` uses, so
  `target/`, `node_modules/`, `.venv/`, and similar generated/vendored trees are skipped
  by default even outside a git repo.
- **Token-accurate chunking with overlap** — sliding GPT-2 (`r50k_base`) token windows
  (`--chunk-tokens`/`--chunk-overlap`), or, with `--raw-text-only`, each file kept as a
  single unchunked record instead.
- **Whitespace normalization** before chunking (line-ending unification, collapsed blank
  runs) so chunk boundaries aren't computed against extraction noise.
- **Quality filtering** — drops chunks that are too short (`--min-chars`) or too
  non-ASCII (`--min-ascii-ratio`), a cheap proxy for garbled/binary extraction output.
- **Exact-duplicate removal** (hash-based, O(n)) — disable with `--no-dedupe`.
- **Seeded shuffle + train/test split** (`--train-ratio`, `--seed`), or skip it entirely
  with `--no-split` for a single combined dataset.
- **Dual output**: JSONL with per-chunk metadata, plus an optional plain-text corpus
  (`--no-emit-text` to skip) formatted as a drop-in for `custom-gpt-10m`'s `data/` folder.
- **Crash-proof per-file extraction** — a panic inside a format crate (real-world PDFs
  are known to trigger this) is caught and reported as a normal skipped file instead of
  aborting the whole batch.
- **Progress bar with ETA** during extraction, plus an end-of-run summary (files
  scanned/extracted/failed, chunks before/after filtering and dedupe, per-extension
  counts kept).

## Build

```bash
cargo build --release
```

Binary at `target/release/corpus-extractor`. No system dependencies (PDF extraction is
pure-Rust via `pdf-extract`; no `poppler`/`libpdf` install required).

## Usage

```bash
corpus-extractor --input /path/to/folder --output dataset_out
```

```bash
# only source code, larger chunks, no train/test split
corpus-extractor --input ./my-project --output out \
  --extensions rs,py,js --chunk-tokens 1024 --no-split

# feed straight into custom-gpt-10m
corpus-extractor --input ~/notes --output out
cp out/train.txt out/test.txt ../../from_scratch/custom-gpt-10m/data/
```

```bash
# one record per file, no token-windowing — useful when a downstream step wants each
# document whole rather than pre-chunked
corpus-extractor --input ./my-notes --output out --raw-text-only
```

### Flag reference

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
| `--threads <N>` | `0` (rayon default: one per logical CPU) | Cap worker threads for extraction. Output is identical regardless of this value — see "Parallel extraction" above. |
| `-h, --help` | — | Print full help. |
| `-V, --version` | — | Print version. |

Every flag's default and reasoning is also documented inline in `src/cli.rs`, kept as the
single source of truth so this table can't silently drift out of sync — re-check there
(or `corpus-extractor --help`) if in doubt.

## The pipeline, stage by stage

```
walk        (src/walk.rs)     -> every file under --input matching --extensions
extract     (src/extract.rs)  -> raw text, per format (plain read / html2text / pdf-extract / epub)
clean       (src/clean.rs)    -> whitespace normalization
chunk       (src/chunk.rs)    -> GPT-2-token-accurate windows, with overlap (or, with
                                  --raw-text-only, the whole cleaned file as one record)
filter      (src/clean.rs)    -> drop chunks that are too short or too non-ASCII
dedupe      (src/dataset.rs)  -> drop exact-duplicate chunks (hash-based, O(n))
split       (src/dataset.rs)  -> seeded shuffle, train/test split
write       (src/dataset.rs)  -> train.jsonl/test.jsonl (+ train.txt/test.txt)
```

Each stage is one small, independently testable module — mirroring
`custom-gpt-10m`'s own `data/` package structure (`sources.py` / `prepare.py` /
`audit.py` / `dataset.py` as separate stages) rather than one monolithic script.

**Directory walking respects `.gitignore`** (via the `ignore` crate — the same walker
`ripgrep` is built on), even when `--input` isn't itself a git repository. This matters
in practice: pointed at a real software project, a naive recursive walk would also
extract everything under `target/`, `node_modules/`, `.venv/`, and similar
generated/vendored trees — not source material for a training corpus, and often large
enough to bury the real content in duplicated/irrelevant chunks.

## Output format

**JSONL** (`train.jsonl`/`test.jsonl`, or `dataset.jsonl` with `--no-split`) — one record
per chunk (or, with `--raw-text-only`, one record per file, always `"chunk_index": 0`):

```json
{"text": "...", "source_path": "/abs/path/file.py", "file_type": "py", "chunk_index": 0, "char_count": 812, "token_count": 512}
```

**Plain text** (`train.txt`/`test.txt`, on by default — disable with `--no-emit-text`) —
chunk text joined by a blank line, matching `custom-gpt-10m`'s own `train.txt` convention
exactly, so it can be copied straight into that project's `data/` folder.

## Known limitations

- **PDF extraction has no OCR.** `pdf-extract` reads a PDF's existing text layer; a
  scanned/image-only PDF correctly extracts to empty or near-empty text, not an error —
  there's no text there to extract without OCR, which this tool doesn't do.
- **A single malformed file can't crash the batch, but it can still take a real toll on
  output quality.** `pdf-extract` (and, defensively, the other format crates too) can
  panic internally on unusual real-world input rather than returning an error — hit for
  real on a PDF with an unusual `DeviceN` colorspace. `extract.rs`'s `catch_panic()`
  catches this and reports it as a normal `[skip]`/`files failed to extract` entry
  instead of ending the run, but a crate that panics on a case like this may also produce
  subtly wrong (not just missing) text on other malformed input it doesn't panic on —
  worth spot-checking output on a large, uncurated real-world folder, not just trusting
  a clean `files failed to extract: 0`.
- **EPUB reading order follows the book's spine**, not filename order — this is the
  same reading order an e-reader would use (`OEBPS/chapter2.xhtml` reading before
  `OEBPS/chapter10.xhtml` if the spine says so, regardless of what a naive
  alphabetical/filename sort would produce). Each chapter is HTML internally, so it's
  converted via the same `html2text` path `.html` files use.
- **Chunking is not code-aware.** A 512-token window can land in the middle of a
  function — the same honest trade-off `custom-gpt-10m`'s own
  [`DATA_PREP_GUIDELINE.md`](../../from_scratch/custom-gpt-10m/docs/DATA_PREP_GUIDELINE.md)
  names for its chat corpus's document boundaries: simple and fast now, a known,
  documented weak spot rather than a hidden one.
- **The quality filter is intentionally minimal** (length + ASCII ratio + some alphabetic
  content) — it catches obvious extraction noise, not domain-specific quality issues. For
  a serious corpus, treat this as a first pass, not a substitute for manually reviewing a
  sample of `train.jsonl`.
