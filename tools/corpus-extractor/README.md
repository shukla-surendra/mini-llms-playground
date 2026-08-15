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

Full flag reference: `corpus-extractor --help` (every flag's default and reasoning is
documented inline in `src/cli.rs`, not duplicated here where it could drift out of sync).

## The pipeline, stage by stage

```
walk        (src/walk.rs)     -> every file under --input matching --extensions
extract     (src/extract.rs)  -> raw text, per format (plain read / html2text / pdf-extract / epub)
clean       (src/clean.rs)    -> whitespace normalization
chunk       (src/chunk.rs)    -> GPT-2-token-accurate windows, with overlap
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
per chunk:

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
