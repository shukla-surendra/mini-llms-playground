# The Data Pipeline: Books + Open-Source Datasets, Reproducibly

The operational reference for building this project's corpus from scratch — what
`make books` / `make data` / `make tokenize` each actually do, in what order, and how
to verify each one before moving to the next. Complementary to
[`BOOKS_CORPUS_INTEGRATION.md`](BOOKS_CORPUS_INTEGRATION.md), which is a historical
narrative log of the *original* books-integration decisions (made once, on
`custom-gpt-10m`, back when this project shared that data via a symlink) — this doc
describes the pipeline as it exists **now**: automated, and independent of any sibling
project's data.

## Why this project no longer symlinks `custom-gpt-10m`'s data

Until now, `custom-gpt-153m/data/books_staging` was a symlink to
`custom-gpt-10m/data/books_staging` (see
[`../../../docs/llm-engineering/34_data_preparation_strategies_for_pretraining.md`](../../../docs/llm-engineering/34_data_preparation_strategies_for_pretraining.md)'s
sibling docs on why *derived* corpus artifacts should never be shared across projects
with different tokenizers/context lengths — books_staging is closer to a shared *raw
input* than a derived artifact, but the underlying principle still applies: this
project now runs its own independent extraction, so a future change to `custom-gpt-10m`'s
books folder, chunk size, or corpus-extractor version can never silently change what
`custom-gpt-153m` trains on.

**Enforced, not just documented.** Every `make` target below (`books`, `books-force`,
`data`, `data-public`, `tokenize`, `tokenize-force`, `audit`) checks `[ -L data ]`
before touching anything and refuses to proceed if `data/` is ever a symlink again —
see the `guard_not_symlinked` definition near the top of the `Makefile`. This exists
specifically because the *original* symlink here almost certainly came from
copy-pasting a sibling project's setup steps; a paragraph in a doc doesn't stop that
from happening a second time, a failing command does. Verified directly: pointing
`data` at a symlink and running `make books` fails loudly with a clear error before
anything is read from or written to the wrong project's directory.

## The three stages, in order

```
make books   ->  data/books_staging/dataset.jsonl        (local PDFs/EPUBs -> JSONL)
make data    ->  data/train.txt, data/test.txt            (+ HF open-source datasets, merged & shuffled)
make tokenize -> data/train.bin, data/test.bin            (final, trainable format)
```

`make data-all` (or `make data-all-public` to skip the gated LMSYS dataset) runs all
three in this exact order with one command. Each stage is also independently
re-runnable — the sections below say what to check after each one, and what state on
disk lets you skip a stage and resume partway through.

### Stage 1 — `make books`: local books -> JSONL

```bash
make books                              # BOOKS_DIR defaults to ~/Downloads/books
make books BOOKS_DIR=~/some/other/path  # override for a one-off
make books-force                        # wipe data/books_staging/ and re-extract from scratch
```

Runs [`tools/corpus-extractor`](../../../tools/corpus-extractor/) against `BOOKS_DIR`,
recursively, over `.pdf`/`.epub` files only (source code/markdown extraction is a
separate, not-yet-wired concern — see "What's deliberately not automated yet" below).
`--chunk-tokens 1024` matches this project's own `context_length` exactly (see
`src/gpt/config.py`), so a "1024-token chunk" here really is one training window's
worth once it reaches `gpt-train` — not an estimate that drifts once real
tokenization happens. `--no-split`: this stage is pure extraction, not corpus-building
— the shuffle and train/test split happen once, across every source (books AND every
HF dataset) together, in Stage 2, not per-source here. Splitting per-source first and
merging split files second would bias the split (a book's chunks would always land in
the same train/test bucket as each other, rather than being independently randomly
assigned like every other document).

**Verify before moving on:**

```bash
wc -l data/books_staging/dataset.jsonl        # one JSON object per chunk
head -1 data/books_staging/dataset.jsonl | python3 -m json.tool   # confirm the schema
```

Read the printed summary (files scanned / extracted OK / failed / chunks kept) — a
large `files failed to extract` count usually means malformed PDFs, not a pipeline
bug; corpus-extractor's own crash-proofing (`catch_panic`, see its README) means a bad
file is skipped and logged, never a run-ending failure.

**Reproducibility note:** corpus-extractor's shuffle/split (irrelevant here, since
`--no-split` is used) and dedupe are seeded/deterministic — re-running `make books`
against an *unchanged* `BOOKS_DIR` reproduces byte-identical output. Adding or
removing files from `BOOKS_DIR` between runs changes the output, as expected — that's
the whole point of `make books-force` existing as an explicit, separate step from
`make books` (which only extracts, it never wipes stale output from a previous run
first — `books-force` is what to reach for after removing/replacing books).

### Stage 2 — `make data` / `make data-public`: merge + build the corpus

```bash
make data           # all registered HF datasets (needs HF_TOKEN for gated LMSYS) + books, if staged
make data-public     # same, minus the gated LMSYS dataset
```

**This runs `tools/corpus-extractor build-corpus` (Rust), not the Python `gpt-data`
CLI** — see [`tools/corpus-extractor/README.md`](../../../tools/corpus-extractor/README.md#build-corpus--hugging-face-datasets---traintxttesttxttest_promptstxt)
for the full flag reference. It downloads every dataset in `tools/corpus-extractor/src/sources.rs`'s
registry (a manually-synced mirror of `src/gpt/data/sources.py`'s `DATASETS` — see
[`DATASETS.md`](DATASETS.md) for the full registry and
[`DATA_PREP_GUIDELINE.md`](DATA_PREP_GUIDELINE.md) for the quality-filter mechanics),
parses each into `(role, text)`-shaped conversations, and — **automatically, with no
flag to remember** — pools in `data/books_staging/dataset.jsonl` if `make books`
already produced it (`Makefile`'s `EXTRA_JSONL_FLAG`, re-checked on every invocation,
passed through as `build-corpus --extra-jsonl`). Delete `data/books_staging/` if you
want an HF-only corpus instead; there's no separate "books off" switch to set anywhere
else.

**Sources download and parse in parallel**, one `rayon` worker thread per registered
source by default (`--threads N` to cap it) — ordinary thread parallelism, not
Python's process-pool workaround, since there's no GIL to route around in Rust and
parquet reads happen columnarly (`arrow`/`parquet`) rather than row-by-row. Verified
deterministic regardless of thread count the same way `extract`'s own parallel
extraction was verified — rayon's `par_iter().map().collect()` preserves the source
list's original order in its result automatically, so the reproducible shuffle
(`--seed`, default 42) holds regardless of which source's download finishes first.

**Python fallback**: `make data-legacy` / `make data-legacy-public` run the original
`gpt-data` CLI (`src/gpt/data/prepare.py`, `--workers N` for its
`ProcessPoolExecutor`-based parallelism) — kept in the tree as a reference/fallback,
not the default. Both paths write the same `train.txt`/`test.txt`/`test_prompts.txt`
shape, so either one's output is a drop-in for the other's; downstream stages
(`gpt-audit`, `gpt-tokenize`/`corpus-extractor tokenize`, `gpt-train`) don't care which
one produced the corpus.

Everything — every HF conversation and every book chunk — is pooled into one list,
shuffled with a fixed seed (`--seed 42` by default), and split train/test together in
that same step (`--train-ratio 0.9` by default) — this is what makes the split fair
across sources rather than biased by extraction order (see Stage 1's note on why
per-source splitting would be wrong). Documents are joined with a plain `"\n\n"`
(`DOCUMENT_SEPARATOR` in `src/gpt/data/prepare.py`) — an earlier version of this
project used GPT-2's real reserved `<|endoftext|>` special token instead, specifically
to give the model a harder document-boundary signal (see
[`BOOKS_CORPUS_INTEGRATION.md`](BOOKS_CORPUS_INTEGRATION.md#step-2--fix-document-boundary-marking-for-both-old-and-new-data)
for that original reasoning); that trade-off was deliberately reverted, so this
project's corpus no longer relies on the special-token mechanism anywhere.

**Verify before moving on:**

```bash
uv run gpt-audit          # or: make audit — noise rate, ASCII ratio, train/test overlap
wc -l data/train.txt data/test.txt
```

Read the printed per-source breakdown (conversations kept per HF dataset, plus the
`extra_documents (books/repos)` line if books were included) — a source showing 0
conversations usually means a real parsing bug (see
`BOOKS_CORPUS_INTEGRATION.md`'s Step 3 for two real historical examples: OASST1's
tree-shaped schema and Dolly's JSONL-only HF repo, both silently returning 0 until
fixed), not something to shrug off as "that source just didn't have much data."

**Reproducibility note:** `--seed 42` (the default) makes the shuffle and split
deterministic given the same input sources — re-running `make data` against unchanged
`data/raw/` and an unchanged `books_staging/dataset.jsonl` reproduces the same
`train.txt`/`test.txt` byte-for-byte. `--skip-download` reuses whatever's already
under `data/raw/` instead of re-fetching from HF, useful for iterating on a filter
change without re-downloading.

### Stage 3 — `make tokenize`: final trainable format

```bash
make tokenize          # skips files whose .bin is already newer than its .txt source
make tokenize-force    # rebuild unconditionally
```

**This runs `tools/corpus-extractor tokenize` (Rust)**, not the Python `gpt-tokenize`
CLI. It streams `data/train.txt`/`data/test.txt` through GPT-2's `r50k_base` tokenizer
(`tiktoken-rs`; this project's `TOKENIZER_NAME` = `"gpt2"` on the Python side — same
tokenizer, no training step needed since it's a fixed public vocabulary, unlike the
RoPE-family siblings' self-trained tokenizer) into flat `uint16` `.bin` files. Training
reads the `.bin` as a memory-mapped array, so the corpus never has to fit in RAM — see
`src/gpt/data/dataset.py`. This is the literal "final trainable format": `gpt-train`
reads `.bin` files directly, never `.txt`.

Each `.bin` gets a `.bin.json` fingerprint (vocab size + a probe string's token ids,
SHA-256-hashed) that `gpt/data/dataset.py::load_token_array()` checks before training
— this is the one place the Rust and Python implementations' output has to be
**byte-identical**, not just equivalent, since either one may write a `.bin` the other
later reads. Verified directly: a `.bin` built by `corpus-extractor tokenize` loads
through Python's `load_token_array()` with no fingerprint mismatch, and the token
counts match exactly.

**Python fallback**: `make tokenize-legacy` / `make tokenize-legacy-force` run the
original `gpt-tokenize` CLI (`src/gpt/data/dataset.py::build_token_bin()`) — kept as a
reference, not the default.

**Verify before training:** `make config` and confirm `paths.corpus` points at the
files you expect; `ls -la data/*.bin` and sanity-check the size roughly tracks the
`.txt` size (a `.bin` is 2 bytes/token, so a very small `.bin` next to a large `.txt`
usually means tokenization silently ran against stale/empty input).

## What's deliberately not automated yet

- **Repo-source extraction** (the `.rs`/`.md`/`.py` extraction pattern
  `BOOKS_CORPUS_INTEGRATION.md`'s Step 5 describes for `custom-gpt-10m`) has no
  `make` target here. `make books` is scoped to `pdf,epub` only. Adding repo sources
  is a real, separate follow-up — run `corpus-extractor extract` by hand with
  `--extensions rs,md,py`, output to a new `data/repos_staging/dataset.jsonl`, and
  pass it via a second `--extra-jsonl` (the flag is repeatable — see
  `corpus-extractor build-corpus --help`) rather than trying to force it through the
  single `EXTRA_JSONL_FLAG` variable this Makefile currently wires up automatically.
- **Wikipedia / non-conversational extra sources** (`BOOKS_CORPUS_INTEGRATION.md`'s
  Step 6) likewise have no `make` target — they used a one-off `ingest_wikipedia.py`
  script producing the same `--extra-jsonl`-compatible JSONL shape, not something this
  Makefile currently builds for you.
- **Adding a new HF open-source dataset** to the mix is a code change, not a Makefile
  change: add a new `DatasetSource` entry to `src/gpt/data/sources.py` (see that
  file's existing entries for the exact shape, and `DATA_PREP_GUIDELINE.md` for the
  schema/quality-filter contract a new source needs to satisfy), then re-run `make
  data`/`make data-all` — no other step in this pipeline needs to change to pick up a
  newly-registered source.

## Full rebuild from nothing

```bash
rm -rf data checkpoints logs
make data-all              # or data-all-public to skip the gated LMSYS dataset
```

This is the exact sequence to confirm the whole pipeline really is reproducible from
an empty `data/` directory — `make books` re-extracts from `BOOKS_DIR` (untouched by
the `rm`, since it lives outside this repo), `make data` re-downloads every HF source
into a fresh `data/raw/`, and `make tokenize` rebuilds the `.bin` files. Nothing in
this sequence depends on any state left over from a previous run.
