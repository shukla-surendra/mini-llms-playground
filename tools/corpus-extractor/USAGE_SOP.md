# Corpus Extractor: Usage SOP (Standard Operating Procedure)

A practical guide to running the corpus-extractor tool, with real-world examples and decision rationale.

## Quick Start

```bash
./target/release/corpus-extractor \
  --input ~/Downloads/books \
  --output ~/Downloads/books_extracted \
  --extensions pdf,epub,mobi,azw3,fb2,txt,md,rtf,chm,pdb \
  --chunk-tokens 1024 \
  --chunk-overlap 100 \
  --no-split
```

This extracts text from all books in `~/Downloads/books/`, chunks them into 1024-token windows, and writes output to `~/Downloads/books_extracted/`.

---

## What This Tool Does

1. **Walks your input directory recursively** — finds all supported file types (PDF, EPUB, MOBI, etc.)
2. **Extracts text** — different extraction logic per format (plain text, HTML-to-text, PDF parsing, EPUB spine-aware reading)
3. **Cleans text** — normalizes whitespace, removes junk
4. **Chunks by token count** — uses GPT-2 tokenization to split into fixed-size windows (not character count)
5. **Filters & deduplicates** — removes low-quality or duplicate chunks
6. **Writes training data** — outputs JSONL (structured) and/or plain text (corpus)

**Why this beats character-based chunking:** A 1024-character chunk might be 150 tokens in code, 300 tokens in prose, or 400 tokens in conversational text. Token-based chunking gives you consistent semantic windows regardless of source format.

---

## Folder Location Decisions

### Input: `~/Downloads/books`

**Why this location:**
- Standard macOS download folder — where books naturally land when fetched
- Separate from project code — keeps training data outside the monorepo
- Easy to organize — we organized 1,110 books into 20 category subfolders here
- Respects `.gitignore` — no bloat in the repo if you accidentally commit a symlink

**Alternative inputs:**
- `~/my-project/` — extract from your own codebase (Python, Rust, JavaScript)
- `./notes/` — relative path to a local notes folder
- `/path/to/research/papers/` — any folder of documents

### Output: `~/Downloads/books_extracted`

**Why this location:**
- Paired with input — `books` → `books_extracted`, obvious relationship
- Outside the repo — training data doesn't live in version control
- Fast access — no network I/O, local SSD read/write
- Easy to version — the extracted dataset is content-addressed (hashes in JSONL metadata), so you can track which books produced which chunks

**Alternative outputs:**
- `./data/` — relative path, useful for per-project corpora
- `~/ml-training/corpus-2026/` — versioned by date
- `s3://my-bucket/corpus/` — if you extend the tool to write to S3

**Why NOT in the repo:**
- Keeps git history clean (1000+ books = 100s of MB of text)
- Training data scales independently of code
- Easier to regenerate if needed (re-run the extractor, not git checkout)

---

## Real-World Examples

### Example 1: Extract Books for LLM Training (What We Did)

**Goal:** Convert 1,110 organized books into a training corpus for a 50M-parameter model.

```bash
./target/release/corpus-extractor \
  --input ~/Downloads/books \
  --output ~/Downloads/books_extracted \
  --extensions pdf,epub,mobi,azw3,fb2,txt,md,rtf,chm,pdb \
  --chunk-tokens 1024 \
  --chunk-overlap 100 \
  --no-split
```

**Key flags explained:**
- `--input ~/Downloads/books` — read from all 20 category subfolders here
- `--output ~/Downloads/books_extracted` — write structured (JSONL) + plain text (TXT)
- `--extensions pdf,epub,mobi,...` — support all common ebook formats (match what you have)
- `--chunk-tokens 1024` — match your model's context window (custom-gpt-50m uses 1024)
- `--chunk-overlap 100` — 10% overlap between consecutive chunks ensures smooth continuity if a window lands near a chapter boundary
- `--no-split` — single `dataset.jsonl` and `dataset.txt`, not train/test split (we'll do that in the training pipeline)

**Output files:**
- `dataset.jsonl` — one JSON object per chunk: `{"text": "...", "source_path": "...", "token_count": 1024, ...}`
- `dataset.txt` — plain text, one chunk per paragraph, ready to copy into `custom-gpt-50m/data/`

**Time to complete:** ~20-30 minutes for 1,110 books (depends on PDF complexity, file I/O speed)

**What NOT to do:** Don't use `--no-split` if you need to feed this into `custom-gpt-10m`'s training pipeline directly (it will expect `train.jsonl` and `test.jsonl` as separate files).

---

### Example 2: Extract Source Code Only

**Goal:** Build a corpus from your own Python/Rust/JavaScript project to train a code-specific model.

```bash
./target/release/corpus-extractor \
  --input ~/my-project \
  --output ~/my-project-corpus \
  --extensions py,rs,js,ts,go,java \
  --chunk-tokens 512 \
  --no-split \
  --no-emit-text
```

**Why these flags:**
- `--extensions py,rs,js,ts,go,java` — code only, skip docs and random PDFs
- `--chunk-tokens 512` — smaller chunks for code (functions/classes are often <512 tokens)
- `--no-emit-text` — skip the `.txt` file, JSONL is enough for code
- `.gitignore` is automatically respected — `target/`, `node_modules/`, `.venv/` are skipped

---

### Example 3: Mix Books + Code + Docs

**Goal:** Train on a blend of books, documentation, and source code.

```bash
# Step 1: Extract books
./target/release/corpus-extractor \
  --input ~/Downloads/books \
  --output ~/corpus-books \
  --no-split

# Step 2: Extract docs
./target/release/corpus-extractor \
  --input ~/my-project/docs \
  --output ~/corpus-docs \
  --extensions md,txt,rst \
  --chunk-tokens 1024 \
  --no-split

# Step 3: Extract code
./target/release/corpus-extractor \
  --input ~/my-project/src \
  --output ~/corpus-code \
  --extensions py,rs,js \
  --chunk-tokens 512 \
  --no-split

# Step 4: Pool them for training (in your training pipeline)
# Concatenate or read them sequentially during training
```

This keeps datasets separate (easier to debug, audit, and remove one if needed) while training on all three.

---

## Flag Reference

### Core Flags

| Flag | Example | What It Does | Default |
|------|---------|-------------|---------|
| `--input` | `~/Downloads/books` | Read from this folder (required) | — |
| `--output` | `~/books_extracted` | Write output here (required) | — |
| `--extensions` | `pdf,epub,py,md` | File types to extract (comma-separated) | `pdf,epub,txt,md,rs,html,js,py` |

### Chunking

| Flag | Example | What It Does | Default |
|------|---------|-------------|---------|
| `--chunk-tokens` | `1024` | Target chunk size (GPT-2 tokens) | `512` |
| `--chunk-overlap` | `100` | Overlap between consecutive chunks (tokens) | `50` |

### Output

| Flag | What It Does | Default |
|------|-------------|---------|
| `--no-split` | Write to `dataset.jsonl` (no train/test split) | train/test split enabled |
| `--no-emit-text` | Skip `.txt` output, JSONL only | emit both |

### Filtering & Quality

| Flag | Example | What It Does | Default |
|------|---------|-------------|---------|
| `--min-tokens` | `50` | Skip chunks shorter than this | `10` |
| `--min-ascii-ratio` | `0.8` | Skip chunks with <80% ASCII (non-UTF8 junk) | `0.95` |

### Other

| Flag | What It Does |
|------|-------------|
| `--help` | Print all flags with full documentation |

---

## Output Format

### JSONL (`dataset.jsonl`)

One JSON object per line, one line per chunk:

```json
{"text": "Chapter 1: Introduction...\n\nThis is the extracted text from...", "source_path": "/absolute/path/to/file.pdf", "file_type": "pdf", "chunk_index": 0, "char_count": 4812, "token_count": 1024}
{"text": "...continued from previous chunk...", "source_path": "/absolute/path/to/file.pdf", "file_type": "pdf", "chunk_index": 1, "char_count": 5200, "token_count": 1024}
```

**Use this for:**
- Training pipelines that need metadata (source file, token count for stats)
- Filtering by file type after extraction
- Auditing (which chunks came from which books)

### Plain Text (`dataset.txt`)

Chunks joined by a blank line separator:

```
Chapter 1: Introduction...
This is the extracted text from...

...continued from previous chunk...

Chapter 2: Deep Dive...
Another chunk starts here...
```

**Use this for:**
- Feeding straight into `custom-gpt-50m/data/` (the training script expects this format)
- Simplicity (no JSON parsing needed)
- Human readability (just text)

---

## Common Pitfalls & Solutions

### Pitfall 1: "Files Failed to Extract: 5"

**Why it happens:** A PDF is scanned (image-only, no text layer), or a file is corrupted.

**Solution:** This is expected with real-world data. The tool skips unparseable files and continues. Check if you want to manually verify those files, but most of the time: ignore and continue.

**Command to identify them:**
```bash
# Run with verbose output (if available in your build)
./target/release/corpus-extractor --input ~/Downloads/books --output ~/out 2>&1 | grep "Failed"
```

### Pitfall 2: "Output is Huge" (100s of MB for 1,110 books)

**Why it happens:** Correct! 1,110 books ≈ 1 billion+ tokens. Plain text output at 1 token ≈ 4 characters = 4GB raw text.

**Solution:** 
- Use `--no-emit-text` to skip the `.txt` file and keep only `.jsonl` (smaller, metadata-rich)
- Extract in batches by category: `--input ~/Downloads/books/Programming_and_Coding` instead of the whole folder
- Stream directly from JSONL into your training pipeline (don't decompress to disk)

### Pitfall 3: Chunks Are Too Large/Small

**Symptom:** Training is slow, or loss is noisy.

**Why:** Chunk size doesn't match your model's context length.

**Solution:** 
- Check your model: `custom-gpt-50m` uses `context_length=1024` tokens
- Set `--chunk-tokens 1024` (match exactly)
- Re-run the extractor if you change it

### Pitfall 4: "Need Different Extensions"

**Symptom:** Your books are `.doc`, `.pptx`, or some other format.

**Solution:** 
1. Check what's supported: `--help` will list supported file types
2. If unsupported: convert to PDF first (LibreOffice CLI, Pandoc, etc.)
3. Or request a PR to add the format (the tool is modular — adding a new format is one new function in `src/extract.rs`)

---

## Integration with Training Pipeline

### Step 1: Extract

```bash
./target/release/corpus-extractor \
  --input ~/Downloads/books \
  --output ~/books_extracted \
  --chunk-tokens 1024 \
  --no-split
```

### Step 2: Copy to Training Data Folder

```bash
# If using custom-gpt-50m
cp ~/books_extracted/dataset.txt ~/projects/2026/mini-llms-playground/from_scratch/custom-gpt-50m/data/books.txt

# Or add to the data pipeline:
# In custom-gpt-50m/src/gpt/data/prepare.py, add:
# extra_docs = load_extra_jsonl("~/books_extracted/dataset.jsonl")
```

### Step 3: Train

```bash
cd ~/projects/2026/mini-llms-playground/from_scratch/custom-gpt-50m
make train STEPS=1000000
```

The training loop will read from `data/` folder (your extracted books + any other corpora).

---

## Best Practices

1. **Start small:** Extract one category first, verify output quality, then scale to all 1,110 books.
2. **Check metadata:** Scan the first 10 lines of `dataset.jsonl` — verify token counts and source paths look right.
3. **Reserve a test set:** Use `--no-split` initially, then manually hold out 10% of JSONL for evaluation.
4. **Track provenance:** Keep `source_path` in your JSONL — later you can audit "which books helped the most" by analyzing loss per source.
5. **Deduplicate across sources:** If you extract books + docs + code, the tool deduplicates within each run. If you want cross-source dedup, merge JSONL files and run a separate dedup pass.

---

## Troubleshooting

### "Command not found: corpus-extractor"

**Solution:** Build it first:
```bash
cd ~/projects/2026/mini-llms-playground/tools/corpus-extractor
cargo build --release
```

Then use the full path:
```bash
./target/release/corpus-extractor --input ... --output ...
```

### "Permission denied: ~/Downloads/books_extracted"

**Solution:** Create the output folder first:
```bash
mkdir -p ~/Downloads/books_extracted
chmod 755 ~/Downloads/books_extracted
```

### "No files matched"

**Symptom:** Output folder is empty despite running the command.

**Why:** No files in `--input` match `--extensions`.

**Solution:** 
```bash
# Check what extensions are actually in your folder
find ~/Downloads/books -type f -name "*.pdf" | wc -l
find ~/Downloads/books -type f -name "*.epub" | wc -l

# Then set --extensions to match
./target/release/corpus-extractor --input ~/Downloads/books --output ~/out --extensions pdf,epub,txt
```

### "PDF extraction is producing empty text"

**Why:** Scanned PDFs (images, no text layer). This is expected.

**Solution:** Ignore them. If many PDFs fail, you may need OCR (outside this tool's scope).

---

## Why We Made This Choice: ~/Downloads/books → ~/Downloads/books_extracted

**Input (`~/Downloads/books`):**
- ✅ Natural landing zone for downloaded books
- ✅ Easy to organize into categories (we organized 1,110 books here)
- ✅ Separate from code (cleaner git history)
- ✅ Respects `.gitignore` (no accidental commits)

**Output (`~/Downloads/books_extracted`):**
- ✅ Obvious pairing with input (books → books_extracted)
- ✅ Local, no network overhead
- ✅ Easy to delete and re-run (no cost to re-extracting if format changes)
- ✅ Can version separately from input (track which extraction params produced which JSONL)

**What NOT to do:**
- ❌ Extract directly into `custom-gpt-50m/data/` (ties your training data to the repo; hard to swap)
- ❌ Extract into `/tmp` (ephemeral, gets cleaned up)
- ❌ Extract into cloud storage (slow iteration, cost)

---

## Next Steps After Extraction

1. **Verify:** Spot-check 10 lines from the JSONL. Are tokens in the right ballpark?
2. **Audit:** `wc -l ~/books_extracted/dataset.jsonl` tells you how many chunks you got
3. **Integrate:** Copy `dataset.txt` into your training folder, or load `dataset.jsonl` in your data pipeline
4. **Train:** Run your model training with the extracted corpus
5. **Evaluate:** Check if model learns from your domain-specific books (compare loss before/after)

---

## Quick Command Cheat Sheet

```bash
# All books, full ebook format support
./target/release/corpus-extractor \
  --input ~/Downloads/books \
  --output ~/books_extracted \
  --extensions pdf,epub,mobi,azw3,fb2,txt,md,rtf,chm,pdb \
  --chunk-tokens 1024 --chunk-overlap 100 --no-split

# Code only, smaller chunks
./target/release/corpus-extractor \
  --input ~/my-project \
  --output ~/code-corpus \
  --extensions py,rs,js,ts \
  --chunk-tokens 512 --no-split

# Documentation only
./target/release/corpus-extractor \
  --input ~/docs \
  --output ~/docs-corpus \
  --extensions md,txt,rst \
  --chunk-tokens 1024 --no-split

# With train/test split (not --no-split)
./target/release/corpus-extractor \
  --input ~/books \
  --output ~/books-split \
  --chunk-tokens 1024
# Produces: train.jsonl, test.jsonl, train.txt, test.txt
```

---

## Questions?

Run `./target/release/corpus-extractor --help` for the authoritative flag docs.

Check `README.md` in this directory for architecture and implementation details.
