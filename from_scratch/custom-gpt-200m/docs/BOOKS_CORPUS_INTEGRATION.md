# Books Corpus Integration: A Running Log

Tracking the actual steps of enriching this project's training corpus with book-derived
text via [`tools/corpus-extractor`](../../../tools/corpus-extractor/), source material:
665 real PDF files (~4.4GB) in `~/Downloads/books` (80 additional `.epub`-named items in
that folder are Apple Books' unpacked local-storage directory bundles, not real zip-based
EPUB files — `corpus-extractor`'s EPUB support doesn't read that format, so they're
excluded from this pass; a real, separate follow-up if that content is wanted too).

This is a genuinely different goal from
[`DATA_PREP_GUIDELINE.md`](DATA_PREP_GUIDELINE.md), worth being explicit about so the two
docs aren't read as the same effort: that doc is about **narrowing** to one specialized
domain. This is about **enriching** the existing general corpus with more diverse,
structurally-coherent, factually-real prose — a broader, not narrower, move.

## Why this isn't just "run the tool and merge the output"

Two problems already diagnosed in this project's other docs directly determine whether
this actually helps or backfires, so the integration has to account for them rather than
naively concatenating extracted text onto `train.txt`:

- **Document-boundary marking**: conversations in the existing corpus are joined with a
  plain `"\n\n"` — already a known weak spot
  ([`TRAINING_QA.md`](TRAINING_QA.md#does-arranging-data-in-a-particular-way-increase-model-performance)).
  Appending book chunks with that same weak separator would make it worse, not better —
  a random training window could now straddle a chat exchange and unrelated book prose.
  Since the corpus is being rebuilt anyway, this is the point to fix that for both the
  existing chat data and the new book data at once, not just pile more content onto the
  existing problem.
- **Mix ratio**: the current chat corpus is 173,706,682 train tokens. Book-derived token
  volume needs to be *measured*, not assumed — if it ends up massively outnumbering the
  chat data, the model sees proportionally less of the `"User: ...\nAssistant: ..."`
  pattern it needs to produce chat-shaped completions at all, which would hurt the exact
  thing this project evaluates with `make eval`/`make test`. The plan is to cap
  book-token volume relative to the chat corpus once real numbers are in, not use
  whatever comes out unfiltered.

## Step 1 — extract and measure, before touching any live data

```bash
./target/release/corpus-extractor \
  --input ~/Downloads/books \
  --output .../custom-gpt-10m/data/books_staging \
  --extensions pdf \
  --chunk-tokens 512 \
  --no-split
```

`--chunk-tokens 512` matches this project's `context_length` exactly, so token counts
here mean the same thing they will once this data reaches `custom-gpt-10m`'s training
loop. `--no-split` because this step is purely measurement — deciding the train/test
split, and the boundary-token fix, happen once the real volume is known and a mix ratio
is chosen, not before. Output goes to `data/books_staging/`, deliberately not
`data/train.txt`/`data/test.txt` — nothing here is live yet.

**First attempt crashed the whole batch, not just one file.** `pdf-extract` (the crate
`corpus-extractor` uses for PDF text extraction) panics internally on some malformed
real-world PDFs instead of returning an `Err` — hit for real on the first run, on a PDF
using an unusual `DeviceN` colorspace. A Rust panic unwinds straight past ordinary
`Result`-based error handling, so this took the entire 665-file run down on one bad file
rather than skipping it. Fixed in `corpus-extractor` itself
(`src/extract.rs`'s `catch_panic()`, wrapping every format's extraction call in
`std::panic::catch_unwind` and converting a caught panic into a normal, loggable `Err`) —
a real, general robustness fix for the tool, not a one-off workaround for this one file.
Re-running with the fix now.

**Result**, measured from the real output (`data/books_staging/dataset.jsonl`, 665 PDFs
scanned):

```
files scanned:            665
files extracted OK:       610   (55 failed — mostly malformed/unusual PDFs, now cleanly
                                  skipped and logged rather than crashing the batch)
chunks kept:               97,621
book corpus tokens:        49,894,028
existing chat corpus tokens: 173,706,682   (data/train.txt, current)
ratio (books / chat):       0.29x
```

Better than the worst case worried about above — books land at roughly **22% of a
combined corpus**, not a volume that would swamp the `"User: ...\nAssistant: ..."`
pattern. No aggressive capping needed; using close to the full extracted set is
reasonable as-is.

## Step 2 — fix document-boundary marking, for both old and new data

Done. `prepare.py` now has a `DOCUMENT_SEPARATOR = "<|endoftext|>"` constant, used to
join every document (chat conversation *and* book chunk) instead of `"\n\n"`.

**A real assumption turned out wrong, caught before running the full rebuild, not
after.** The original plan assumed `dataset.py`'s existing `tokenizer.encode(text,
disallowed_special=())` was already enough to make `"<|endoftext|>"` collapse to GPT-2's
real reserved token id at tokenization time. Verified empirically instead of trusting
that: it doesn't. `disallowed_special=()` only stops the tokenizer from *raising* on
seeing the string — without `allowed_special={"<|endoftext|>"}` too, the literal string
tokenizes as 7 ordinary subword pieces (`<`, `|`, `end`, `of`, `text`, `|`, `>`), not the
single special token id `50256`. Fixed in both `dataset.py`'s `encode_raw()` (training)
and `inference/generate.py`'s prompt encoding (consistency at inference time), both now
passing `allowed_special={DOCUMENT_SEPARATOR}`. Confirmed with a direct test:
`encode("<|endoftext|>", disallowed_special=())` → 7 tokens; adding `allowed_special` →
`[50256]`, one token, the real id.

## Step 3 — decide and apply the mix ratio, rebuild `train.txt`/`test.txt`

Done — `build_corpus()` extended with an `extra_documents` parameter (pooled and
shuffled in with the chat conversations, same "shuffle everything together" principle
the five HF sources already used — see `TRAINING_QA.md`), wired through `gpt-data
--books-jsonl data/books_staging/dataset.jsonl`. Held-out prompts (`test_prompts.txt`)
still come only from chat conversations, unchanged — a book excerpt has no natural
"held-out Assistant reply" to construct a prompt from.

**A second, larger discovery made in the process, not something introduced by this
change**: rebuilding surfaced that 3 of the project's supposed "5 sources" have
apparently never actually contributed any real conversations, going back to the
*original* corpus build already used by the paused `10m`/`E=512` runs:

- **OASST1**: its real Hugging Face parquet schema is a flat per-message tree table
  (`message_id`/`parent_id`/`message_tree_id`/`role`) — not the `conversation`/
  `messages` list-of-dicts shape `extract_turns_conversation()` expects, nor the flat
  `instruction`/`output` shape `extract_turns_instruction()` expects. Every row silently
  fails to parse into turns; `load_conversations()` has been returning 0 for this source.
  Fixing this for real needs tree reconstruction (grouping rows by `message_tree_id`,
  walking `parent_id` links into linear paths) — real, separate engineering work, not
  attempted here.
- **Dolly** (`zidankhan/databricks-dolly-15k`): the actual HF repo stores
  `databricks-dolly-15k.jsonl` — a plain JSONL file, never parquet. `download_source()`
  is hardcoded to `allow_patterns=["*.parquet"]`, so this source has never been
  downloadable at all through this pipeline. Needs a JSONL ingestion path, not attempted
  here.
- **LMSYS-Chat-1M**: gated, no `HF_TOKEN` configured in this environment — expected to
  be empty per the project's own docs (`make data-public` explicitly excludes it), not a
  new finding.

Net effect: this project's chat corpus has, in practice, only ever really been
**UltraChat 200k + SmolTalk** (2 sources), not the 5 described throughout `README.md`/
`DATASETS.md`/`DATA_PREP_GUIDELINE.md`. Those docs' "5-source" framing is now known to be
inaccurate and should be corrected once OASST1/Dolly are actually fixed or the framing
is updated to reflect reality — flagged here rather than silently left for someone to
rediscover later.

**Final numbers, the corpus actually in `data/train.txt`/`data/test.txt` now**:

```
train.txt: 218,517,382 tokens  (180,000 chat conversations + 87,858 book docs)
test.txt:   24,242,077 tokens  ( 20,000 chat conversations +  9,763 book docs)
total:     242,759,459 tokens  — up from the previous 192,969,401 (+25.8%)
```

Previous `train.txt`/`test.txt`/`test_prompts.txt` backed up to `data/backup_pre_books_merge/`
before this rewrite — the paused `10m` (step 481,399) and `custom-e512-l6-h8-c512`
(step ~7,226) checkpoints would silently train against different data than they started
with if ever resumed after this point; restore the backup first if that matters for a
specific resume.
