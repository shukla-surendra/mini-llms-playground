# Books (and Repo Source) Corpus Integration: A Running Log

Scope grew past just books partway through (Step 5) — this doc's filename is now
slightly narrower than its actual content, kept as-is rather than renamed to avoid
breaking the cross-links from `README.md`/`CODE_WALKTHROUGH.md`/`LLM_DEV_GUIDE.md`.

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

## Step 4 — targeting `50m` instead of `10m`: three more real bugs found and fixed

After the `10m` run's `best_test_loss` plateaued well above its pre-merge best for 46k+
steps with no recovery (see the "13 hours" / "1.5 hours" investigation elsewhere in this
project's history), the decision was made to stop chasing `10m` and build the best
possible corpus for a `50m` run instead — `50m`'s `embed_size=512` sits right at this
project's own computed embedding/block parameter balance point
([`MODEL_SIZING_GUIDE.md`](MODEL_SIZING_GUIDE.md)), unlike `10m`'s deeply
embedding-dominated `160`.

Rebuilding surfaced three more real, independent bugs — none introduced by this pass,
all pre-existing and silently costing real data:

1. **Generic parquet-availability fix.** Not every HF dataset repo publishes parquet on
   its main branch — `databricks/databricks-dolly-15k` (and its `zidankhan` mirror, the
   one actually registered) only ever shipped a raw `.jsonl` file. Hugging Face
   auto-converts every public dataset to parquet regardless, published at a standard
   `refs/convert/parquet` ref — `prepare.py`'s `download_source()` now falls back to that
   ref whenever the main branch has none, which fixes *any* future JSONL-only source the
   same way, not just this one.
2. **OASST1 tree reconstruction.** Its real raw schema is a flat table of individual
   messages forming reply trees (`message_id`/`parent_id`/`rank`), not the list-of-dicts
   shape every other "conversation"-schema source uses — structurally unparseable by the
   generic `extract_turns()` dispatch, confirmed empirically (0 conversations, silently).
   `prepare.py`'s new `load_oasst_conversations()` reconstructs one linear conversation
   per tree, walking from each root and picking the best-ranked reply at each branch
   (rank 0 = highest quality, per OASST1's own human ranking) — English-only, matching
   the rest of this corpus. `sources.py`'s OASST1 entry now correctly declares
   `schema="oasst_tree"`, dispatched separately from the generic loader.
3. **`min_turns=3` was silently killing every single-turn source.** Dolly's
   `extract_turns_instruction()` (and `HuggingFaceH4/no_robots`'s, mostly) always
   produces exactly 2 turns — one prompt, one reply. `build_corpus()`'s old default of
   `min_turns=3` meant **100%** of single-turn instruction data was discarded, not just
   unusually short multi-turn conversations, which is presumably what the threshold was
   meant to catch. Lowered the default to `2` — the real minimum for "a conversation
   happened at all" — after confirming the bug directly (`min_turns=3` → 0 Dolly
   conversations parsed from a file that parses to 76/100 at `min_turns=2`, from the
   *same* input).

Also added [`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
as a genuinely new source (not a recovery) — ~9.5k entirely human-written prompts and
replies across 10 categories, no model anywhere in the loop, unlike UltraChat/SmolTalk's
synthetic generation. **License: CC BY-NC 4.0 — non-commercial use only**, flagged the
same way this registry already flags LMSYS's gated status and UltraChat's OpenAI-terms
question.

Books were also **re-extracted at `--chunk-tokens 1024`** (was `512`, matching `10m`'s
context) to match `50m`'s `context_length=1024` — 48,912 chunks, down from 97,621 at the
smaller size (same books, roughly half as many chunks since each is twice as long).

**Final corpus, all fixes applied, real numbers:**

```
HuggingFaceH4/ultrachat_200k   100,000 conversations
OpenAssistant/oasst1              3,446 conversations   (0 before the tree-reconstruction fix)
zidankhan/databricks-dolly-15k   12,173 conversations   (0 before the min_turns fix)
HuggingFaceTB/smoltalk          100,000 conversations
HuggingFaceH4/no_robots          17,110 conversations   (new source)
lmsys/lmsys-chat-1m                    0 conversations   (gated, no HF_TOKEN — expected)
                                 --------
TOTAL chat                      232,729 conversations
book documents (1024-token)      48,912

train.txt: 209,783,994 tokens  (209,456 chat + 44,020 book docs)
test.txt:   23,438,110 tokens  ( 23,273 chat +  4,892 book docs)
total:     233,222,104 tokens
```

Six sources now genuinely contributing (five real + one correctly-excluded-gated), not
two — the "5-source" documentation gap flagged in Step 3 is now actually closed, not
just noted.

(A `data/backup_pre_50m_corpus/` was made at this point but turned out to be a mislabeled
duplicate, not a real snapshot — removed in Step 5 below, where the mistake was caught;
`data/backup_pre_books_merge/` remains the one backup that actually matters.)
Verified end-to-end against the real files before calling this done: `encode_raw()` at
`allowed_special`, `effective_context_length` at `1024`, and `get_batch()` all run
cleanly against the actual `train.txt`/`test.txt` on disk, not just the numbers reported
by `gpt-data`.

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

## Step 5 — merged in three more repos' `.rs`/`.md`/`.py` source

Added `~/projects/2026/eng-skills`, `~/projects/2026/OxideOS`, and
`~/projects/2026/platform-lab` — `corpus-extractor` run on each separately
(`--extensions rs,md,py --chunk-tokens 1024`, matching `50m`'s `context_length`), all
three clean (zero failed files, zero panics):

```
eng-skills:    114 files ->  1,216 chunks  (1,211 md,   5 py)
OxideOS:       206 files ->  1,051 chunks  (  165 md, 886 rs)
platform-lab:  870 files ->  2,231 chunks  (1,766 md, 335 py, 130 rs)
```

`private_profile/` was deliberately **not** included — real employer/client identifiers
live there by design (see the global privacy convention in `~/.claude/CLAUDE.md`), and
it wasn't part of what was actually asked for.

**The `--books-jsonl` flag became `--extra-jsonl` (repeatable)** — the old name stopped
being accurate the moment a second, non-book source existed. `gpt-data` now takes
`--extra-jsonl` multiple times to pool several `corpus-extractor` runs into one build in
a single pass, rather than requiring a manual `cat` of JSONL files beforehand.
`prepare.py`'s `load_book_chunks()` renamed to `load_extra_documents()` to match — it
was always source-agnostic (just reads a `{"text": ...}` JSONL), only the name implied
otherwise.

**A backup-labeling mistake, corrected rather than left**: the intermediate "chat +
books, no repos" corpus state (from Step 3/4) was never actually captured in a real
backup — the two directories that were supposed to hold it
(`data/backup_pre_50m_corpus/`, `data/backup_pre_repos_merge/`) were both copied
*after* later rewrites had already happened, making them byte-identical duplicates of
other states rather than genuine snapshots of that specific point. Both removed rather
than left around as misleading names. Nothing was actually lost — that intermediate
state is exactly reproducible on demand (`seed=42`, deterministic):

```bash
uv run gpt-data --skip-download --extra-jsonl data/books_staging_1024/dataset.jsonl
```

The one backup that actually matters — `data/backup_pre_books_merge/`, the original,
narrow, 2-real-source chat-only corpus from before any of this — is still correctly
in place, untouched.

**Final numbers, the corpus actually in `data/train.txt`/`data/test.txt` now**:

```
6 chat sources: UltraChat 100,000 + OASST1 3,446 + Dolly 12,173 + SmolTalk 100,000 +
                No Robots 17,110 + LMSYS 0 (gated) = 232,729 conversations
extra documents: 48,912 book chunks + 4,498 repo chunks = 53,410

train.txt: 213,428,560 tokens  (209,456 chat + 48,069 extra docs)
test.txt:   23,834,626 tokens  ( 23,273 chat +  5,341 extra docs)
total:     237,263,186 tokens
```

## Step 6 — GSM8K (math reasoning) and Simple English Wikipedia (factual grounding)

Both chosen against specific, observed weaknesses from this project's own QA reports
(`reports/qa_report_10m_step*.html`) — the model attempting zero real arithmetic on word
problems, and getting basic facts wrong (never naming Paris for "capital of France") —
not general corpus growth for its own sake.

**GSM8K** (`openai/gsm8k`, MIT): registered in `sources.py` as `schema="instruction"` —
its `question`/`answer` columns are literally the exact keys
`extract_turns_instruction()` already looks for, so this needed zero new parsing code,
unlike OASST1/Dolly earlier. 16,860 conversations kept (both the `main` and `socratic`
train splits). Answers include GSM8K's standard inline calculator annotations
(`48/2=24`) — left as-is; that's explicit worked arithmetic, not noise.

**Simple English Wikipedia** (`wikimedia/wikipedia`, config `20231101.simple`,
CC-BY-SA-3.0/GFDL): not conversational, so it doesn't fit the chat-source registry —
ingested the same way books/repos are, via a one-off chunking script
(`ingest_wikipedia.py`, same GPT-2 tokenizer, same 1024-token/100-overlap window as
`corpus-extractor`, same JSONL output shape) producing a file mergeable through the
existing `--extra-jsonl` path with no new merge code. 241,787 articles, 45,756 stub
articles (<200 chars) dropped, 222,029 chunks kept — **71,245,373 tokens**, notably
larger than the ~37M initially estimated from raw file size alone.

**A real, worth-stating-plainly composition shift**: extra documents (books + repos +
Wikipedia) now number 275,439 — more than the 249,589 chat conversations, by raw
document count. This corpus has moved further from "mostly chat-turn-shaped" than any
previous step in this log. Proceeded because it was explicitly requested
("add both and update dataset"), not silently — flagged here for the record.

**Final corpus, all sources, verified end-to-end against the real files on disk:**

```
7 chat sources: UltraChat 100,000 + OASST1 3,446 + Dolly 12,173 + SmolTalk 100,000 +
                No Robots 17,110 + GSM8K 16,860 + LMSYS 0 (gated) = 249,589 conversations
extra documents: 48,912 books + 4,498 repos + 222,029 Wikipedia = 275,439

train.txt: 280,330,103 tokens  (224,630 chat + 247,895 extra docs)
test.txt:   31,367,398 tokens  ( 24,959 chat +  27,544 extra docs)
total:     311,697,501 tokens
```
