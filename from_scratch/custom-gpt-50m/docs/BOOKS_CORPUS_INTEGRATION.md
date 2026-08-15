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

Not yet done. Planned: replace the `"\n\n"` separator in `custom-gpt-10m/src/gpt/data/prepare.py`'s
`build_corpus()` with a real reserved token (`<|endoftext|>`) — already confirmed
low-cost, since `dataset.py`'s `encode_raw()` already calls `tokenizer.encode(text,
disallowed_special=())`, meaning special tokens already pass through unchanged.

## Step 3 — decide and apply the mix ratio, rebuild `train.txt`/`test.txt`

Not yet done — blocked on Step 1's real numbers.
