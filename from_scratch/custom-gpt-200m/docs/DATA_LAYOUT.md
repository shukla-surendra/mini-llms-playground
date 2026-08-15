# Corpus layout: what is shared, what is not, and why it matters

## Run order

The tokenizer is a **prerequisite**, not an option — the model's embedding table is
sized to its vocabulary, and token ids are meaningless without it.

```bash
make tokenizer    # 1. train this project's 32K BPE on data/train.txt
make tokenize     # 2. build data/*.bin from data/*.txt using that tokenizer
make train        # 3. train (also runs step 2 if the .bin is stale)
```

`make tokenize` and `make train` both depend on a `tokenizer-check` target that fails
immediately with an explanation if `tokenizer/tokenizer.json` is missing, rather than
letting the failure surface somewhere deeper.

Retraining the tokenizer invalidates **every `.bin` and every checkpoint**: different
merges mean different ids, and different ids index different embedding rows. The
`.bin` files rebuild automatically (see the fingerprint guard below); checkpoints
cannot be migrated and must be retrained from scratch.

## The layout

```
from_scratch/
  custom-gpt-10m/data/          <- physically holds the shared inputs
      raw/                      downloaded HF parquet          ~5.6 GB
      hf_cache/                 datasets cache                 ~6.1 GB
      books_staging*/           corpus-extractor JSONL         ~740 MB
      repos_staging/  wikipedia_staging/
  custom-gpt-153m/data/         <- REAL directory
      raw -> ../../custom-gpt-10m/data/raw          (symlink, shared)
      hf_cache -> ...                               (symlink, shared)
      train.txt  test.txt                           OWN
      train.bin  train.bin.json                     OWN
  custom-gpt-200m/data/         <- same pattern
```

**Shared**: downloads and extractor output. These are raw inputs — independent of
tokenizer, of context length, and of model size. Duplicating ~12 GB of them per
project buys nothing.

**Never shared**: anything *derived*. `train.txt`/`test.txt` (a corpus build, whose
document chunking is chosen to match a context length) and `.bin` (a token stream,
which is meaningless under a different vocabulary).

Until now `custom-gpt-50m/data` and `custom-gpt-153m/data` were symlinks to the whole
of `custom-gpt-10m/data`, so all three shared derived artifacts too.

## Why sharing `.bin` across tokenizers is dangerous, not just untidy

A `.bin` is a bare `uint16` array. Nothing in the bytes records which vocabulary the
ids index into. So if a project with a different tokenizer writes `train.bin` into a
shared directory, the other projects do **not** crash — every id is still a valid row
number, just the wrong one. Training continues, loss is garbage, and there is no error
to trace.

That was a live risk here: `custom-gpt-{10m,50m,153m}` use GPT-2's 50,257-token
vocabulary, while this project uses a 32,768-entry vocabulary of its own.

**The guard.** `build_token_bin` now writes a `<name>.bin.json` sidecar holding a
behavioural fingerprint of the tokenizer that produced it:

```json
{ "tokenizer": { "n_vocab": 32768, "probe_ids": 13, "probe_sha256": "dd0a9f266680ef97" },
  "tokens": 9728825, "source": "data/train.txt" }
```

`load_token_array` compares that against the tokenizer configured now and **refuses to
load on a mismatch**, naming both fingerprints. The probe string deliberately mixes
words, a digit run, the document separator and a non-ASCII character, so it also
catches the subtler case: a tokenizer *retrained at the same vocabulary size* produces
different merges, which would otherwise silently invalidate existing `.bin` files with
no size change to notice.

Verified against the real failure: a 32,768-vocabulary `.bin` presented to a
50,257-vocabulary tokenizer is rejected with both fingerprints printed, rather than
loaded.

## Chunk size is a soft preference, not a correctness issue

`DATASET.md` chunks extracted documents to roughly one context window — 512 for the
10m project, 1024 for 50m/153m, 2048 here. It is worth matching, but it is **not** a
correctness constraint, and the reason is in `dataset.py`: `get_batch` samples random
`context_length`-token windows from one flat concatenated stream, ignoring document
boundaries entirely. A mismatched chunk size only means a training window is more
likely to straddle several unrelated documents rather than see one coherently — a
quality nudge, not a broken run.

So sharing a 1024-chunked corpus with a 2048-context model is legal and merely
suboptimal. Sharing a `.bin` across tokenizers is neither.

## Giving an existing project its own data directory

```bash
cd from_scratch
rm custom-gpt-50m/data                     # remove the whole-directory symlink
mkdir -p custom-gpt-50m/data
for d in raw hf_cache books_staging books_staging_1024 repos_staging wikipedia_staging; do
  ln -s "../../custom-gpt-10m/data/$d" "custom-gpt-50m/data/$d"
done
# then either rebuild the corpus, or link the shared build if it is genuinely identical:
ln -s ../../custom-gpt-10m/data/train.txt custom-gpt-50m/data/train.txt
ln -s ../../custom-gpt-10m/data/test.txt  custom-gpt-50m/data/test.txt
make tokenize                              # builds this project's OWN .bin
```

**Do not run this while that project is training.** `train.bin` is held open as a
memmap for the life of the run.
