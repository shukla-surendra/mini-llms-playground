# Dataset

**Status: the corpus this project needs does not exist yet.** This file is the
collection plan, not a description of something already built.

`data/` is currently a symlink to `custom-gpt-10m`'s corpus — 280M train tokens of
chat/books/Wikipedia shared with the 10m and 50m siblings. That corpus is correctly
sized for a ~50M model and **roughly 9x too small for this one**. What it becomes here
is the *fine-tuning* set, not the pretraining set — see "The corpus you already have"
below.

Companion docs: [`docs/GPU_TRAINING.md`](docs/GPU_TRAINING.md) for the training budget
and the `.bin` token pipeline; [`docs/MODEL_SIZING_GUIDE.md`](docs/MODEL_SIZING_GUIDE.md)
for why this model is 152.8M parameters.

## How much

**2.5B tokens.** That is exactly what the configured run consumes, since a step is one
micro-batch:

```
steps x batch_size x context_length = 150,000 x 16 x 1024 = 2.46B tokens
```

Against 152,791,296 parameters that is **16 tokens per parameter** — a little under
Chinchilla-optimal (~20:1, i.e. 3.06B) so the run fits inside ~24 GPU-hours. `gpt-train`
prints the resolved budget at startup; read it before walking away.

If the full amount is not reachable, shrink the model rather than starving it:

| Fresh tokens collected | Do this |
|---|---|
| 2.5–3B | Train 153M as configured |
| 1–2.5B | Still train 153M; accept 1.5–2.5 epochs (repetition costs little up to ~4) |
| 600M–1B | Drop to ~110M (`GPT_EMBED_SIZE=640`) and cut `GPT_STEPS` to match |
| < 600M | Don't scale up — the 50m model is already matched to what you have |

The repetition allowance is not a guess: training on repeated data is close to as good
as fresh data for the first ~4 epochs, and decays toward worthless by ~16. Which is also
why the existing 280M corpus tops out at justifying ~56M parameters (280M x 4 epochs
≈ 1.1B effective ÷ 20) — almost exactly the sibling 50m model.

## Sources

**Recommended: [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).**
It is the corpus behind SmolLM-135M — a model within 12% of this one's size — so the mix
is validated *at this scale* rather than extrapolated down from frontier-model runs.
Three configs in one dataset:

| Config | Available | Role |
|---|---|---|
| `fineweb-edu-dedup` | 220B tokens | CommonCrawl filtered by an educational-quality classifier |
| `cosmopedia-v2` | 28B tokens | Synthetic textbooks and stories — very clean, dense prose |
| `python-edu` | 4B tokens | Educational Python source |

### The mixture, for 2.5B

| Share | Tokens | Source | Why |
|---|---|---|---|
| 70% | 1.75B | `fineweb-edu-dedup` | General world knowledge and factual grounding — the gap the QA reports keep showing |
| 25% | 0.60B | `cosmopedia-v2` | Weighted higher than a frontier run would: a 153M model absorbs clean, low-noise textbook prose far better than raw web text |
| 5% | 0.13B | `python-edu` | Small, but the QA prompt set tests a Python function, so omitting it is not free |

**Alternatives**, if a single source is preferred:
[`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
`sample-10BT` (simplest — one config, already the right order of magnitude), or
SlimPajama-627B (more diverse: arXiv, StackExchange, books, GitHub).

**Avoid The Pile.** Its Books3 component makes the licensing untenable for a public repo.
FineWeb-Edu and Cosmopedia are ODC-By/permissive and safe to name in a public README.

## The corpus you already have

The inherited 280M-token corpus (UltraChat, SmolTalk, OASST1, Dolly, No Robots, GSM8K,
plus books/Wikipedia/repos — full detail in [`docs/DATASETS.md`](docs/DATASETS.md), with
the registry itself in [`src/gpt/data/sources.py`](src/gpt/data/sources.py)) should
**not** be poured into the pretraining mix.

Use it as a **fine-tuning set after pretraining**: pretrain on the 2.5B general corpus,
then fine-tune on the chat data for an hour or two of extra GPU time. Mixing them
instead makes the two compete for gradient signal — a dilution the 50m project's own
`DATASET.md` already flags, where extra documents came to outnumber chat conversations
and the `User:/Assistant:` pattern got proportionally less exposure per step.

## Mechanics

### Two ways in: `--extra-jsonl`, or straight to `.txt`

`gpt-data --extra-jsonl <path>` takes `{"text": ...}` JSONL and pools it in with the
chat sources, shuffled, at the configured train/test ratio. It is **repeatable** — pass
it once per source:

```bash
gpt-data --extra-jsonl data/fineweb.jsonl --extra-jsonl data/cosmopedia.jsonl
```

That path is worth using when you want the corpus *builder's* behaviour (quality
filters, consistent shuffling, matched train/test split). For a pure bulk pretraining
mix, writing documents joined by `<|endoftext|>` into a `.txt` and running
`gpt-tokenize --file` is simpler and skips a round-trip through JSONL.

> **Was broken until now.** `cli/prepare_data.py` in this project was a stale copy that
> imported a since-renamed `load_book_chunks` and exposed only a single-source
> `--books-jsonl`, so `gpt-data` failed at import on *every* invocation. It also
> defaulted `--min-turns` to 3, which silently discards 100% of single-turn sources
> like Dolly — the exact bug `docs/DATASETS.md` documents as already found and fixed.
> The fix lived only in `custom-gpt-10m`, because `src/gpt/data/` was gitignored while
> `src/gpt/cli/` was tracked, so the two halves drifted apart unnoticed. Synced from
> 10m; verify with `gpt-data --list`.

### `.bin` files concatenate

Token files are flat `uint16` arrays with no header, so mixing sources is just `cat`:

```bash
gpt-tokenize --file data/fineweb.txt
gpt-tokenize --file data/cosmopedia.txt
gpt-tokenize --file data/python_edu.txt
cat data/fineweb.bin data/cosmopedia.bin data/python_edu.bin > data/train.bin
```

The blend ratio is therefore just how much of each you tokenize — changing it does not
mean re-running a pipeline. No shuffling step is needed either: `get_batch` samples
random `context_length` windows from the whole stream, so physical order in the file
does not affect training (only *which* sources are present, and in what proportion).

Keep a held-out `test.bin` from the same mixture — a test set drawn only from the old
chat corpus would measure something the model is no longer primarily trained on.

### Budget for disk and bandwidth

| | Size |
|---|---|
| 2.5B tokens as `.txt` | ~10 GB |
| 2.5B tokens as `.bin` (uint16) | **~5 GB** |
| Downloaded parquet | ~8 GB |

Tokenize on the laptop (free) and upload only the `.bin` — half the size, and it means a
rented GPU never spends billed minutes re-tokenizing. Give the instance ~100 GB of EBS
rather than the 30 GB default.
