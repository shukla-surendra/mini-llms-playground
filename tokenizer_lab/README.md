# Tokenizer Lab

Hands-on tokenization exploration — notebooks and docs, grounded in this repo's own
**real, already-trained tokenizers** rather than toy examples:

| Tokenizer | Vocab size | Where it's from | Notable trait |
| --- | --- | --- | --- |
| `custom-gpt-6m` | 4,096 | [`../from_scratch/custom-gpt-6m/data/tokenizer.json`](../from_scratch/custom-gpt-6m/data/tokenizer.json) | Smallest vocab, has an explicit `<unk>` fallback token |
| `custom-gpt-50m` | 50,257 | tiktoken's public `"gpt2"` encoding | The real, unmodified GPT-2 vocabulary |
| `custom-gpt-350m` | 32,768 | [`../from_scratch/custom-gpt-350m/tokenizer/tokenizer.json`](../from_scratch/custom-gpt-350m/tokenizer/tokenizer.json) | Splits every digit individually — no `<unk>`, pure byte-level fallback |

This section exists for **conceptual foundation first, then hands-on poking** — read
[`../docs/llm-engineering/09_tokenization.md`](../docs/llm-engineering/09_tokenization.md)
before or alongside the notebooks here if any term below is unfamiliar. This isn't a
replacement for that chapter; it's where you actually run the code and look at the real
output instead of just reading about it.

## Setup

```bash
cd tokenizer_lab
uv sync
uv run jupyter lab
```

`uv sync` installs `jupyter`, `tokenizers`, `tiktoken`, `transformers`, and `matplotlib`
into this project's own `.venv` — isolated from every sibling project's environment,
same convention as the rest of this repo.

## Layout

- **`notebooks/`** — runnable exploration, in order:
  1. [`01_comparing_this_repos_tokenizers.ipynb`](notebooks/01_comparing_this_repos_tokenizers.ipynb) —
     compare the three already-trained tokenizers above: compression, digit-splitting,
     byte-level fallback (including a real, verified bug in `custom-gpt-6m`'s tokenizer),
     and implementing one BPE merge step by hand.
  2. [`02_building_a_custom_tokenizer.ipynb`](notebooks/02_building_a_custom_tokenizer.ipynb) —
     train a tokenizer yourself from scratch, deliberately reproduce Notebook 01's
     `custom-gpt-6m` bug in miniature, then fix it the way `custom-gpt-350m` actually does.
  3. [`03_hugging_face_tokenizers_theory_and_usage.ipynb`](notebooks/03_hugging_face_tokenizers_theory_and_usage.ipynb) —
     the `transformers`-library layer on top of raw `tokenizers`: `AutoTokenizer`,
     wrapping a custom `tokenizer.json` in `PreTrainedTokenizerFast` (the exact pattern
     this repo's real `export_vllm.py` scripts use), and padding/`attention_mask` theory.
- **`docs/`** — write-ups of what you actually find. Nothing here yet on purpose — this
  is meant to fill up with your own observations as you explore, not pre-written
  conclusions. `docs/TEMPLATE.md` has a suggested shape (analogy → mechanism → what
  surprised you) if you want a starting structure rather than a blank page.

## Why real tokenizers instead of a toy example

A hand-rolled 20-word toy vocabulary hides exactly the things worth understanding: how
BPE actually degrades on out-of-distribution text, why a 32K vocab and a 50K vocab
compress the same English paragraph differently, what byte-level fallback actually looks
like on non-Latin script. This repo already has three trained tokenizers sitting on disk
with real, different design choices behind them — better material to explore than
anything built specifically to be explored.
