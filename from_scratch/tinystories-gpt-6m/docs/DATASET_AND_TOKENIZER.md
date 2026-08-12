# Dataset & Tokenizer: Why TinyStories, Why a Custom Small Vocabulary

## The goal this project is actually optimizing for

Not "smart." Not "knows facts." Just: **generate short text that's grammatically
sensible and locally coherent** — not a random-looking jumble of tokens. That's a much
narrower, much more achievable target than a general-purpose chatbot, and it changes
almost every design decision in this project relative to
[`../../custom-gpt-153m/`](../../custom-gpt-153m/).

## Why TinyStories, specifically

[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) is a
dataset built for exactly this goal — it's not a general web-scrape, it's ~2.1M short
children's stories, generated to use a **deliberately restricted vocabulary and simple
sentence structure** (the kind of vocabulary a 3-4 year old would understand). The
dataset's own research finding (Eldan & Li, 2023, "TinyStories: How Small Can Language
Models Be and Still Speak Coherent English?") is directly relevant here: models with as
few as 1-10 million parameters, trained on this dataset, produce grammatically correct,
locally coherent short stories — something models of the same tiny size trained on
general web text (this repo's `custom-gpt-153m`'s LMSYS-derived data, at a much larger
153M param scale, still shows repetitive/degenerate output per its own
[`docs/LLM_DEV_GUIDE.md`](../../custom-gpt-153m/docs/LLM_DEV_GUIDE.md#12-why-outputs-still-look-repetitive))
essentially cannot achieve.

**The mechanism, in plain terms**: a small model has limited capacity
([`../../../docs/llm-engineering/01_neurons_layers_and_networks.md`](../../../docs/llm-engineering/01_neurons_layers_and_networks.md)).
General web text uses a huge, varied vocabulary and covers an enormous range of topics,
sentence structures, and registers — far more than a few-million-parameter model can
adequately represent. TinyStories deliberately narrows *what the model needs to learn to
represent well*, so a small model's limited capacity is actually sufficient to cover it.
This is the single biggest lever behind "meaningful, not garbage" at this model size —
bigger than any specific architecture tweak.

## Why a custom, small tokenizer vocabulary — not GPT-2's 50,257-token one

`../../custom-gpt-153m/` reuses GPT-2's tokenizer directly (see
[`../../../docs/llm-engineering/09_tokenization.md`](../../../docs/llm-engineering/09_tokenization.md)).
This project trains its **own** small BPE tokenizer (via Hugging Face's `tokenizers`
library) instead, with `vocab_size=4096` — about 8% of GPT-2's vocabulary. Why this
matters specifically for a tiny model:

```
Embedding table size = vocab_size × embed_size   (from
  ../../../docs/llm-engineering/05_embeddings_the_general_idea.md)

At embed_size=256:
  GPT-2 vocab (50,257):  50,257 × 256 ≈ 12.9M params — for JUST the embedding table
  This project's vocab (4,096):  4,096 × 256 ≈ 1.05M params

For a model whose ENTIRE parameter budget is ~5.85M, spending 12.9M on the embedding
table alone isn't just wasteful — it's larger than the whole model's actual budget.
```

Since TinyStories itself uses a restricted vocabulary (per the dataset's own design), a
custom-trained tokenizer specific to this corpus needs far fewer distinct tokens to
represent it well than a general-purpose tokenizer built for arbitrary web/code text —
`prepare_dataset.py` trains this tokenizer directly on a sample of the training stories,
so its vocabulary is shaped by exactly the text this model will actually see.

## The tokenizer: byte-level BPE, trained fresh

`prepare_dataset.py` uses Hugging Face's [`tokenizers`](https://github.com/huggingface/tokenizers)
library (the same library underlying `transformers`, already used in
[`../../../fine_tuning/tinyllama-1.1b-lora/`](../../../fine_tuning/tinyllama-1.1b-lora/)),
with:

- **`models.BPE`** — the same Byte-Pair Encoding algorithm from
  [`../../../docs/llm-engineering/09_tokenization.md`](../../../docs/llm-engineering/09_tokenization.md#the-middle-ground-subword-tokenization-bpe),
  trained on this project's own data instead of reusing GPT-2's.
- **`pre_tokenizers.ByteLevel`** — operates on raw UTF-8 bytes rather than
  pre-split words, which guarantees **zero out-of-vocabulary failures**: any input text,
  however unusual, can always be represented as some sequence of byte-level tokens, even
  if a specific word never appeared in training.
- **A `<|endoftext|>` special token** — inserted between every story during
  tokenization (`prepare_dataset.py`'s `tokenize_split` function), so the model learns
  where one story ends and another begins, rather than seeing the whole corpus as one
  undifferentiated stream of text.

## The data pipeline, concretely

```
1. Download roneneldan/TinyStories via Hugging Face `datasets`
   (2.1M train stories, 22k validation stories available; this project uses a
   configurable subset via --max-samples, not the full set, to keep local
   training time on a laptop reasonable)

2. Train a byte-level BPE tokenizer (vocab_size=4096) on the training subset

3. Tokenize every story, insert <|endoftext|> between stories

4. Write tokens to train.bin / val.bin as compact uint16 numpy arrays
   (this is the standard nanoGPT-style approach: tokenizing once and storing
   raw token IDs on disk is far faster to load repeatedly during training
   than re-tokenizing text on every epoch)
```

Run it: `python prepare_dataset.py --max-samples 100000 --vocab-size 4096`
(defaults match this — see [`../README.md`](../README.md) for the full quickstart).
