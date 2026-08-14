# Continuing Training on a New Dataset

Companion to [`TRAINING.md`](TRAINING.md),
[`HOW_MUCH_TRAINING_IS_ENOUGH.md`](HOW_MUCH_TRAINING_IS_ENOUGH.md), and
[`DATASET_AND_TOKENIZER.md`](DATASET_AND_TOKENIZER.md). Answers: *"I trained this model
on TinyStories — can I later keep training that same model on a different dataset?"*

## Short answer

Yes — this is a real, standard technique called **continued pretraining** (sometimes
"domain-adaptive pretraining"), and it's mechanically the *same* resume logic
[`TRAINING.md`](TRAINING.md#resume-behavior) already uses. But there's one hard
requirement the code didn't originally support, and getting it wrong silently corrupts
the model rather than throwing an error — this doc covers both the concept and the fix.

## The one hard requirement: the tokenizer must stay the same

Recall from [`DATASET_AND_TOKENIZER.md`](DATASET_AND_TOKENIZER.md): this project trains
its **own** BPE tokenizer, fit to whatever corpus is passed to `prepare_dataset.py`. If
you point `prepare_dataset.py` at a new dataset and let it train a *new* tokenizer (the
default behavior), that new tokenizer's vocabulary — the actual mapping from token ID to
text — will be **different** from the one your checkpoint was trained with, even if
`vocab_size` comes out to the identical number (4,096 either way).

**Why this breaks silently, not loudly**: `train.py`'s resume-compatibility check (per
[`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md#resume-compatibility-check)) only compares
`vocab_size` — a single integer. Two independently-trained tokenizers with the same
`vocab_size` pass that check every time, because the check has no way to know the
*content* of the vocabulary changed. The model would load fine and "train," but every
token ID would now mean something different than what the checkpoint's embedding table
([`../../../docs/llm-engineering/05_embeddings_the_general_idea.md`](../../../docs/llm-engineering/05_embeddings_the_general_idea.md))
learned it to mean — token ID 37 might have meant "cat" during original training and mean
something unrelated in the new tokenizer. The model wouldn't error; it would just quietly
learn from scrambled associations, and generation quality would degrade in a way that's
confusing to debug after the fact.

## The fix: `--reuse-tokenizer`

`prepare_dataset.py` now supports tokenizing a new dataset with an **existing**
tokenizer, instead of always training a fresh one:

```bash
python prepare_dataset.py \
  --dataset some/other-dataset \
  --reuse-tokenizer data/tokenizer.json \
  --out-dir data_v2
```

This loads the tokenizer you already have (from your original TinyStories run) and uses
it, unchanged, to tokenize the new dataset — writing output to a **separate** directory
(`data_v2/`, not overwriting `data/`) so your original training data and the new data
coexist and you can compare or return to either.

## The full workflow

```bash
# 1. Prepare the new dataset, reusing the EXISTING tokenizer
python prepare_dataset.py \
  --dataset your/new-dataset \
  --reuse-tokenizer data/tokenizer.json \
  --out-dir data_v2

# 2. Point training at the new data directory, and raise STEPS past
#    wherever your current checkpoint already is (per TRAINING.md's
#    "resuming doesn't train past STEPS" gotcha)
DATA_DIR=data_v2 STEPS=8000 python train.py
```

`train.py` will load `tinystories_gpt_checkpoint_latest.pt` exactly as it would for an
ordinary resume, confirm the (now genuinely matching) `vocab_size`, and continue training
from those weights — the model doesn't start over, it picks up from everything it already
learned on TinyStories and keeps updating those same weights using the new data's
batches.

## What this actually does to the model, conceptually

This isn't fine-tuning in the LoRA sense
([`../../../fine_tuning/tinyllama-1.1b-lora/`](../../../fine_tuning/tinyllama-1.1b-lora/)'s
approach, which freezes the base model and trains small additional adapter matrices).
This is **the exact same full-parameter training mechanism** as the original run — every
weight, including the embedding table, keeps receiving gradient updates
([`../../../docs/llm-engineering/03_how_neural_networks_learn.md`](../../../docs/llm-engineering/03_how_neural_networks_learn.md)),
just now computed against new-dataset batches instead of TinyStories batches. The
distinction between "continuing to train the same model on new data" and "training from
scratch" is entirely about **which weights you start from** — random initialization
versus an already-trained checkpoint — not a different training mechanism.

### Does it retain previous knowledge? Yes, partially — it's a spectrum, not yes/no

This is a well-documented phenomenon called **catastrophic forgetting** — the honest
answer is some of it, to a degree that depends on how different the new data is and how
long you train on it, not a clean "keeps everything" or "loses everything." The full
mechanism (why gradient descent has no way to protect old-task weights), what determines
how much is retained, and the general mitigations (replay, lower learning rate, LoRA,
regularization to a reference model) are covered in
[Chapter 28 — Catastrophic Forgetting and Continual Training](../../../docs/llm-engineering/28_catastrophic_forgetting_and_continual_training.md).
This section covers only what's specific to *measuring and mitigating it in this
project's own code*.

### You can actually *measure* forgetting, not just worry about it

This is directly checkable with what's already in this project. Keep your **original**
TinyStories `data/val.bin` around (don't delete it when you create `data_v2/`), and
periodically evaluate the model *against it* while training on the new data — a second,
held-out check that isn't part of what's currently being trained on:

```python
# after loading a checkpoint mid-continuation-training, using inference.py's
# load_model_and_tokenizer plus train.py's estimate_loss/load_tokens helpers:
from train import load_tokens, estimate_loss

old_val_tokens = load_tokens("data/val.bin")   # the ORIGINAL TinyStories val set

# estimate_loss(model, train_tokens, val_tokens, ...) reports BOTH a "train" and "val"
# figure from whatever two token sets you pass it — passing the same old_val_tokens for
# both arguments here just means the returned "train" and "val" entries both describe
# the SAME thing (loss against the original TinyStories val set); either entry works.
old_result = estimate_loss(model, old_val_tokens, old_val_tokens, ctx_len=256, bsz=32, device=device)
print("loss against ORIGINAL TinyStories val set:", old_result["val"])
```

If `old_val_tokens`' loss climbs over the course of continuation training, that's forgetting,
directly measured, not inferred — and how steeply it climbs tells you exactly how much of
a problem it actually is for your specific new dataset, rather than relying on the general
guidance above. This turns "would it retain previous knowledge" from a theoretical
question into something you can answer with a number, for your specific case.

### Applying Chapter 28's mitigations here

Of Chapter 28's mitigations, **replay** (mixing TinyStories back into the new-data batches)
and **a lower learning rate for the continuation phase** are the two that apply directly
to this project's plain full-parameter continuation setup — LoRA and reference-model
regularization are fine-tuning-specific techniques this project's continued-pretraining
path doesn't use. If your actual goal is training on *many* datasets over time, not just
switching once, see [`CONTINUAL_TRAINING_LOW_RESOURCE.md`](CONTINUAL_TRAINING_LOW_RESOURCE.md) —
it builds directly on the replay idea here with tooling for mixing from every prior round
(not just the last one) and reversible checkpoints between rounds.

## When you'd actually want a genuinely new tokenizer instead

If the new dataset's vocabulary is different enough from TinyStories that reusing the old
tokenizer would be badly inefficient (e.g., switching from children's stories to source
code — every line would fall back to inefficient byte-level splitting for text the old
tokenizer never saw patterns for), a new tokenizer might genuinely be the better choice.
But in that case, **you cannot meaningfully resume from the old checkpoint at all** — the
embedding table and `lm_head` are shaped to, and semantically tied to, the old
vocabulary. The honest options at that point are: train a new model from scratch on the
new data (this project's normal from-scratch path, just pointed at different data), or
look into more advanced vocabulary-extension/embedding-remapping techniques that are
genuinely out of scope for this project's current code.
