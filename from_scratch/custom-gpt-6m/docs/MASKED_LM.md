# Masked Language Modeling: A Second, Bidirectional Pretraining Objective

## The mechanism, if you need a refresher

`src/gpt/model_mlm.py` and `src/gpt/training/trainer_mlm.py` (`gpt-train-mlm`) implement BERT-style masked language modeling — a
different pretraining objective from the causal (next-token) LM the rest of this project
trains (`src/gpt/model.py`/`src/gpt/training/trainer.py`). If the concept itself (what makes an objective
"self-supervised," why BERT's objective needs bidirectional attention while GPT's doesn't)
is unfamiliar, the first-principles treatment is
[`../../../docs/llm-engineering/12_the_pretraining_objective_and_why_data_dominates.md`](../../../docs/llm-engineering/12_the_pretraining_objective_and_why_data_dominates.md) —
this doc covers only what's specific to this project's implementation and real, observed
numbers.

## Why this needed a second model class, not just a training-loop flag

Causal LM and masked LM aren't the same architecture with a different loss bolted on —
they need fundamentally different attention. Causal LM predicts token `t+1` from tokens
`0..t`, so position `t` must never see position `t+1` or later (the causal mask). Masked
LM predicts a *masked* token from *every other position in the sequence, including ones
after it* — bidirectional attention is the entire point, since the model needs the right
context, not just the left context, to fill in a masked word. `model_mlm.py`'s
`MaskedLMTinyStories` reuses `src/gpt/model.py`'s `GPTBlock` unchanged, just constructed with
`causal=False` — same block, same math, the causal mask simply isn't applied.

## The masking policy: BERT's 80/10/10 rule, exactly

For each token position, independently: with probability `GPT_MASK_PROB` (default 0.15) it's
*selected*. Among selected positions: 80% are replaced with a reserved `[MASK]` id, 10%
are replaced with a random real token, 10% are left unchanged. **Loss is computed on all
selected positions**, regardless of which of the three things happened to the input at
that position — this is deliberate, not incidental:

- If only literally-masked positions contributed to loss, the model could learn to only
  pay attention to positions containing the special `[MASK]` token — useless at inference
  time, when there's no `[MASK]` token in real text at all.
- The 10% "replaced with a random token" case forces the model to actually verify each
  position's plausibility using context, rather than trivially copying the input token
  through — including positions that *weren't* replaced.
- The 10% "left unchanged" case means the model can never be *certain* a given position is
  or isn't the prediction target just by checking whether it looks like `[MASK]`,
  reducing the train/inference mismatch (real text never contains `[MASK]`) that a
  100%-masked policy would create.

See `apply_bert_masking` in [`model_mlm.py`](../src/gpt/model_mlm.py) for the exact implementation.

## The one tokenizer wrinkle: no `[MASK]` token exists

This project's BPE tokenizer (see
[`DATASET_AND_TOKENIZER.md`](DATASET_AND_TOKENIZER.md)) was trained for the causal-LM
project and only has two special tokens, `<unk>` (id 0) and `<|endoftext|>` (id 1) — no
`[MASK]`. Retraining the tokenizer to add one would also invalidate every existing
causal-LM checkpoint (the vocabulary and its ids are baked into `token_emb`/`lm_head`).
Instead, `MaskedLMTinyStories` reserves **id `vocab_size` (4096)** — one past the real
tokenizer's range, which the tokenizer can never actually produce — as a synthetic
`[MASK]` id, and sizes `token_emb` to `vocab_size + 1` rows to hold it. `mlm_head` still
only outputs `vocab_size` classes, since `[MASK]` itself is never a valid prediction
target (see [`model_mlm.py`](../src/gpt/model_mlm.py)'s module docstring for why this also means
no weight tying with `token_emb`, unlike the causal-LM model).

## Real training run, actually executed on this project's own MacBook (MPS)

`GPT_STEPS=1200`, default `GPT_BATCH_SIZE=32`/`GPT_CONTEXT_LENGTH=256`/`GPT_MASK_PROB=0.15`, same
100,000-story tokenized data as the causal-LM path:

```
[model] 6,906,112 parameters (mask_token_id=4096)

step   train_loss   val_loss
0      8.3747       8.3750   <- near-random (ln(4096) ≈ 8.32, same starting point as causal LM)
150    5.9276       5.9262
300    5.7559       5.7933
600    5.6149       5.6632
900    5.5392       5.6064
1199   5.5262       5.5635   <- final
```

Total wall-clock: **~4 minutes** for 1200 steps (vs. the causal-LM run's ~15 minutes for
4000 steps in [`TRAINING.md`](TRAINING.md)) — not a fair speed comparison, since this run
is fewer steps and the two losses aren't computed over the same thing (see next section).

**Reading these numbers — why this isn't directly comparable to the causal-LM loss
curve**: both start near `ln(4096) ≈ 8.32` (random guessing over the vocabulary), which is
a real, meaningful sanity check that the objective is wired correctly. But the causal-LM
loss in `TRAINING.md` is averaged over *every* position in every sequence (every position
has a real next-token target), while this loss is averaged over only the ~15% of
positions selected for masking, and — critically — each masked position gets to see
context from *both directions*, a strictly easier prediction problem than causal LM's
left-context-only setup. The faster-looking drop here is a property of the objective
being different, not evidence that masked LM is "better" or trains faster in some
general sense.

## What was deliberately left out, and why

- **Next-Sentence Prediction (NSP)** — BERT's original paper trained two objectives
  jointly (masked LM + NSP, a binary "do these two segments follow each other"
  classification task); later work (RoBERTa) found NSP contributed little and dropped it.
  This project follows RoBERTa's finding and implements masked LM alone.
- **Whole-word masking** — masking whole words (all of a multi-subword-token word
  together) rather than independent per-token masking, a later BERT refinement. Skipped
  for the same reason as elsewhere in this project: added complexity not worth it at this
  scale and dataset (TinyStories' restricted vocabulary means most words are already
  single tokens under this project's 4,096-size BPE vocab, so the distinction matters
  less here than it would for a larger, subword-heavy vocabulary).
- **Using this checkpoint for anything downstream** — this project trains the masked-LM
  objective as a real, working demonstration of the mechanism itself; fine-tuning this
  checkpoint for a downstream classification task (BERT's actual intended use pattern) is
  a natural next exercise, not implemented here.

## The gotcha: masked-LM loss and causal-LM loss are not on the same scale, ever

Worth repeating in its own section since it's the single easiest mistake to make reading
these two docs side by side: **do not compare `val_loss` between `TRAINING.md` and this
doc as if a lower number means "the better objective" or "the better-trained model."**
They're loss values over different position subsets, under different attention
visibility, solving different tasks. The only valid same-objective comparison is
masked-LM-vs-masked-LM (e.g., this run vs. a longer one), or causal-LM-vs-causal-LM.
