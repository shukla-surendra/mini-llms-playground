# Contrastive Self-Supervised Learning: SimCSE + InfoNCE on the Causal Backbone

## The mechanism, if you need a refresher

`src/gpt/model_contrastive.py` and `src/gpt/training/trainer_contrastive.py` (`gpt-train-contrastive`) implement a third pretraining objective
on this project's data — contrastive self-supervised learning, in the SimCSE (Gao et al.,
2021) style. If the concept itself (what a positive/negative pair is, why in-batch
negatives work, what InfoNCE actually computes) is unfamiliar, the first-principles
treatment is
[`../../../docs/llm-engineering/12_the_pretraining_objective_and_why_data_dominates.md`](../../../docs/llm-engineering/12_the_pretraining_objective_and_why_data_dominates.md) —
this doc covers only what's specific to this project's implementation and real, observed
numbers.

## Why this needed no new backbone architecture at all

Unlike masked LM (which needed bidirectional attention — see
[`MASKED_LM.md`](MASKED_LM.md)), the contrastive objective reuses `src/gpt/model.py`'s
`TinyStoriesGPT` **completely unchanged** as an encoder. Two additions sit on top of it,
in `model_contrastive.py`:

1. **Last-token pooling**: `hidden[:, -1, :]` — the last position's hidden state, since
   causal attention means it's the only position that has attended to the *entire*
   sequence. This is exactly the technique real causal-LM-based embedding models (E5-
   mistral, LLM2Vec, GTR) use: a decoder-only model was never architecturally an encoder,
   but its last-token representation is a legitimate whole-sequence summary anyway.
2. **A small projection head** (`Linear → GELU → Linear`) mapping the pooled hidden state
   down to a 128-dim embedding, L2-normalized so similarity reduces to a plain dot product.

## The SimCSE trick: dropout as the only augmentation

Contrastive learning needs *pairs* — two different "views" that should map to similar
embeddings. Image contrastive methods (SimCLR) get views via crops/color jitter; text
methods often need paraphrase pairs or back-translation. SimCSE's insight: **skip
augmentation entirely** — pass the exact same input through the same model twice, in
training mode. Dropout randomly zeroes different neurons on each forward call, so the two
resulting embeddings are close but not identical — a real, if noisy, positive pair,
manufactured for free from unlabeled data with no augmentation pipeline needed.

## The loss: InfoNCE over in-batch negatives

For a batch of `B` sequences, two forward passes produce `z1` and `z2` (each `B × 128`,
L2-normalized). `z1[i]` and `z2[i]` are a positive pair; `z1[i]` and `z2[j]` for any `j !=
i` are negatives — every *other* sequence currently in the batch, free, no separate
negative-mining step:

```python
logits = z1 @ z2.T / temperature   # (B, B); logits[i, j] = cosine_sim(z1_i, z2_j) / temperature
labels = torch.arange(B)            # the positive pair is always at the diagonal
loss = 0.5 * (cross_entropy(logits, labels) + cross_entropy(logits.T, labels))
```

This is literally a `B`-way classification problem: "given `z1[i]`, which of the `B` items
in `z2` is its true pair?" — cross-entropy over a similarity-score row *is* InfoNCE. The
`temperature` (default 0.05) sharpens the similarity distribution before softmax — lower
values push the model to separate the true positive from negatives more aggressively.
Averaging the two directions (`z1`-finds-`z2` and `z2`-finds-`z1`) is the standard
symmetric formulation (also used by CLIP), not specific to text.

## Real training run, actually executed on this project's own MacBook (MPS)

`GPT_STEPS=600`, default `GPT_BATCH_SIZE=32` (so 31 in-batch negatives per anchor),
`GPT_TEMPERATURE=0.05`, same tokenized data as the other two objectives:

```
[model] 5,951,872 parameters

step   train_loss   val_loss   val_acc
0      0.8997       0.8332     0.9396
75     0.0235       0.0346     0.9938
150    0.0133       0.0140     0.9979
225    0.0088       0.0113     0.9979
300    0.0016       0.0022     1.0000
375    0.0016       0.0013     1.0000
450    0.0009       0.0013     1.0000
```

`val_acc` here is retrieval accuracy: for how many anchors is the true positive the
single highest-similarity match among all 32 candidates in the batch.

**Reading these numbers honestly — this saturates almost immediately, and that's an
expected property of this exact setup, not a sign of a bug or of unusually fast
learning**: even at step 0, before any training, `val_acc` starts at **0.94**, far above
the `1/32 ≈ 0.03` a genuinely random encoder would score. That's because `z1[i]` and
`z2[i]` come from the *same tokens* — only dropout differs between the two passes — so
even an untrained model's two passes end up more similar to each other than to a
different sequence's passes, just from sharing the same input. By step 300, `val_acc`
hits a ceiling of 1.000 and loss keeps dropping toward zero mainly by making the *margin*
between positive and negative similarity scores larger, not by resolving any remaining
ambiguity — there isn't any left to resolve. **This is the honest limitation of
same-input-only positive pairs at small batch size**: the task never required learning
real semantic similarity *between different sentences*, only self-consistency under
dropout noise for the *same* sentence — a real, useful signal for some purposes (it's
genuinely how SimCSE trains sentence-embedding models that do generalize), but this
project's version doesn't demonstrate that generalization, only the training mechanism
itself.

## What was deliberately left out, and why

- **Hard negative mining** — real production contrastive systems often deliberately
  include *difficult* negatives (near-misses) rather than relying purely on whatever's in
  the random batch, specifically because in-batch-only negatives saturate quickly, as
  seen above. Out of scope here to keep the implementation focused on the core mechanism.
- **A downstream evaluation task** (e.g., semantic textual similarity benchmarks) to
  actually measure whether the resulting embeddings generalize beyond this training
  setup's easy same-input-pair task. This project demonstrates the training mechanism
  works end-to-end, not that its specific output embeddings are production-quality.
- **Larger batch sizes to increase negative-pair difficulty** — the number of in-batch
  negatives is `batch_size - 1`; a larger batch is a real, simple lever for a harder
  (more informative) contrastive task, left as a natural follow-on experiment.

## The gotcha: this project's `estimate_loss` can't use `torch.no_grad()` on MPS

Every other `estimate_loss` in this project (`trainer.py`, `trainer_mlm.py`) wraps evaluation
in `@torch.no_grad()`, since eval never needs gradients. This one can't, unconditionally,
on Apple Silicon: `nn.MultiheadAttention` with `dropout_p > 0` routes into a fast
inference-only code path built on `F.scaled_dot_product_attention` as soon as gradient
tracking is off, and `NotImplementedError: scaled_dot_product_attention for MPS does not
support dropout` is a real error this project hit running exactly that combination during
development — not a hypothetical edge case. Since this objective's evaluation genuinely
needs dropout to stay active (that's what makes `z1 != z2`), `estimate_loss` in
[`trainer_contrastive.py`](../src/gpt/training/trainer_contrastive.py) keeps gradients enabled specifically on
MPS (harmless here — nothing calls `.backward()` on the resulting graph, so it's discarded
immediately) while still using `torch.no_grad()` on CUDA/CPU, where this MPS-specific
kernel limitation doesn't apply.
