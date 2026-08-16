# Catastrophic Forgetting and Continual Training

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 2B — Training at Scale
(appended after the original numbered catalog, alongside
[Chapter 25](25_efficient_attention_flash_and_sdpa.md),
[Chapter 26](26_distributed_training_ddp_and_fsdp.md), and
[Chapter 27](27_checkpointing_and_resuming_training.md) — see [Chapter 0](00_roadmap.md)'s
reading-order note). Builds on [Chapter 3](03_how_neural_networks_learn.md)'s gradient
descent mechanism and [Chapter 27](27_checkpointing_and_resuming_training.md)'s resume
mechanics — this chapter is about what happens when a resumed run trains on *different*
data than the checkpoint was originally trained on.

## In Plain English

A model has exactly one set of weights, and everything it knows lives in that one shared
set — there's no separate storage per skill or per dataset the way a filesystem has
separate files. If you train fully on dataset A, then switch entirely to dataset B,
gradient descent has no concept of "this weight matters for A, protect it" — at every
step it moves every weight in whatever direction reduces loss on the *current* batch,
full stop. If B's optimal weight values genuinely conflict with A's, continued training
erodes A-specific behavior, because nothing in the mechanism is protecting it. This is
**catastrophic forgetting**, and it's caused by training *order*, not by whether the old
data still exists somewhere — having dataset A sitting on disk the whole time doesn't
help if the model only ever sees it in an earlier, disconnected phase.

## The First-Principles Explanation

### Sequential vs. joint training: the difference has a name

- **Sequential** (train fully on A, then switch to B): loss on A tends to rise again by
  the end of the B phase, because nothing in that phase penalizes forgetting A.
- **Joint/shuffled** (mix A and B into one training stream from the start): every batch is
  a mixed signal from both, so the converged weights are a genuine compromise across both
  distributions — no forgetting phase to speak of, because there was never a phase where
  only one distribution's gradients were flowing.

If the goal is *both* skills long-term, joint training from the start is structurally
immune to this failure mode in a way sequential training isn't — but joint training isn't
always available (the new dataset may not exist yet when the original training happened,
which is exactly the continued-pretraining scenario below).

### What actually determines how much is retained, when training continues sequentially

| Factor | Less forgetting | More forgetting |
|---|---|---|
| Similarity between old and new data | Similar | Very different |
| Steps trained on new data alone | Few | Many |
| Whether old data is mixed in | Mixed throughout (replay) | Never revisited |
| Learning rate during continuation | Lower (gentler updates) | Same/higher than original |

None of these eliminate forgetting outright — they trade off how much of it happens
against how much the new data actually gets learned. A learning rate low enough to
prevent all forgetting also prevents the model from meaningfully adapting to the new data
at all; the useful middle ground is a genuine tuning problem, not a solved one.

### Replay: the practical, easy-to-implement mitigation

Rather than switching training data entirely, **replay** (also called rehearsal) mixes a
fraction of old data into every new batch, so the old distribution keeps getting
reinforced throughout continued training instead of only at the start. This directly
counteracts the mechanism described above by never letting the "current batch" be
exclusively new data. Across *many* sequential rounds (not just one old→new transition),
replaying only from the immediately-previous round isn't enough — round 3 can still
gradually erode round 1's patterns if round 1 is never revisited again several rounds
later; replay needs to draw from *every* prior round, not just the most recent one, to
avoid that slow compounding.

### Why fine-tuning is the scenario where this bites hardest

Fine-tuning a pretrained model on a narrow dataset is the classic catastrophic-forgetting
case, often worse than the phased-pretraining scenario above: the base model is already
converged on broad knowledge, and fine-tuning data is usually small and narrow, seen many
times over comparatively few steps — gradients push hard in one direction with nothing
counterbalancing the original distribution. This is why instruction-tuned/RLHF'd models
sometimes get measurably worse at general tasks even as they improve on the narrow tuning
target — a known trade-off inherent to the mechanism, not a bug in a specific
implementation. Mitigations here, roughly cheapest/most common to most involved:

- **Low learning rate + few epochs** — the cheapest lever; drifts less from the pretrained
  optimum by construction.
- **LoRA/adapters** — freeze the base weights entirely, train a small low-rank delta;
  forgetting is structurally limited to what that small delta can express, since the
  original weights never change at all (see [Chapter 17](17_lora_and_qlora.md)).
- **Data mixing/replay** — the same technique as above, applied to fine-tuning.
- **KL/regularization to a reference model** — explicitly penalize the fine-tuned model
  for diverging too far from the frozen original's output distribution, part of what RLHF
  does (see [Chapter 19](19_rlhf_and_dpo.md)).

A smaller model has less redundant capacity to "absorb" new data without overwriting old
knowledge than a large one — forgetting tends to be *more* visible at small parameter
counts, not less, which is worth expecting going in rather than treating as a surprise
when it shows up.

### The silent failure mode specific to continued pretraining: tokenizer mismatch

Continuing to train an existing checkpoint on new data (not fine-tuning in the LoRA
sense — the *same* full-parameter mechanism, just starting from trained weights instead
of random initialization) has one hard requirement beyond the forgetting trade-offs
above: the tokenizer must be **identical**, not merely the same `vocab_size`. Training a
fresh tokenizer on new data — even one whose vocabulary happens to come out to the exact
same size — produces a different mapping from token ID to text. A resume-compatibility
check that only compares `vocab_size` (a single integer, per
[Chapter 27](27_checkpointing_and_resuming_training.md)'s architecture check) passes
every time regardless, because it has no way to know the vocabulary's *content* changed.
The model loads and "trains" without error; every token ID now silently means something
different than what the checkpoint's embedding table learned it to mean, and generation
quality degrades in a way that's confusing to debug after the fact, because nothing
failed loudly.

The fix is mechanical: reuse the exact tokenizer file the checkpoint was trained with,
rather than training a new one, whenever continuing training on new data is the goal. If
the new data's vocabulary is different enough that reusing the old tokenizer would be
badly inefficient (switching domains entirely — say, prose to source code), that's a real
signal a genuinely new tokenizer is the better choice — but at that point resuming from
the old checkpoint isn't meaningfully possible at all, since the embedding table and
output head are shaped to, and semantically tied to, the old vocabulary; the honest option
is training a new model from scratch on the new data.

## Grounded in This Repo's Code

[`from_scratch/custom-gpt-10m/docs/LLM_DEV_GUIDE.md`](../../from_scratch/custom-gpt-10m/docs/LLM_DEV_GUIDE.md)
works through the sequential-vs-joint question directly, including a concrete fix for
seeing per-dataset results without paying the sequential-forgetting cost: train on the
shuffled union, but keep separate validation sets per source and log loss against each
independently — a small change to the eval loop that runs the eval helper once per
held-out split instead of one combined split.
[`from_scratch/custom-gpt-6m/docs/CONTINUING_TRAINING_ON_NEW_DATA.md`](../../from_scratch/custom-gpt-6m/docs/CONTINUING_TRAINING_ON_NEW_DATA.md)
implements the tokenizer-mismatch fix as `prepare_dataset.py --reuse-tokenizer`, and shows
forgetting being *measured*, not just discussed — keeping the original dataset's `val.bin`
around and periodically evaluating a continuation run against it, so whether (and how
much) forgetting is happening becomes a number you can watch rather than a theoretical
worry.
[`from_scratch/custom-gpt-6m/docs/CONTINUAL_TRAINING_LOW_RESOURCE.md`](../../from_scratch/custom-gpt-6m/docs/CONTINUAL_TRAINING_LOW_RESOURCE.md)
extends replay to *many* sequential rounds with `scripts/build_replay_mix.py` (sampling
from every prior round's already-tokenized data, not just the last one) plus reversible
checkpoint snapshots (`make snapshot`/`make restore-snapshot`) so a bad round can be
rolled back instead of permanently degrading the model.

## Deep-Dive: Why Having the Old Data On Disk Doesn't Help by Itself

A tempting but wrong intuition: "forgetting can't be a real problem, the old dataset is
still right there on disk, nothing was deleted." This misunderstands where a model's
knowledge actually lives. Gradient descent only ever responds to what's in the *current*
batch — the data sitting unused on disk contributes exactly zero gradient signal to the
current training step. Availability and training exposure are entirely different things;
only the latter affects what the weights currently encode. This is precisely why replay
works and "just don't delete the old data" doesn't: replay puts old data back into actual
batches the model trains on, while an unused file on disk, however complete, influences
nothing.

## Try It Yourself

- Using this repo's per-source validation logging idea, sketch (or implement) evaluating
  a joint-trained model's loss against dataset A's held-out set and dataset B's held-out
  set separately — confirm both stay reasonable, in contrast to what a sequential
  train-A-then-B run's A-loss would look like by the end of the B phase.
- Walk through what a resume-compatibility check that only compares `vocab_size` would do
  if handed two independently-trained tokenizers that happen to produce the same
  `vocab_size` — confirm for yourself it passes, and explain why that's the dangerous
  case, not the case where sizes visibly differ.

## Common Misconceptions

- **"Catastrophic forgetting only matters for full fine-tuning of huge pretrained
  models."** It's a general property of gradient descent on a shared weight set —
  visible at any scale, including small from-scratch models trained sequentially on
  multiple datasets, and often *more* visible at small scale due to limited capacity.
- **"If the old dataset still exists on disk, the model hasn't really forgotten it."**
  Availability isn't training exposure — only data actually appearing in training batches
  affects the weights; an unused file on disk contributes no gradient signal at all.
- **"Two tokenizers with the same `vocab_size` are interchangeable."** They can have
  completely different token-ID-to-text mappings despite matching sizes — this is exactly
  the silent-failure trap a `vocab_size`-only compatibility check misses.

## Practice Questions

1. Explain, mechanically, why sequential training (A fully, then B fully) causes A's loss
   to rise during the B phase, while joint/shuffled training of the same two datasets
   doesn't show that pattern.
2. A resume loads successfully, `vocab_size` matches, and no error is raised — under what
   specific condition could the run still be silently corrupted, and how would you detect
   it?
3. Why does replay need to draw from *every* prior round in a multi-round continual
   setup, rather than just the immediately preceding round?

## Key Terms

- **Catastrophic forgetting**: degradation of previously-learned behavior caused by
  continued training on different data, driven by training order, not data availability.
- **Replay / rehearsal**: mixing a fraction of old data into new training batches to keep
  reinforcing the old distribution throughout continued training.
- **Continued pretraining**: resuming full-parameter training of an existing checkpoint on
  new data — mechanically identical to an ordinary resume, but requiring the tokenizer to
  match exactly, not just in `vocab_size`.
- **Tokenizer mismatch**: two tokenizers with matching `vocab_size` but different
  token-ID-to-text mappings, which passes a size-only compatibility check while silently
  scrambling what every token ID means to the model.
