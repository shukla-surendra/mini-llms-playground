# RLHF, DPO, and Preference Optimization

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 3 — Fine-Tuning. Builds on
[Chapter 18](18_instruction_tuning_and_sft.md)'s observation that SFT teaches a model to
produce *plausible* responses but not to distinguish *better* ones. This chapter is the
mechanism that closes that specific gap — covered for completeness of the landscape, even
though (as the deep-dive explains) neither technique is what this repo's own projects
use.

## In Plain English

SFT trains a model to imitate example responses. But real usefulness often depends on
*preference*, not just plausibility — given two reasonable responses, which one is
actually more helpful, accurate, or safe? That's not something a single "correct answer"
per training example can teach. RLHF and DPO are two different ways of training a model
directly on **comparisons** — "response A is better than response B" — rather than single
target outputs.

## The First-Principles Explanation

### Why SFT alone has a real, structural limitation

An SFT training example has exactly one target response. The loss
([Chapter 3](03_how_neural_networks_learn.md#step-2-the-loss-function)) pushes the model
toward reproducing *that* response's pattern — but real questions often have many
reasonable responses of varying quality, and SFT has no mechanism to express "these two
different responses are both plausible, but one is meaningfully better." Preference-based
training exists specifically to supply that missing signal.

### RLHF: the original, more complex approach

**RLHF (Reinforcement Learning from Human Feedback)** is a multi-stage pipeline:

```
Stage 1: Collect preference data
  Humans (or, increasingly, another capable model — "RLAIF") are shown
  pairs of responses to the same prompt and pick which is better.

Stage 2: Train a REWARD MODEL
  A separate model is trained to predict human preference — given a
  (prompt, response) pair, output a single score estimating how good a
  human would rate it. This uses the pairwise comparison data directly:
  the reward model is trained so it scores the preferred response higher
  than the rejected one.

Stage 3: Reinforcement learning (typically PPO — Proximal Policy
Optimization) against the reward model
  The SFT model generates responses; the reward model scores them; PPO
  updates the SFT model's weights to increase expected reward — using
  reinforcement learning (not supervised learning) because there's no
  single "correct" target sequence anymore, only a scalar reward signal
  to increase.
```

**Why RLHF is genuinely complex and resource-intensive**: at Stage 3, you need **multiple
full model copies in memory simultaneously** — the policy model being trained, a frozen
reference copy of it (to prevent it from drifting too far and "reward hacking" — finding
weird outputs that fool the reward model without actually being good), the reward model
itself, and often a separate value/critic model PPO uses internally. This is a
meaningfully heavier setup than SFT or plain LoRA fine-tuning, and PPO training is also
known to be less stable and more hyperparameter-sensitive than ordinary supervised
training.

### DPO: the simpler, mathematically equivalent alternative

**DPO (Direct Preference Optimization)** achieves a similar end result — a model that
prefers better responses — **without** a separate reward model or any reinforcement
learning loop at all. The key insight: it's possible to derive a loss function, directly
in terms of the model's own output probabilities on preferred vs. rejected responses,
that's mathematically shown to optimize toward the same objective RLHF's full
reward-model-plus-RL pipeline does.

```
DPO training data: the SAME shape as RLHF's Stage 1 —
  (prompt, preferred_response, rejected_response) triples

DPO loss (conceptually): increase the model's relative probability of
  generating the preferred response over the rejected one, compared to
  a frozen reference copy of the model — computed directly via a
  modified cross-entropy-style loss, no reward model, no PPO, no
  separate rollout/generation step during training.
```

**Why this matters practically**: DPO needs only **two** models in memory (the model
being trained, and a frozen reference copy) — no separate reward model, no value network
— a meaningfully lighter, more stable, and simpler-to-implement setup than full RLHF,
while targeting the same underlying goal. This is *why* DPO has become the more common
choice for teams without the infrastructure/expertise investment RLHF's full pipeline
demands.

## Deep-Dive: Neither Technique Is What This Repo's Projects Use, and Why That's the Honest Right Call

Both `custom-gpt-153m` and `fine_tuning/tinyllama-1.1b-lora/` stop at
[Chapter 18](18_instruction_tuning_and_sft.md)'s SFT stage — worth being explicit about,
rather than implying more happened than actually did:

- **RLHF is genuinely out of reach for this curriculum's "no GPU, MacBook" constraint** —
  needing multiple full model copies simultaneously (even with LoRA reducing each
  individual model's trainable-parameter footprint per
  [Chapter 17](17_lora_and_qlora.md), the *frozen* copies still occupy full memory) pushes
  well past what's practical on consumer hardware for anything beyond the smallest
  models.
- **DPO is more tractable, but still adds real requirements this repo's projects
  deliberately don't take on**: preference-pair data (prompt + two responses + a
  preference label) is a different, harder-to-source data shape than SFT's single-
  response-per-example format both existing datasets
  (`HuggingFaceH4/ultrachat_200k`, TinyStories) already provide directly.

**The honest, correct framing — now confirmed, not just inferred**:
[`base_models/tinyllama-1.1b-base-serving/docs/MODEL_DETAILS.md`](../../base_models/tinyllama-1.1b-base-serving/docs/MODEL_DETAILS.md)
pulled the real model card directly and confirmed it explicitly: `TinyLlama-1.1B-Chat`,
the base model `fine_tuning/tinyllama-1.1b-lora/` fine-tunes further, was put through
**both** SFT (on a UltraChat variant) **and** DPO (via TRL's `DPOTrainer` on
`openbmb/UltraFeedback`, 64k GPT-4-ranked completions) by its original creators, following
Hugging Face's own Zephyr recipe — before this repo ever touches it. This repo's own LoRA
fine-tuning is an *additional* SFT pass on top of a model that already completed the full
SFT→DPO pipeline this chapter describes, not a from-scratch demonstration of that
pipeline. The same pattern holds for
[`base_models/smollm2-135m-base-serving/docs/MODEL_DETAILS.md`](../../base_models/smollm2-135m-base-serving/docs/MODEL_DETAILS.md)'s
findings on SmolLM2's `-Instruct` sibling (SFT + DPO on UltraFeedback again) — though
notably, [`fine_tuning/smollm2-135m-dolly-lora/`](../../fine_tuning/smollm2-135m-dolly-lora/)
deliberately fine-tunes the **base**, non-instruct SmolLM2 checkpoint instead, precisely
to demonstrate SFT's effect starting from *before* any of this pipeline has run — see
that project's [`docs/APPROACH.md`](../../fine_tuning/smollm2-135m-dolly-lora/docs/APPROACH.md).
Knowing precisely where in this pipeline your own work sits — pretraining, SFT, or
preference tuning — is itself a real skill, distinct from being able to execute every
stage yourself.

## Trade-offs

| Technique | Upside | Cost |
|---|---|---|
| RLHF (PPO-based) | The original, most-studied approach; explicit reward model can be reused/audited separately | Multiple full model copies in memory; PPO training instability; most complex to implement correctly |
| DPO | Same underlying goal, no separate reward model or RL loop, more stable | Still needs preference-pair data (harder to source than plain SFT data); simpler but not zero-cost |
| Stopping at SFT (what this repo does) | Achievable on modest hardware, real quality improvement over a raw base model | Model can't distinguish "acceptable" from "actually the best" response among several plausible ones |

## Common Misconceptions

- **"RLHF and DPO are unrelated techniques."** They target the mathematically related
  objective — DPO was specifically derived as a simpler alternative that reaches a
  similar outcome to RLHF's reward-model-plus-PPO pipeline, not an unrelated idea.
- **"A '-Chat' or '-Instruct' model has definitely been through RLHF/DPO."** Not
  guaranteed — some instruction-tuned open models are SFT-only; the exact pipeline varies
  by creator and is usually documented in the model's own release materials, worth
  checking rather than assuming.
- **"You need RLHF or DPO to get any benefit from fine-tuning."** SFT alone, as both this
  repo's projects demonstrate, produces a real, measurable behavior change — RLHF/DPO
  address a specific, further refinement (preference among plausible responses), not a
  precondition for fine-tuning to be worthwhile at all.

## Practice Questions

1. Why does RLHF's Stage 3 need a *frozen reference copy* of the policy model, in
   addition to the copy actually being trained?
2. Explain, at a high level, why DPO can achieve a similar training objective to RLHF
   without ever training a separate reward model.
3. This repo's `fine_tuning/tinyllama-1.1b-lora/` project fine-tunes an already
   `-Chat`-suffixed model. What does that imply about which pipeline stages have already
   happened before this repo's own training even starts?

## Key Terms

- **Reward model**: a model trained to predict human (or AI) preference between
  responses, used as the training signal in RLHF's Stage 3.
- **PPO (Proximal Policy Optimization)**: the reinforcement learning algorithm commonly
  used to optimize a policy model against a reward model in RLHF.
- **RLAIF**: RLHF where an AI model, not a human, provides the preference labels.
- **DPO (Direct Preference Optimization)**: a simpler alternative to RLHF that trains
  directly on preference pairs via a modified loss, without a separate reward model or RL
  loop.
- **Preference pair**: a (prompt, preferred response, rejected response) training example
  — the data shape both RLHF's reward-model stage and DPO consume.
