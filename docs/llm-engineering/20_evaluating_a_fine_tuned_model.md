# Evaluating a Fine-Tuned Model

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 3 — Fine-Tuning. Closes out
Part 3. Builds on every prior chapter — this is how you actually know whether all of it
(PEFT/LoRA, SFT, chat templates) produced something better, not just something different.

## In Plain English

After fine-tuning, you have two models: the original and the fine-tuned one. "Did it
work?" is a genuinely harder question than it sounds — a single loss number, by itself,
often doesn't answer it, and the right evaluation methodology depends on what the
fine-tuning was actually trying to change.

## The First-Principles Explanation

### Why loss/perplexity alone is a weaker signal here than it was for pretraining

Recall [Chapter 15](00_roadmap.md#part-2--pretraining-building-a-model-from-zero)
(pretraining evaluation): `val_loss` directly tracks the pretraining objective (predict
the next token well), so it's a strong, direct signal there. For a fine-tuned model,
loss on a held-out set of the *fine-tuning* data still measures "how well does the model
predict this specific data" — but that's often not quite the actual question. A model can
achieve excellent loss on held-out instruction-following examples while still producing
responses a human would judge as awkward, unhelpful, or subtly wrong in ways cross-entropy
loss doesn't directly penalize. Loss is a necessary sanity check (a fine-tune with rapidly
increasing loss is clearly broken), not a sufficient one.

### The real evaluation methods, and what each actually tells you

```
1. QUALITATIVE / SIDE-BY-SIDE COMPARISON
   Generate responses from BOTH the original and fine-tuned model, on the
   SAME prompts, and read them side by side. The most direct, always-
   applicable method — genuinely answers "did the behavior actually
   change the way I intended," which no automated metric fully captures.

2. HELD-OUT LOSS / PERPLEXITY
   Loss on fine-tuning-format data the model never trained on. Useful as
   a sanity check and for comparing between fine-tuning RUNS of the SAME
   kind (e.g., "did lora_r=32 do better than lora_r=16 on this exact
   data"), less useful as a sole verdict on real-world quality.

3. BENCHMARK SUITES (MMLU, HellaSwag, and similar)
   Standardized, automated test sets measuring specific capabilities
   (general knowledge, commonsense reasoning, etc.). Useful for comparing
   against other published models on a shared scale — genuinely
   overkill/out of scope for a small-scale, narrow-purpose fine-tune like
   this repo's own projects, but worth knowing these exist as the
   standard tool once evaluation needs to be rigorous and comparable
   across many models.

4. LLM-AS-JUDGE
   Use a separate, more capable model to read and score/compare outputs
   from the model(s) being evaluated — a real, increasingly common
   technique for getting a consistent, semi-automated quality signal at
   a scale hand-reading every output doesn't allow. Not infallible (the
   judge model has its own biases and blind spots), but a genuinely
   useful middle ground between pure human reading and pure automated
   metrics.
```

### The right method depends on what the fine-tuning was actually for

- **Fine-tuned for a specific behavior/format/tone** (like this curriculum's own
  fine-tuning projects) → **qualitative side-by-side comparison is the primary tool** —
  the actual question ("does it now respond in the target style/format") is best answered
  by directly reading real outputs against the original model's outputs on the same
  prompts.
- **Fine-tuned for a measurable capability** (e.g., improved math reasoning) → a
  benchmark testing that specific capability is more appropriate, since "better at math"
  has a more objective, testable definition than "better tone."
- **Comparing many candidate fine-tuning runs at scale** (many hyperparameter
  combinations, more than a human can read through individually) → LLM-as-judge or
  held-out loss as a first-pass filter, with qualitative review of the top candidates.

## Deep-Dive: Designing a Fair Before/After Comparison

A few real, easy-to-get-wrong details matter for a comparison to actually be meaningful:

- **Same prompts, same decoding settings, for both models** — comparing the original
  model at `temperature=1.0` against the fine-tuned model at `temperature=0.7` conflates
  the fine-tuning's effect with a decoding-setting difference
  ([`../../from_scratch/tinystories-gpt-6m/docs/TEMPERATURE_AND_SAMPLING.md`](../../from_scratch/tinystories-gpt-6m/docs/TEMPERATURE_AND_SAMPLING.md)),
  making the comparison uninterpretable.
- **Prompts the fine-tuning data didn't directly contain** — evaluating only on prompts
  taken verbatim from the training set risks mistaking memorization for genuine
  behavioral improvement; held-out or genuinely novel prompts test whether the change
  actually generalizes.
- **A mix of prompts squarely inside the target behavior, and some slightly outside it**
  — the first checks the fine-tuning worked at all; the second checks whether it
  generalized reasonably or overfit narrowly to the exact training distribution.

## Trade-offs

| Method | Upside | Cost |
|---|---|---|
| Qualitative side-by-side | Directly answers "did the actual behavior change as intended," no proxy metric involved | Doesn't scale to hundreds of comparisons; some subjectivity in judgment |
| Held-out loss/perplexity | Cheap, automatic, good for comparing runs of the same kind | Doesn't directly measure real-world response quality |
| Benchmark suites | Standardized, comparable across models | Often measures general capability, not the specific narrow behavior a small fine-tune targets |
| LLM-as-judge | Scales past what manual reading allows, more consistent than ad-hoc spot checks | Judge model has its own blind spots/biases; an added dependency, not a ground truth |

## Common Misconceptions

- **"Lower fine-tuning loss always means a better fine-tuned model."** Not necessarily —
  very low loss on the fine-tuning data specifically can indicate overfitting/
  memorization rather than genuinely improved, generalizing behavior; this is the same
  train/val gap reasoning from
  [Chapter 4](04_hyperparameter_tuning.md#using-trainloss-vs-testloss-as-your-tuning-feedback-signal),
  applied to fine-tuning instead of pretraining.
- **"A benchmark score is the objective, final word on quality."** Benchmarks measure
  specific, predefined capabilities — a high score doesn't guarantee good performance on
  a narrow, specific behavior a small fine-tune actually targeted, and a fine-tune can
  legitimately succeed at its real goal while barely moving a general benchmark's needle.
- **"You need automated metrics to evaluate a fine-tune properly."** For a small, targeted
  fine-tuning project, actually reading a reasonable number of real before/after outputs
  side by side is often the single most informative evaluation step available — not a
  fallback for when "real" evaluation isn't feasible.

## Practice Questions

1. Why can a fine-tuned model achieve excellent held-out loss on fine-tuning-format data
   while still producing responses a human would judge as low quality?
2. Design a fair before/after comparison for a model fine-tuned to answer questions in a
   more concise style — name the specific things you'd hold constant between the two
   models' generations, and why each one matters.
3. When would benchmark suites (MMLU-style) actually be the *wrong* primary evaluation
   tool for a fine-tuning project, even though they're the most "rigorous"-sounding
   option?

## Key Terms

- **Held-out evaluation data**: data not used during training, used to check whether
  results generalize rather than reflecting memorization.
- **Benchmark suite**: a standardized, automated test set measuring specific model
  capabilities, enabling comparison across different models on a shared scale.
- **LLM-as-judge**: using a separate, more capable model to score or compare another
  model's outputs, as a semi-automated middle ground between manual review and pure
  metrics.
- **Side-by-side comparison**: generating from both models on identical prompts under
  identical decoding settings, and directly comparing the outputs.
