# The Real Training Run

Actually executed: `uv run train_lora.py --max-samples 4000 --batch-size 4 --grad-accum 4
--epochs 3` on Apple Silicon MPS.

## Setup

```
[model] trainable params: 4,884,480 || all params: 139,399,488 || trainable%: 3.5039
```

Matches [Chapter 16](../../../docs/llm-engineering/16_fine_tuning_landscape.md#deep-dive-why-peft-specifically-matters-for-this-curriculums-no-gpu-constraint)'s
prediction directly: only 3.5% of the model's parameters were ever trained — every other
weight stayed exactly as SmolLM2-135M's original authors trained it.

- **4,000 training examples** (a random subset of Dolly-15k's ~15,011 rows, shuffled with
  a fixed seed for reproducibility)
- **3 epochs**, effective batch size 16 (`batch_size=4 × grad_accum=4`)
- **750 total optimizer steps**

## Timing — real, observed

```
train_runtime: 3808 seconds (~63.5 minutes)
train_samples_per_second: 3.15
train_steps_per_second: 0.197
```

Slower per-step than [`../../../from_scratch/tinystories-gpt-6m/`](../../../from_scratch/tinystories-gpt-6m/)'s
training (~5-6 steps/sec there) — expected and consistent with
[Chapter 16](../../../docs/llm-engineering/16_fine_tuning_landscape.md)'s reasoning:
even with only 3.5% of parameters trainable, every forward/backward pass still runs
through the *entire* 135M-parameter frozen base model (LoRA reduces the **training
memory/gradient** cost, not the per-step **compute** cost of the forward pass itself —
worth being precise about this distinction, since it's a common point of confusion).

## Loss curve — real, logged every 10 steps

```
step (epoch)    loss
0.04             2.607   <- start
0.24             2.152
0.44             2.199
0.64             2.149
0.84             2.156
1.04             2.139
1.24             2.034
1.44             2.154
1.64             2.101
1.84             2.201
2.04             2.175
2.24             2.176
2.44             2.023
2.64             2.140
2.84             1.988
3.00 (final)     2.153
```

## An honest limitation of this run's measurement, worth naming directly

Unlike [`../../../from_scratch/tinystories-gpt-6m/`](../../../from_scratch/tinystories-gpt-6m/)'s
training loop (which computes a separate, smoothed `val_loss` on held-out data every
`eval_interval` steps — see
[`../../../from_scratch/tinystories-gpt-6m/docs/READING_TRAINING_OUTPUT.md`](../../../from_scratch/tinystories-gpt-6m/docs/READING_TRAINING_OUTPUT.md#train-2395-and-val-2465--the-smoothed-periodic-evaluation-losses)),
`train_lora.py` (like the sibling
[`../../tinyllama-1.1b-lora/train_tinyllama_lora.py`](../../tinyllama-1.1b-lora/train_tinyllama_lora.py)
it's modeled on) only logs raw, per-batch training loss — no held-out validation set was
configured for this run. This is a real, honest gap:

- **What this loss curve tells you**: training is proceeding (loss dropped from ~2.6 to
  ~2.0-2.15 and largely plateaued there), and nothing pathological happened (no NaN, no
  runaway increase).
- **What this loss curve does NOT tell you**: whether the model is overfitting the
  specific 4,000 training examples rather than genuinely generalizing — per
  [Chapter 4](../../../docs/llm-engineering/04_hyperparameter_tuning.md#using-trainloss-vs-testloss-as-your-tuning-feedback-signal)'s
  train/val gap diagnostic, that question requires a held-out set this run didn't
  measure against.

**Why [`BEFORE_AFTER_COMPARISON.md`](BEFORE_AFTER_COMPARISON.md) is the evaluation that
actually matters here**: per
[Chapter 20](../../../docs/llm-engineering/20_evaluating_a_fine_tuned_model.md#why-lossperplexity-alone-is-a-weaker-signal-here-than-it-was-for-pretraining),
loss alone is a weaker signal for a fine-tuned instruction-following model than for
pretraining — the qualitative, side-by-side comparison is what actually answers "did this
work," and that comparison's prompts were never part of the 4,000-example training set,
giving real (if informal) evidence the improvement generalizes beyond memorized training
examples.
