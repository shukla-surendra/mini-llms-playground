# Training: Process, Hyperparameters, and What to Expect on a MacBook (MPS)

## The mechanism, if you need a refresher

`src/gpt/training/trainer.py` implements the exact four-step loop from
[`../../../docs/llm-engineering/03_how_neural_networks_learn.md`](../../../docs/llm-engineering/03_how_neural_networks_learn.md)
(forward pass → loss → backprop → gradient descent) — this doc assumes that mechanism is
understood and focuses on what's specific to *this* project: the actual hyperparameter
values, why they're set the way they are, and real, observed performance on a MacBook.
If you want the live progress-bar output itself decoded term by term (`loss=`, `lr=`,
`train=`, `val=`, `step/s`, and the rest), see
[`READING_TRAINING_OUTPUT.md`](READING_TRAINING_OUTPUT.md). For what "epoch" means for
this project and how to actually decide when to stop training, see
[`HOW_MUCH_TRAINING_IS_ENOUGH.md`](HOW_MUCH_TRAINING_IS_ENOUGH.md).

## Hyperparameters, and the reasoning behind each

All configurable via environment variables (see [`../README.md`](../README.md)'s
quickstart for the full list) — defaults shown here:

| Hyperparameter | Default | Reasoning |
|---|---|---|
| `GPT_BATCH_SIZE` | 32 | At this model's tiny size (~5.85M params) and `context_length=256`, a real batch of 32 fits comfortably in unified memory on any modern Mac — no need for the heavy gradient-accumulation workaround `custom-gpt-153m` uses for its much larger model |
| `GPT_GRAD_ACCUM` | 1 | Not needed at this scale — see above |
| `GPT_LR` | 3e-4 | A standard starting point for small Transformer training; see
[`../../../docs/llm-engineering/04_hyperparameter_tuning.md`](../../../docs/llm-engineering/04_hyperparameter_tuning.md)'s note that learning rate is the single most sensitive hyperparameter |
| `GPT_MIN_LR` | 3e-5 | The cosine-decay floor — see [`../../../docs/llm-engineering/03_how_neural_networks_learn.md`'s warmup/decay explanation](../../../docs/llm-engineering/03_how_neural_networks_learn.md#deep-dive-what-the-learning-rate-schedule-is-actually-doing) |
| `GPT_STEPS` | 5000 | Enough for this model/dataset combination to move well past random-output loss (~8.3, i.e. `ln(vocab_size)`) into coherent-short-sentence territory — see real numbers below |
| `GPT_EVAL_INTERVAL` | 250 | How often `train_loss`/`val_loss` are logged to `logs/train_eval_history_6m_causal.csv` |

## Real training run, actually executed on this project's own MacBook (MPS)

This isn't a projection — it's what actually happened running
`GPT_STEPS=4000 GPT_BATCH_SIZE=32 make train` on Apple Silicon MPS, with the default
100,000-story subset (~22.4M training tokens):

```
[device] using mps
[model] 5,853,184 parameters

step     train_loss   val_loss   val_perplexity
0        8.372        8.368      4308.6   <- near-random (ln(4096) ≈ 8.32)
200      4.378        4.395        81.1
400      3.859        3.887        48.8
1000     3.106        3.160        23.6
2000     2.674        2.728        15.3
3000     2.498        2.571        13.1
3999     2.442        2.471        11.8   <- final
```

Total wall-clock: **~15 minutes**, including dataset download, tokenizer training, and
all 20 evaluation passes. The real generated sample from this exact checkpoint is in
[`../README.md`'s results section](../README.md#real-results-from-this-projects-own-training-run),
and the complete, raw record of every evaluation is in
[`../logs/train_eval_history_6m_causal.csv`](../logs/train_eval_history_6m_causal.csv).

**Reading these numbers**: `ln(vocab_size) = ln(4096) ≈ 8.32` is the loss a model gets by
guessing *uniformly at random* over the vocabulary — the starting point (`8.372`) is
almost exactly there, confirming the model starts knowing nothing, as expected. A
perplexity of ~4,309 at step 0 means "on average, the model is as uncertain as if it were
choosing uniformly among ~4,309 tokens" — dropping to ~49 by step 400 means that
uncertainty has already fallen by roughly two orders of magnitude in under a minute of
wall-clock training.

## Throughput on Apple Silicon MPS

Observed: roughly **5-6 steps/second** at `batch_size=32`, `context_length=256` (so
~40,000-50,000 tokens/second) after the first few steps' MPS graph-compilation warmup —
the first 1-3 steps are noticeably slower (multiple seconds each) before settling into
this steady rate. At this throughput, `GPT_STEPS=4000` completes in roughly **12-15
minutes** end to end, including periodic evaluation passes.

## Resume behavior

Identical mechanism to
[`../../custom-gpt-153m/docs/MIGRATION.md`](../../custom-gpt-153m/docs/MIGRATION.md)'s
approach: `src/gpt/training/trainer.py` writes `checkpoints/6m/causal/latest.pt` every
`GPT_SAVE_EVERY` steps and on `Ctrl+C`, and checks saved hyperparameters against the
current run's config before resuming — the general mechanism (why checkpoints need to be
self-describing, why this check exists, atomic saves) is
[Chapter 27 — Checkpointing and Resuming Training](../../../docs/llm-engineering/27_checkpointing_and_resuming_training.md);
[Chapter 2](../../../docs/llm-engineering/02_parameters_vs_hyperparameters.md#try-it-yourself)
covers the same check from the parameters-vs-hyperparameters angle. Resume by simply
re-running `make train` — `gpt-train --no-resume` (or `make train-fresh`) forces a fresh start.

### The gotcha: resuming doesn't automatically train further past `GPT_STEPS`

`GPT_STEPS` (default `5000`) is a **target step count**, not "how many additional steps to
run." Resume logic computes `start_step = saved_checkpoint_step + 1` and then runs
`for step in range(start_step, steps)`. If your saved checkpoint is already at or past
`GPT_STEPS - 1`, this loop is empty (or nearly so) — the run will complete almost instantly,
print the same final numbers your checkpoint already had, and **train nothing new**, even
though it looks like a normal completed run. This is expected behavior, not a bug — but
it's genuinely easy to mistake for "I just trained more and got no improvement." To
actually continue training past a checkpoint that's already at your current `GPT_STEPS`
target, **raise `GPT_STEPS`** for the resumed run:

```bash
GPT_STEPS=10000 make train
```

See [`HOW_MUCH_TRAINING_IS_ENOUGH.md`](HOW_MUCH_TRAINING_IS_ENOUGH.md) for how to judge
whether doing this is actually likely to help, based on the loss trend at the point
training last stopped. If what you actually want is to continue training this checkpoint
on a *different* dataset (not just more of the same one), see
[`CONTINUING_TRAINING_ON_NEW_DATA.md`](CONTINUING_TRAINING_ON_NEW_DATA.md) — same resume
mechanism, but with one hard requirement (the tokenizer must match) that's easy to get
wrong silently.

## Diagnosing a bad run

Same `train_loss` vs. `val_loss` diagnostic as
[`../../../docs/llm-engineering/04_hyperparameter_tuning.md`](../../../docs/llm-engineering/04_hyperparameter_tuning.md#using-train_loss-vs-test_loss-as-your-tuning-feedback-signal):

- Both decreasing together → healthy.
- `train_loss` decreasing, `val_loss` flat/rising → overfitting; try more training stories
  (`--max-samples`), or increase `dropout`.
- Both stuck high → underfitting; try a higher `GPT_LR`, more `GPT_STEPS`, or check
  `logs/train_eval_history_6m_causal.csv` for an actual bug (loss should never be `NaN` or wildly
  oscillating — if it is, the learning rate is very likely too high).
