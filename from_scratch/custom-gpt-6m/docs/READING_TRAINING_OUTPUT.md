# Reading the Training Progress Bar, Term by Term

Companion to [`TRAINING.md`](TRAINING.md). While `gpt-train` (`src/gpt/training/trainer.py`) runs, it prints a live,
constantly-updating line like this (from [`tqdm`](https://github.com/tqdm/tqdm), the
progress-bar library used here):

```
training:  25%|██▎       | 250/1000 [00:51<02:28,  5.05step/s, loss=2.563, lr=4.53e-05, train=2.395, val=2.465]
```

This doc is that line, decoded piece by piece — every term maps to something specific in
[`../src/gpt/training/trainer.py`](../src/gpt/training/trainer.py).

## The parts `tqdm` generates automatically

```
training:  25%|██▎       | 250/1000 [00:51<02:28,  5.05step/s, ...]
```

| Part | Meaning |
|---|---|
| `training:` | The bar's label — set by `desc="training"` in `trange(start_step, steps, desc="training", unit="step")` |
| `25%` | Fraction complete: `250 / 1000` |
| `██▎` | The visual bar itself — purely cosmetic, same information as the percentage |
| `250/1000` | **Current step / total steps** — `250` is how many training steps have run so far; `1000` is `GPT_STEPS` from [`TRAINING.md`'s hyperparameter table](TRAINING.md#hyperparameters-and-the-reasoning-behind-each) (the total this run was configured for) |
| `[00:51<02:28, ...]` | `00:51` = elapsed wall-clock time so far; `02:28` = tqdm's *estimated* remaining time, computed from the observed steps/second rate — an estimate, not a guarantee, since throughput can vary (see [`TRAINING.md`'s note on MPS warmup](TRAINING.md#throughput-on-apple-silicon-mps)) |
| `5.05step/s` | Measured throughput: training steps completed per second, averaged over a recent window — directly comparable to the real, observed **5-6 steps/second** figure in [`TRAINING.md`](TRAINING.md#throughput-on-apple-silicon-mps) |

## The parts this project adds explicitly: `loss`, `lr`, `train`, `val`

These come from `trainer.py`'s own code, not from `tqdm` itself:

```python
postfix = {"loss": f"{loss.item():.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
if latest_eval:
    postfix.update(latest_eval)   # adds "train" and "val"
progress.set_postfix(**postfix)
```

### `loss=2.563` — this single step's raw training loss

The cross-entropy loss computed on **just this one batch** of `batch_size` sequences
(step 250's `get_batch` call), before the gradient-descent update happens — per
[`CODE_WALKTHROUGH.md`'s four-step-loop section](CODE_WALKTHROUGH.md#the-four-step-loop).

**Why this number jumps around noisily, step to step**: it's computed from one random
batch, and different batches genuinely vary in difficulty (some contain more predictable
text than others) — this is expected noise, not a bug. Don't judge training health from
`loss` alone; that's exactly what `train`/`val` below are for.

### `lr=4.53e-05` — the current learning rate

The actual learning rate this specific step is using, computed by `lr_for_step(step)` —
per [`CODE_WALKTHROUGH.md`'s warmup/cosine-decay section](CODE_WALKTHROUGH.md#lr_for_step--warmup-then-cosine-decay).
Watching this value confirms the schedule is behaving as expected: it should ramp up from
near-zero during the first ~2% of steps, then smoothly decrease toward `GPT_MIN_LR` for the
rest of the run. If `lr` looks flat or wrong, that's a real signal something in the
schedule configuration is off.

### `train=2.395` and `val=2.465` — the smoothed, periodic evaluation losses

**Not** computed from one batch — these come from `estimate_loss`, which runs every
`GPT_EVAL_INTERVAL` steps (default 250 — notice this line is *exactly* at step 250, the first
evaluation point after the initial one at step 0) and averages the loss over
`GPT_EVAL_BATCHES` separate batches for both the training set and a held-out validation set.
Because they're averaged over many batches, `train`/`val` are far less noisy than the raw
`loss` figure, and are the numbers actually worth watching for a training-health trend —
exactly the diagnostic pair covered in
[`../../../docs/llm-engineering/04_hyperparameter_tuning.md`'s train/val diagnostic
loop](../../../docs/llm-engineering/04_hyperparameter_tuning.md#using-train_loss-vs-test_loss-as-your-tuning-feedback-signal).

**Why `train` and `val` stay fixed between evaluation points**: they only update every
`GPT_EVAL_INTERVAL` steps — the line shown above will display the *same* `train=2.395,
val=2.465` for every step from 250 up to (but not including) the next evaluation at step
500, even as `loss` and `lr` keep changing every single step. This is expected — not a
sign the evaluation is "stuck."

## Reading the whole line as one diagnostic snapshot

```
training:  25%|██▎       | 250/1000 [00:51<02:28,  5.05step/s, loss=2.563, lr=4.53e-05, train=2.395, val=2.465]
```

Translated: *"250 of 1000 steps done, running at ~5 steps/sec with about 2m28s left at
this rate. This specific batch's loss was 2.563 (noisy, ignore in isolation). The
learning rate has ramped up to 4.53e-05 so far. As of the last full evaluation (also at
step 250), the averaged training loss was 2.395 and the averaged validation loss was
2.465 — close to each other, both trending down from where they started, which is the
healthy pattern."*

The gap between `train` (2.395) and `val` (2.465) here is small — a good sign, per the
overfitting/underfitting diagnostic in
[`../../../docs/llm-engineering/04_hyperparameter_tuning.md`](../../../docs/llm-engineering/04_hyperparameter_tuning.md#using-train_loss-vs-test_loss-as-your-tuning-feedback-signal).
A `val` noticeably and increasingly higher than `train` as training progresses would be
the signal to watch for instead.

## Where these same numbers end up permanently

Every evaluation point (not just what's visible in the live progress bar) is also written
to [`../logs/train_eval_history_6m_causal.csv`](../logs/train_eval_history_6m_causal.csv) — the full,
persistent record, including `val_perplexity` and `best_val_loss`, useful for reviewing
an entire run's trend after the fact rather than only watching it live.
