# What Is an Epoch, and How Do You Know When to Stop Training?

Companion to [`TRAINING.md`](TRAINING.md) and
[`READING_TRAINING_OUTPUT.md`](READING_TRAINING_OUTPUT.md). Both questions in this doc's
title are answered using this project's own **real, actual training run** — not a
hypothetical — pulled directly from [`../logs/train_eval_history.csv`](../logs/train_eval_history.csv).

## What Is an Epoch

**One epoch = one full pass through the entire training dataset.** If you have 1,000
training examples and you've shown the model all 1,000 of them once, that's one epoch,
regardless of how many gradient-descent steps ([`../../../docs/llm-engineering/03_how_neural_networks_learn.md`](../../../docs/llm-engineering/03_how_neural_networks_learn.md))
it took to do it.

### Converting steps to epochs for this project, with real numbers

From [`data/meta.json`](../data/meta.json), this project's actual training set has
**22,425,706 tokens**. Each training step consumes `batch_size × context_length` tokens —
with this project's defaults, `32 × 256 = 8,192` tokens per step. So:

```
steps per epoch ≈ train_tokens / (batch_size × context_length)
                ≈ 22,425,706 / 8,192
                ≈ 2,738 steps
```

At step 4,000 (roughly where this project's own training was at when this doc was
written), that's `4,000 / 2,738 ≈ 1.46` — **about one and a half epochs.**

### An important nuance: this project's "epoch" is a statistical approximation, not exact

[`../train.py`](../train.py)'s `get_batch` function picks a **random** starting position
for every batch:

```python
ix = torch.randint(0, max_start, (bsz,))
```

This means training does **not** walk through the dataset sequentially, non-overlapping
chunk by chunk, the way a classic `DataLoader`-based training loop (with shuffling and no
replacement) does. Some tokens get sampled multiple times before others get sampled even
once. After ~2,738 steps, each token has been seen *on average* about once — but "one
epoch" here is a useful approximation for reasoning about training progress, not a
literal guarantee that every single token was shown to the model exactly once. This is a
deliberate simplicity trade-off (the same one nanoGPT-style implementations commonly
make) — worth knowing about rather than assuming epoch-counting here works identically to
every other training setup you might read about.

## How Do You Know How Much Training Is Enough?

There's no fixed answer in steps or epochs ahead of time — it depends on watching the
actual training signal and reading it correctly. Here's this project's real evaluation
history, in full:

```
step    train_loss   val_loss   val_perplexity   improved
0       8.372        8.368      4308.6            yes
1000    3.106        3.160        23.6            yes
2000    2.674        2.728        15.3            yes
3000    2.498        2.571        13.1            yes
4000    2.395        2.465        11.8            yes
4250    2.426        2.503        12.2            NO  <- val went UP slightly
4500    2.420        2.450        11.6            yes (new best)
4750    2.390        2.464        11.8            NO  <- val went UP slightly again
4999    2.388        2.417        11.2            yes (new best, run's final step)
```

This run's actual final evaluation (step 4,999, the last step of a 5,000-step run) landed
on a **new best** — `val_loss=2.417`, right after the step-4,750 dip. This is Signal #1
and Signal #2 both visible in the same real data: noisy step to step (the 4,250 and 4,750
dips), but the underlying trend was still heading down at the point training stopped —
not yet at the flattened, no-further-improvement point this doc's stopping rule describes.
Concretely: this particular run likely stopped a bit *before* the point of maximum value
from more training, not after it — see [`TRAINING.md`](TRAINING.md#resume-behavior) for
how to resume and push further with `make train-resume` / `STEPS=<higher number>`.

### Signal #1: the rate of improvement, not just the direction

Look at how much `val_loss` dropped over each successive 1,000-step block:

```
step 0    -> 1000:  8.368 -> 3.160   (Δ = -5.208)
step 1000 -> 2000:  3.160 -> 2.728   (Δ = -0.432)
step 2000 -> 3000:  2.728 -> 2.571   (Δ = -0.157)
step 3000 -> 4000:  2.571 -> 2.465   (Δ = -0.106)
```

This is a textbook **diminishing-returns curve** — each additional block of training
steps buys a smaller and smaller improvement. This is completely normal and expected (not
a sign anything is wrong) — but it's exactly the pattern you watch to judge "is more
training still worth it." Early on, 1,000 more steps bought a massive improvement (-5.2);
by step 3,000-4,000, the same 1,000 steps bought a tenth of that (-0.1). At some point
further out, this curve will flatten close to zero — that's the point additional training
genuinely stops being worth the wall-clock time.

### Signal #2: a single non-improving evaluation is noise, not a stop sign

Notice `improved=NO` at step 4,250 and again at step 4,750 — `val_loss` briefly ticked
*up* (2.465 → 2.503, then 2.450 → 2.464). **This is normal noise, not evidence training
has "gone bad."** `estimate_loss` averages over `EVAL_BATCHES` (20) *randomly sampled*
batches each time — different random batches naturally produce a somewhat different
average from one evaluation to the next, especially once the loss curve has mostly
flattened and the remaining differences are small relative to this sampling noise. The
real signal to watch for is a **sustained** run of several consecutive non-improving
evaluations, not any single one.

This is exactly why [`../train.py`](../train.py) tracks `best_val_loss` and only
overwrites the "best" checkpoint when a new evaluation actually beats it:

```python
improved = losses["val"] < best_val_loss
if improved:
    best_val_loss = losses["val"]
    torch.save(payload, best_checkpoint_path)
    torch.save(payload, checkpoint_path)
```

Concretely: even though the step-4,250 and step-4,750 evaluations were non-improving
blips, `tinystories_gpt_checkpoint.pt` (the "best" / serving checkpoint) skipped straight
past both of them and now holds the **step 4,999** weights — the actual final,
best-so-far evaluation of this run. The model served via `api_server.py` never gets
contaminated by an intermediate, slightly-worse-by-chance evaluation. You never have to
guess which checkpoint was best; the training loop already tracked it for you.

### Signal #3: the train/val gap — the overfitting check

At the run's final step (4,999): `train_loss=2.388`, `val_loss=2.417` — a gap of about
`0.029`, if anything *smaller* than the `0.074` gap seen at step 4,750. Per
[`../../../docs/llm-engineering/04_hyperparameter_tuning.md`'s diagnostic](../../../docs/llm-engineering/04_hyperparameter_tuning.md#using-train_loss-vs-test_loss-as-your-tuning-feedback-signal):

- **Gap staying small and roughly stable** (what's happening here) → healthy, training is
  still generalizing, not just memorizing.
- **Gap steadily widening over many consecutive evaluations** (train keeps dropping,
  val stalls or rises) → overfitting — the real signal to actually stop, or to add more
  training data / increase `dropout`, not just a data point to note in passing.

This project hasn't shown that widening-gap pattern at any point across the run — which is
itself useful information: it suggests there was still real headroom for more training
steps before overfitting would become the limiting factor, consistent with Signal #1
showing the loss was still (slowly) decreasing all the way through the run's final step.

### Signal #4: does the generated text actually sound better — the goal itself

Loss and perplexity are proxies, not the actual goal. This project's stated goal (per
[`../README.md`](../README.md)) is "grammatically sensible, locally coherent text, not
garbage" — a judgment call, not purely a number. Periodically running
`python inference.py --prompt "..."` on the current best checkpoint and reading the
output is a real, legitimate part of deciding when training is "enough," alongside the
loss curve — a lower validation loss that doesn't translate into noticeably better
generated text is a sign you're optimizing the proxy past the point it still tracks the
actual goal.

## A Practical Stopping Rule, Combining All Four Signals

```
Keep training while:
  - val_loss is still trending down over multi-hundred-step windows (Signal #1), AND
  - the train/val gap isn't steadily widening (Signal #3)

Consider stopping once:
  - val_loss has failed to improve for several consecutive evaluations in a row
    (not just one — Signal #2), AND/OR
  - the per-1,000-step improvement has dropped close to the "noise floor" you can
    already see between adjacent evaluations (Signal #1), AND
  - generated samples aren't noticeably improving anymore (Signal #4)

Either way: `tinystories_gpt_checkpoint.pt` always holds the best checkpoint seen so
far, regardless of how much further training continues past that point (this is
"early stopping" via best-checkpoint tracking, already built into train.py) — so
there's little downside to letting a run continue a while past its likely-optimal
point, since you can always serve the best checkpoint rather than the final one.
```

This project's own run ended at step 4,999 (a 5,000-step budget) still in the "keep
training, still improving, gap still small" zone per this rule — the run hit its
configured step budget before hitting a real stopping signal. That's a legitimate,
honest outcome (`STEPS` was a budget decision, not a claim that 4,999 was the optimal
stopping point) — resuming with `make train-resume` (or `STEPS=10000 make train-resume`)
and re-checking Signal #1's improvement rate is the direct next step if squeezing out
more quality is the goal.
