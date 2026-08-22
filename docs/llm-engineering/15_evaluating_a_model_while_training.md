# Evaluating a Model While It's Still Training

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 2 — Pretraining: Building a
Model From Zero. Builds on [Chapter 4](04_hyperparameter_tuning.md)'s train-loss-vs-
test-loss diagnostic — this chapter assumes that pairing is already familiar and answers
two narrower questions Chapter 4 doesn't: what an **epoch** actually is once training
samples random windows instead of walking the dataset sequentially, and how to combine a
handful of signals into an actual decision about *when to stop*.

## In Plain English

Watching a loss number go down doesn't by itself tell you when to stop training — a
single tick upward can be pure noise, and "the number is still falling" doesn't mean
another hour of training is worth the wall-clock time. Deciding when training has done
enough means reading several signals together: is the *rate* of improvement still
meaningful, is a single bad evaluation actually a sustained trend, is the gap between
training and validation loss staying small, and — the thing loss is a proxy for in the
first place — does the model's actual output look better.

## The First-Principles Explanation

### What an epoch is, and why it's an approximation here

One epoch is one full pass through the training set. For a training loop that samples a
**random** window of tokens on every step (`ix = torch.randint(0, max_start, (bsz,))`,
the standard approach in nanoGPT-style trainers), there is no literal walk through the
data in fixed-size, non-overlapping chunks the way a shuffled `DataLoader` provides —
some tokens get sampled more than once before others are sampled at all. "Epochs" here is
still a useful unit for reasoning about training progress:

```
steps per epoch ≈ train_tokens / (batch_size × context_length)
```

but it's a statistical approximation of average coverage, not a guarantee every token was
seen exactly once. Worth knowing explicitly rather than assuming every training setup
counts epochs the same way.

### Four signals, read together

None of these is sufficient alone — each catches a failure mode the others miss.

1. **Rate of improvement, not just direction.** Compare `val_loss` drop over successive
   equal-sized step windows. A healthy run shows a diminishing-returns curve: large drops
   early, progressively smaller drops later. That's expected, not a warning sign — but
   watching how close the per-window improvement has gotten to zero is the direct signal
   for "is more training still worth the time."
2. **A single non-improving evaluation is noise, not a stop sign.** Evaluation loss is
   itself estimated from a sample of batches, so it carries its own sampling noise —
   especially once the curve has mostly flattened and remaining differences are small
   relative to that noise. The signal worth acting on is a **sustained** run of several
   non-improving evaluations in a row, not any single one.
3. **The train/val gap — reuses [Chapter 4](04_hyperparameter_tuning.md#using-train_loss-vs-test_loss-as-your-tuning-feedback-signal)'s
   diagnostic as a stopping criterion, not just a tuning one.** A gap that stays small and
   roughly stable means training is still generalizing. A gap that widens steadily over
   many evaluations (train still falling, val stalled or rising) is the real signal to
   stop, or to intervene (more data, more dropout) — not just a data point to note.
4. **Does the generated output actually sound better.** Loss and perplexity are proxies
   for the real goal, not the goal itself. A validation loss that keeps improving without
   the generated text improving alongside it is a sign the proxy has stopped tracking what
   actually matters. (What perplexity actually is, precisely, and why reading it instead
   of raw loss changes intuition but not information: [Chapter 29](29_perplexity_understanding_and_interpreting_it.md).)

### A practical stopping rule combining all four

```
Keep training while:
  - val_loss is still trending down over multi-hundred-step windows (signal 1), AND
  - the train/val gap isn't steadily widening (signal 3)

Consider stopping once:
  - val_loss has failed to improve for several consecutive evaluations in a row
    (signal 2, not just one), AND/OR
  - the per-window improvement has dropped close to the noise floor already visible
    between adjacent evaluations (signal 1), AND
  - generated samples aren't noticeably improving anymore (signal 4)
```

Because best-checkpoint tracking (see below) means there's little cost to training past
the optimal point — you can always serve the best checkpoint rather than the final one —
this rule is a guide for *when more training stops being worth the wall-clock*, not a
hard correctness boundary the way, say, a shape mismatch is.

## Grounded in This Repo's Code

Every from-scratch project here tracks "best" separately from "latest," specifically so a
temporarily-worse evaluation never contaminates what gets served:

```python
# from_scratch/custom-gpt-6m/src/gpt/training/trainer.py
improved = losses["val"] < best_val_loss
if improved:
    best_val_loss = losses["val"]
    torch.save(payload, best_checkpoint_path)
    torch.save(payload, checkpoint_path)   # the serving checkpoint
```

A real run from this project's own evaluation history makes signals 1 and 2 concrete in
the same data: `val_loss` dropped 8.368 → 3.160 → 2.728 → 2.571 → 2.465 over steps
0/1000/2000/3000/4000 (a textbook diminishing-returns curve — signal 1), while two
individual evaluations inside that same run (steps 4,250 and 4,750) briefly ticked *up*
before the run's final step landed on a new best (signal 2: noise, not a trend). See
[`from_scratch/custom-gpt-6m/docs/HOW_MUCH_TRAINING_IS_ENOUGH.md`](../../from_scratch/custom-gpt-6m/docs/HOW_MUCH_TRAINING_IS_ENOUGH.md)
for this run's complete evaluation table and per-window arithmetic, and
[`from_scratch/custom-gpt-10m/docs/LLM_DEV_GUIDE.md`](../../from_scratch/custom-gpt-10m/docs/LLM_DEV_GUIDE.md)
for the same `best`/`latest` split applied to the `custom-gpt` project's checkpoint
layout.

## Deep-Dive: Why Best-Checkpoint Tracking Changes the Cost/Benefit of "Training Too Long"

Without separate best-checkpoint tracking, overshooting the optimal stopping point is a
real cost — the final checkpoint (whatever it happens to be) is what gets served, so
training well past the point of diminishing returns risks serving a model that has
started to overfit. With it, the downside of continuing past the optimal point shrinks to
"wasted wall-clock time," because the best-so-far weights are preserved regardless of how
much further the run continues. This is why the practical stopping rule above is framed
as a wall-clock efficiency question rather than a strict correctness one — the mechanism
that makes that framing safe is the `improved` check itself, not a property of the loss
curve.

## Try It Yourself

- Pull a real `train_eval_history.csv` from a run you have (or the one referenced above)
  and compute the per-1,000-step `val_loss` delta for each window — confirm for yourself
  that it shrinks over the course of the run, and estimate by eye roughly where it
  approaches the noise floor visible between adjacent evaluations.
- Find one evaluation point in that same history where `val_loss` ticked up from the
  previous one, and confirm the run's `best_val_loss` tracking correctly skipped over it
  rather than treating it as the new best.

## Common Misconceptions

- **"If `val_loss` isn't at its lowest possible value, training failed."** No — the
  question is whether *more* training is still worth the time, not whether the absolute
  minimum has been reached; a run can be stopped well short of its theoretical floor and
  still be the right call.
- **"One non-improving evaluation means training has plateaued."** As signal 2 explains,
  evaluation loss carries its own sampling noise; a single tick is not evidence of a
  trend, only a sustained run of them is.
- **"Lower validation loss always means better generated output."** Usually true early in
  training, but the two can decouple — signal 4 exists specifically because loss is a
  proxy, and proxies can stop tracking the real goal before they stop improving
  numerically.

## Practice Questions

1. A run's `val_loss` improves for 8 consecutive evaluations, then fails to improve for
   the next 3. Using the four-signal framework, what would you check next before deciding
   whether to stop?
2. Why does random-window batch sampling make "epoch" an approximation rather than an
   exact count, and why is it still a useful unit despite that?
3. Explain why best-checkpoint tracking changes what "training too long" actually costs
   you, compared to a setup that only ever saves the most recent checkpoint.

## Key Terms

- **Epoch**: one full pass through the training data; approximate, not exact, under
  random-window sampling.
- **Diminishing returns (loss curve)**: each additional block of training steps buys a
  smaller improvement than the block before it — expected, not a sign of a problem.
- **Best-checkpoint tracking**: saving a separate checkpoint only when an evaluation
  actually improves on the best seen so far, decoupling "what gets served" from "how long
  the run happened to continue."
