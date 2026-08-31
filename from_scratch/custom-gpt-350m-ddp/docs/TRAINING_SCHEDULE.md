# Training Step Count & the Learning-Rate Schedule

What the **learning rate** is, and the full mechanism of the warmup/cosine-decay
schedule, is covered in
[Chapter 3 — How Neural Networks Learn](../../../docs/llm-engineering/03_how_neural_networks_learn.md#deep-dive-what-the-learning-rate-schedule-is-actually-doing)
and grounded in this project's training loop in
[Chapter 13 — The Training Loop, Mechanism by Mechanism](../../../docs/llm-engineering/13_the_training_loop_mechanism_by_mechanism.md).
This doc only covers what's specific to *this* project: what `steps` actually counts in
`src/gpt/training/trainer.py`, how that number drives the LR schedule, and — the question
that actually matters day to day — how to tell whether a longer run is still worth it.

## What is a learning rate, in one paragraph

Training repeatedly nudges every model weight in the direction that reduces the loss
(gradient descent — [Chapter 3](../../../docs/llm-engineering/03_how_neural_networks_learn.md)).
The **learning rate** is the size of that nudge: `new_weight = old_weight - (learning_rate
× gradient)`. Too large, and updates overshoot and training destabilizes; too small, and
training crawls and wastes compute. It's the single most impactful hyperparameter in the
whole run ([Chapter 4](../../../docs/llm-engineering/04_hyperparameter_tuning.md)).

## What `steps` means in this codebase

A **step is one micro-batch forward/backward pass** (`batch_size=1`,
`context_length` tokens) — it is *not* one optimizer update.
`optimizer.step()` only fires every `grad_accum_steps` steps
(`trainer.py`, the `is_accum_boundary` check). So for the default `10m` config:

```
TrainConfig.steps = 1_000_000        # micro-batches (forward/backward passes)
grad_accum_steps  = 32
→ real optimizer updates  = 1,000,000 / 32        = 31,250
→ tokens processed        = 1,000,000 × 512       = 512,000,000
```

`processed_tokens` and `est_epoch` in `logs/train_eval_history_<label>.csv` are derived
directly from this: `est_epoch = processed_tokens / len(train_tokens)`. Divide your
corpus's token count (printed at the start of every `make train` run) into the total token
budget above to get how many epochs the configured `steps` value actually represents.

## The LR schedule is tied to `TrainConfig.steps`, not to wall-clock or epochs

`lr_for_step` (`trainer.py`):

```python
warmup_steps = max(200, int(train_cfg.steps * 0.02))   # first 2% of steps: linear warmup
decay_steps  = train_cfg.steps - warmup_steps           # the rest: cosine decay to min_lr
```

The practical consequence: **changing `steps` reshapes the entire schedule**, not just
where training stops. Doubling `steps` doesn't just "train longer at the same LR curve" —
it stretches the cosine decay over twice as many steps, so the LR at any given step number
is *higher* than it would have been under the shorter schedule. This is why you can't judge
"has training converged" purely from the loss number — you have to know where you are on
*this run's specific* cosine curve.

## Is a longer run still worth it? A three-question framework

Don't judge from a raw loss plateau alone — check these three, in order:

1. **Where are you on the LR curve?** If current LR is still close to peak (early in the
   decay), most of the schedule's fine-tuning phase — the part of a cosine schedule that
   reliably squeezes out the last real gains as LR approaches `min_lr` — hasn't happened
   yet. A stall here is not evidence of convergence.
2. **How many epochs has the budget spent?** A small model (few params relative to corpus
   size) is usually *data-underfit*, not data-saturated — repeated epochs still teach it
   something. Diminishing returns show up as epoch count climbs, not epoch 0.6.
3. **Is `best_test_loss` actually stuck, or just its noisy single-sample estimate?** Judge
   by the tracked running minimum (`best_test_loss` in the CSV, which also gates
   `checkpoints/<label>/best.pt`), not the raw per-eval `test_loss` column — with a small
   `eval_batches` and `batch_size=1`, that column is sampled from very few tokens per eval
   and swings well beyond the real underlying trend. If you want a less noisy read on the
   current state without touching the training budget, raise `TrainConfig.eval_batches`.

Only once (1) LR has meaningfully left the near-peak region *and* (2) epoch count is no
longer small *and* (3) `best_test_loss` (not the noisy per-eval sample) has genuinely
stopped improving for a sustained stretch, is a plateau real rather than an artifact of
which phase of the schedule you're reading it from.

## Reading a live run yourself

```bash
# best_test_loss progression over the run (every ~400 eval rows)
awk -F, 'NR==1{next} {print $2, $8}' logs/train_eval_history_10m.csv | awk 'NR%400==1'

# step of the most recent genuine improvement (col 9 "improved" == 1)
awk -F, 'NR==1{next} $9==1 {print $2, $6, $8}' logs/train_eval_history_10m.csv | tail -10

# current progress: step, LR, best_test_loss, and % of the configured step budget
tail -1 logs/train_eval_history_10m.csv
```

Cross-check against `TrainConfig.steps` and `grad_accum_steps` in `src/gpt/config.py` for
the current total-budget math (optimizer updates, total tokens, implied epochs).

## Key terms

- **Learning rate**: the size of each gradient-descent weight update.
- **Warmup**: the first portion of training where LR ramps up from ~0 to its peak, so
  large early (noisy, randomly-initialized-weight) gradients don't destabilize training.
- **Cosine decay**: the LR schedule shape used after warmup — a smooth decrease from peak
  `lr` to `min_lr` following a cosine curve, so updates get finer as training progresses.
- **Step** (this codebase): one micro-batch forward/backward pass — distinct from one
  optimizer update, which happens every `grad_accum_steps` steps.
- **Epoch**: one full pass over the training corpus; `est_epoch` in the eval log is an
  estimate derived from tokens processed so far, not a discrete counted event.
