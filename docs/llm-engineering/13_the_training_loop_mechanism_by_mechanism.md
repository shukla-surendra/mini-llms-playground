# The Training Loop, Mechanism by Mechanism

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 2 — Pretraining: Building a
Model From Zero. Builds on [Chapter 3](03_how_neural_networks_learn.md)'s four-step loop
(forward → loss → backward → gradient descent) and [Chapter 4](04_hyperparameter_tuning.md)'s
hyperparameter vocabulary. Grounded primarily in
[`from_scratch/custom-gpt-6m/src/gpt/training/trainer.py`](../../from_scratch/custom-gpt-6m/src/gpt/training/trainer.py),
since it's the sibling project's simpler loop
([`custom-gpt-153m/tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py)) with a
few extra mechanisms layered on that a real training loop needs once model or batch size
grows past "fits trivially in memory."

## In Plain English

Chapter 3's four-step loop is the *what*. This chapter is the *how* — the specific
engineering mechanisms a real training loop adds on top of that core loop to run
correctly, resume safely, and use hardware efficiently: batching many examples together
efficiently, simulating a bigger batch than fits in memory at once, running parts of the
computation in lower-precision arithmetic to go faster, and trading extra compute for
memory when memory is the real constraint.

## The First-Principles Explanation

### The loop, one mechanism at a time

Every training step in
[`train.py`](../../from_scratch/custom-gpt-6m/src/gpt/training/trainer.py)'s `main()` does these things,
in this order:

1. **Sample a batch** — `get_batch` picks `batch_size` random starting positions in the
   token stream and slices out `context_length`-token windows, `x` (input) and `y` (the
   same window shifted by one token — the causal-LM target). Randomly sampled, not
   sequential, so the model doesn't see stories in a fixed order every pass.
2. **Forward pass under autocast** — `with torch.autocast(...)`: matmul-heavy ops run in
   a lower-precision dtype when `AMP=1` (see "Mixed precision" below); the loss itself
   (cross-entropy) still accumulates in fp32 for numerical stability.
3. **Backward pass, scaled** — `scaler.scale(loss / grad_accum_steps).backward()`.
   Dividing by `grad_accum_steps` before `.backward()` is what makes gradient
   accumulation work: each of the `grad_accum_steps` micro-batches contributes an
   equally-weighted fraction of the eventual averaged gradient, so the accumulated
   gradient after all of them ends up mathematically equivalent to computing it on one
   large batch of size `batch_size × grad_accum_steps` at once.
4. **Optimizer step, every `grad_accum_steps` micro-batches** — gradient clipping
   (`clip_grad_norm_`, caps the gradient's overall magnitude to prevent a single bad batch
   from taking a destructively large step), then `scaler.step(optimizer)` /
   `scaler.update()` (a no-op passthrough to plain `optimizer.step()` when the scaler is
   disabled), then `optimizer.zero_grad()` to clear gradients before the next
   accumulation cycle.
5. **Learning-rate schedule** — `lr_for_step` computes a fresh learning rate every step:
   linear warmup for the first ~2% of steps, then cosine decay to `MIN_LR` — see
   [Chapter 3's warmup/decay deep-dive](03_how_neural_networks_learn.md#deep-dive-what-the-learning-rate-schedule-is-actually-doing)
   for why warmup exists at all.

### Gradient accumulation: simulating a bigger batch than fits in memory

If `batch_size=8` is all that fits in memory but you want the gradient-noise
characteristics of `batch_size=32`, run 4 micro-batches of 8, summing (via the `/
grad_accum_steps` scaling above) their gradients before one optimizer step. The model
never sees a real batch of 32 examples processed simultaneously — it sees the *gradient
signal* of one, spread across 4 forward/backward passes. Slower wall-clock than a real
large batch (no parallelism gained across the accumulated micro-batches), but it makes an
otherwise memory-infeasible effective batch size trainable at all.

### Mixed precision: which numbers actually change, and why it's not free

Floating-point precision controls how many bits represent a number's fractional part
(mantissa) and its exponent range. `float32` (the default) uses 8 exponent bits and 23
mantissa bits. Autocast runs specific ops — the ones actually accuracy-tolerant and
matmul-heavy — in a narrower dtype while leaving others (reductions, the loss itself) in
fp32:

- **`float16`**: 5 exponent bits, 10 mantissa bits. Narrower exponent range than fp32
  means gradients *can* underflow to exactly zero during backward — this is why fp16
  autocast is always paired with `GradScaler`: it multiplies the loss by a large scale
  factor before backward (so small gradients don't round to zero), then unscales before
  the optimizer step.
- **`bfloat16`**: 8 exponent bits (same range as fp32 — this is the actual design intent,
  "fp32's range, less precision"), 7 mantissa bits. Doesn't underflow the way fp16 does,
  so no scaler is needed — at the cost of coarser precision within that range.

The *speedup* from either is hardware-dependent, not a property of the dtype alone — see
[`from_scratch/custom-gpt-6m/docs/EFFICIENT_TRAINING.md`](../../from_scratch/custom-gpt-6m/docs/EFFICIENT_TRAINING.md)
for real, measured numbers showing this concretely (a real speedup on the hardware that
has dedicated low-precision matmul units, near-zero benefit on hardware that doesn't).

### Gradient checkpointing: recompute instead of remember

Normally, every intermediate activation computed during the forward pass is kept in memory
because backward needs it to compute gradients. Gradient checkpointing instead discards
most activations right after the forward pass and **recomputes them from a saved
checkpoint** (each block's input) during backward — trading extra forward compute (one
extra forward pass per checkpointed block, per step) for a large memory reduction, since
only the checkpointed inputs, not every intermediate tensor, need to stay resident.

## Grounded in This Repo's Code

All three mechanisms above (gradient accumulation was already present; autocast/scaler and
gradient checkpointing are additions) are real, runnable flags in
[`train.py`](../../from_scratch/custom-gpt-6m/src/gpt/training/trainer.py):

```python
amp_dtype = {"cuda": torch.float16, "mps": torch.bfloat16, "cpu": torch.bfloat16}[device]
amp_enabled = use_amp and device != "cpu"
use_scaler = amp_enabled and device == "cuda"
scaler = torch.amp.GradScaler(enabled=use_scaler)
...
with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
    logits = model(xb)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
scaler.scale(loss / grad_accum_steps).backward()
```

and in [`model.py`](../../from_scratch/custom-gpt-6m/src/gpt/model.py)'s
`TinyStoriesGPT.forward`:

```python
if self.grad_checkpoint and self.training:
    h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
else:
    h = block(h)
```

Run any combination yourself: `ATTN_IMPL=sdpa AMP=1 GRAD_CHECKPOINT=1 make train` (see
[`EFFICIENT_TRAINING.md`](../../from_scratch/custom-gpt-6m/docs/EFFICIENT_TRAINING.md)
for what each combination actually measured on this project's own hardware).

## Deep-Dive: Why Gradient Checkpointing and Gradient Accumulation Solve Different Problems

It's easy to conflate these since both are "tricks for training bigger than your hardware
naively allows," but they address different bottlenecks. Gradient accumulation lets you
simulate a bigger **batch** than fits in memory — the model and its activations for one
micro-batch still have to fit. Gradient checkpointing lets you fit a bigger **model** (or
longer context, or larger batch) into the *same* memory budget by reducing how much of any
single forward pass's activations stay resident at once. A training setup that's memory-
constrained by model/activation size, not batch size, gains nothing from more accumulation
steps — it needs checkpointing (or a smaller model, or more memory) instead. Real large-
model training pipelines commonly use both simultaneously, because they solve genuinely
independent constraints.

## Try It Yourself

- Run `GRAD_ACCUM_STEPS=4 BATCH_SIZE=8 make train-fresh` for a short number of steps
  (`STEPS=200`) and compare the loss trajectory in `logs/train_eval_history.csv` against
  `BATCH_SIZE=32 GRAD_ACCUM_STEPS=1` — the two should produce a very similar loss curve
  (same effective batch size), confirming accumulation is mathematically equivalent to a
  real larger batch, just slower.
- Measure memory yourself: run the two-line `torch.mps.driver_allocated_memory()` /
  `torch.cuda.max_memory_allocated()` check (whichever matches your hardware) before and
  after a single forward+backward pass with `GRAD_CHECKPOINT=0` vs `GRAD_CHECKPOINT=1`, and
  compare against the real numbers already recorded in `EFFICIENT_TRAINING.md`.

## Common Misconceptions

- **"Gradient accumulation and a real larger batch produce identical wall-clock speed."**
  They produce a mathematically equivalent *gradient*, not identical speed — accumulation
  runs the micro-batches sequentially, gaining none of the parallelism a hardware-native
  larger batch would get.
- **"Mixed precision always gives roughly a 2x speedup."** Only on hardware with dedicated
  low-precision matmul units that actually accelerate the narrower dtype (e.g. CUDA tensor
  cores for fp16) — see the real measured MPS numbers in `EFFICIENT_TRAINING.md`, where the
  speedup is within noise.
- **"Gradient checkpointing saves memory during evaluation too."** It only affects the
  backward pass — evaluation (`model.eval()`, `torch.no_grad()`) never runs backward, so
  there's no activation-memory cost to save in the first place; checkpointing is inert
  during eval by design.

## Practice Questions

1. Why does gradient accumulation require dividing the loss by `grad_accum_steps` before
   calling `.backward()`, rather than dividing after?
2. `float16` autocast requires `GradScaler`; `bfloat16` autocast doesn't. Explain the
   exponent-range difference that makes this true.
3. A training run is memory-constrained by a single very long input sequence, not by batch
   size. Which of the three mechanisms in this chapter actually helps, and which doesn't —
   and why?

## Key Terms

- **Gradient accumulation**: summing gradients across several micro-batches before one
  optimizer step, simulating a larger effective batch size than fits in memory at once.
- **Autocast**: running specific, precision-tolerant ops in a lower-precision dtype while
  keeping numerically sensitive ops (reductions, loss accumulation) in fp32.
- **`GradScaler`**: multiplies the loss by a scale factor before `float16` backward to
  prevent small gradients from underflowing to zero, then unscales before the optimizer
  step — not needed for `bfloat16`, which doesn't have this underflow problem.
- **Gradient (activation) checkpointing**: discarding intermediate activations after the
  forward pass and recomputing them from a saved input during backward, trading extra
  compute for reduced memory.
