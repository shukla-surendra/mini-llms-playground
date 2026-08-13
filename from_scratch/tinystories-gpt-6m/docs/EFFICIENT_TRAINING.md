# Efficient Training: Fused Attention, Mixed Precision, Gradient Checkpointing

## The mechanism, if you need a refresher

Three independent efficiency mechanisms, each a toggle in [`../train.py`](../train.py) and
[`../model.py`](../model.py), each trading a different resource for a different one. If
the concepts themselves (why attention has a quadratic cost, what floating-point precision
actually controls, what "recompute instead of store" means) are unfamiliar, the
first-principles treatment is
[`../../../docs/llm-engineering/25_efficient_attention_flash_and_sdpa.md`](../../../docs/llm-engineering/25_efficient_attention_flash_and_sdpa.md)
and
[`../../../docs/llm-engineering/13_the_training_loop_mechanism_by_mechanism.md`](../../../docs/llm-engineering/13_the_training_loop_mechanism_by_mechanism.md) —
this doc only covers what's specific to *this project's* implementation and real,
observed numbers on this MacBook.

## The three flags

| Env var | Values | What it controls | Reasoning |
|---|---|---|---|
| `ATTN_IMPL` | `naive` (default) / `sdpa` | Which attention kernel `CausalSelfAttention` in [`../model.py`](../model.py) uses | `naive` = the original explicit `nn.MultiheadAttention` + materialized `seq_len × seq_len` mask; `sdpa` = `F.scaled_dot_product_attention`, which never materializes the full mask matrix and is eligible for PyTorch's fused/flash-attention kernels |
| `AMP` | `0` (default) / `1` | Whether the forward pass + loss run under `torch.autocast` | See "What AMP actually does on this machine" below — the mechanism is real, but its benefit is device-dependent |
| `GRAD_CHECKPOINT` | `0` (default) / `1` | Whether each `GPTBlock`'s activations are recomputed during backward instead of kept in memory | Classic compute-for-memory trade — see real numbers below |

## The math: why SDPA can be faster than the naive path

Both compute the exact same thing — `softmax(QKᵀ/√d_k)V` — so **the output is numerically
identical** (up to floating-point rounding); this is a kernel-implementation change, not an
architecture change. The naive path in `nn.MultiheadAttention` builds the full
`seq_len × seq_len` score matrix, applies the additive `-inf` mask, then softmaxes and
multiplies — every one of those intermediate tensors is materialized in memory.
`F.scaled_dot_product_attention` with `is_causal=True` can instead use fused kernels
(flash-attention-style on CUDA; a fused Metal kernel on MPS) that compute the same result
block-by-block without ever writing the full `seq_len × seq_len` matrix to memory — the
quadratic-in-`seq_len` *memory* cost drops even though the quadratic *compute* cost is
unavoidable (that's inherent to attention itself, not a kernel choice).

## What AMP actually does on this machine

`torch.autocast` runs whitelisted ops (matmuls, mainly) in a lower-precision dtype while
keeping the accumulation/reduction ops in fp32 — but which lower-precision dtype, and
whether it actually speeds anything up, is hardware-dependent:

- **CUDA**: `float16` autocast + `GradScaler` (fp16 has a narrow exponent range, so
  gradients can underflow to zero without the scaler's loss-scaling trick) — this is where
  AMP gives its classic ~2x speedup, using GPU tensor cores that specifically accelerate
  fp16 matmuls.
- **MPS (this machine)**: `bfloat16` autocast, no scaler needed (bf16 has fp32's exponent
  range, just less mantissa precision, so it doesn't underflow the way fp16 does) — but
  Apple Silicon's matmul units don't get the same dramatic speedup from bf16 that CUDA
  tensor cores get from fp16, so the benefit here is smaller and, at this model's tiny
  size, within noise (see benchmark below).
- **CPU**: no meaningful effect — `train.py` detects this and prints a warning instead of
  silently doing nothing (see `[amp] AMP=1 requested but device=cpu` in the source).

This is why `train.py` selects `amp_dtype` per-device rather than hardcoding `float16`
everywhere — using `float16` on MPS/CPU would either error or silently degrade quality for
no speed benefit.

## Real benchmark, actually run on this project's own MacBook (MPS)

`STEPS=100`, default `BATCH_SIZE=32`/`CONTEXT_LENGTH=256`/5.85M-parameter model, no eval
passes (`EVAL_INTERVAL` set above `STEPS` so the 100 steps are pure training throughput):

| Config | Steady-state rate | 100-step wall time |
|---|---|---|
| `ATTN_IMPL=naive` (baseline) | 4.79 step/s | 22.2s |
| `ATTN_IMPL=sdpa` | 5.18 step/s | 20.8s |
| `ATTN_IMPL=sdpa AMP=1` | 4.85 step/s | 22.2s |
| `ATTN_IMPL=sdpa AMP=1 GRAD_CHECKPOINT=1` | 3.65 step/s | 28.8s |

**Reading these numbers honestly**: SDPA gives a real but modest ~8% throughput
improvement at this scale — the win would be larger at a longer `context_length`, since
the naive path's `seq_len × seq_len` mask materialization gets proportionally worse as
`seq_len` grows, and this project's `context_length=256` is short. AMP is a wash here —
consistent with the "What AMP actually does" section above, this is exactly what's
expected on MPS at this model size, not a bug. Gradient checkpointing is **slower**, not
faster — that's the trade working as designed (see memory numbers below), and it's the
one mechanism of the three that should be judged on memory, not speed.

Separate memory measurement (`torch.mps.driver_allocated_memory()` immediately after one
forward+backward pass, `batch_size=64`/`context_length=256`, same model, isolated
measurement so it's not confounded by `train.py`'s optimizer/eval state):

| Config | Driver-allocated memory after backward |
|---|---|
| `GRAD_CHECKPOINT=0` | 6,806 MB |
| `GRAD_CHECKPOINT=1` | 2,511 MB |

A **63% reduction** — this is the actual mechanism working correctly: instead of keeping
every `GPTBlock`'s intermediate activations alive for backward, `torch.utils.checkpoint`
discards them after the forward pass and recomputes them from each block's saved input
during backward, at the cost of the ~24% extra wall-clock seen in the throughput table
above (one extra forward pass per block, per training step). At this model's tiny size the
memory saved isn't needed — the value of this flag is real at model/batch sizes where
memory, not compute, is the binding constraint, which is exactly why it's the standard
technique for training much larger models than this one on the same hardware.

## What was deliberately left out, and why

- **True flash-attention (the CUDA-specific fused kernel from the original paper), not
  just SDPA's fallback path** — `F.scaled_dot_product_attention` automatically selects a
  backend kernel per-device; on CUDA with a compatible GPU it *can* dispatch to a real
  flash-attention kernel, but this project only has MPS/CPU available, so what's actually
  exercised here is MPS's fused kernel, not flash-attention proper. The API usage and the
  memory-avoidance mechanism are the same either way — this project doesn't claim to have
  benchmarked flash-attention specifically, only SDPA.
- **`torch.compile`** — a real, separate speedup mechanism (kernel fusion via graph
  compilation) that's out of scope here to keep this comparison to exactly the three flags
  named above; a natural next benchmark, not a current gap.

## The gotcha: `GRAD_CHECKPOINT=1` needs `model.training=True` to do anything

`TinyStoriesGPT.forward` only checkpoints blocks when `self.grad_checkpoint and
self.training` are both true (see [`../model.py`](../model.py)) — during `estimate_loss`'s
`model.eval()` phase, checkpointing is silently skipped (correctly — there's no backward
pass during eval, so there's nothing to save memory on, and checkpointing would just add
pure overhead). If you're benchmarking checkpointing's effect and only look at eval-time
memory, you'll see no difference and wrongly conclude the flag isn't working — the effect
is real, but only during the training step's backward pass.
