# Efficient Attention: Flash Attention and SDPA

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 2B — Training at Scale
(appended after the original numbered catalog — see [Chapter 0](00_roadmap.md)'s reading
order note for why this and [Chapter 26](26_distributed_training_ddp_and_fsdp.md) sit here
despite the higher numbers). Builds directly on
[Chapter 10](10_transformer_architecture.md)'s attention mechanism — this chapter assumes
`Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` is already understood and covers *how it's computed
efficiently*, not what it computes.

## In Plain English

The naive way to implement attention builds a full table of "how much should every
position attend to every other position" in memory, then processes that whole table at
once. That table's size grows with the *square* of the sequence length — double the input
length, and the table is 4x bigger. Flash attention (and PyTorch's more general
`scaled_dot_product_attention` API) computes the exact same mathematical result without
ever building that full table — it processes small blocks at a time and combines the
results incrementally, so the memory cost stops growing quadratically even though the
computation itself still does.

## The First-Principles Explanation

### Why naive attention's memory cost is quadratic

For a sequence of length `L`, `QKᵀ` produces an `L × L` matrix of raw attention scores —
every position's relevance to every other position. That matrix has to exist somewhere in
memory before softmax and the multiply by `V` can happen. At `L=256` that's 65,536 entries
per attention head — manageable. At `L=8192` (a real production context length) that's
67 million entries *per head, per layer, per item in the batch* — this is what makes naive
attention's memory cost the practical bottleneck on long sequences, not the FLOPs.

### The flash-attention idea: tiling and online softmax

Flash attention (Dao et al., 2022) restructures the *same* computation to avoid ever
materializing the full `L × L` matrix:

1. Split `Q`, `K`, `V` into small blocks that fit in the GPU's fast on-chip SRAM (much
   smaller than the GPU's main memory, but far faster to read/write).
2. Process block-pairs one at a time: compute a *partial* attention score matrix for just
   this block pair, run a numerically-stable incremental ("online") softmax update that
   combines this block's contribution with the running result so far, without needing the
   full row of scores at once.
3. After all block pairs are processed, the running result *is* the exact same output
   naive attention would have produced — this is an exact algorithm, not an approximation.

The net effect: memory cost drops from `O(L²)` to roughly `O(L)`, and because SRAM
reads/writes are far faster than main-memory reads/writes, it's often faster in wall-clock
too, even though it does slightly more *total* arithmetic (recomputing some values during
the backward pass rather than storing them, a checkpointing-flavored trade — see
[Chapter 13](13_the_training_loop_mechanism_by_mechanism.md)'s gradient-checkpointing
section for the same underlying idea applied to a different part of the network).

### Where `F.scaled_dot_product_attention` (SDPA) fits in

Flash attention is one specific fused-kernel implementation. PyTorch's
`torch.nn.functional.scaled_dot_product_attention` is a higher-level API that computes the
same math and automatically dispatches to whichever backend kernel is available and fastest
on the current hardware — a real flash-attention kernel on a compatible CUDA GPU, a
different fused kernel on Apple Silicon (MPS), or a memory-efficient fallback elsewhere.
The API is the same either way; what runs underneath depends on hardware.

## Grounded in This Repo's Code

[`from_scratch/tinystories-gpt-6m/model.py`](../../from_scratch/tinystories-gpt-6m/model.py)'s
`CausalSelfAttention` implements both the naive path and the SDPA path side by side, so the
same class can be run either way via `attn_impl`:

```python
out = F.scaled_dot_product_attention(
    q, k, v,
    dropout_p=self.attn_dropout_p if self.training else 0.0,
    is_causal=self.causal,
)
```

`is_causal=self.causal` is doing the mask's job without ever building the explicit
`torch.triu(..., float("-inf"))` matrix the naive path constructs — the causal constraint
is applied *inside* the fused kernel, block by block, rather than as a separate
materialized tensor added to the scores beforehand. See
[`docs/EFFICIENT_TRAINING.md`](../../from_scratch/tinystories-gpt-6m/docs/EFFICIENT_TRAINING.md)
for the real, measured throughput difference this makes on this project's own hardware —
and for why the difference is modest at this project's short `context_length=256`
specifically (the quadratic-memory problem this chapter describes gets *worse* at longer
sequences than this project uses).

## Deep-Dive: Why This Is an Exact Algorithm, Not an Approximation

A common misreading is that flash attention "approximates" attention for speed, the way
some efficiency tricks trade accuracy for performance. It doesn't — the online-softmax
technique it uses is mathematically exact: standard softmax normalizes by a sum computed
over the *whole* row, but that sum can be built up incrementally (processing one block,
updating a running max and running sum, then correcting previously-computed partial results
if a new, larger max is found) and arrive at exactly the same final value as computing the
whole row at once. This is the same numerical-stability trick behind the ordinary
`softmax(x - max(x))` shift (subtracting the max before exponentiating to avoid overflow),
just applied incrementally across blocks instead of once across a whole row.

## Try It Yourself

- In this repo, run the same short training session with `ATTN_IMPL=naive` and then
  `ATTN_IMPL=sdpa` (`STEPS=100` is enough to see steady-state throughput) and compare
  `step/s` in the console output — a direct, hands-on measurement of the mechanism this
  chapter describes, not a hypothetical.
- Work out the naive attention matrix's memory footprint by hand for `context_length=256`
  vs. `context_length=2048` (this project's vs. a more realistic production context
  length), `num_heads=8`, `float32`, one layer, batch size 1 — confirm for yourself that
  the growth is quadratic, not linear.

## Common Misconceptions

- **"Flash attention changes what attention computes."** It doesn't — same
  `softmax(QKᵀ/√d_k)V` result, different memory-access pattern to get there. This is a
  systems-level optimization, not an architectural or algorithmic change to the model.
- **"SDPA always means flash attention is running."** SDPA is a dispatcher — it picks
  whichever fused kernel is available for the current hardware and input shapes. On
  hardware without a compatible flash-attention kernel (including this project's own
  Apple Silicon MPS setup), it dispatches to a different fused kernel instead — still
  faster/leaner than the naive materialized-mask path, but not literally "flash attention."
- **"The quadratic cost this chapter describes is about compute (FLOPs)."** The *compute*
  cost of attention is already quadratic in sequence length regardless of implementation —
  that part is unavoidable. What flash attention/SDPA fix is the quadratic *memory* cost of
  materializing the full score matrix, which is avoidable.

## Practice Questions

1. Explain, in your own words, why online softmax lets flash attention avoid materializing
   the full `L × L` score matrix while still producing an exact result.
2. If a naive attention implementation's peak memory at `context_length=256` is `X`, what
   would you predict it to be at `context_length=1024`, and why?
3. Why can the same `F.scaled_dot_product_attention` call produce different underlying
   kernel behavior on two different machines, even though the Python code is identical?

## Key Terms

- **Quadratic attention memory cost**: the `L × L` score matrix naive attention builds
  grows with the square of sequence length `L`.
- **Online (incremental) softmax**: computing softmax's normalization incrementally across
  blocks of a row rather than requiring the whole row in memory at once — the core trick
  making flash attention exact rather than approximate.
- **SDPA (`scaled_dot_product_attention`)**: PyTorch's hardware-dispatching attention API —
  same math, kernel chosen automatically per device.
