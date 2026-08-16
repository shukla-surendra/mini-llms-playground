# Architecture: Why This Size, Why This Shape

## The model, in one table

| Hyperparameter | Value | Compare to `custom-gpt-153m` |
|---|---|---|
| `vocab_size` | 4,096 (custom-trained, see [`DATASET_AND_TOKENIZER.md`](DATASET_AND_TOKENIZER.md)) | 50,257 (GPT-2's) |
| `context_length` | 256 | 1,024 |
| `embed_size` | 256 | 768 |
| `num_heads` | 8 | 12 |
| `num_layers` | 6 | 16 |
| `dropout` | 0.1 | 0.1 |
| **Total parameters** | **~5.85M** | **~152.8M** |

This is the exact same architecture *family* as
[`../../custom-gpt-153m/tiny_llm.py`](../../custom-gpt-153m/tiny_llm.py) — decoder-only,
pre-norm residual blocks, weight-tied output head — just every dimension scaled down. If
any term here (embedding, attention, causal mask, MLP, weight tying) is unfamiliar, the
full first-principles explanation is
[`../../../docs/llm-engineering/10_transformer_architecture.md`](../../../docs/llm-engineering/10_transformer_architecture.md) —
this doc only covers the *sizing* decisions, not the mechanism itself.

## Why these specific sizes, not smaller or larger

- **`vocab_size=4096`, not 50,257** — covered fully in
  [`DATASET_AND_TOKENIZER.md`](DATASET_AND_TOKENIZER.md); the short version is that a
  smaller, corpus-specific vocabulary keeps the embedding table from dominating a small
  model's entire parameter budget.
- **`context_length=256`, not 1,024** — TinyStories stories are short (typically a few
  hundred words); a long context window this model would rarely use is wasted capacity
  (recall from
  [`../../../docs/llm-engineering/01_neurons_layers_and_networks.md`](../../../docs/llm-engineering/01_neurons_layers_and_networks.md),
  every dimension here has a real parameter/compute cost) and also makes training slower,
  since attention's compute cost grows with sequence length.
- **`embed_size=256`, `num_layers=6`** — chosen to land in the low-single-digit-millions
  of parameters, matching the TinyStories paper's own finding that models in roughly this
  range (they tested 1M-33M) already produce coherent short stories — going meaningfully
  larger wouldn't be *wrong*, but it works against the explicit goal here (fast to train
  on a laptop, not maximal quality).

## Parameter count, computed exactly (same method as `custom-gpt-153m`'s README)

```
Token embedding:     vocab_size × embed_size        = 4,096 × 256   = 1,048,576
Positional embedding: context_length × embed_size    =   256 × 256   =    65,536

Per Transformer block (embed_size=256, num_heads=8):
  Attention:
    in_proj_weight:  3 × E × E = 3 × 256 × 256                       =   196,608
    in_proj_bias:    3 × E                                            =       768
    out_proj_weight: E × E                                            =    65,536
    out_proj_bias:   E                                                 =       256
  MLP:
    E × 4E + 4E  +  4E × E + E                                       =   525,568
  LayerNorms (×2, each 2E):                                          =     1,024
  Per-block total:                                                    =   789,760

All 6 blocks:        789,760 × 6                                     = 4,738,560
Final LayerNorm:      2 × embed_size                                  =       512

TOTAL (weight tying means lm_head adds nothing extra):
  1,048,576 + 65,536 + 4,738,560 + 512 = 5,853,184 (~5.85M parameters)
```

Confirmed against the real model at runtime: `model.num_parameters()` in
[`../src/gpt/model.py`](../src/gpt/model.py) reports this exact figure — see
[`TRAINING.md`](TRAINING.md) for the actual logged output from a real training run.

## What was deliberately left out, and why

- **No RoPE, no RMSNorm, no SwiGLU, no GQA** — the modern refinements named in
  [`../../../docs/llm-engineering/10_transformer_architecture.md`'s deep-dive](../../../docs/llm-engineering/10_transformer_architecture.md#deep-dive-why-this-specific-set-of-design-choices-and-what-modern-models-change)
  aren't used here, for the same reason `custom-gpt-153m` doesn't use them: at this
  scale, with this short a context length, learned positional embeddings + GELU-MLP +
  LayerNorm are simpler to understand and debug, and aren't the bottleneck holding this
  project back from its actual goal.
- **Efficient/fused attention — implemented, not left out.** The original version of this
  doc listed the standard-library `nn.MultiheadAttention` path as the only option; that's
  now `ATTN_IMPL=naive`, and `ATTN_IMPL=sdpa`
  (`F.scaled_dot_product_attention`, fused/flash-eligible kernels) is a real,
  benchmarked alternative — see
  [`EFFICIENT_TRAINING.md`](EFFICIENT_TRAINING.md) for the math and real measured
  numbers on this machine.
- **No KV cache in the training/inference code** — worth naming explicitly since it's a
  real production concern (`platform-lab/fundamentals/gpu_infrastructure/`'s serving
  chapters cover it in depth): at `context_length=256` and this model size, generation is
  already fast enough on a laptop that KV caching's complexity isn't worth adding for
  this project's goal. A natural "next step" exercise, not a current gap.
