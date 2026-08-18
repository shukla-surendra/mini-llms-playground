# Technical specification — `custom-gpt-nano`

Companion to the [top-level README](../README.md) (which covers *how to run this
project*) and the code itself (which covers *the mechanism, line by line* — see the
README's "Reading order"). This doc is the single-page reference for *the exact
numbers*: what kind of architecture this is, its full spec, where every one of its
807,040 parameters lives, and how it compares to this workspace's other six
`from_scratch/` projects.

## What "Transformer" means, precisely

**An architecture, not an algorithm, a model, or a framework** — a structural blueprint
for arranging neural-network layers, first published in "Attention Is All You Need"
(2017). Four terms worth keeping distinct, since they get conflated constantly:

| Term | Answers the question | This project's answer |
|---|---|---|
| **Architecture** | How are the layers structured and connected? | Transformer (decoder-only) |
| **Model** | One specific trained instance of an architecture | `custom-gpt-nano` itself — this exact config, trained on `data/corpus.txt` |
| **Algorithm** | How does training actually update the weights? Not architecture-specific. | Backpropagation + AdamW gradient descent (`train.py`) |
| **Framework** | What software implements it? | PyTorch |

The Transformer's defining structural feature is **self-attention**: every token's
representation is updated by directly looking at every other allowed token in the
sequence in a single step (see `model.py`'s `CausalSelfAttention`), rather than only its
immediate neighbor the way older RNN-based architectures worked — this is both why it
captures long-range relationships better and why a whole sequence can be processed in
parallel instead of token-by-token. Three architecture *families* exist under that one
umbrella:

- **Encoder-only** (e.g. BERT) — bidirectional attention (every token can see every
  other token, including *later* ones), used for understanding/classification tasks,
  not generation.
- **Decoder-only** (e.g. the GPT family, and this project) — **causal** attention only
  (a token can see itself and everything *before* it, never after — see the causal mask
  in `model.py`), used for generation, one token at a time.
- **Encoder-decoder** (the original 2017 paper's design) — an encoder reads the full
  input, a decoder generates output attending back to it; the classic architecture for
  translation-style tasks.

This project, and every other `from_scratch/` project in this workspace, is
**decoder-only**. Deep dive: `../../docs/llm-engineering/06_nlp_architecture_landscape.md`
and `../../docs/llm-engineering/10_transformer_architecture.md`.

## Full specification

| | |
|---|---|
| Architecture family | Decoder-only Transformer (causal self-attention) |
| Parameters | **807,040** |
| Context length (`block_size`) | 64 tokens |
| Embedding size (`n_embd`) | 128 |
| Attention heads (`n_head`) | 4 (32 dimensions each) |
| Layers (`n_layer`) | 4 |
| Feed-forward expansion | 4x (128 → 512 → 128) |
| Nonlinearity | GELU |
| Normalization | LayerNorm, **pre-norm** placement (`x + sublayer(norm(x))`) |
| Position encoding | Learned absolute position embeddings (one vector per position 0..63) |
| Weight tying | Token embedding and output (`lm_head`) share one weight matrix |
| Tokenizer | Character-level, 43-symbol vocabulary, built from `data/corpus.txt` |
| Training objective | Causal language modeling (next-token prediction), cross-entropy loss |
| Optimizer | AdamW, `lr=3e-3`, constant (no warmup/decay schedule) |
| Attention implementation | Hand-written (explicit Q/K/V matmuls) — not the fused `F.scaled_dot_product_attention` |

## Where the 807,040 parameters live

Exact counts, computed from the config above — reproduce with `make config`:

```
token_embedding (43 x 128)                      5,504    ( 0.7%)
position_embedding (64 x 128)                    8,192    ( 1.0%)
4 x transformer_block                          793,088    (98.3%)
  each block:
    ln1 (LayerNorm)                                256
    attn.qkv_proj  (128 -> 384, +bias)          49,536
    attn.out_proj  (128 -> 128, +bias)          16,512
    ln2 (LayerNorm)                                256
    mlp.fc_in      (128 -> 512, +bias)          66,048
    mlp.fc_out     (512 -> 128, +bias)          65,664
    ---------------------------------------------------
    per-block total                            198,272
final LayerNorm                                     256
lm_head                                               0    (reused token_embedding.weight)
=====================================================
total                                          807,040
```

Two things worth noticing in this table:

1. **The embedding tables are ~2% of the model, combined.** That's the entire reason
   this project uses a 43-symbol character-level tokenizer instead of GPT-2's
   50,257-word one — see `../README.md`'s "one design choice that matters most"
   section and `src/nanogpt/tokenizer.py`'s docstring for the full reasoning.
2. **The MLP (feed-forward) sublayer, not attention, is the bigger half of every
   block** — 131,712 params (66,048 + 65,664) vs. attention's 66,048. This holds for
   virtually every Transformer at any scale: the 4x-expansion feed-forward layers
   dominate the parameter count, while attention dominates the *compute* cost at long
   sequence lengths (see Computational complexity below) — parameter count and compute
   cost scale with different things.

## Shape trace through one forward pass

Concrete shapes for a training batch (`batch_size=32`, `block_size=64`, `n_embd=128`,
`n_head=4`, `head_size=32`), matching the code in `src/nanogpt/model.py`:

```
input token ids                                  (32, 64)
  -> token_emb + pos_emb                         (32, 64, 128)
  -> Block x4:
       ln1                                       (32, 64, 128)
       qkv_proj                                  (32, 64, 384)  -> split into
         q, k, v                                 (32, 64, 128) each
       reshape to heads                          (32, 4, 64, 32) each
       attention scores  q @ k^T / sqrt(32)      (32, 4, 64, 64)
       causal mask + softmax                     (32, 4, 64, 64)
       weighted sum  scores @ v                  (32, 4, 64, 32)
       merge heads back                          (32, 64, 128)
       out_proj                                  (32, 64, 128)
       residual add                              (32, 64, 128)
       ln2 -> mlp (128->512->128)                (32, 64, 128)
       residual add                              (32, 64, 128)
  -> final LayerNorm                             (32, 64, 128)
  -> lm_head                                     (32, 64, 43)   <- logits over the vocabulary
```

The last dimension only ever changes at the very start (43 → 128, entering the model)
and the very end (128 → 43, `lm_head`) — every Transformer block in between preserves
`n_embd` exactly, which is what makes stacking arbitrarily many of them (just changing
`n_layer`) structurally trivial: each block's output is a valid input to the next.

## Computational complexity

- **Attention**: the `q @ k^T` step above produces a `(T, T)` matrix per head — cost
  scales as **O(T² · C)** where `T` = sequence length, `C` = `n_embd`. This is why
  context length (`block_size`) is expensive to grow: doubling it quadruples attention's
  compute cost, not just doubles it. At `T=64` this is trivial; at the 1024-2048 context
  lengths this workspace's larger sibling projects use, it's the dominant cost, which is
  exactly why those projects use the fused `F.scaled_dot_product_attention` kernel
  instead of this hand-written version — see
  `../../docs/llm-engineering/25_efficient_attention_flash_and_sdpa.md`.
- **MLP**: cost scales as **O(T · C²)** — linear in sequence length, quadratic in
  embedding size. At this project's tiny `C=128`, negligible; at hundreds-of-billions
  scale, this is where most training FLOPs actually go.

## How this compares to the other six `from_scratch/` projects

See [`../models.md`](../models.md) for the full picture; the short version:

| | `custom-gpt-nano` | "classic" family (6m/10m/50m/153m) | "modern" family (200m/350m) |
|---|---|---|---|
| Position encoding | Learned absolute embeddings | Learned absolute embeddings | RoPE (rotary) |
| Normalization | LayerNorm | LayerNorm | RMSNorm |
| Feed-forward | Plain GELU-MLP | Plain GELU-MLP | SwiGLU (gated) |
| Attention impl | Hand-written matmuls | `F.scaled_dot_product_attention` | `F.scaled_dot_product_attention` |
| Tokenizer | Character-level (43) | GPT-2 BPE (50,257) or custom (4,096) | Custom-trained BPE (32,768) |

`custom-gpt-nano` and the "classic" family share the *same* Transformer sub-variant
(learned position embeddings, LayerNorm, plain GELU-MLP) — the only real differences are
tokenizer and whether attention is hand-written or fused. The "modern" family is a
genuinely different, newer sub-variant (RoPE/RMSNorm/SwiGLU) used by Llama-family models
and most current open-weight LLMs; `../custom-gpt-200m/` and `../custom-gpt-350m/` are
the place to see that variant in this workspace. Deep dive on both position-encoding
choices and the RMSNorm/SwiGLU pair:
`../../docs/llm-engineering/11_positional_encoding_variants_rope_and_beyond.md` and
`../../docs/llm-engineering/35_normalization_and_mlp_variants_rmsnorm_and_swiglu.md`.
