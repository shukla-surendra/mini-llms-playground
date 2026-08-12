# The Transformer Architecture, Line by Line

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 1 — Foundations. Builds on
[Chapter 7](07_history_how_we_got_here.md) (why this architecture exists),
[Chapter 8](08_what_is_a_language_model.md) (what it's trained to do), and
[Chapter 9](09_tokenization.md) (what actually goes in). This is the longest chapter in
Part 1 because it's the one that matters most — every later chapter, in every part of
this curriculum, assumes this architecture as known.

## In Plain English

A decoder-only Transformer, like the one in this repo, is a stack of identical
processing blocks. Each block does two things to the text it's looking at: first, it lets
every word look back at every earlier word and decide how much attention to pay to each
one (self-attention); second, it processes that result through a small standalone neural
network (the MLP). Stack enough of these blocks, and the model builds up an increasingly
rich understanding of the text — from "what words are these" in early layers toward
"what's actually being said" in later ones. `tiny_llm.py`'s model has 16 of these blocks
stacked on top of each other.

## The First-Principles Explanation

### The full pipeline, top to bottom

```
Input: token IDs, e.g. [15496, 11, 995, 0]  (from Chapter 9's tokenizer)
   │
Token Embedding    — look up a learned vector for each token ID
   +
Positional Embedding — look up a learned vector for each POSITION (0, 1, 2, 3...)
   │
   ▼  (these two are added together, element-wise)
Dropout
   │
   ▼
GPTBlock × N       — N identical blocks, each containing:
   ├── LayerNorm
   ├── Causal Self-Attention
   ├── (residual connection: add the block's input back in)
   ├── LayerNorm
   ├── MLP (feed-forward network)
   └── (residual connection: add again)
   │
   ▼
Final LayerNorm
   │
   ▼
LM Head (Linear layer)  — project back to vocabulary size
   │
   ▼
Output: logits, one score per vocabulary token, for EVERY input position
```

Every one of these pieces exists to solve a specific problem — walking through each,
grounded directly in `tiny_llm.py`'s `TinyGPT` class (line 282).

### Token embeddings: from arbitrary integers to meaningful vectors

A token ID like `15496` is an arbitrary index — the model can't do meaningful arithmetic
on the raw integer (there's no reason token 15497 should be "similar" to 15496). The
**embedding layer** is a lookup table: one learned vector (length `embed_size`, 768 in
this repo) per vocabulary entry. During training, these vectors get adjusted so that
tokens used in similar contexts end up with similar vectors — this is what people mean by
embeddings capturing "semantic meaning": it's an emergent property of training, not
something explicitly programmed.

```python
self.token_emb = nn.Embedding(vocab_size, embed_size)   # TinyGPT.__init__, line 286
```

### Positional embeddings: telling the model *where* each token is

Self-attention (below) treats its input as a **set**, not a sequence — without extra
information, it has no inherent notion of "this token comes before that one." Positional
embeddings fix this: a second lookup table, indexed by *position* (0, 1, 2, ...) rather
than token identity, added element-wise to the token embedding.

```python
self.pos_emb = nn.Embedding(context_length, embed_size)   # line 287
...
pos = torch.arange(seq_len, device=x.device)
h = self.token_emb(x) + self.pos_emb(pos)                 # forward(), line 311
```

This specific approach — a learned embedding table for positions — is one of several
valid options (others include the fixed sinusoidal encoding from the original Transformer
paper, and rotary position embeddings/RoPE used in many modern models); this repo uses
the simplest, most directly interpretable version.

### Self-attention: the mechanism that lets tokens look at each other

This is the core of the architecture, and the direct implementation of the idea named in
[Chapter 7](07_history_how_we_got_here.md#generation-4-attention-is-the-architecture-transformer-2017).
For each token, self-attention computes three vectors from its embedding — a **Query**
(what am I looking for), a **Key** (what do I contain, for others to match against), and
a **Value** (what do I actually offer if attended to) — and uses them to decide how much
every *other* token should contribute to this token's updated representation:

```
attention_score(i, j) = Query_i · Key_j    (how relevant is token j to token i)
attention_weight = softmax(attention_scores, scaled)   (normalize into a distribution)
output_i = sum over j of (attention_weight(i,j) × Value_j)
```

In `tiny_llm.py`, this entire mechanism — including the Query/Key/Value projections — is
handled by a single, standard PyTorch layer:

```python
self.attn = nn.MultiheadAttention(
    embed_dim=embed_size, num_heads=num_heads, dropout=dropout, batch_first=True,
)   # CausalSelfAttention.__init__, line 230
```

**Multi-head** means this Query/Key/Value process runs `num_heads` (12, here) times in
parallel, each on a smaller slice of the embedding (`embed_size / num_heads = 64`
dimensions per head), each potentially learning to attend to different *kinds* of
relationships (one head might learn to track grammatical subject-verb agreement, another
might track topical relevance — this specialization emerges from training, it isn't
assigned).

### The causal mask: the detail that makes this GPT-like, not bidirectional

Without restriction, self-attention lets every token look at every *other* token,
including ones that come *after* it. For a decoder-only, next-token-prediction model,
this would be cheating — the model would be allowed to "peek" at the answer it's supposed
to predict. The **causal mask** prevents this: at position `t`, attention is only allowed
to look at positions `0` through `t`, never `t+1` or beyond.

```python
causal_mask = torch.triu(
    torch.full((seq_len, seq_len), float("-inf"), device=x.device),
    diagonal=1,
)   # CausalSelfAttention.forward(), line 240
```

This builds an upper-triangular matrix of `-inf` values (everything *above* the diagonal
— i.e., "future" positions relative to each row) — adding `-inf` to an attention score
before softmax forces that score's contribution to become exactly `0` after softmax,
which is precisely how "cannot look at future tokens" gets enforced numerically. This one
mechanism is what separates a GPT-style decoder from an encoder (like BERT), which
*does* allow full bidirectional attention — a genuinely different design choice for a
genuinely different objective (BERT is trained to fill in masked-out tokens using
context from *both* directions, not to generate text left-to-right).

### The MLP block: per-token processing after attention mixes information

After self-attention lets tokens exchange information, each token's representation is
passed independently through a small feed-forward network — same weights applied to
every position, but no cross-token interaction happens here (that already happened in
attention):

```python
self.net = nn.Sequential(
    nn.Linear(embed_size, 4 * embed_size),   # expand: 768 -> 3072
    nn.GELU(),                                # nonlinearity
    nn.Linear(4 * embed_size, embed_size),   # project back: 3072 -> 768
    nn.Dropout(dropout),
)   # MLP.__init__, line 257
```

The expand-then-contract pattern (4x the embedding size, then back down) is a
long-standing convention across Transformer architectures — the larger intermediate
dimension gives the network more room to compute a richer, higher-dimensional
transformation before compressing back to the model's standard width.

### Residual connections and LayerNorm: why deep stacks of blocks actually train

Two supporting mechanisms make it possible to stack many blocks (16, here) without
training collapsing:

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))    # residual: add the ORIGINAL x back in
    x = x + self.mlp(self.ln_2(x))     # residual again
    return x
```

- **Residual (skip) connections** — each sub-layer's output is *added to*, not used to
  *replace*, its input. This gives gradients a direct path backward through the network
  during training (they can flow through the `+` unchanged), which is what makes very
  deep networks trainable at all — without residuals, gradients tend to vanish or
  explode across many stacked layers.
- **LayerNorm** — normalizes activations (roughly: rescales them to have consistent
  mean/variance) before each sub-layer, stabilizing training. This specific arrangement —
  LayerNorm applied *before* attention/MLP, not after — is called **pre-norm**, and is
  the modern standard specifically because it trains more stably at depth than the
  original Transformer paper's post-norm arrangement.

### Weight tying: one clever parameter-saving trick

```python
self.lm_head.weight = self.token_emb.weight   # TinyGPT.__init__, line 296
```

The final layer (`lm_head`, projecting from `embed_size` back up to `vocab_size` logits)
and the initial token embedding layer are **the same matrix**, just used in opposite
directions — one converts token ID → vector, the other converts vector → per-token score.
This is a deliberate, common design choice (not an accident): it's motivated by the
intuition that "how similar is this output vector to token X's representation" is a
sensible way to score token X as a candidate next token, and it saves a substantial
number of parameters (this single tied matrix accounts for `vocab_size × embed_size` =
~38.6M parameters — see the parameter breakdown in
[`from_scratch/custom-gpt-153m/README.md`](../../from_scratch/custom-gpt-153m/README.md#parameter-count-current-config)
— that would otherwise need to exist twice).

## Grounded in This Repo's Code: The Full Stack, Assembled

```python
class TinyGPT(nn.Module):
    def forward(self, x):
        pos = torch.arange(seq_len, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)   # embeddings (this chapter)
        h = self.drop(h)
        for block in self.blocks:                    # N × GPTBlock
            h = block(h)                              # attention + MLP + residuals
        h = self.ln_f(h)                              # final LayerNorm
        return self.lm_head(h)                        # -> logits (Chapter 8)
```

Every line in this function is something this chapter has now explained from first
principles — there is no remaining unexplained "magic" between raw token IDs going in
and logits coming out.

## Deep-Dive: Why This Specific Set of Design Choices, and What Modern Models Change

`tiny_llm.py` uses a deliberately simple, standard-library-heavy implementation
(`nn.MultiheadAttention` rather than a hand-rolled attention mechanism) — a reasonable
choice for a learning project. Production-scale models commonly diverge in a few specific
ways, worth naming so you recognize them elsewhere:

- **RoPE (Rotary Position Embeddings)** instead of learned positional embeddings — encodes
  position via a rotation applied to the Query/Key vectors themselves, which generalizes
  better to sequence lengths longer than what was seen during training (a real limitation
  of this repo's fixed-size learned `pos_emb` table).
- **RMSNorm** instead of LayerNorm — a simplified normalization that skips re-centering
  (only rescales), cheaper to compute, used in LLaMA and many modern models.
- **SwiGLU** instead of GELU-based MLPs — a gated variant of the feed-forward block,
  used in LLaMA-family models (including `TinyLlama`, fine-tuned in this repo's
  `fine_tuning/` track) for a modest but real quality improvement.
- **Grouped-Query Attention (GQA)** — reduces the number of Key/Value projections
  relative to Query projections, primarily an *inference*-time memory optimization
  (directly relevant to `platform-lab/fundamentals/gpu_infrastructure/`'s KV-cache
  chapters), not something this repo's small model needs to bother with.

## Try It Yourself

- In `tiny_llm.py`, temporarily print `causal_mask` for a short `seq_len` (e.g. 5) and
  look at the actual matrix of `0` and `-inf` values — confirm for yourself that position
  0 can only attend to position 0, position 1 can attend to positions 0-1, and so on.
- Change `num_heads` from 12 to a different divisor of `embed_size` (768) — e.g. 8 — and
  re-run training briefly. The model still works (the total attention computation is the
  same shape), but the per-head dimensionality changes; this is a good way to build
  intuition for what "multi-head" actually controls.

## Common Misconceptions

- **"More attention heads always means a better model."** Not directly — head count
  trades off against per-head dimensionality for a fixed `embed_size`; it's an
  architectural hyperparameter with real trade-offs, not a dial that only goes one
  direction.
- **"The causal mask is applied to the tokens, not the attention scores."** It's applied
  to the *attention scores* before softmax (as `-inf`, forcing post-softmax weight to
  `0`) — the tokens themselves are never modified or hidden, the model just can't
  *attend* to future ones.
- **"Weight tying is a compression trick that hurts quality."** It's a deliberate
  architectural choice motivated by a real intuition about embedding/output symmetry, and
  is standard practice in GPT-2-class and many other models, not a corner cut purely for
  size.

## Practice Questions

1. Walk through, in order, every transformation a single token ID undergoes from input to
   output logits — name each layer and what it does.
2. Why would removing the residual connections (the `x +` in `GPTBlock.forward`) make a
   16-layer network much harder to train, even if the attention and MLP math stayed
   identical?
3. Explain, precisely, how the causal mask numerically prevents position 3 from being
   influenced by position 5's information, tracing through the `-inf` → softmax step.

## Key Terms

- **Embedding**: a learned lookup table converting a discrete ID (token or position) into
  a dense vector.
- **Self-attention**: the mechanism letting each position compute a weighted combination
  of all other (allowed) positions' representations, via Query/Key/Value projections.
- **Multi-head attention**: running several smaller attention computations in parallel,
  each potentially specializing in different relationship types.
- **Causal mask**: the mechanism restricting attention to only earlier (or the same)
  positions, required for autoregressive, left-to-right generation.
- **Residual (skip) connection**: adding a sub-layer's input back to its output, enabling
  stable training of deep stacks.
- **LayerNorm / pre-norm**: a normalization step stabilizing activations, applied before
  each sub-layer in the modern ("pre-norm") convention.
- **Weight tying**: sharing the same weight matrix between the input embedding and output
  projection layers.
- **RoPE, RMSNorm, SwiGLU, GQA**: modern architectural refinements used in many current
  models (including `TinyLlama`) beyond this repo's simpler, standard-library
  implementation.
