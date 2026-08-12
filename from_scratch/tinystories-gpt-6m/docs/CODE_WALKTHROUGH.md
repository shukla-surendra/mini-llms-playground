# Code Walkthrough: `model.py` and `train.py`, Line by Line

Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (which covers *sizing* decisions) and
[`TRAINING.md`](TRAINING.md) (which covers *hyperparameters*). This doc is the third
piece: every mechanism in [`../model.py`](../model.py) and [`../train.py`](../train.py)
explained precisely — the math, why that specific API/approach was chosen, and what the
real alternatives are (including what larger models like `TinyLlama`, fine-tuned in
[`../../../fine_tuning/tinyllama-1.1b-lora/`](../../../fine_tuning/tinyllama-1.1b-lora/),
do differently). If a term here is unfamiliar at the concept level (what is attention,
what is backprop), start with
[`../../../docs/llm-engineering/00_roadmap.md`](../../../docs/llm-engineering/00_roadmap.md)
first — this doc assumes those concepts and goes straight to *this file's* specific
implementation choices.

## `model.py`

### `CausalSelfAttention`

```python
self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout, batch_first=True)
```

**The math**: for every position, three vectors are computed as linear projections of the
input — Query (Q), Key (K), Value (V) — then:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

`QKᵀ` produces a `seq_len × seq_len` matrix of raw relevance scores (how much position
*i* should attend to position *j*). The `/√d_k` scaling exists because dot products grow
large as dimension `d_k` grows, which would push softmax into a region with near-zero
gradient — a numerical stability fix, not decoration. Softmax turns each row into a
probability distribution; multiplying by V produces, for each position, a weighted blend
of every other (allowed) position's Value vector.

**Why `nn.MultiheadAttention`, not hand-written Q/K/V matrices**: PyTorch's built-in layer
does the same math but fuses the three projections into one matrix multiply internally
(`in_proj_weight` is actually `3 × embed_size × embed_size`), faster than three separate
`nn.Linear` calls. The trade-off: less visible/hackable internals — production
codebases (nanoGPT, LLaMA reference implementations) typically hand-roll attention
instead, specifically so they can swap in custom variants (sliding-window attention,
grouped-query attention). This project deliberately keeps the simpler, standard-library
path.

**The causal mask, exactly**:
```python
causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
```
`torch.triu(..., diagonal=1)` keeps only the values *strictly above* the diagonal — so
`causal_mask[i][j] = -inf` whenever `j > i` (a future position relative to `i`), and `0`
otherwise. PyTorch adds this to the raw `QKᵀ/√d_k` scores before softmax; `exp(-inf) = 0`
after softmax, so that future position contributes exactly zero weight to position `i`'s
output. It's recomputed every forward call (not cached), since `seq_len` differs between
training (fixed at `context_length`) and generation (grows token by token).

**A real, faster alternative**: `torch.nn.functional.scaled_dot_product_attention`
(PyTorch 2.0+) has an `is_causal=True` flag that applies this same masking internally
using fused/flash-attention kernels, meaningfully faster because it never materializes
the full `seq_len × seq_len` mask matrix in memory — a real cost that grows quadratically
with sequence length. Production LLMs almost universally use flash-attention-style fused
kernels for exactly this reason; the explicit mask tensor here is the more
readable/educational choice, not the fastest one.

### `MLP`

```python
nn.Linear(embed_size, 4*embed_size) -> GELU -> nn.Linear(4*embed_size, embed_size) -> Dropout
```

**The math**: `output = W₂ · GELU(W₁·x + b₁) + b₂`, applied **independently per token
position** — no cross-token mixing happens here (that already happened in attention).

**Why 4× expansion**: convention inherited from the original Transformer paper —
empirically, a wider intermediate computation space before compressing back down
outperforms keeping width constant throughout. LLaMA-family models (including
`TinyLlama`) use a different ratio (~2.67×) combined with **SwiGLU** — a gated variant
`(W₁x) ⊙ SiLU(W₂x)` that empirically outperforms this simpler GELU-MLP at the cost of one
extra weight matrix. This project uses the simpler design deliberately, for the same
reason as [`../../custom-gpt-153m/`](../../custom-gpt-153m/): fewer moving parts to
debug, and not the bottleneck holding output quality back at this scale.

**Why GELU over ReLU**: `ReLU(x) = max(0, x)` has zero gradient for any negative input —
a neuron permanently in negative territory can get "stuck" and stop learning entirely
(the "dying ReLU" problem). GELU (`x · Φ(x)`, a smooth Gaussian-CDF-based approximation)
has no hard zero region and trains more reliably for Transformers specifically — why GELU
(or its close cousin SiLU/Swish) is the near-universal MLP activation in modern
Transformers.

### `GPTBlock`

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

**Pre-norm, not post-norm**: the original 2017 Transformer normalized *after* each
sublayer (`LayerNorm(x + sublayer(x))`). Pre-norm (used here, and in essentially every
modern LLM) trains more stably at depth — gradients flowing backward through many stacked
blocks pass through fewer normalization operations on their most direct path, avoiding a
real instability post-norm exhibits in deep stacks. This is an empirically-motivated
switch the field made, not a stylistic preference.

**Why the residual (`x + ...`) connection is load-bearing**: the derivative of `x + f(x)`
with respect to `x` includes a clean `1` term regardless of how complicated `f` is —
giving backpropagated gradients a direct path through the network. Without residuals,
gradients must pass through every sublayer's full transformation, and across 6 (or 16, or
96) stacked layers this compounding tends to vanish toward zero or explode. Residual
connections are the specific mechanism that makes training deep networks tractable —
remove them and this model would very likely fail to train past a handful of layers.

**LayerNorm's math**: `y = γ · (x - mean(x))/√(var(x) + ε) + β`, computed per-token
across the `embed_size` dimension (not across the batch, unlike BatchNorm — LayerNorm is
used specifically because sequence lengths vary and BatchNorm's batch-dependent
statistics behave badly for variable-length sequential data). `γ`/`β` are learned,
letting the network undo the normalization if that turns out to help. LLaMA-family models
use **RMSNorm** instead — skips mean-centering, only rescales by root-mean-square,
cheaper to compute, roughly comparable quality.

### `TinyStoriesGPT`

```python
self.token_emb = nn.Embedding(vocab_size, embed_size)   # a 4096×256 lookup table
self.pos_emb = nn.Embedding(context_length, embed_size)  # a 256×256 lookup table
h = self.token_emb(x) + self.pos_emb(pos)
```

`nn.Embedding` is a matrix; indexing it by token/position ID is a lookup, not a matrix
multiply — cheap regardless of how large the table is. Token and position embeddings are
combined by **element-wise addition** — the model receives one vector per position
encoding both "which token" and "which position," and has to learn to disentangle them
itself through training. **Alternative**: **RoPE** (rotary position embeddings) doesn't
add anything to the input at all — it *rotates* Q and K vectors inside attention by an
angle depending on position, encoding *relative* position directly into the attention
score, and generalizing to sequence lengths longer than any seen in training (this
project's fixed 256-row `pos_emb` table has no representation at all for position 257).

```python
self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
self.lm_head.weight = self.token_emb.weight   # weight tying
```

Setting `lm_head.weight` to literally the *same tensor object* as `token_emb.weight`
(not a copy): saves `vocab_size × embed_size` parameters (~1M here) that would otherwise
be duplicated, and means gradients from both the input-embedding lookup and the
output-projection step update the same underlying matrix — motivated by the intuition
that "how similar is this hidden vector to token X's embedding" is a sensible way to
score token X as a next-token candidate. Standard in GPT-2-class models; some larger
models *un-tie* these past a certain scale, since the optimal input-embedding and
output-projection matrices tend to diverge somewhat as models grow.

```python
torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

Weights start neither at zero (a network of all-zero weights has ~zero gradient almost
everywhere and can't learn) nor too large (large initial weights can saturate
nonlinearities or cause unstable early gradients). `std=0.02` is GPT-2's exact value — a
reasonable, empirically-validated default, not something derived from first principles
for this specific model size (more principled schemes like Xavier/Kaiming initialization
scale the std based on layer width, which this simpler fixed value doesn't do).

```python
for block in self.blocks:
    h = block(h)
h = self.ln_f(h)
return self.lm_head(h)
```

The final `ln_f` LayerNorm, applied *after* the last block and *before* `lm_head`, exists
specifically because of the pre-norm design: every block normalizes its own *input*, not
its output, so without this final norm the last block's raw, un-normalized output would
feed directly into `lm_head` — inconsistent with how every other layer received its
input.

## `train.py`

### `get_batch`

Picks `batch_size` random starting positions in the token stream, slices out
`context_length`-token windows; `y` is `x` shifted right by one position — the standard
next-token-prediction framing (at every position, the target is "whatever token comes
next"). Random windows, not a sequential sweep through the data, give better stochastic
coverage of the dataset across training steps.

### The four-step loop

```python
logits = model(xb)
loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
(loss / grad_accum_steps).backward()
optimizer.step()
```

Forward pass → loss → backprop → gradient descent, exactly as covered in
[`../../../docs/llm-engineering/03_how_neural_networks_learn.md`](../../../docs/llm-engineering/03_how_neural_networks_learn.md).
Cross-entropy specifically because this is classification-shaped (predict *which* of
4,096 tokens comes next); it penalizes the model proportionally to `-log(probability
assigned to the correct token)`, so a confident wrong answer is penalized far more
than an unconfident one.

### `clip_grad_norm_(model.parameters(), max_norm=1.0)`

Rescales the *entire* gradient vector (across all parameters combined, as one vector) if
its overall L2 norm exceeds `1.0`, without changing its direction — a cheap safeguard
against the rare batch producing an unusually large gradient that would otherwise
destabilize a single update step.

### AdamW, not plain SGD

Plain gradient descent (`w -= lr × gradient`) uses one fixed step size for every
parameter equally. AdamW tracks a running average of each parameter's gradient *and* its
squared gradient, using both to adapt that parameter's effective step size individually —
parameters with consistently large gradients get relatively smaller steps, and vice
versa. This adaptivity is why AdamW, not SGD, is the near-universal choice for
Transformer training (SGD remains preferred in some vision/CNN contexts with different
convergence properties).

### `lr_for_step` — warmup then cosine decay

At step 0, weights are random and gradients are noisy/large — taking full-size steps
immediately risks destabilizing training before it starts, so the learning rate ramps
linearly from ~0 up to `lr` over the first ~2% of steps. After that, it decays along a
cosine curve down to `min_lr` — smaller late-training steps let the model settle
precisely into a good solution rather than oscillating around it. **Alternatives**:
linear decay (simpler, less common now), constant LR with no decay (works, usually
converges to a slightly worse final loss), one-cycle schedules (popular in the vision
world). Cosine decay is simply today's empirically-favored default for Transformers.

### Resume-compatibility check

```python
compatible = (ckpt.get("embed_size") == embed_size and ckpt.get("num_heads") == num_heads and ...)
```

Loading a saved `state_dict` into a model built with *different* `embed_size`/
`num_layers`/`vocab_size` than it was trained with produces a shape-mismatch crash at
best, silently wrong behavior at worst — this check catches the mismatch before
attempting the load, falling back to a fresh run instead. Same mechanism, same reasoning,
as [`../../custom-gpt-153m/`](../../custom-gpt-153m/)'s resume logic.
