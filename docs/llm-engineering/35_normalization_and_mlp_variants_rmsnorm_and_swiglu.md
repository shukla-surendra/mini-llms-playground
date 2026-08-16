# Normalization and MLP Variants: RMSNorm vs. LayerNorm, SwiGLU vs. GELU-MLP

Part of the [LLM Engineering Curriculum](00_roadmap.md), Chapter 35 (Part 1B — appended
after the original catalog, same reason as Parts 2B/2C/3B: avoids renumbering
already-written chapters; belongs right after [Chapter 11](11_positional_encoding_variants_rope_and_beyond.md)
in reading order). Builds directly on
[Chapter 10](10_transformer_architecture.md#the-mlp-block-per-token-processing-after-attention-mixes-information)'s
introduction to the MLP block and pre-norm residual pattern — that chapter covers *why*
normalization and a feed-forward block exist at all, using this repo's simplest versions
(`nn.LayerNorm`, a GELU MLP). This chapter is the deeper dive: the two newer variants
`custom-gpt-200m`/`350m` actually use instead, and why.

## In Plain English

**Normalization** is a volume knob. Every layer's output can drift toward very large or
very small numbers as it passes through a deep stack — normalization rescales things back
to a sane range before the next layer has to deal with them, the same way a sound
engineer rides the fader so no single input clips or gets lost. **LayerNorm** and
**RMSNorm** are two different, closely related volume-knob designs. LayerNorm centers the
signal first (subtracts the average) *and* rescales it, then lets the network re-shift the
result with a learned offset. RMSNorm skips the centering step entirely — it only
rescales, nothing else — on the finding that the centering step wasn't actually pulling
its weight.

**The MLP block** is the "think about this token on its own" step, right after attention
has let tokens exchange information with each other. A plain (GELU) MLP is a single
expand-then-contract pass — squeeze the token through a wider hidden layer, then back
down. **SwiGLU** does something genuinely different: it computes *two* parallel paths
through that wider layer, and uses one of them as a learned **gate** that decides, per
hidden unit, how much of the other path to actually let through — like a mixing desk
where one fader doesn't just control volume, it controls how much another fader's signal
gets heard at all.

## The First-Principles Explanation

### RMSNorm: the exact formula, and what's missing on purpose

LayerNorm (Chapter 10's version):

```
LayerNorm(x) = γ · (x − mean(x)) / sqrt(var(x) + ε) + β
```

Two learned parameter vectors: `γ` (scale) and `β` (shift/bias). Two statistics computed
per token: mean and variance.

RMSNorm ([`custom-gpt-200m/src/gpt/model.py`](../../from_scratch/custom-gpt-200m/src/gpt/model.py)):

```
RMSNorm(x) = γ · x / sqrt(mean(x²) + ε)
```

One learned parameter vector (`γ` only — no `β`). One statistic: the root-mean-square of
`x`, computed *without first subtracting the mean*. Two things are gone relative to
LayerNorm: the mean-centering step, and the learned shift. The empirical finding behind
this (Zhang & Sennrich, 2019) is that **re-centering doesn't meaningfully help
transformer quality** — the re-scaling (controlling the *magnitude* of activations) is
what actually stabilizes training; subtracting the mean turned out to be along for the
ride, not load-bearing. Removing it removes real, measurable compute (a mean reduction
and a subtraction across every element) and one full parameter vector per norm, for
negligible quality cost.

### SwiGLU: the exact formula, and why "gating" is a different mechanism

GELU-MLP (Chapter 10): `down(GELU(up(x)))` — one nonlinearity applied to one projection.

SwiGLU: `down(SiLU(gate(x)) ⊙ up(x))` — **two** separate linear projections of the same
input (`gate` and `up`), a nonlinearity (`SiLU`, a smooth variant of `ReLU`: `x · sigmoid(x)`)
applied only to the gate branch, then the gate and up branches are multiplied together
**elementwise** (`⊙`) before the final down-projection.

The mechanism this adds isn't "a different activation function" — GELU vs. SiLU is a
minor detail. It's the **elementwise multiplication of two different learned
projections of the input**. A plain MLP can only scale a hidden unit's contribution by a
fixed nonlinear function of *that unit's own pre-activation*. A gated MLP lets one entire
learned projection of the input (`gate`) modulate — multiplicatively, per hidden unit —
a completely separate learned projection of the same input (`up`). That's a strictly
more expressive computation per parameter: the network can learn to suppress or amplify
different parts of the `up` projection based on a different view of the same token,
rather than every hidden unit's gate being a fixed function of only itself.

### The parameter-matching trick: `f = 8/3 · E`

A GELU-MLP is two matrices, `E → 4E` and `4E → E`: `2 × E × 4E = 8E²` parameters. SwiGLU
is *three* matrices (`gate`, `up`, `down`), each `E ↔ f` where `f` is the gated hidden
size: `3Ef` parameters. Naively using the same `4E` width for `f` would make SwiGLU cost
50% more than a GELU-MLP for the same `embed_size` — not a fair comparison of "which
mechanism is better," just "which one has more parameters." Setting `f = (8/3)E` instead
makes `3Ef = 3E·(8/3)E = 8E²` — **exactly** the same parameter cost as the GELU-MLP. This
is precisely why `custom-gpt-200m`'s `ffn_hidden` defaults to `2368` at `embed_size=896`
(`2368 / 896 ≈ 2.643`, close to `8/3 ≈ 2.667` — see the Deep-Dive below for why it isn't
exact) rather than `4 × 896 = 3584`.

## Grounded in This Repo's Code

**RMSNorm** — [`custom-gpt-200m/src/gpt/model.py`](../../from_scratch/custom-gpt-200m/src/gpt/model.py):

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # gamma only — no beta

    def forward(self, x):
        # Compute in fp32 even under autocast: the mean-square of a bf16 vector loses
        # precision exactly where the norm is most sensitive.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)
```

Note the explicit fp32 upcast — a real, deliberate detail, not incidental: mixed-precision
training runs most matmuls in bf16, but this project computes the norm's statistics in
full precision regardless, because the mean-square is exactly the quantity a norm is most
sensitive to losing precision on.

**SwiGLU** — same file:

```python
class SwiGLU(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout):
        super().__init__()
        self.gate = nn.Linear(embed_size, hidden_size, bias=False)
        self.up = nn.Linear(embed_size, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
```

Note `bias=False` on all three — a separate, related design choice this project makes
uniformly (see Deep-Dive), not something specific to SwiGLU itself.

**Both used in the same pre-norm residual pattern Chapter 10 already covers** —
`GPTBlock.forward()` is structurally identical between the two projects, just with
different norm/MLP classes substituted in:

```python
# custom-gpt-153m (Chapter 10)          # custom-gpt-200m (this chapter)
def forward(self, x):                   def forward(self, x, cos, sin):
    x = x + self.attn(self.ln_1(x))         x = x + self.attn(self.norm_1(x), cos, sin)
    x = x + self.mlp(self.ln_2(x))          x = x + self.mlp(self.norm_2(x))
    return x                                return x
```

The residual/pre-norm *pattern* (why `x = x + ...` and why the norm goes inside the
branch, not on the residual path) doesn't change at all between the two — only which
concrete norm and MLP class fill those two slots.

## Deep-Dive: Why It's Built This Way

**Why `custom-gpt-200m` made both switches together, not independently** — its own
`model.py` docstring frames all four of its departures from the sibling family as one
coherent set of choices for a model "aimed at reasoning rather than novelty," not
individually novel ideas: RoPE, RMSNorm, SwiGLU, and no-biases are exactly the
architecture used by LLaMA and most subsequent open-weight model families — this project
adopts an already-converged-on modern recipe rather than inventing its own.

**Why `2368` isn't exactly `8/3 × 896`** — `8/3 × 896 = 2,389.33` (not an integer, and not
matching the `2368` actually used). Real implementations commonly round `f` to a multiple
of a hardware-friendly number (64, 128, or similar) for kernel efficiency, which nudges
the exact value slightly off the theoretical ratio without meaningfully changing the
parameter-matching argument. When this repo's own `custom-gpt-350m` was scaffolded (see
[the from-scratch practical work](../../from_scratch/custom-gpt-350m/)), its `ffn_hidden`
was chosen the same way — computed toward the `8/3` ratio, then rounded to a multiple of
32 (`2720` at `embed_size=1024`, where `8/3 × 1024 = 2730.67`).

**Why `bias=False` everywhere in this project, not just in SwiGLU** — a `nn.Linear`'s
bias term is one more learned parameter per output unit; the empirical finding this
project's docstring cites (again following the modern open-weight consensus, e.g. LLaMA)
is that removing biases costs essentially nothing in quality at this parameter scale
while being strictly cheaper. Worth being precise that this is a *separate* design choice
from SwiGLU itself — a GELU-MLP could just as easily drop its biases, and a SwiGLU
implementation could just as easily keep them; this project bundles both choices, but
they aren't logically coupled.

**What was deliberately NOT changed**: the pre-norm residual placement itself. Both
architecture families in this repo normalize *inside* each sub-layer's branch
(`x + branch(norm(x))`), never on the residual path itself (`norm(x + branch(x))`,
"post-norm," the original 2017 Transformer's arrangement) — Chapter 10 already covers why
pre-norm trains more stably at depth; nothing about switching to RMSNorm or SwiGLU
touches that placement decision at all.

## Try It Yourself

- **Verify RMSNorm's formula by hand.** Take a small random tensor, compute
  `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)` yourself in a Python shell,
  and compare against `RMSNorm(dim)(x)` (with `weight` initialized to all-ones, its
  default) — they should match exactly.
- **Confirm the SwiGLU parameter-matching trick, empirically, with this repo's own exact
  counter.** Run `make config` in `custom-gpt-200m` and read `param_breakdown()`'s
  `swiglu_mlp` line item; hand-compute `3 × embed_size × ffn_hidden` and confirm it
  matches; separately hand-compute what a GELU-MLP at the same `embed_size` would have
  cost (`8 × embed_size²`) and confirm the two are close (not identical, per the rounding
  note above).
- **Feel the difference between LayerNorm's centering and RMSNorm's lack of it.** Take a
  tensor with a large, deliberately nonzero mean (e.g. `torch.randn(8) + 100`), run it
  through both `nn.LayerNorm(8)` and `RMSNorm(8)` (both freshly initialized), and compare
  the outputs — LayerNorm's output will be centered near zero regardless of the input's
  offset; RMSNorm's won't, since it never subtracts anything.

## Common Misconceptions

- **"RMSNorm is just LayerNorm without a bias parameter."** Close, but the more important
  omission is the **mean-subtraction step**, not merely the learned `β`. A version of
  LayerNorm with `β` removed but mean-centering kept would be a different, less common
  variant — RMSNorm specifically removes the centering.
- **"SwiGLU has three matrices, so it must be more expensive than a two-matrix GELU-MLP."**
  False at matched hidden width — exactly the point of the `f = 8/3·E` sizing: three
  smaller matrices can cost the same as two larger ones. It only costs more if `f` is left
  at the naive `4E` width instead of being resized.
- **"The gate in SwiGLU is basically the same idea as a normal activation function, just
  with extra steps."** No — a normal activation (GELU, ReLU, SiLU on its own) is a fixed
  nonlinear function applied to a value using only that value's own information. A gate is
  a *learned, separate projection of the whole input* multiplying another *learned,
  separate projection of the whole input* — a fundamentally different, multiplicative
  interaction between two full linear transforms, not a nonlinearity bent to a new shape.
- **"Switching from LayerNorm/GELU to RMSNorm/SwiGLU is switching the architecture."**
  Structurally, no — both are pre-norm, residual-connected, two-sublayer-per-block
  transformers, exactly the pattern Chapter 10 covers. Only the two concrete classes
  filling the "norm" and "MLP" slots change; `GPTBlock.forward()`'s shape is identical.

## Practice Questions

1. RMSNorm removes mean-centering, and the empirical claim is that this costs
   essentially nothing in quality. What would you actually expect to observe in training
   if that claim were *false* for a given model/dataset — what signal would show up, and
   where?
2. Why does `f = (8/3)·E` make a 3-matrix SwiGLU cost the same as a 2-matrix `4E`-hidden
   GELU-MLP? Derive it from the two parameter-count formulas rather than just quoting the
   ratio.
3. `SwiGLU.forward()` multiplies `F.silu(self.gate(x))` by `self.up(x)` elementwise. What
   would be structurally different (not just "worse") about a version that instead added
   the two branches together instead of multiplying them?
4. `custom-gpt-200m`'s `RMSNorm.forward()` explicitly upcasts to fp32 before computing the
   mean-square, then casts back down. Why does this matter more for a norm's statistics
   specifically than it would for, say, a plain matrix multiply elsewhere in the network?

## Key Terms

- **RMSNorm**: normalization that rescales by root-mean-square only — no mean-centering,
  no learned shift — cheaper than LayerNorm with comparable empirical quality.
- **LayerNorm**: normalization that both centers (subtracts mean) and rescales (divides
  by standard deviation), with two learned parameters (scale and shift) — covered in
  [Chapter 10](10_transformer_architecture.md).
- **SwiGLU**: a gated MLP variant — two parallel input projections (`gate`, `up`)
  combined by elementwise multiplication (after a SiLU nonlinearity on the gate branch),
  then projected back down.
- **Gating (in an MLP)**: one learned projection of the input multiplicatively controlling
  how much of a separate learned projection passes through, per hidden unit — a
  fundamentally different mechanism from applying a fixed nonlinearity to a single
  projection.
- **SiLU (Sigmoid Linear Unit)**: `x · sigmoid(x)`, the smooth activation function used on
  SwiGLU's gate branch — closely related to, but distinct from, GELU.
- **Pre-norm**: normalizing *inside* each sub-layer's branch (`x + branch(norm(x))`)
  rather than on the residual path — the arrangement both architecture families in this
  repo use, unaffected by which specific norm/MLP variant fills the slots (Chapter 10).
