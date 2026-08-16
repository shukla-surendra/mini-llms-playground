# Positional Encoding Variants, Deeper (RoPE and Beyond)

Part of the [LLM Engineering Curriculum](00_roadmap.md), Chapter 11. Builds directly on
[Chapter 10](10_transformer_architecture.md#positional-embeddings-telling-the-model-where-each-token-is)'s
introduction to positional embeddings — that chapter covers *why* a transformer needs
position information at all and walks through this repo's simplest implementation
(learned absolute position embeddings, `custom-gpt-10m`/`50m`/`153m`). This chapter is the
deeper dive promised there: the mechanism, math, and trade-offs of **RoPE** (rotary
position embeddings), used by `custom-gpt-200m`/`350m`, and why a newer project in the
same repo switched away from the simpler approach.

## In Plain English

Imagine a classroom where every student sits in a numbered seat, and the teacher has a
seating chart taped to the wall — seat 1 is "front-left," seat 2 is "front-left-plus-one,"
and so on, all the way up to however many seats the room has. That's a **learned position
embedding**: a fixed chart, memorized in advance, with exactly as many entries as there
are seats. It works great — until a new student needs to sit in seat 51, and the chart
only goes up to 50. There's no entry for that seat. It isn't fuzzy or degraded, it simply
doesn't exist.

RoPE throws out the seating chart entirely and gives every student a **compass instead**.
Each student knows their own position by how far they'd have to rotate to face the front
of the room — student 1 rotates a little, student 50 rotates a lot, student 51 just keeps
rotating a little further, no chart lookup required. And here's the genuinely useful part:
if two students want to know how far apart *they* are from each other, they don't need to
know the room's layout at all — they just compare how differently they're rotated
relative to each other. That relative comparison is exactly what attention (Chapter 10)
actually needs — not "where am I in the room," but "how far is that other token from me."

## The First-Principles Explanation

### Recap: what a learned position embedding actually is

From Chapter 10: `self.pos_emb = nn.Embedding(context_length, embed_size)` — a table with
one learned row per position, added to the token embedding once, at the very start:

```python
pos = torch.arange(seq_len, device=x.device)
h = self.token_emb(x) + self.pos_emb(pos)
```

This is an **absolute** encoding: position 5 always gets exactly the same vector added to
it, regardless of what else is in the sequence. Every layer of the network then sees that
positional signal only indirectly, mixed into the hidden state from the very first line.

### The real problem this creates: a hard, architectural ceiling

`self.pos_emb` is a lookup table with exactly `context_length` rows. Position
`context_length` (say, position 512 in a model trained with `context_length=512`) has no
row to look up — not a degraded answer, an **index that doesn't exist**. This repo's own
`MODEL_SIZING_GUIDE.md` calls this out directly: *"a model trained at `context_length=512`
cannot later be resumed or fine-tuned at a longer context — position index 600 simply has
no learned embedding row; there's nothing to look up."* Changing `context_length` after
training starts is a one-way door, architecturally, for exactly this reason.

### RoPE's actual mechanism: rotate, don't look up

RoPE (Su et al., 2021, *RoFormer*) makes a different choice: instead of adding a
position-dependent vector to the token embedding once, it **rotates** the Query and Key
vectors, inside attention, by an angle that depends on absolute position — and it does
this at *every layer*, not once at the start.

Treat each adjacent pair of dimensions in a Q or K vector as the (x, y) coordinates of a
2D point. Rotating that point by angle `θ·m` (where `m` is the token's position) is
literally a 2D rotation:

```
x' = x·cos(θ·m) − y·sin(θ·m)
y' = x·sin(θ·m) + y·cos(θ·m)
```

Do this for every dimension pair, with a different rotation *speed* `θ` per pair (more on
that below), and you've rotated the whole vector by an amount that encodes its position —
with **zero new learned parameters**. `cos`/`sin` are computed from position, not trained.

### Why this gives you *relative* position for free — the actual payoff

Attention computes `Q·Kᵀ` — a dot product between a query at position `m` and a key at
position `n`. Rotating two vectors by angles `θm` and `θn` and then taking their dot
product is mathematically equivalent to taking the dot product of the *unrotated* vectors
and then rotating by `θ(m−n)` — **the absolute positions cancel out, and only the
difference `m−n` survives**. This is the load-bearing property: attention scores end up
depending only on how far apart two tokens are, never on where they sit in the sequence in
absolute terms. Token 5 attending to token 3 (a gap of 2) produces the identical
positional contribution as token 500 attending to token 498 — same gap, same effect. A
learned absolute table has no such property; row 5 and row 500 are two unrelated,
independently-learned vectors.

### Why some dimensions rotate fast and others rotate slowly

`θ` isn't one number — it's a different rotation *frequency* per dimension pair, decaying
geometrically:

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
```

Early dimension pairs (small index) get a high frequency — they complete a full rotation
after only a few tokens, so they're sensitive to *fine-grained*, short-range position
differences. Late dimension pairs get a low frequency — they rotate very slowly across the
whole sequence, so they carry *coarse*, long-range positional structure. This is directly
analogous to the original Transformer paper's fixed sinusoidal encoding (which also uses
a spread of frequencies) — RoPE's actual innovation isn't "use multiple frequencies," it's
*where* that signal gets applied (rotating Q/K inside every attention call, not added to
the embedding once) and the relative-position algebra that falls out of doing it that way.

`theta` (the base of that geometric decay, `rope_theta`, default `10000.0`) sets how far
apart the fastest and slowest frequencies are — raising it stretches the slowest
wavelength longer, which is the standard lever used to extend a trained model's usable
context after the fact (not implemented in this repo, but the reason the field exists as a
tunable `ModelConfig` value rather than a hardcoded constant).

## Grounded in This Repo's Code

**The learned-table family** (`custom-gpt-10m`/`50m`/`153m`) — already walked through in
[Chapter 10](10_transformer_architecture.md); the short version, for contrast: one
`nn.Embedding(context_length, embed_size)`, added once, in `TinyGPT.forward()`/`encode()`.

**The RoPE family** (`custom-gpt-200m`/`350m`) — three pieces, in
[`custom-gpt-200m/src/gpt/model.py`](../../from_scratch/custom-gpt-200m/src/gpt/model.py):

```python
def build_rope_cache(head_dim, max_seq_len, theta, device=None, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)          # (seq, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    seq_len = x.shape[-2]
    cos = cos[:seq_len].view(1, 1, seq_len, -1)
    sin = sin[:seq_len].view(1, 1, seq_len, -1)
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)
```

`build_rope_cache` precomputes every position's `(cos, sin)` pair up front, once, as
non-trainable **buffers** (`self.register_buffer("rope_cos", cos, persistent=False)`,
`persistent=False` meaning they're rebuilt from config on load, never saved in a
checkpoint — there's nothing to save, they're derived, not learned). `apply_rope` is the
rotation formula above, applied to whole batched tensors at once. Inside
`CausalSelfAttention.forward()`:

```python
q, k, v = self.in_proj(x).chunk(3, dim=-1)
...
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)          # <- V is never rotated — see Common Misconceptions
out = F.scaled_dot_product_attention(q, k, v, ...)
```

Note exactly where this happens: **inside every attention call, at every layer** — not
once at the input, unlike `pos_emb`.

### A live example of the "relative position, not absolute" property: incremental decoding

The KV-cache generation code added to this project family (`inference/generate.py`,
[Chapter 21](21_inference_mechanics_decoding_sampling_and_kv_cache.md)) is a genuine,
running example of RoPE's relative-position property paying off. A cached decode step
rotates only the *one new token*, by its own absolute position, and concatenates it with
**already-rotated** keys from every earlier step:

```python
cos = self.rope_cos[start_pos:start_pos + seq_len]
sin = self.rope_sin[start_pos:start_pos + seq_len]
...
if past_kv is not None:
    past_k, past_v = past_kv
    k = torch.cat([past_k, k], dim=2)   # mixing rotation angles from different calls — and it's still correct
```

This only works because each token's rotation is a fixed function of *its own* absolute
position, computed once and never revisited — mixing K vectors rotated in different
forward passes is safe precisely because rotation isn't relative to "the current call," it
was baked in at each token's own position from the start.

## Deep-Dive: Why It's Built This Way

**Why `custom-gpt-200m` switched, specifically** — straight from that project's own
`model.py` docstring: *"Position 2049 in a model trained at 2048 simply has no row, so the
context window is a permanent, architectural ceiling... RoPE instead rotates Q and K by an
angle proportional to absolute position, so attention scores depend only on the relative
offset between two tokens. Nothing is learned per position, so there is no table to run
off the end of."* This wasn't a change made for novelty — it's a direct response to a
named, documented limitation of the simpler approach used by the project's own smaller
siblings.

**Parameter cost**: a learned `pos_emb` costs `context_length × embed_size` real,
trained parameters — small relative to a model's total (`MODEL_SIZING_GUIDE.md` puts it at
~1% for the `10m` preset) but not zero. RoPE costs **exactly zero** learned parameters —
`rope_cos`/`rope_sin` are computed, not trained, and explicitly excluded from checkpoints
(`persistent=False`).

**Compute cost, honestly**: RoPE isn't free at runtime — it adds an elementwise
rotation (a few multiplies and adds per Q/K vector) at *every layer's* attention call,
where a learned table only ever costs one lookup at the very start. This is a real,
measurable cost, just a small one relative to the surrounding matmuls — this repo doesn't
claim otherwise, it's a genuine trade (zero parameters and no context ceiling, in exchange
for a small recurring compute cost) not a strictly-better-in-every-way replacement.

**What extending context after training actually requires**: raising `rope_theta` after
the fact (context-extension techniques like "NTK-aware scaling" or linear interpolation)
is the standard lever in the wider field — genuinely possible with RoPE in a way it isn't
with a learned table, but **not implemented in this repo**; worth knowing the door exists,
not a claim this codebase walks through it.

**What was deliberately left out here**: ALiBi (Attention with Linear Biases) is another
well-known relative-position scheme — instead of rotating Q/K, it adds a fixed,
distance-proportional penalty directly to attention scores before the softmax. Mentioned
for completeness (it solves the same "avoid a fixed table" problem RoPE does, via a
different mechanism), not implemented anywhere in this repo, so no code-grounded treatment
of it is given here.

## Try It Yourself

- **Verify the relative-position property empirically.** In a Python shell, build a small
  `head_dim` (say 8), call `build_rope_cache(8, 20, 10000.0)`, rotate two fixed vectors `q`
  and `k` at positions `(5, 3)` and separately at `(500, 498)` (same gap, 2, both times),
  and confirm `q_rotated @ k_rotated` is the same in both cases — direct, hands-on proof
  of the claim in "Why this gives you relative position for free" above.
- **Compare parameter counts directly.** Run `make config` in both `custom-gpt-153m`
  (learned `pos_emb`) and `custom-gpt-200m` (RoPE) and look at each `param_breakdown()` —
  one has a `position_embedding` line item, the other doesn't have one at all.
- **Trigger the context ceiling on purpose.** In `custom-gpt-200m`, call
  `model(torch.zeros(1, model.context_length + 1, dtype=torch.long))` — read the
  `ValueError` `TinyGPT.forward()` raises (*"the RoPE cache is only built that far"*) and
  compare it to what would happen in `custom-gpt-153m` if you tried the equivalent
  (an `IndexError` out of `nn.Embedding`, not a clean, named error) — RoPE still has a
  ceiling at `context_length` (the precomputed cache doesn't extend itself), the real
  difference is what determines that ceiling: an architectural constant that must be
  fixed before training (`pos_emb`'s row count) vs. a cache size that's a pure engineering
  choice, rebuildable at load time to any length without retraining.

## Common Misconceptions

- **"RoPE means unlimited context length."** No — `custom-gpt-200m`'s own `rope_cos`/
  `rope_sin` buffers are built only up to `context_length` at construction time, and
  `TinyGPT.forward()` raises a `ValueError` past that, same as any other fixed-size
  architecture. What RoPE actually buys is that this ceiling is a *rebuildable cache size*,
  not a *trained, fixed parameter count* — genuinely different, but "unlimited" overstates
  it.
- **"Position is added to the input once, same as `pos_emb`, just computed differently."**
  No — this is the single most important mechanical difference. `pos_emb` touches the
  hidden state exactly once, before the first block. RoPE rotates Q and K **inside every
  attention call, at every layer** — there is no single "position gets added here" moment.
- **"V gets rotated too, since Q and K do."** No — `apply_rope` is called on `q` and `k`
  only; `v` passes through unrotated. Rotation exists to make the `Q·Kᵀ` *similarity score*
  position-aware; `V` just carries content to be weighted-summed, which doesn't need a
  position signal of its own.
- **"Higher `rope_theta` always means better long-context performance."** Not
  automatically — `theta` sets frequency *spacing*, and changing it after training without
  further adaptation is a real intervention with its own failure modes, not a free dial.
  This repo doesn't implement or claim to validate any specific context-extension
  technique; treat "raise theta to extend context" as a pointer to further reading, not a
  recipe demonstrated here.

## Practice Questions

1. A learned `pos_emb` table and RoPE both eventually hit a maximum sequence length they
   can handle. What's the actual difference between the two ceilings — is one truly
   architectural and the other not, or is the difference something else?
2. Why does rotating *both* Q and K (rather than just one of them) produce a *relative*
   position signal in the dot product, instead of just shifting the absolute signal
   around?
3. `apply_rope` treats each pair of adjacent dimensions independently, with its own
   rotation frequency. What would likely go wrong (or simply not happen) if every
   dimension pair used the *same* frequency instead of a geometrically decaying spread?
4. The KV-cache decode loop concatenates key vectors that were rotated in different,
   separate forward-pass calls. Why is this safe for RoPE specifically — what property
   would have to hold for this to also be safe with some other, hypothetical positional
   scheme?

## Key Terms

- **Absolute position embedding**: a learned vector added per position, independent of
  any other token — this repo's `custom-gpt-10m`/`50m`/`153m` approach (Chapter 10).
- **RoPE (Rotary Position Embedding)**: rotating Q and K vectors by an angle proportional
  to absolute position, inside every attention call, so that attention scores depend only
  on relative position — this repo's `custom-gpt-200m`/`350m` approach.
- **Relative position encoding**: any scheme (RoPE, ALiBi, and others) where the
  positional signal attention actually uses depends on the *distance* between two tokens,
  not either token's absolute location.
- **`rope_theta`**: the base controlling how fast rotation frequency decays across
  dimension pairs — the lever used, elsewhere in the field, to extend a trained model's
  usable context.
- **Rotation frequency / dimension pair**: RoPE rotates each adjacent pair of vector
  dimensions at its own speed; early pairs rotate fast (fine-grained position), late pairs
  rotate slow (coarse, long-range position).
- **Context ceiling**: the maximum sequence length a model can process at all — for a
  learned table, a fixed row count set at training time; for RoPE, a precomputed cache
  size, rebuildable without retraining.
