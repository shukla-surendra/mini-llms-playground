# Inference Mechanics: Decoding, Sampling, and KV Cache

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 4 — Serving: Turning a
Trained Model Into Something You Can Talk To. Builds on
[Chapter 8](08_what_is_a_language_model.md)'s autoregressive generation loop and "greedy
vs. sampling" introduction — this chapter goes one level deeper: the exact math behind
temperature, top-k, and top-p, why they compose the way they do, and why real serving
systems need a KV cache even though the model in this repo doesn't have one.

## In Plain English

Once a model can predict "what token comes next," turning that into actual generated text
means repeatedly picking one token from that prediction and feeding it back in. *How* you
pick — always the single most likely token, or something with room for variety — is
decoding strategy. Temperature, top-k, and top-p are three independent knobs on the
"something with room for variety" side, each cutting the problem differently: temperature
reshapes how confident the distribution is allowed to look, top-k/top-p throw away the
unlikely tail before you even sample from what's left. And every one of those repeated
predictions naturally recomputes attention over the whole sequence so far — which is fine
for a short generation on a small model, and a genuine bottleneck at production scale,
which is exactly the problem the KV cache exists to solve.

## The First-Principles Explanation

### Where temperature sits in the pipeline

Recall the generation loop: the model produces **logits** (raw, unnormalized scores, one
per vocabulary token), and **softmax** turns those into a probability distribution.
Temperature is applied *between* these two steps, rescaling logits before softmax runs:

```
raw logits  →  divide by temperature  →  softmax  →  probability distribution  →  sample
```

```
scaled_logit_i = logit_i / temperature
probability_i  = exp(scaled_logit_i) / sum(exp(scaled_logit_j) for all j in vocabulary)
```

Dividing every logit by the same number changes how *spread out* the resulting
probabilities are without changing their relative order — the highest-logit token still
has the highest probability at any temperature. What changes is the **gap** between the
top choice and everything else: lower temperature widens that gap (sharper, more
deterministic-leaning), higher temperature narrows it (flatter, more varied — and, past a
point, more likely to pick something incoherent).

**A worked example.** Four candidate tokens with raw logits `[4.0, 3.0, 1.0, 0.5]`:

```
temperature = 1.0:  softmax([4.0, 3.0, 1.0, 0.5])         ≈ [0.62, 0.23, 0.031, 0.019]
temperature = 0.5:  softmax([8.0, 6.0, 2.0, 1.0])          ≈ [0.87, 0.12, 0.002, 0.001]
temperature = 2.0:  softmax([2.0, 1.5, 0.5, 0.25])         ≈ [0.42, 0.26, 0.10, 0.08]
```

At `temperature=0.5` the top choice dominates even more than at 1.0; at `temperature=2.0`
the third and fourth candidates become genuinely competitive. Temperature doesn't invent
randomness from nowhere — it redistributes probability mass the model already computed. A
token the model assigned near-zero probability stays near-zero at any reasonable
temperature.

### Why temperature → 0 is exactly greedy decoding, not an approximation of it

As temperature approaches 0, dividing logits by it drives the scaled logits toward
±infinity, so softmax collapses to assigning ~1.0 to the single highest-logit token and ~0
to everything else — precisely what `argmax` (greedy decoding) does directly. This is a
genuine mathematical limit, not a coincidence: `temperature=0` and greedy decoding are the
same operation reached by two different mechanical paths, which is why real
implementations special-case `temperature=0` to call `argmax` directly rather than divide
by zero.

### Top-k and top-p: truncating the tail, a different operation from temperature

Temperature reshapes the *whole* distribution's sharpness without removing any candidate
from consideration. Top-k and top-p do the opposite kind of thing — they throw away part
of the distribution *before* sampling:

- **Top-k**: keep only the `k` highest-probability tokens, renormalize, sample from those
  alone. Simple, but a fixed `k` is either too permissive (a peaked distribution where
  only 2 tokens are plausible still samples from `k=40`) or too restrictive (a flat
  distribution where 100 tokens are all reasonable still caps at `k=40`).
- **Top-p (nucleus sampling)**: keep the smallest set of highest-probability tokens whose
  cumulative probability exceeds `p`, renormalize, sample from that set. This adapts to
  the shape of the distribution at each step — a confident step keeps a small set, an
  uncertain step keeps a larger one — which is why top-p is generally preferred over a
  fixed top-k alone, and why real systems commonly apply both together (top-k as a hard
  ceiling, top-p as the adaptive cut within it).

These are genuinely different mechanisms from temperature and are commonly composed:
temperature reshapes sharpness, then top-k/top-p truncate the reshaped distribution's
tail, then sampling picks one token from what remains.

### Repetition penalty: a third, independent lever

Downweighting logits for tokens that appeared recently (dividing them by a
`repetition_penalty` factor before the rest of the pipeline runs) discourages the model
from looping on the same phrase — a practical fix for a real failure mode of small models
and greedy-leaning decoding, orthogonal to both temperature and truncation.

## Grounded in This Repo's Code

[`from_scratch/tinystories-gpt-6m/inference.py`](../../from_scratch/tinystories-gpt-6m/inference.py)'s
`sample_next_token` implements temperature, top-k, and top-p in exactly the order
described above:

```python
def sample_next_token(logits, temperature=0.8, top_k=40, top_p=0.9):
    logits = logits / max(temperature, 1e-5)          # temperature: reshape sharpness
    vals, idx = torch.topk(logits, top_k, dim=-1)      # top-k: hard truncation first
    probs = torch.softmax(vals, dim=-1)
    # ... cumulative-sum cutoff at top_p: adaptive truncation within the top-k set
    chosen = torch.multinomial(probs, num_samples=1)  # sample from what remains
```

`generate()` in the same file is the full autoregressive loop, and it is the concrete
reason a KV cache matters: every iteration calls `model(window)` on the **entire**
context-length window again, from scratch — token 1 gets attended over on every single
step of generation after it, recomputed identically each time, because nothing from the
previous step's attention computation is reused. At this repo's scale (a context length
of 256, generations of a few hundred tokens, one request at a time on a laptop CPU/MPS
device) that recomputation is cheap enough to not matter — see
[`from_scratch/tinystories-gpt-6m/docs/SERVING.md`](../../from_scratch/tinystories-gpt-6m/docs/SERVING.md)
for why this project's serving setup is a legitimate choice at this scale, not a shortcut.
[`from_scratch/tinystories-gpt-6m/docs/TEMPERATURE_AND_SAMPLING.md`](../../from_scratch/tinystories-gpt-6m/docs/TEMPERATURE_AND_SAMPLING.md)
and each `custom-gpt` project's [`docs/API_SERVER.md`](../../from_scratch/custom-gpt-153m/docs/API_SERVER.md)
cover this project's own default values and request-field specifics.

## Deep-Dive: What a KV Cache Actually Avoids

`CausalSelfAttention` computes, at every position, an attention output from that
position's query against every earlier position's key and value vectors. Those key/value
vectors depend only on the tokens already generated — not on any token generated
afterward. Recomputing them for the same earlier tokens on every subsequent generation
step (as this repo's `generate()` does) is pure wasted work: the same keys and values for
token 1 get produced identically whether this is decoding step 1 or step 200.

A **KV cache** stores each layer's key and value tensors for every token as it's
generated, so a new generation step only needs to compute attention for the *one new*
token's query against the already-cached keys/values, appending its own new key/value
pair to the cache for future steps. This turns each decoding step's attention cost from
"recompute over the whole sequence so far" into "compute for one new token, reuse the
rest" — the standard technique behind every production LLM-serving engine, and the reason
inference throughput at scale is dominated by memory bandwidth (moving the cache) rather
than raw compute, once a KV cache is in place. It is a serving-engineering technique, not
a change to what the model computes — the output is identical either way, exactly as
[Chapter 25](25_efficient_attention_flash_and_sdpa.md)'s flash-attention/SDPA distinction
is a memory-access optimization, not a mathematical one.

This repo's own `generate()` doesn't implement one — a deliberate, honest scope choice at
this model's size, not an oversight (see "What's out of scope" in
[`SERVING.md`](../../from_scratch/tinystories-gpt-6m/docs/SERVING.md)). The
`platform-lab` repo's `gpu_infrastructure`/serving chapters, which this curriculum's
Part 4 hands off to for production-scale serving, cover KV cache memory budgeting under
real concurrent request load, where it stops being optional.

## Try It Yourself

- Using this chapter's four-token worked example, compute the approximate probabilities
  at `temperature=0.1` and at `temperature=5.0` by reasoning about direction and
  magnitude, without doing exact softmax arithmetic.
- In `inference.py`, generate the same prompt with `--greedy` and with sampling at
  `temperature=0.8` several times — observe that greedy is identical every run while
  sampled output varies, a direct, hands-on confirmation of temperature=0 being a genuine
  limit rather than "low randomness."
- Trace through `generate()`'s loop by hand for 3 generation steps on a short prompt and
  count how many times token 1's key/value vectors are effectively recomputed — that
  count is exactly what a KV cache eliminates.

## Common Misconceptions

- **"Temperature adds randomness the model didn't already have an opinion about."** It
  reshapes probabilities the model already computed; a token the model considers
  near-impossible stays near-impossible at any reasonable temperature.
- **"Temperature and top-k/top-p do the same thing."** Temperature reshapes the whole
  distribution's sharpness; top-k/top-p truncate it, discarding the tail entirely before
  sampling — related but genuinely different mechanisms, commonly used together.
- **"A KV cache changes what the model outputs."** It doesn't — same attention math, same
  output, computed without redundant work. Exactly the same "systems optimization, not an
  algorithmic change" distinction as flash attention/SDPA.

## Practice Questions

1. Why does `temperature=0` require special-casing to `argmax` rather than just letting
   `logit / 0` flow through the normal division-then-softmax path?
2. A prompt generates coherent text at `temperature=0.7` but becomes word salad at
   `temperature=1.5`. Describe what's happening to the underlying probability distribution
   between those two settings.
3. Explain, in your own words, why a KV cache's benefit grows with sequence length and
   number of concurrent requests, and why this repo's own inference setup doesn't need one
   to be usable.

## Key Terms

- **Temperature**: a pre-softmax rescaling of logits that sharpens (low) or flattens
  (high) the resulting probability distribution without changing token ranking.
- **Top-k / top-p (nucleus) sampling**: truncating the candidate set to the `k` highest-
  probability tokens, or to the smallest set whose cumulative probability exceeds `p`,
  before sampling.
- **Repetition penalty**: downweighting logits for recently-generated tokens to discourage
  looping.
- **KV cache**: storing each layer's key/value vectors for already-generated tokens so
  each new decoding step only computes attention for the new token, instead of
  recomputing the whole sequence.
