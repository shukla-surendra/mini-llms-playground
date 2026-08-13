# Temperature: What It Is, and Exactly How It Reshapes the Output

Companion to [`SERVING.md`](SERVING.md) and
[`../../../docs/llm-engineering/08_what_is_a_language_model.md`](../../../docs/llm-engineering/08_what_is_a_language_model.md#how-a-token-actually-gets-chosen-greedy-vs-sampling)
(the curriculum's general treatment). This doc goes one level deeper specifically on
temperature — the actual math, a worked numerical example, and what it looks like in
this project's own [`inference.py`](../inference.py).

## In Plain English

Temperature controls how "confident" the model's next-token choice is allowed to be
before a token gets picked. Low temperature makes the model stick close to its single
most-likely guess every time (safe, but repetitive). High temperature flattens things out,
giving less-likely tokens a real chance of being picked (more varied, but riskier —
crank it too high and the output stops making sense at all). It doesn't change *what the
model knows* — it changes how that knowledge gets turned into a specific choice.

## The First-Principles Explanation

### Where temperature actually sits in the pipeline

Recall the generation loop from
[`../../../docs/llm-engineering/02_what_is_a_language_model.md`](../../../docs/llm-engineering/08_what_is_a_language_model.md):
the model outputs **logits** (raw, unnormalized scores, one per vocabulary token), and
**softmax** turns those into a probability distribution. Temperature is applied
**between** these two steps — it rescales the logits *before* softmax runs:

```
raw logits  →  divide by temperature  →  softmax  →  probability distribution  →  sample
```

### The exact math

```
scaled_logit_i = logit_i / temperature

probability_i = exp(scaled_logit_i) / sum(exp(scaled_logit_j) for all j in vocabulary)
```

Dividing every logit by the same number before softmax changes how *spread out* the
resulting probabilities are, without changing their *relative order* — the token that had
the highest logit still has the highest probability at any temperature. What changes is
the **gap** between the top choice and everything else.

## A Worked Numerical Example

Suppose, at some position, the model outputs these raw logits for just 4 candidate
tokens (a real vocabulary has 4,096 in this project, but the mechanism is identical —
this is a toy slice to make the arithmetic followable by hand):

```
Token:      "the"    "a"     "dog"    "purple"
Raw logit:   4.0      3.0     1.0       0.5
```

**At `temperature = 1.0`** (no rescaling — logits pass through unchanged):
```
softmax([4.0, 3.0, 1.0, 0.5]) ≈ [0.62, 0.23, 0.031, 0.019]  (renormalized over just these 4)
```
"the" is clearly favored, but "a" still has a real ~23% chance.

**At `temperature = 0.5`** (divide every logit by 0.5, i.e. double them):
```
scaled logits: [8.0, 6.0, 2.0, 1.0]
softmax(...) ≈ [0.87, 0.12, 0.002, 0.001]
```
"the" now dominates even more heavily — the gap between the top choice and the rest
*widened*. Lower temperature → sharper, more confident, more deterministic-leaning
distribution.

**At `temperature = 2.0`** (divide every logit by 2.0, i.e. halve them):
```
scaled logits: [2.0, 1.5, 0.5, 0.25]
softmax(...) ≈ [0.42, 0.26, 0.10, 0.08]
```
The gap between "the" and the others *shrank* substantially — "dog" and even "purple"
(a much less likely word here) now have a real, meaningfully higher chance of being
picked than at temperature 1.0. Higher temperature → flatter, more varied, more
"willing to take a risk" distribution.

**The key insight this example makes concrete**: temperature doesn't add randomness from
nowhere — it redistributes probability *mass* the model already computed, making the
existing distribution sharper or flatter. A token the model assigned essentially zero
probability to will still have essentially zero probability at any reasonable
temperature; temperature reshapes the *shape* of the distribution, it doesn't override
what the model actually learned.

## Grounded in This Project's Code

[`../inference.py`](../inference.py)'s `sample_next_token`:

```python
def sample_next_token(logits, temperature=0.8, top_k=40, top_p=0.9):
    logits = logits / max(temperature, 1e-5)   # <- exactly the division from the math above
    ...
    probs = torch.softmax(vals, dim=-1)         # <- softmax runs AFTER the temperature scaling
```

The `max(temperature, 1e-5)` guards against a literal divide-by-zero if `temperature=0`
were passed through this path — but semantically, `temperature=0` should just mean
"always pick the single most likely token," which is exactly what `--greedy` /
`do_sample=False` already does directly via `torch.argmax`, without touching this
function at all. See the next section for why these two are mathematically the same
thing in the limit.

## Deep-Dive: Why Temperature → 0 Is the Same as Greedy Decoding

As temperature approaches 0, dividing logits by it makes the scaled logits approach
±infinity — the gap between the top logit and every other logit becomes enormous. Softmax
of a distribution with one value approaching +infinity and the rest approaching relatively
−infinity collapses to assign probability ~1.0 to the single highest-logit token and ~0
to everything else. That's *exactly* what `argmax` (greedy decoding) does directly, just
reached by a different mechanical path. This is why `temperature=0.0` and `--greedy` are
described as equivalent in
[`../../../docs/llm-engineering/08_what_is_a_language_model.md`'s misconceptions
section](../../../docs/llm-engineering/08_what_is_a_language_model.md#common-misconceptions) —
not an approximation, a genuine mathematical limit.

## Practical Guidance for This Project's `/generate` Endpoint

| Temperature | Effect | When to use it |
|---|---|---|
| 0.0 (or `do_sample: false`) | Fully deterministic, same output every time | Debugging — confirming the model itself, not sampling randomness, produced a specific output |
| 0.5 - 0.7 | Safe, coherent, somewhat repetitive | When coherence matters more than variety |
| 0.8 (this project's default) | A balance — real variety, usually still coherent | General use |
| 1.0 - 1.3 | Noticeably more varied, occasional odd word choices | Exploring different completions of the same prompt |
| \> 1.5 | Frequently incoherent | Rarely useful — mostly demonstrates what "too high" looks like |

Since this model is small (~5.85M parameters) and trained on a narrow, simple dataset
(per [`DATASET_AND_TOKENIZER.md`](DATASET_AND_TOKENIZER.md)), it has less "headroom" than
a large model before high temperature breaks coherence — worth expecting the useful range
here to be somewhat narrower than what you'd see quoted for a large frontier model.

## Common Misconceptions

- **"Temperature adds randomness the model didn't already have an opinion about."** No —
  as the worked example shows, it reshapes probabilities the model already computed; a
  token the model considers near-impossible stays near-impossible regardless of
  temperature.
- **"Higher temperature makes the model 'more creative' in a meaningful sense."** It makes
  output more *varied*, which is a different claim — variety isn't the same as quality or
  genuine creativity; at high enough temperature, "varied" just becomes "wrong."
- **"Temperature and top-k/top-p do the same thing."** Temperature reshapes the *whole*
  distribution's sharpness; top-k/top-p *truncate* it (throw away the unlikely tail
  entirely before sampling from what's left) — related tools, genuinely different
  mechanisms, and commonly used together (as this project's defaults do).

## Practice Questions

1. Using this doc's 4-token example, what would the approximate probabilities be at
   `temperature = 0.1`? (You don't need exact softmax arithmetic — reason about the
   direction and magnitude of the effect.)
2. Why does `temperature = 0` require special-casing (via `argmax`/greedy) rather than
   just passing `temperature=0` through the normal division-then-softmax code path?
3. A prompt generates coherent text at `temperature=0.7` but becomes word-salad at
   `temperature=1.5`. Explain what's happening to the underlying probability distribution
   between those two settings.
