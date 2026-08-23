# Scaling Laws: Why Bigger Models, and When They Stop Helping

Part of the [LLM Engineering Curriculum](00_roadmap.md), Chapter 14. Builds on
[Chapter 12](12_the_pretraining_objective_and_why_data_dominates.md)'s argument that the
corpus, not the architecture, sets the ceiling on what a model can learn, and on
[Chapter 10](10_transformer_architecture.md)/[Chapter 11](11_positional_encoding_variants_rope_and_beyond.md)/
[Chapter 35](35_normalization_and_mlp_variants_rmsnorm_and_swiglu.md)'s coverage of what
each architectural piece actually is. This chapter is the piece that ties them together
into an answer to a very practical question: **given a parameter budget, how do you
actually decide `context_length`, `embed_size`, `num_heads`, and `num_layers` — and how
much training data does that architecture actually need?** Not folklore, not "bigger is
always better" — the real formulas this repo's own `ModelConfig.param_count()` computes,
applied as a decision procedure.

## In Plain English

Think of building a model like hiring a team for a research project. You could hire one
genius who works alone (very "wide," very few "layers" of collaboration), or a hundred
mediocre people passing work down an assembly line (very "deep," very little individual
capability at each step) — for the same total headcount budget. Neither extreme is
obviously right; the real question is what kind of problem you're solving. And separately
from team size: a brilliant team given only one afternoon to read background material
will underperform a smaller team given a month to actually study — headcount and study
time both matter, and there's a right *ratio* between them, not an independent "more is
always better" for either one alone.

## The First-Principles Explanation

### Two separate scaling questions, not one

"Scaling a model" is really two independent decisions that get conflated in casual
conversation:

1. **Model size** — how many parameters. This is an *architecture* decision:
   `context_length`, `embed_size`, `num_heads`, `num_layers` (and, for the SwiGLU family,
   `ffn_hidden` — [Chapter 35](35_normalization_and_mlp_variants_rmsnorm_and_swiglu.md)).
2. **Data size** — how many tokens the model actually trains on. This is a *training
   budget* decision: `steps × batch_size × context_length` ([Chapter 13](13_the_training_loop_mechanism_by_mechanism.md)).

The historical mistake this field made and then corrected (see "Deep-Dive" below) was
scaling #1 aggressively while under-scaling #2 — training ever-bigger models on
roughly the same amount of data. The correction, known as **Chinchilla scaling**
(Hoffmann et al., 2022), is the finding that these two numbers need to grow **together**,
in a specific ratio, for a fixed compute budget to produce the best possible model.

### Why width and depth are not interchangeable levers

This repo's own `ModelConfig.param_count()` (identical formula-shape across every
`from_scratch` project) makes the actual cost of each lever exact, not approximate. For
the learned-position family (`custom-gpt-10m/50m/153m`):

```
per_block = 12·E² + 13·E
total     = V·E + C·E + layers·(12·E² + 13·E) + 2·E
```

Two things to notice immediately: **`embed_size` (E) enters quadratically** (`E²`, from
attention's Q/K/V/output projections and the MLP's `E×4E` matrices), while **`num_layers`
enters linearly** (`layers ×`, a straight multiplier on a fixed per-block cost). This is
verifiable, not folklore — comparing this repo's own `10m` (`E=160`) and `30m` (`E=384`)
presets, both at `num_layers=6`:

```
per_block(160) = 12·160² + 13·160 =   309,280
per_block(384) = 12·384² + 13·384 = 1,774,464
                                     ---------
                       ratio ≈ 5.74x     (embed_size only grew 2.4x)
```

Doubling `num_layers` roughly doubles the block-parameter total and roughly doubles
per-step compute (one more sequential pass). Doubling `embed_size` roughly **quadruples**
the block-parameter total (`per_block`'s `E²` term) — a fundamentally steeper cost curve
for the same nominal "make it bigger" instinct. This is the first concrete answer to "how
many and what layers": **depth is the cheap, roughly-linear lever; width is the
expensive, quadratic one** — they are not interchangeable ways of spending the same
budget.

### The embedding-vs-block crossover point

At small `embed_size`, the **token embedding** (`V × E`, linear in `E`) can dominate the
total parameter count more than the transformer blocks do — a model that's mostly a
lookup table, not much of a reasoning engine. The GPT-2-style family's own
`MODEL_SIZING_GUIDE.md` solves for exactly where this flips, at its GPT-2 tokenizer's
`vocab_size=50,257` and (using `num_layers=6`, the `10m` preset's own depth, as the
reference point):

```
V·E = layers·(12E² + 13E)   →   E ≈ 697
```

Below `E≈697`, the embedding table dominates (at the `10m` preset's actual `E=160`, it's
80.6% of the total — per that project's own README); above it, block parameters take over
and every added unit of `embed_size` is buying real transformer capacity, not
lookup-table bulk. This crossover point moves with `vocab_size` and `num_layers` — it is
not a universal constant, but the *method* (set the two formulas equal, solve for `E`) is
a real, reusable piece of the decision procedure.

**The RoPE family needs the same method applied to its own, different formula** — worth
being explicit about, since the two architectures' costs aren't shaped alike. RoPE has no
`pos_emb` term, and its per-block cost is `4E² + 3Ef + 2E` where `f` (`ffn_hidden`) is
conventionally set to `≈(8/3)E` ([Chapter 35](35_normalization_and_mlp_variants_rmsnorm_and_swiglu.md)),
which simplifies the block cost to `≈12E² + 2E` — coincidentally close in shape to the
GPT-2 family's `12E² + 13E`, but not identical, and driven by a genuinely different vocab
size. Solving it for `custom-gpt-200m`'s own real numbers (`vocab_size=32,768`,
`num_layers=18`) gives `E ≈ 152` — a much lower crossover point than the GPT-2 family's
697, almost entirely because RoPE's smaller, digit-aware vocabulary
([Chapter 9](09_tokenization.md)) costs far less per unit of `E` to embed. `custom-gpt-200m`'s
actual `E=896` sits comfortably past its own crossover point either way — the two
families just cross over at very different widths, which is exactly why "is 896 wide
enough to be block-dominated" needs the *right* formula plugged in, not either family's
number borrowed on faith.

### `num_heads`: free in parameter count, not free in behavior

Changing `num_heads` while holding `embed_size` fixed costs **zero** extra parameters —
the Q/K/V/output projection matrices are still `embed_size`-shaped regardless of how many
ways that dimension gets split for the attention computation. What changes is
`head_dim = embed_size / num_heads`: more heads means more, narrower attention patterns
per layer; fewer heads means fewer, wider ones. `MODEL_SIZING_GUIDE.md` also flags a real,
measured hardware consequence, not just a theory concern: very small or non-power-of-two
`head_dim` values can fall outside what fused/flash-attention-eligible kernels
([Chapter 25](25_efficient_attention_flash_and_sdpa.md)) are tuned for, which is why this
repo's own presets favor `head_dim` in the `32`–`128` range (e.g. `head_dim=64` across
`30m`/`153m`/`200m`), with the `10m` preset's `head_dim=20` called out explicitly as an
exception tuned for small total size, not clean hardware alignment.

### The data side: the Chinchilla ratio, in this repo's own numbers

Once an architecture (and therefore a real parameter count) is fixed, the training-budget
question is: how many tokens should this specific model actually see? Chinchilla's
finding, in simplified form: **roughly 20 tokens per parameter** is compute-optimal — a
model trained on meaningfully less than that is *undertrained* (it would have learned more
from the same compute spent on more tokens instead of a bigger model), and one trained on
meaningfully more is past the point where more data on this exact architecture is the best
use of additional compute. This repo's own `TrainConfig` docstrings compute this ratio
directly, for real configured runs, not as an afterthought:

```
custom-gpt-153m:  tokens = 150,000 × 16 × 1024 = 2.46B   →  ~16 tokens/param  (near-optimal)
custom-gpt-200m:  tokens = 150,000 × 16 × 2048 = 4.92B   →  ~24 tokens/param  (a bit past optimal)
custom-gpt-350m:  tokens = 150,000 × 16 × 2048 = 4.92B   →  ~14 tokens/param  (under-optimal —
                                                              steps copied from 200m, not retuned)
```

`custom-gpt-350m`'s under-training isn't a hypothetical example — it's a real, currently
unresolved gap in this repo's own configuration, flagged in that project's own `config.py`
comment as something to fix (raise `steps` toward ~215,000) before a real training run,
not a mistake to imitate.

### Worked example: what does $100 on L4 GPUs actually buy?

Turning the ratio above into a real answer means converting a dollar budget into a FLOP
budget, then solving `C ≈ 6ND` (with `D ≈ 20N`, so `C ≈ 120N²`) for `N`.

**Budget → GPU-hours.** L4 pricing varies by provider and spot-vs-on-demand: roughly
$0.22/hr (GCP spot) to $0.80/hr (typical on-demand). $100 buys 125–455 GPU-hours
depending which.

**GPU-hours → compute.** L4's peak dense BF16 is 242 TFLOPS, but real training throughput
lands at 20–40% Model FLOPs Utilization (MFU) — L4 is a 300GB/s-memory-bandwidth card
(roughly 1/7th an A100's), and training is memory-bandwidth-bound per FLOP, not just
FLOP-bound, so peak specs are never realistic.

**Compute → N.** Solving `N = sqrt(C / 120)` across that price/MFU range lands
consistently in the **400M–1B parameter** range — roughly 500–700M as the realistic
planning number (on-demand pricing, first-attempt MFU), ~1B as an optimistic ceiling
(aggressive spot pricing, well-tuned run).

**The single-GPU sanity check this chapter's own memory math makes possible.** Does a
500M–1B model even need [Chapter 26](26_distributed_training_ddp_and_fsdp.md)'s multi-GPU
strategies? At 16 bytes/param (mixed-precision AdamW), that is 8–16GB of static state —
it fits on *one* 24GB L4. The multi-GPU discussion only becomes load-bearing past roughly
1.2B params on a single L4 (16GB+ static, no room left for activations). A $100 budget on
this hardware doesn't reach the regime where sharding is *necessary* — it reaches the
regime where sharding is merely *available* — and splitting across multiple L4s (no
NVLink, PCIe-only interconnect) would eat into the already-modest MFU further, making one
well-utilized L4 the better $/model choice here, not a multi-GPU setup.

**Why Chinchilla-optimal isn't the only valid target.** The `N` above assumes the whole
budget funds one clean, compute-optimal run. Real small-model releases routinely ignore
the 20:1 ratio on purpose: SmolLM2 (~1.7B params) was trained on roughly 11 trillion
tokens — **over 6,000 tokens/parameter**, not 20. The reason: training cost is paid once,
inference cost is paid forever, so a model meant to be *served* a lot is worth
over-training well past compute-optimal in exchange for a smaller, cheaper-to-run final
size. "The biggest model my $100 can Chinchilla-optimally train" and "the best model my
$100 lets me keep querying afterward" are different optimization targets that land on
different `N` — this chapter's ratio is a starting point for that decision, not a verdict.

**What that size model can actually do, and why context length is a separate question
from parameter count.** Real ~1B-class open models (Llama-3.2-1B, SmolLM2) land around
45–57% on instruction-following/grade-school-math benchmarks and meaningfully lower on
harder reasoning — competent at narrow, well-scoped tasks (instruction-following,
summarization, context-grounded QA), weak at multi-step reasoning and broad factual
recall, a hard function of how many facts 1B parameters can physically store, not a
training-recipe fix. Context length, separately, is an architecture choice, not a
consequence of parameter count — see [Chapter 21](21_inference_mechanics_decoding_sampling_and_kv_cache.md)
for the KV-cache mechanism a long context depends on at serving time. Grouped-query
attention (fewer KV heads than query heads) is what makes a long context's KV-cache
memory affordable at all; this repo's own `custom-gpt-word` project's hand-written
attention (full multi-head, an explicit `(B,H,T,T)` score matrix, no GQA) is a concrete,
small-scale illustration of the *un*-optimized version of both problems — no GQA to
shrink the KV-cache side, and `O(T²)` memory in the attention computation itself, which is
exactly why that project's own `block_size=12` is a sane choice and not an arbitrary small
number.

## Grounded in This Repo's Code

Every number above comes from a real, callable function, not a spreadsheet alongside the
code:

```python
# custom-gpt-153m/src/gpt/config.py — ModelConfig.param_count()
def param_count(self) -> int:
    e, c, v, layers = self.embed_size, self.context_length, self.vocab_size, self.num_layers
    token_emb = v * e
    pos_emb = c * e
    per_block = 12 * e * e + 13 * e
    final_ln = 2 * e
    return token_emb + pos_emb + layers * per_block + final_ln
```

`make config`/`gpt-config` in any `from_scratch` project prints this exact number, plus a
full `param_breakdown()` (token embedding / position embedding / transformer blocks /
final norm, each with a percentage share) — the crossover-point argument above is directly
checkable against real output, not just the formula. `make presets`/`gpt-config --list`
prints every named preset's exact parameter count side by side, computed from the same
formula — the actual `10m`/`30m`/`50m`/`153m`/`200m`/`350m` progression this repo uses is
sized this way, not by trial and error.

`TrainConfig`'s own class docstring, in every project, computes the token budget and its
Chinchilla ratio explicitly as a code comment — not a separate document that can drift out
of sync with the actual configured `steps`/`batch_size`/`context_length` values.

## Deep-Dive: Why It's Built This Way

**A short, honest history, since "scaling laws" gets invoked loosely**: Kaplan et al.
(2020) first showed loss follows a predictable power-law curve as compute, parameters, and
data each scale up — but under-weighted how *data* specifically should scale alongside
parameters, which led much of the field toward "train ever-bigger models on roughly fixed
data" for a couple of years. Hoffmann et al.'s 2022 Chinchilla paper trained many
models at different size/data combinations for the *same* compute budget and found the
earlier prescription was leaving real performance on the table — a properly-data-matched
70B model (Chinchilla itself) beat a 280B model (Gopher) trained on the same compute but
proportionally less data. The ~20-tokens-per-parameter figure comes from that correction,
not the original scaling-laws paper.

**Why this repo's own models sit nowhere near the regime where these laws were actually
measured, and that's fine**: Chinchilla's own experiments spanned roughly 70M to 16B
parameters, trained on up to hundreds of billions of tokens. This repo's models
(`5.85M`–`350M` parameters) are toy-scale by comparison — the *ratio* (~20 tokens/param)
is still a reasonable rule of thumb to aim for, and this repo's own `TrainConfig`
docstrings do exactly that, but treat it as a useful heuristic transplanted from a
different regime, not a law proven to hold with the same precision at this scale.

**Why depth being "the cheap lever" doesn't mean maximize depth** — `MODEL_SIZING_GUIDE.md`
names a real, opposite-direction limitation at this repo's small scale specifically:
shrinking `num_layers` below the `10m` preset "barely moves" total parameter count,
*because* at `E=160` the embedding table so thoroughly dominates (80.6%) that `num_layers`
changes mostly move training *speed* (more/fewer sequential passes), not model size. The
general "depth is linear, width is quadratic" cost relationship is always true; whether
changing depth is *worth doing* for a given goal (size vs. speed vs. capability) depends on
which side of the embedding-crossover point (above) the model actually sits on.

**Diminishing returns and the data ceiling** — even a perfectly Chinchilla-sized model is
still bounded by [Chapter 12](12_the_pretraining_objective_and_why_data_dominates.md)'s
argument: scaling laws describe how loss improves *given* a certain corpus's quality and
diversity; they don't promise that corpus itself is good enough for the model to become
genuinely capable, only that the compute was spent efficiently against whatever loss floor
that corpus permits. "Scaling laws stop helping" in the literal sense once you're past the
compute-optimal ratio for a fixed compute budget (more of either lever alone starts buying
less than a balanced increase of both would); they stop helping in a *deeper* sense the
moment data quality, not compute allocation, becomes the actual bottleneck.

## Try It Yourself

- **Reproduce the `10m` vs `30m` ratio yourself.** Run `GPT_EMBED_SIZE=160 make config` and
  `GPT_EMBED_SIZE=384 make config` in `custom-gpt-153m` (holding `num_layers` fixed at its
  default), read off the `transformer_blocks` line from each `param_breakdown()`, and
  confirm the ~5.74x ratio by hand.
- **Verify the two families really do cross over at different widths — and that
  `ffn_hidden` has to scale with `embed_size` for the RoPE family's test to mean anything.**
  Reproduce this chapter's `E≈697` (GPT-2-style, `vocab=50,257`, `num_layers=6`) and
  `E≈152` (RoPE, `vocab=32,768`, `num_layers=18`) by hand from each family's own formula.
  Then check them against real output: `GPT_EMBED_SIZE=700 make config` in
  `custom-gpt-10m` (not `153m` — `10m` is the project this `E≈697` figure's `num_layers=6`
  actually matches) lands `token_embedding` around 42% and `transformer_blocks` around
  57% — close, not exact, since the two-term crossover formula omits `10m`'s smaller
  `position_embedding` term entirely. `GPT_EMBED_SIZE=150 GPT_FFN_HIDDEN=400 make config`
  in `custom-gpt-200m` (note: **both** overrides — `GPT_EMBED_SIZE` alone leaves
  `ffn_hidden` stuck at its preset default, silently breaking the `f≈(8/3)E` assumption
  the crossover formula depends on) lands almost exactly 50/50 — RoPE's formula has no
  `pos_emb` term to omit, so it tracks the two-term prediction more tightly.
- **Audit a real run's Chinchilla ratio yourself.** Pick any `from_scratch` project, read
  its `TrainConfig.steps`/`batch_size`/`context_length` defaults, compute
  `tokens = steps × batch_size × context_length`, divide by `ModelConfig().param_count()`,
  and compare your ratio against what that project's own docstring claims — `custom-gpt-350m`
  is a real, currently-uncorrected example of a ratio worth recomputing.

## Common Misconceptions

- **"Twice the layers means twice the model."** Only approximately true, and only once
  you're past the embedding-crossover point — below it, doubling layers barely moves total
  size at all, because the embedding table (unaffected by `num_layers`) still dominates.
- **"A bigger model is always better, given enough training."** Directly contradicted by
  Chinchilla's own headline result: a smaller, properly data-matched model beat a larger,
  under-trained one at the *same* compute budget. Parameter count alone is not the
  right thing to compare.
- **"Scaling laws mean I should always max out depth for a fixed parameter budget."**
  False — depth being the linear (cheap) lever doesn't mean unlimited depth is free of
  trade-offs; very deep, very narrow architectures can be harder to train and slower
  per-step for the same total parameter count as a shallower, wider one, and this repo's
  own `head_dim` guidance (32–128) exists precisely because extreme splits in the other
  direction (very many heads, very narrow `head_dim`) cause real, separate problems.
- **"The ~20 tokens/parameter ratio is a hard rule this repo's models must hit exactly."**
  It's a compute-optimal *target* under Chinchilla's specific experimental regime, applied
  here as a reasonable heuristic at a much smaller scale — this repo's own
  `custom-gpt-153m` (~16:1) and `custom-gpt-200m` (~24:1) both sit close-but-not-exactly on
  it by design choice, not because the ratio is a strict pass/fail threshold.

## Practice Questions

1. Given a fixed parameter budget, why does the embedding-crossover point mean the "right"
   split between `embed_size` and `num_layers` depends on `vocab_size` — what would
   change about the crossover `E` if `vocab_size` were doubled, holding everything else
   fixed?
2. `num_heads` doesn't change parameter count at all, holding `embed_size` fixed. Given
   that, what's actually being decided when you choose `num_heads=8` vs `num_heads=16` at
   the same `embed_size`?
3. `custom-gpt-350m`'s token budget currently sits at ~14 tokens/parameter, copied
   unchanged from `custom-gpt-200m`'s `steps=150_000`. Using this chapter's ratio
   reasoning, roughly what `steps` value would bring it to ~20:1, and what does leaving it
   at 14:1 actually cost in practice (not just "the ratio is wrong")?
4. Chinchilla's own experiments were run at 70M–16B parameters. What, specifically, makes
   it reasonable to still use its ~20:1 ratio as a *heuristic* for a 5.85M-parameter model,
   and what would make you doubt the ratio still applies well at that much smaller scale?
5. A $100 GPU budget solves out to a ~700M-parameter Chinchilla-optimal model. Using the
   16-bytes/param memory math from [Chapter 26](26_distributed_training_ddp_and_fsdp.md),
   determine whether that model requires FSDP/multi-GPU sharding to train on a single
   24GB GPU, and explain why "the model doesn't fit" is a different question from
   "training would be faster with more GPUs."

## Key Terms

- **Scaling laws**: empirically-observed, roughly power-law relationships between a
  model's loss and its parameter count, training data volume, and compute budget.
- **Chinchilla scaling (compute-optimal training)**: the corrected finding (Hoffmann et
  al., 2022) that model size and training-data volume must grow together, in roughly a
  fixed ratio (~20 tokens/parameter), for a given compute budget to produce the best
  possible model — as opposed to scaling model size while under-scaling data.
- **Embedding-vs-block crossover point**: the `embed_size` at which a model's token
  embedding and its transformer blocks contribute equal parameter shares — below it, the
  model is embedding-dominated; above it, block parameters (real transformer capacity)
  dominate.
- **Width (in this context)**: `embed_size` — the quadratic-cost lever
  (`per_block ∝ E²`).
- **Depth (in this context)**: `num_layers` — the linear-cost lever
  (`total ∝ layers × per_block`).
- **`head_dim`**: `embed_size / num_heads` — free in parameter count, but a real lever on
  attention expressiveness and on which fused/flash-attention kernels are eligible to run
  (see [Chapter 25](25_efficient_attention_flash_and_sdpa.md)).
- **Tokens-per-parameter ratio**: total training tokens (`steps × batch_size ×
  context_length`) divided by a model's parameter count — the practical, code-computable
  form of the Chinchilla ratio used throughout this repo's own `TrainConfig` docstrings.
- **Model FLOPs Utilization (MFU)**: the fraction of a GPU's peak theoretical FLOPS
  actually achieved during real training — typically 20-50%, since training is often
  memory-bandwidth-bound rather than purely compute-bound, which is why a compute budget
  calculated from peak specs alone overstates what's actually achievable.
