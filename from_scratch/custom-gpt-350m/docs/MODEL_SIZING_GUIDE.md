# Model Sizing Guide: Every `ModelConfig` Field, What It Actually Costs, and How to Pick It

`src/gpt/config.py` declares every architecture knob in one place:

```python
# GPT-2 BPE via tiktoken. Declared here so parameter counts can be computed without
# loading the tokenizer; verified against the real tokenizer at training time.
TOKENIZER_NAME = "gpt2"
VOCAB_SIZE = 50257


@dataclass(frozen=True)
class ModelConfig:
    """Architecture. `param_count()` is exact — it mirrors model.py's actual layers."""

    context_length: int = 512
    embed_size: int = 160
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE
```

[`docs/CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md#configpy--every-knob-computed-not-hardcoded)
covers *why* this is a frozen dataclass with a computed `param_count()`. This doc is the
complementary piece: what each individual field actually controls, what increasing it
costs (in real parameter/compute numbers from this project's own formula, not estimates),
its hard limitations, and what value fits which use case.

## `TOKENIZER_NAME` / `VOCAB_SIZE` — not really a tunable knob

**What it is**: which tokenizer this model is built for (GPT-2 BPE, via `tiktoken`) and
its exact vocabulary size, declared as plain constants rather than obtained by loading
the tokenizer and reading `.n_vocab`.

**Why constants instead of loading the tokenizer**: per the comment, `param_count()` (and
everything that calls it — `make config`, `make presets`, the checkpoint-compatibility
check) needs to work without paying the cost of loading `tiktoken`'s merge table just to
print a number. The trade-off this creates — a hardcoded number that *could* silently
drift from the real tokenizer — is closed at training time instead:
`training/trainer.py` explicitly checks `tokenizer.n_vocab != model_cfg.vocab_size` and
raises before training starts, so a mismatch fails loudly at the one point it would
actually corrupt something, not silently.

**Why this isn't really "tunable" the way the fields below are**: `vocab_size` must equal
whatever tokenizer is actually in use — setting it to a different number without
switching tokenizers doesn't shrink the model in any meaningful sense, it just makes the
embedding table the wrong size for the token IDs the tokenizer actually produces, which
`is_compatible()`/the startup check will refuse. The only real way to change this number
is to train and switch to a genuinely different tokenizer — a one-way door, covered in
[`DATA_PREP_GUIDELINE.md`](DATA_PREP_GUIDELINE.md#3-a-domain-fit-tokenizer-not-gpt-2s-off-the-shelf-one)
and [`DATA_PREP_GUIDELINE.md`'s closing section](DATA_PREP_GUIDELINE.md#the-one-way-door-worth-naming-explicitly).

**Impact of vocab size on the model that *is* fixed**: it linearly scales both
`token_embedding` (`vocab_size × embed_size`) and the tied `lm_head` (free, since weights
are shared — see `model.py`'s weight-tying, covered in `CODE_WALKTHROUGH.md`), and it sets
how many logits get computed and cross-entropy'd over at every single token position
during training and generation. At GPT-2's 50,257-token vocabulary, this is *the* reason a
small model's parameter budget is embedding-dominated — see the `embed_size` section below
for exactly how much.

## `context_length` (default `512`) — how many tokens the model can see at once

**What it controls**: the size of the learned position-embedding table
(`pos_emb = nn.Embedding(context_length, embed_size)`) and, mechanically, the longest
sequence the model can process in one forward pass at all — `TinyGPT.forward()` builds
`pos = torch.arange(seq_len)` and indexes `pos_emb` with it, so a sequence longer than
`context_length` has no position embedding to look up and cannot be run through the model.

**Impact of increasing it**:
- *Parameter cost*: `context_length × embed_size` — linear, and usually small. At the
  `10m` preset, `512 × 160 = 81,920` parameters — 0.8% of the total (per `make config`'s
  own breakdown). Doubling `context_length` doubles a very small slice of the budget.
- *Compute cost*: quadratic, not linear — attention computes a score for every
  token-pair in the window (`seq_len × seq_len`), so doubling `context_length`
  roughly **quadruples** the attention computation's cost per forward/backward pass
  (mitigated but not eliminated by the `sdpa` kernel — see
  [Chapter 25](../../../docs/llm-engineering/25_efficient_attention_flash_and_sdpa.md) —
  the underlying score matrix is still `seq_len²`-shaped regardless of kernel).
- *Coverage per training step*: a larger window means each `get_batch()` sample covers
  more of a document's actual context — useful when source material has long-range
  structure worth learning (a book chapter, a full function definition), not just
  short independent exchanges.

**Real, hard limitation**: position embeddings here are **learned and absolute** — a
plain lookup table, not a relative/rotary scheme (RoPE and similar are curriculum
territory: [Chapter 11 — Positional Encoding Variants, planned](../../../docs/llm-engineering/00_roadmap.md#part-1--foundations-llm-history--architecture)).
Concretely, this means **a model trained at `context_length=512` cannot later be resumed
or fine-tuned at a longer context** — position index 600 simply has no learned embedding
row; there's nothing to look up. Changing `context_length` after training has started is
architecturally the same kind of one-way door as changing `embed_size`/`num_heads`/
`num_layers` — `is_compatible()` in `checkpoint.py` refuses the mismatched resume outright
rather than guessing.

**What value for what usage**:
| Use case | Suggested `context_length` |
|---|---|
| Fast laptop CPU/MPS iteration, short chat-turn-shaped text (this project's default corpus) | `512` |
| Source material with longer natural units (book chapters, long-form docs) | `1024` (the `50m`/`153m` presets already use this) |
| Real GPU-scale training | `1024`+, budget permitting — but every doubling costs ~4x the attention compute, so this is a real budget decision, not a free quality knob |

## `embed_size` (default `160`) — the single biggest lever on model size

**What it controls**: the width of every token's vector representation — the dimension
flowing through the entire network (`token_emb`, `pos_emb`, every attention/MLP
projection, the residual stream). Every other per-layer dimension in `model.py` is
derived from it.

**Why it dominates parameter count more than any other field**: from `config.py`'s own
exact formula, `per_block = 12·E² + 13·E` — the transformer-block cost grows
**quadratically** with `embed_size`, not linearly, because attention's QKV/output
projections and the MLP's 4x-widen/narrow projections are all `E×E`-or-`E×4E`-shaped
matrices. This is verifiable, not an estimate — comparing this project's own `10m`
(`E=160`) and `30m` (`E=384`) presets, both at `num_layers=6`:

```
per_block(160) = 12·160² + 13·160 =   309,280
per_block(384) = 12·384² + 13·384 = 1,774,464
                                     ---------
                        ratio ≈ 5.74x   (embed_size only grew 2.4x)
```

`6 × 309,280 = 1,855,680` — exactly matches the `10m` preset's real
`transformer_blocks` count from `make config`'s breakdown. Squaring the growth is the
concrete, checkable reason `embed_size` moves total parameter count far faster than
`num_layers` does for the same-sized change.

**The crossover point worth knowing**: `token_embedding` (`vocab_size × E`) grows only
*linearly* in `E`, while `blocks` grows *quadratically* — so which one dominates depends
on `E`. Solving `vocab_size·E = layers·(12E² + 13E)` for this project's fixed
`vocab_size=50,257` at `num_layers=6` gives **E ≈ 697**. Below that, like the `10m`
preset's `E=160`, the token embedding dominates (80.6% of the total, per README) and
increasing `embed_size` mostly buys "a bigger lookup table," not more reasoning capacity.
Above it, block parameters take over and every extra unit of `embed_size` is genuinely
adding transformer capacity, not embedding-table bulk — which is exactly why the `153m`
preset (`E=768`) sits just past that crossover.

**Hard constraint**: `embed_size % num_heads` must be `0` (`__post_init__` validates this
and raises with an exact explanation) — `embed_size` and `num_heads` aren't independent
choices; see below.

**What value for what usage**:
| Use case | Suggested `embed_size` |
|---|---|
| Prove the pipeline mechanics fast, laptop-scale (this project's `10m`/`tiny` default) | `128`–`192` |
| Meaningfully past the embedding-dominated regime, real quality gains per added parameter | `700`+ — but this requires real (GPU) compute, not laptop CPU/MPS |
| Matching a specific sibling project exactly | `384` (`30m`), `512` (`50m`), `768` (`153m`) — see `make presets` |

## `num_heads` (default `8`) — how attention is split, not how big it is

**What it controls**: splits `embed_size` into `num_heads` parallel attention subspaces,
each of `head_dim = embed_size / num_heads` — letting the model attend to several
different relationship patterns per layer simultaneously, per
[Chapter 10](../../../docs/llm-engineering/10_transformer_architecture.md).

**Impact of changing it (holding `embed_size` fixed)**: **zero effect on parameter
count** — the QKV/output projection matrices are still `embed_size`-shaped regardless of
how many ways that dimension gets split for the attention computation itself. What
changes is `head_dim`, and with it the expressiveness/specialization trade-off: more
heads means each one is narrower (less capacity to represent its own attention pattern);
fewer, wider heads means fewer distinct relationship types can be attended to per layer.
This is a real architectural trade-off, not a free parameter.

**A concrete, hardware-level consequence found in this project directly**: `head_dim`
size affects which fused attention kernel actually gets selected when `attn_impl="sdpa"`
— very small or non-power-of-two `head_dim` values (the `10m` preset's `160/8=20` is
exactly this) can fall outside what flash-attention-eligible kernels are tuned for,
influencing real, measured training throughput (see
[`TRAINING_SCHEDULE.md`](TRAINING_SCHEDULE.md) for the actual benchmark this project ran).
Choosing `num_heads` isn't purely an architecture-quality decision here — it has a
measurable performance side effect too.

**Hard constraint**: same one as `embed_size` — `embed_size % num_heads == 0`, validated
at construction, with the exact remainder explained in the raised error.

**What value for what usage**: keep `head_dim` (`embed_size / num_heads`) in a
conventional range — very small `head_dim` (well under 32) is generally accepted in the
broader literature to under-represent each head's attention pattern. This project's own
presets aren't perfectly consistent on this (`10m`: `head_dim=20`; `30m`/`153m`:
`head_dim=64`) — a real, visible artifact of `10m`'s presets being tuned primarily for
small total parameter count on a laptop, not for clean `head_dim` sizing. If picking a
custom combination, favor `head_dim` values of `32`–`128` where practical.

## `num_layers` (default `6`) — depth, and the cheapest way to change compute

**What it controls**: how many `GPTBlock`s (attention + MLP) are stacked — the network's
depth.

**Impact of increasing it**: **linear**, not quadratic, in both parameter count
(`layers × per_block`) and compute per step (each layer is one more sequential pass a
token's representation goes through) — a much gentler cost curve than `embed_size`'s
quadratic one for a similarly-sized nominal change. Depth generally helps the model
compose multi-step, hierarchical patterns (each layer refines what previous layers
produced); very deep stacks are made trainable at all by the pre-norm residual pattern
`GPTBlock.forward()` uses (`x = x + attn(ln(x))`, `x = x + mlp(ln(x))`) — without the
residual path, gradients have a much harder time propagating cleanly through many stacked
layers.

**Real limitation specific to this project's small scale**: per the README's own
observation, shrinking `num_layers` below the `10m` preset "barely moves" total parameter
count — because at `E=160`, `token_embedding` so thoroughly dominates (80.6%) that
`num_layers` changes mostly move **training speed** (linearly more/fewer sequential
blocks per forward/backward pass — directly the throughput/steps-per-second story in
[`TRAINING_SCHEDULE.md`](TRAINING_SCHEDULE.md)), not the parameter budget. Depth is the
lever to reach for when the goal is "train faster/slower at roughly the same size," and
`embed_size` is the lever for "meaningfully change model size."

**What value for what usage**:
| Use case | Suggested `num_layers` |
|---|---|
| Fastest laptop iteration (`tiny`/`10m` presets) | `4`–`6` |
| Real capability at meaningful scale (`153m` preset) | `16`+ — firmly GPU-training territory given the linear-but-real compute cost |

## `dropout` (default `0.1`) — the only field that does nothing at inference

**What it controls**: the probability of zeroing activations during training, applied in
four distinct places per forward pass: inside `CausalSelfAttention` on the attention
*weights themselves* when `attn_impl="sdpa"` (`dropout_p` passed straight into
`F.scaled_dot_product_attention`), on the attention block's output (`self.dropout(out)`/
`self.dropout(self.out_proj(out))`, both `attn_impl` paths), inside `MLP` after its final
linear layer, and once globally on the summed token+position embedding in
`TinyGPT.forward()`. It's regularization: preventing the network from over-relying on any
single connection by randomly removing some every step.

**Impact of increasing it**: more regularization — directly useful when training data is
small relative to model capacity, which is exactly the failure mode
[`DATA_PREP_GUIDELINE.md`](DATA_PREP_GUIDELINE.md#8-size-the-corpus-to-the-training-budget-honestly)
names ("small corpus + small vocab + narrow domain... overfits faster"). Too high, and it
does the opposite of help — training becomes noisier and can genuinely underfit, wasting
steps on a harder optimization problem for a corpus that wasn't actually at overfitting
risk.

**A cross-cutting effect worth knowing about**: `dropout_p > 0` during training is part
of what can steer PyTorch's SDPA kernel-backend selection away from the fastest
fused/flash-eligible path on some hardware/versions (see
[`TRAINING_SCHEDULE.md`](TRAINING_SCHEDULE.md) again) — `dropout` isn't purely a
data-science knob here, it can have a measurable training-throughput side effect too,
same as `num_heads` above.

**Hard constraint**: `__post_init__` requires `0.0 <= dropout < 1.0` — `1.0` (drop
*everything*) is rejected outright, not silently accepted as a degenerate no-signal
model.

**One thing that's easy to forget**: dropout only ever matters during training.
`model.eval()` (used by every inference/serving/evaluation path — `checkpoint.load_model`
sets it by default) disables all four dropout layers automatically; the stored value has
zero effect once a checkpoint is being used to generate text, and no effect on parameter
count either.

**What value for what usage**:
| Use case | Suggested `dropout` |
|---|---|
| Default, this project's current corpus/scale | `0.1` — a standard, safe default |
| Test loss diverging from train loss (real overfitting — see [`TRAINING_SCHEDULE.md`'s three-question framework](TRAINING_SCHEDULE.md#is-a-longer-run-still-worth-it-a-three-question-framework)) | raise toward `0.2`–`0.3` |
| Large corpus relative to model size, maximizing signal per step, overfitting isn't a live risk | lower toward `0.0` |
