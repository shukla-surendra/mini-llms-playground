# LoRA & QLoRA, Mechanism by Mechanism

Part of the [LLM Engineering Curriculum](00_roadmap.md), Part 3 — Fine-Tuning. Builds on
[Chapter 16](16_fine_tuning_landscape.md)'s PEFT category — this chapter is exactly what
LoRA's inserted "small number of new parameters" actually are, the math behind why they
work, and QLoRA's extension of the same idea, including a real hardware constraint that
directly shapes which projects in this repo use which technique. See also
[`personal_notes_lora_and_peft_alternatives.md`](personal_notes_lora_and_peft_alternatives.md)
for a condensed, standalone quick-reference version of this same material.

## In Plain English

LoRA doesn't touch the model's original weight matrices at all — it leaves them frozen
and adds a small, separate "correction" alongside each one, computed as the product of
two much smaller matrices. Only those two small matrices get trained. The insight making
this work: the *change* a model needs to adapt to a new task tends to be far simpler
(lower "rank," in the linear-algebra sense) than the full complexity of the original
weight matrix — so a much smaller number of parameters is enough to capture it.

## The First-Principles Explanation

### The actual math: low-rank decomposition

For any weight matrix `W` (shape `d × k`) in the frozen base model that LoRA targets,
instead of directly updating `W`, LoRA adds a correction `ΔW`, decomposed as the product
of two small matrices:

```
ΔW = B × A

where:
  A has shape (r × k)   — "down-projects" into a small r-dimensional space
  B has shape (d × r)   — "up-projects" back out to the original d dimension
  r (the "rank")  <<  min(d, k)   — the whole point: r is small, often 8-64,
                                     while d and k can be in the thousands

Effective forward pass through this layer:
  output = W·x + ΔW·x = W·x + B·(A·x)
```

`W` never changes — it stays exactly as the pretrained model had it. Only `A` and `B` are
trained (`W` is frozen). The parameter count for `A` and `B` combined is `r×k + d×r` —
compare against `W`'s own `d×k` parameters: when `r` is small (say 16) and `d`/`k` are in
the thousands, `A`+`B`'s combined size is a tiny fraction of `W`'s — this is the exact
mechanism behind [Chapter 16](16_fine_tuning_landscape.md#deep-dive-why-peft-specifically-matters-for-this-curriculums-no-gpu-constraint)'s
"0.4% of parameters trainable" figure.

### `alpha` — the scaling factor, and why it exists separately from `r`

LoRA's actual contribution to the forward pass is scaled: `output = W·x + (alpha/r)·B·(A·x)`.
`alpha` controls how strongly the learned correction influences the output, independent of
`r` (the rank/capacity of that correction). A common convention (used in this repo's own
config, `lora_alpha=32` with `lora_r=16`) sets `alpha` to roughly `2×r` — worth knowing as
a real, empirically-common starting ratio, not a hard rule.

### Which weight matrices actually get LoRA applied — `target_modules`

LoRA isn't applied to *every* weight in a Transformer — it's applied to specific,
chosen matrices, per [Chapter 10](10_transformer_architecture.md)'s architecture:

```python
# fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py
lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",      # attention's Q/K/V/output projections
        "gate_proj", "up_proj", "down_proj",           # the MLP block's matrices
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
```

This targets essentially every major weight matrix in each Transformer block — both the
attention projections ([Chapter 10](10_transformer_architecture.md#self-attention-the-mechanism-that-lets-tokens-look-at-each-other))
and the MLP's matrices ([Chapter 10](10_transformer_architecture.md#the-mlp-block-per-token-processing-after-attention-mixes-information),
though `TinyLlama`'s SwiGLU-based MLP names them `gate_proj`/`up_proj`/`down_proj` rather
than this curriculum's simpler GELU-MLP naming — same conceptual role). Targeting more
matrices generally captures more of what fine-tuning could change, at the cost of more
trainable parameters — a real, tunable trade-off, not a fixed requirement.

## Grounded in This Repo's Code, End to End

```python
base_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype, ...)
base_model.config.use_cache = False
base_model.gradient_checkpointing_enable()

model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()
```

`get_peft_model` (from the `peft` library) performs the actual insertion described above
— it walks `base_model`'s modules, finds every one named in `target_modules`, freezes its
original weight, and attaches a new `A`/`B` pair. `gradient_checkpointing_enable()` is a
separate, complementary memory-saving technique worth naming precisely: it trades
recomputation for memory by *not* storing every intermediate activation during the
forward pass (needed for backprop's chain rule, per
[Chapter 3](03_how_neural_networks_learn.md#step-3-backpropagation--the-chain-rule-applied-systematically)),
instead recomputing them on demand during the backward pass — independent of LoRA itself,
but commonly used alongside it since both are aimed at the same constraint (fitting
training into limited memory).

## Deep-Dive: QLoRA, and Why It Isn't What This Repo Uses on a MacBook

**QLoRA** extends LoRA with one more idea: keep the frozen base model's weights in a
**quantized**, 4-bit format (not the 16/32-bit precision LoRA alone uses) — dramatically
shrinking the base model's memory footprint, while the trainable `A`/`B` LoRA matrices
still train in higher precision on top. This is a real, widely-used technique, and it's
worth understanding precisely why it *isn't* what either fine-tuning project in this repo
uses:

**The actual constraint**: QLoRA's 4-bit quantization is implemented by the
`bitsandbytes` library, whose custom quantization kernels are built specifically for
NVIDIA CUDA GPUs. `bitsandbytes` has historically had little to no working support for
Apple Silicon's MPS backend — meaning QLoRA, as commonly implemented, simply isn't a
functioning option on a MacBook's GPU today. This is a real, checkable hardware
limitation, not a design preference — it's the direct reason
[`fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py`](../../fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py)
loads the base model at `torch.float16` (via `model_dtype()`, checked against MPS
availability), not 4-bit — plain LoRA on a float16 base model, not QLoRA, is the correct
choice for this hardware.

**What this means practically for a MacBook-based fine-tuning setup**: you get LoRA's
full training-memory advantage (Chapter 16's dramatic reduction, since gradients/optimizer
state still only apply to the small `A`/`B` matrices), but not QLoRA's *additional*
reduction in how much memory the frozen base model itself occupies. For a model in the
hundreds-of-millions-to-low-single-billions-of-parameters range (like `TinyLlama-1.1B`,
or the smaller base model this curriculum's next chapters use), plain LoRA at float16 is
comfortably within a modern MacBook's unified memory — QLoRA's extra squeeze becomes
essential mainly for meaningfully larger base models (7B+), where it isn't available on
MPS regardless.

## Deep-Dive: LoRA's Neighbors in the PEFT Family

LoRA is the dominant PEFT technique (and the only one this repo's own projects use), but
it's worth knowing what else lives in the same category from
[Chapter 16](16_fine_tuning_landscape.md) — each makes a different, more aggressive trade
between "how few parameters get trained" and "how much control that gives you":

| Technique | Basic idea | Relative trainable-parameter count |
|---|---|---|
| **LoRA** | Low-rank matrices added alongside frozen weight matrices (this chapter) | Low |
| **QLoRA** | LoRA + a 4-bit quantized frozen base model (above) | Low (same as LoRA — the reduction is in *base model* memory, not trainable params) |
| **Adapters** (bottleneck adapters) | Small feed-forward modules inserted *between* existing layers, rather than added alongside a weight matrix | Low |
| **Prefix Tuning** | Learn a set of "virtual token" vectors prepended to every layer's input, steering attention without changing any weight matrix at all | Very low |
| **Prompt Tuning** | Similar to prefix tuning, but learns virtual vectors only at the input embedding layer, not injected into every layer | Very low |
| **(IA)³** | Learn per-channel scaling vectors that rescale activations, rather than adding new matrices | Very low |
| **BitFit** | Train *only* the existing bias terms already in the model, nothing new added | Very low |

**The pattern across this whole family**: every one of these answers [Chapter 16](16_fine_tuning_landscape.md)'s
same question — "freeze almost everything, train something small" — differently. LoRA's
specific answer (low-rank matrices alongside existing weights) has become the dominant
default because it hits a strong practical balance: enough capacity to meaningfully shift
behavior (unlike the very-low-capacity methods at the bottom of the table), while still
being dramatically cheaper than full fine-tuning — this repo's own projects' choice of
LoRA specifically, not one of its lower-capacity neighbors, reflects that same trade-off.

### LoRA isn't LLM-specific

Worth knowing explicitly: the low-rank-decomposition idea this chapter explains has
nothing inherently to do with language models — it's a general technique for adapting
*any* large neural network cheaply. It's also widely used for image-generation models
(e.g., Stable Diffusion), where a LoRA adapter can teach a base model a specific visual
style, character, or object, with multiple such adapters swappable on top of the same
frozen base model — the exact same "small correction alongside frozen weights" mechanism
this chapter derives, applied to a completely different architecture and modality.

## Trade-offs

| Choice | Upside | Cost |
|---|---|---|
| Higher LoRA rank (`r`) | More capacity to represent the fine-tuning change | More trainable parameters, more memory/compute (though still far less than full fine-tuning) |
| More `target_modules` | Captures more of what full fine-tuning could change | More trainable parameters |
| LoRA (float16 base) on MPS | Actually works on Apple Silicon today | No QLoRA-style further reduction in base-model memory |
| QLoRA (4-bit base) | Meaningfully lower base-model memory, enabling larger models on constrained hardware | Effectively CUDA-only via `bitsandbytes` — not a working option on MPS as of this writing |

## Try It Yourself

- In [`../../fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py`](../../fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py),
  try changing `--lora-r` from 16 to 4, re-run `model.print_trainable_parameters()`, and
  confirm the trainable-parameter count drops roughly in proportion — direct, observed
  confirmation of the `r×k + d×r` formula's linear dependence on `r`.

## Common Misconceptions

- **"LoRA modifies the original model weights, just efficiently."** No — the original
  weights (`W`) are never touched at all; they stay completely frozen. LoRA adds a
  parallel, separate correction (`B×A`) that gets combined with `W`'s output at
  inference/training time.
- **"QLoRA is strictly better than LoRA, so you should always use it if possible."** It's
  a different trade-off (lower base-model memory, at the cost of quantization's own
  precision loss and, currently, CUDA-only tooling) — not a strictly-better upgrade,
  especially given it's not available on MPS at all.
- **"A higher LoRA rank always produces a better fine-tuned model."** More capacity isn't
  automatically better — per [Chapter 4](04_hyperparameter_tuning.md)'s general
  hyperparameter-tuning reasoning, an unnecessarily high rank just adds trainable
  parameters (and overfitting risk on small datasets) without a guaranteed quality gain.

## Practice Questions

1. Write out the shapes of `A` and `B` for a weight matrix `W` of shape `2048×2048` with
   LoRA rank `r=16`, and compute the parameter-count reduction versus fine-tuning `W`
   directly.
2. Why does `alpha` exist as a setting separate from `r`, rather than just using a larger
   `r` to increase the correction's influence?
3. Explain, precisely, why QLoRA's memory advantage over plain LoRA is about the *frozen
   base model*, not about the trainable LoRA matrices themselves — and why that means
   QLoRA's benefit scales with base model size.

## Key Terms

- **Low-rank decomposition**: representing a matrix update as the product of two much
  smaller matrices, exploiting the idea that the needed update has much lower complexity
  ("rank") than the full weight matrix.
- **Rank (`r`)**: the shared inner dimension of LoRA's `A`/`B` matrices — the capacity
  knob for how much the correction can represent.
- **`alpha`**: the scaling factor controlling how strongly the LoRA correction influences
  the frozen base model's output.
- **`target_modules`**: which specific weight matrices in the model get a LoRA adapter
  attached.
- **Gradient checkpointing**: trading recomputation for memory by not storing every
  forward-pass activation, recomputing them during backprop instead.
- **QLoRA**: LoRA combined with a 4-bit quantized frozen base model, implemented via
  `bitsandbytes` — effectively CUDA-only, not currently functional on Apple Silicon MPS.
- **Adapters (bottleneck adapters)**: small feed-forward modules inserted between
  existing layers, a different PEFT approach than LoRA's alongside-the-weight-matrix
  correction.
- **Prefix / Prompt Tuning**: learning virtual token vectors (prepended at every layer,
  or just at the input embedding) to steer behavior, without adding to or modifying any
  weight matrix.
- **(IA)³ / BitFit**: even lower-capacity PEFT methods — learned per-channel activation
  scaling, and training only existing bias terms, respectively.
