# LLM Engineering Curriculum: From First Principles to Building Your Own

This is a from-the-ground-up curriculum on how LLMs actually work — starting from **deep
learning fundamentals** (since that's genuinely where LLMs come from, not a separate
field), through history, architecture, pretraining, fine-tuning, and serving — written to
go from **layman's terms to advanced LLM-engineer depth**, and grounded, wherever
possible, in the **real, runnable code already in this repo** rather than abstract
examples. When a chapter says "here's how attention works," it means "here's the exact
`CausalSelfAttention` class in [`tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py)
that implements it" — not a diagram with no code behind it.

## Why this curriculum exists, and how it relates to the rest of this repo

The top-level [`README.md`](../../README.md) and [`docs/README.md`](../README.md)
describe *what* this repo contains (a from-scratch GPT and a LoRA fine-tuning setup) and
*how* to run each one. This curriculum is the missing third piece: **why** each part is
built the way it is — the concepts, history, and design decisions underneath the code,
explained well enough that you could rebuild it yourself, or explain it clearly to
someone else at any level of seniority.

**Relationship to `platform-lab/fundamentals/gpu_infrastructure/`**: that track (in a
sibling repo) covers the *hardware and fleet-infrastructure* layer — GPU architecture,
NCCL, Kubernetes GPU scheduling, multi-node serving at scale. This curriculum covers the
*model-building* layer — what a neural network and a transformer actually are, how
training and fine-tuning work, what serving means at the mechanism level. Where the two
genuinely overlap (KV cache, quantization, serving engines), this curriculum explains the
*concept* and links out to that track's deeper infrastructure treatment rather than
duplicating it.

## The six parts

### Part 0 — Deep Learning Foundations

LLMs are not a separate field from deep learning — they're a specific application of it.
This part is the general neural-network vocabulary and mechanism every later chapter
assumes: what a neuron/parameter/hyperparameter actually is, how a network learns at all,
and where NLP-specific architectures (including the Transformer) sit in the broader
landscape.

| # | Chapter | Status |
|---|---|---|
| 1 | [Neurons, Layers, and Neural Networks](01_neurons_layers_and_networks.md) | **written** |
| 2 | [Parameters vs. Hyperparameters](02_parameters_vs_hyperparameters.md) | **written** |
| 3 | [How Neural Networks Learn: Loss, Backprop, Gradient Descent](03_how_neural_networks_learn.md) | **written** |
| 4 | [Hyperparameter Tuning: What to Tune and How](04_hyperparameter_tuning.md) | **written** |
| 5 | [Embeddings: The General Idea](05_embeddings_the_general_idea.md) | **written** |
| 6 | [The NLP Architecture Landscape](06_nlp_architecture_landscape.md) | **written** |

### Part 1 — Foundations: LLM History & Architecture

The "how did we get here, and what actually *is* this thing, specifically for language"
layer, building directly on Part 0's general neural-network vocabulary.

| # | Chapter | Status |
|---|---|---|
| 7 | [History: How We Got Here](07_history_how_we_got_here.md) | **written** |
| 8 | [What Is a Language Model, Really](08_what_is_a_language_model.md) | **written** |
| 9 | [Tokenization: Turning Text Into Numbers](09_tokenization.md) | **written** |
| 10 | [The Transformer Architecture, Line by Line](10_transformer_architecture.md) | **written** |
| 11 | [Positional Encoding Variants, Deeper (RoPE and beyond)](11_positional_encoding_variants_rope_and_beyond.md) | **written** |

### Part 1B — Architecture Variants: What the Newer `from_scratch` Project Changed, and Why

Appended after the original catalog, same reason as Part 2B/2C/3B: avoids renumbering
already-written chapters. Belongs right after Chapter 11 in reading order — see "Reading
order" below. `custom-gpt-153m`'s architecture (Chapters 10-11) is one valid transformer
recipe; `custom-gpt-200m`/`350m` make four different choices (RoPE instead of learned
position embeddings, RMSNorm instead of LayerNorm, SwiGLU instead of a GELU MLP, no
biases). Chapter 11 already covers the first swap; this chapter covers the other two,
comparing both architecture families side by side throughout rather than describing the
newer one in isolation.

| # | Chapter | Status |
|---|---|---|
| 35 | [Normalization and MLP Variants: RMSNorm vs. LayerNorm, SwiGLU vs. GELU-MLP](35_normalization_and_mlp_variants_rmsnorm_and_swiglu.md) | **written** |

### Part 2 — Pretraining: Building a Model From Zero

What actually happens inside [`tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py)'s
training loop, and why each piece is there.

| # | Chapter | Status |
|---|---|---|
| 12 | [The Pretraining Objective & Why Data Dominates](12_the_pretraining_objective_and_why_data_dominates.md) | **written** |
| 13 | [The Training Loop, Mechanism by Mechanism](13_the_training_loop_mechanism_by_mechanism.md) | **written** |
| 14 | [Scaling Laws: Why Bigger Models, and When They Stop Helping](14_scaling_laws_why_bigger_models_and_when_they_stop_helping.md) | **written** |
| 15 | [Evaluating a Model While It's Still Training](15_evaluating_a_model_while_training.md) | **written** |

### Part 2C — Data Preparation: The Practical Pipeline Behind "Data Dominates"

Appended after the original catalog, same reason as Part 2B: avoids renumbering
already-written chapters. Belongs right after Chapter 12 in reading order — see "Reading
order" below. Chapter 12 argues *why* the corpus, not the objective or architecture, sets
the ceiling on what a model can learn; this chapter is the practical follow-up — the actual
levers (collection, filtering, deduplication, mixture weighting vs. capping, packing and
document-boundary masking, tokenization, base-vs-SFT format) a real pipeline pulls, each
tied to an observed failure mode in a real trained checkpoint's QA report.

| # | Chapter | Status |
|---|---|---|
| 34 | [Data Preparation Strategies for Pretraining](34_data_preparation_strategies_for_pretraining.md) | **written** |

### Part 2B — Training at Scale: Efficiency, Distribution, and Continuity

Appended after the original catalog above rather than inserted between 15 and 16, to avoid
renumbering already-written chapters — see "Reading order" below for where these actually
belong in sequence. All four dig into what Part 2's training loop needs once model size,
context length, hardware budget, or run duration stop being "trivially small": 25-26 cover
scaling a single run's efficiency and distributing it across processes; 27-28 cover what a
run needs to survive interruption and repeated/sequential training without silently
corrupting or forgetting what it already learned.

| # | Chapter | Status |
|---|---|---|
| 25 | [Efficient Attention: Flash Attention and SDPA](25_efficient_attention_flash_and_sdpa.md) | **written** |
| 26 | [Distributed Training: DDP, FSDP, and the Parallelism Landscape](26_distributed_training_ddp_and_fsdp.md) | **written** |
| 27 | [Checkpointing and Resuming Training](27_checkpointing_and_resuming_training.md) | **written** |
| 28 | [Catastrophic Forgetting and Continual Training](28_catastrophic_forgetting_and_continual_training.md) | **written** |

### Part 2D — Interpreting Evaluation Metrics

Appended after the original catalog, same reason as Part 2B/2C: avoids renumbering
already-written chapters. Belongs right after Chapter 15 in reading order — see "Reading
order" below. Chapters 4 and 15 both lean on loss/perplexity as the signal to watch during
training without fully explaining perplexity itself; this chapter is that explanation.

| # | Chapter | Status |
|---|---|---|
| 29 | [Perplexity: What It Actually Means and How to Read It](29_perplexity_understanding_and_interpreting_it.md) | **written** |

### Part 3 — Fine-Tuning: Adapting an Existing Model

What actually happens inside
[`train_tinyllama_lora.py`](../../fine_tuning/tinyllama-1.1b-lora/train_tinyllama_lora.py),
and the wider landscape of techniques it's one instance of.

| # | Chapter | Status |
|---|---|---|
| 16 | [The Fine-Tuning Landscape: Full FT, PEFT, Prompting, RAG](16_fine_tuning_landscape.md) | **written** |
| 17 | [LoRA & QLoRA, Mechanism by Mechanism](17_lora_and_qlora.md) | **written** |
| 18 | [Instruction Tuning & Supervised Fine-Tuning (SFT)](18_instruction_tuning_and_sft.md) | **written** |
| 19 | [RLHF, DPO, and Preference Optimization](19_rlhf_and_dpo.md) | **written** |
| 20 | [Evaluating a Fine-Tuned Model](20_evaluating_a_fine_tuned_model.md) | **written** |

### Part 3B — Distillation: Compressing a Larger Model Into a Smaller One

Appended after the original catalog, same reason as Part 2B: avoids renumbering
already-written chapters. Belongs right after Part 3 (Fine-Tuning) in reading order — see
"Reading order" below. Covers training a smaller **student** model to imitate a larger
**teacher** model's outputs, including the practical, licensing-aware recipe for using a
popular hosted model (Claude, GPT, etc.) — or, with zero terms-of-service exposure, one of
this repo's own `base_models/` checkpoints — as that teacher.

| # | Chapter | Status |
|---|---|---|
| 32 | [Knowledge Distillation, Mechanism by Mechanism](32_knowledge_distillation_mechanism_by_mechanism.md) | **written** |
| 33 | [Distilling Production Models Into a Local Model (Claude, GPT, and Other Popular Teachers)](33_distilling_production_models_into_a_local_model.md) | **written** |

### Part 4 — Serving: Turning a Trained Model Into Something You Can Talk To

What actually happens inside
[`inference.py`](../../from_scratch/custom-gpt-153m/inference.py) and
[`api_server.py`](../../from_scratch/custom-gpt-153m/api_server.py), and how that
generalizes to production-scale serving.

| # | Chapter | Status |
|---|---|---|
| 21 | [Inference Mechanics: Decoding, Sampling, and KV Cache](21_inference_mechanics_decoding_sampling_and_kv_cache.md) | **written** |
| 22 | [From Script to API: Serving a Model for Real](22_from_script_to_api_serving_a_model_for_real.md) | **written** |
| 23 | The Serving-Engine Ecosystem (vLLM and Friends) | planned |
| 31 | [Publishing a Model: The Hugging Face Hub Workflow](31_publishing_a_model_the_hugging_face_hub_workflow.md) | **written** |

Chapter 31 is numbered out of sequence for the same reason Part 2B's chapters are — appended
after the original catalog to avoid renumbering, but belongs here in reading order: what to
do with a trained checkpoint once serving (21-22) is understood.

### Part 5 — The Practical Toolkit

| # | Chapter | Status |
|---|---|---|
| 24 | The Tools Landscape: What Each Library Actually Does | planned |

## Reading order

**If you're starting from zero, including "what even is a neural network"**: read Part 0
straight through (1 → 6) first — it's the foundation everything else, including basic ML
knowledge you may have picked up elsewhere, gets made precise and consistent here. Then
Part 1 (7 → 10), which is LLM-specific vocabulary built on Part 0's general one.

**If you already know standard deep learning (neurons, backprop, gradient descent) and
want the LLM-specific material**: skip to Part 1 directly, referencing Part 0 chapters
only if a specific term (e.g., "what exactly is a hyperparameter") needs a refresher.

**If you want the full architecture picture, including the newer `from_scratch` project's
choices**: read Part 1B (Chapter 35) right after Chapter 11, despite the higher chapter
number — appended after the original catalog to avoid renumbering, but belongs right
after positional encoding in reading order, before moving on to Part 2.

**If you're following Part 2 (Pretraining) start to finish**: read Chapter 34 (Part 2C)
immediately after Chapter 12, despite the higher chapter number — it's the practical
data-pipeline follow-up to "why data dominates," appended after the original catalog to
avoid renumbering. Then continue 13 → 15, read Chapter 29 (Part 2D) right after 15 — the
perplexity explanation both 4 and 15 assume — and read Part 2B (25 → 28) right after that,
before moving on to Part 3 (Fine-Tuning) — same appended-after-the-catalog reasoning.

**If you want fine-tuning specifically**: skim Part 1's architecture chapter (10) for
vocabulary, then go straight to Part 3, run alongside
[`fine_tuning/tinyllama-1.1b-lora/`](../../fine_tuning/tinyllama-1.1b-lora/)'s training
script.

**If you want to build a local model from a stronger model's outputs (distillation)**:
read Part 3B (32 → 33) right after Part 3's own chapters (16 → 20), despite the higher
chapter numbers — appended after the original catalog to avoid renumbering, but belongs
right after fine-tuning in reading order. Chapter 33 covers the terms-of-service question
before the pipeline itself — read that section even if you skip the rest.

**If your question is "how do I actually serve this to users"**: Part 4 (21 → 22, then 31
for publishing the trained checkpoint itself), and for anything beyond a single local
machine, continue into `platform-lab/fundamentals/gpu_infrastructure/`'s Phase 5 (LLM
Serving) chapters, which this curriculum's Part 4 hands off to explicitly.

**If you're resuming, continuing, or repeating training runs**: 27 (checkpointing
mechanics) and 28 (forgetting/continual training) apply regardless of which Part you're
otherwise reading — they're relevant the moment a run is interrupted, resumed, or pointed
at new data after already having trained on something else.

## Chapter house style

Every chapter follows the same shape, since the whole point is layman's-terms-to-expert
in one pass, not two separate documents:

- **In Plain English** — a simple, jargon-free explanation first, using ordinary analogies.
- **The First-Principles Explanation** — the actual mechanism, precisely, no hand-waving.
- **Grounded in This Repo's Code** — the exact class/function in this repo that implements
  it, with real line references (Part 0 chapters ground concepts in this same code where
  the general DL concept has a direct instance in it — e.g., `nn.Linear`, `nn.Embedding`
  — even though Part 0 itself is framework-general, not LLM-specific).
- **Deep-Dive: Why It's Built This Way** — the design reasoning, trade-offs, and
  alternatives that were rejected and why.
- **Try It Yourself** — a concrete, hands-on exercise using this repo's actual code.
- **Common Misconceptions** — the specific wrong mental models people bring to this topic.
- **Practice Questions** — check whether the concept actually landed.
- **Key Terms** — a glossary of the vocabulary introduced.
