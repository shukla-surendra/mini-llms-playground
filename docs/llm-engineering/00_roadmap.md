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
| 11 | Positional Encoding Variants, Deeper (RoPE and beyond) | planned |

### Part 2 — Pretraining: Building a Model From Zero

What actually happens inside [`tiny_llm.py`](../../from_scratch/custom-gpt-153m/tiny_llm.py)'s
training loop, and why each piece is there.

| # | Chapter | Status |
|---|---|---|
| 12 | [The Pretraining Objective & Why Data Dominates](12_the_pretraining_objective_and_why_data_dominates.md) | **written** |
| 13 | [The Training Loop, Mechanism by Mechanism](13_the_training_loop_mechanism_by_mechanism.md) | **written** |
| 14 | Scaling Laws: Why Bigger Models, and When They Stop Helping | planned |
| 15 | Evaluating a Model While It's Still Training | planned |

### Part 2B — Training at Scale: Efficiency and Distribution

Appended after the original catalog above rather than inserted between 15 and 16, to avoid
renumbering already-written chapters — see "Reading order" below for where these actually
belong in sequence. Both dig into what Part 2's training loop needs once model size, context
length, or hardware budget stop being "trivially small."

| # | Chapter | Status |
|---|---|---|
| 25 | [Efficient Attention: Flash Attention and SDPA](25_efficient_attention_flash_and_sdpa.md) | **written** |
| 26 | [Distributed Training: DDP, FSDP, and the Parallelism Landscape](26_distributed_training_ddp_and_fsdp.md) | **written** |

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

### Part 4 — Serving: Turning a Trained Model Into Something You Can Talk To

What actually happens inside
[`inference.py`](../../from_scratch/custom-gpt-153m/inference.py) and
[`api_server.py`](../../from_scratch/custom-gpt-153m/api_server.py), and how that
generalizes to production-scale serving.

| # | Chapter | Status |
|---|---|---|
| 21 | Inference Mechanics: Decoding, Sampling, and KV Cache | planned |
| 22 | From Script to API: Serving a Model for Real | planned |
| 23 | The Serving-Engine Ecosystem (vLLM and Friends) | planned |

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

**If you're following Part 2 (Pretraining) start to finish**: read Part 2B (25 → 26)
immediately after Part 2's own chapters (12 → 15), despite the higher chapter numbers —
they were appended after the original catalog to avoid renumbering already-written
chapters, but belong right after "how the training loop works" in reading order, before
moving on to Part 3 (Fine-Tuning).

**If you want fine-tuning specifically**: skim Part 1's architecture chapter (10) for
vocabulary, then go straight to Part 3, run alongside
[`fine_tuning/tinyllama-1.1b-lora/`](../../fine_tuning/tinyllama-1.1b-lora/)'s training
script.

**If your question is "how do I actually serve this to users"**: Part 4, and for
anything beyond a single local machine, continue into `platform-lab/fundamentals/
gpu_infrastructure/`'s Phase 5 (LLM Serving) chapters, which this curriculum's Part 4
hands off to explicitly.

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
