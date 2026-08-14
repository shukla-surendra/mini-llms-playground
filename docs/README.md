# Docs Index

This page is the repo-wide map. Each track keeps its own operational docs (how to run
things) next to its code, since commands and paths are specific to that folder — this
page is for orientation: which track to read, and where its docs actually live.

## Learn the concepts: [`llm-engineering/`](llm-engineering/00_roadmap.md)

Before or alongside running either track's code, the
**[LLM Engineering Curriculum](llm-engineering/00_roadmap.md)** covers *why* everything
below is built the way it is — history, tokenization, the Transformer architecture,
pretraining, fine-tuning techniques, and serving — from layman's terms up to advanced
depth, with every concept grounded in this repo's actual code, not generic examples.
Start there if the goal is understanding, not just running commands.

## From-scratch track

Three experiments, all training a GPT-style model's architecture and training loop
completely from scratch, at different scales and for different goals:

### [`custom-gpt-153m`](../from_scratch/custom-gpt-153m/)

A ~153M-parameter model trained on conversational data (LMSYS Chat-1M plus several
public chat datasets) — the goal is learning the full stack at meaningful scale, not
polished output quality.

| Doc | Covers |
|---|---|
| [`README.md`](../from_scratch/custom-gpt-153m/README.md) | Architecture, parameter-count breakdown, quickstart, dataset prep, training/resume flow, Colab compatibility |
| [`docs/LLM_DEV_GUIDE.md`](../from_scratch/custom-gpt-153m/docs/LLM_DEV_GUIDE.md) | Quickstart map: which [`llm-engineering/`](llm-engineering/00_roadmap.md) chapter covers each pipeline stage (data → tokenize → train → checkpoint → infer → serve), plus this project's exact command for each |
| [`docs/API_SERVER.md`](../from_scratch/custom-gpt-153m/docs/API_SERVER.md) | Running and calling the FastAPI serving endpoint (mechanism: [Chapter 22](llm-engineering/22_from_script_to_api_serving_a_model_for_real.md)) |
| [`docs/MIGRATION.md`](../from_scratch/custom-gpt-153m/docs/MIGRATION.md) | Moving a training run between a cloud GPU machine and a local Mac (checkpoint sync, resume; mechanism: [Chapter 27](llm-engineering/27_checkpointing_and_resuming_training.md)) |

### [`custom-gpt-10m`](../from_scratch/custom-gpt-10m/)

The same code as `custom-gpt-153m` (architecture, GPT-2 tokenizer/vocab, training loop) —
only `context_length`/`embed_size`/`num_heads`/`num_layers` shrunk to a ~10M-parameter
config, so the whole pipeline trains and iterates fast on a laptop CPU/MPS. The goal is
proving the mechanics (data prep → train → infer → eval → serve), not output quality —
see `tinystories-gpt-6m` below for the sibling project that optimizes for quality at a
similar scale instead. Environment managed by `uv`; every workflow step has a `make`
target.

| Doc | Covers |
|---|---|
| [`README.md`](../from_scratch/custom-gpt-10m/README.md) | Parameter-count breakdown for the shrunk config, `uv`/`make`-based quickstart |
| [`docs/DATASETS.md`](../from_scratch/custom-gpt-10m/docs/DATASETS.md) | The five-source corpus, filters, and the raw-data-to-`train.txt` pipeline |
| [`docs/LLM_DEV_GUIDE.md`](../from_scratch/custom-gpt-10m/docs/LLM_DEV_GUIDE.md) | Quickstart map: which [`llm-engineering/`](llm-engineering/00_roadmap.md) chapter covers each pipeline stage, plus this project's exact command for each |
| [`docs/API_SERVER.md`](../from_scratch/custom-gpt-10m/docs/API_SERVER.md) | Running and calling the FastAPI serving endpoint (mechanism: [Chapter 22](llm-engineering/22_from_script_to_api_serving_a_model_for_real.md)) |
| [`docs/MIGRATION.md`](../from_scratch/custom-gpt-10m/docs/MIGRATION.md) | Moving a training run between a cloud GPU machine and a local Mac (mechanism: [Chapter 27](llm-engineering/27_checkpointing_and_resuming_training.md)) |

### [`tinystories-gpt-6m`](../from_scratch/tinystories-gpt-6m/)

A ~5.85M-parameter model trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(a dataset built with a deliberately restricted vocabulary) — the goal is the opposite
trade-off from `custom-gpt-153m`: fast to train on a laptop (~15 minutes end to end on
Apple Silicon MPS) and genuinely coherent output, not maximal scale.

| Doc | Covers |
|---|---|
| [`README.md`](../from_scratch/tinystories-gpt-6m/README.md) | Quickstart, real training results and generated output from this project's own run |
| [`docs/DATASET_AND_TOKENIZER.md`](../from_scratch/tinystories-gpt-6m/docs/DATASET_AND_TOKENIZER.md) | Why TinyStories, why a custom small vocabulary instead of GPT-2's |
| [`docs/ARCHITECTURE.md`](../from_scratch/tinystories-gpt-6m/docs/ARCHITECTURE.md) | Every sizing decision, full parameter-count derivation |
| [`docs/TRAINING.md`](../from_scratch/tinystories-gpt-6m/docs/TRAINING.md) | Hyperparameters, real MPS throughput, resume, diagnosing a bad run |
| [`docs/SERVING.md`](../from_scratch/tinystories-gpt-6m/docs/SERVING.md) | Inference and API server, what's deliberately out of scope at this size (mechanism: [Chapters 21](llm-engineering/21_inference_mechanics_decoding_sampling_and_kv_cache.md)-[22](llm-engineering/22_from_script_to_api_serving_a_model_for_real.md)) |

## Fine-tuning track

Two LoRA fine-tuning experiments, at different points on the "already instruction-tuned"
vs. "true base model" spectrum. Each has a standalone counterpart in the
[base-models track](#base-models-track) below, serving that same model's original,
unmodified checkpoint on its own.

### [`tinyllama-1.1b-lora`](../fine_tuning/tinyllama-1.1b-lora/)

LoRA fine-tuning of the pretrained, **already chat-tuned** `TinyLlama/TinyLlama-1.1B-
Chat-v1.0` model on `HuggingFaceH4/ultrachat_200k` — an incremental adaptation.

| Doc | Covers |
|---|---|
| [`README.md`](../fine_tuning/tinyllama-1.1b-lora/README.md) | Install, LoRA training (with resume/checkpoint knobs), serving, MacBook/MPS-specific notes |

### [`smollm2-135m-dolly-lora`](../fine_tuning/smollm2-135m-dolly-lora/)

LoRA fine-tuning of `HuggingFaceTB/SmolLM2-135M`, a genuine **base model with no chat
template**, on `databricks/databricks-dolly-15k` — teaching instruction-following
essentially from scratch, specifically so a before/after comparison shows a dramatic,
unmistakable difference rather than an incremental one.

| Doc | Covers |
|---|---|
| [`README.md`](../fine_tuning/smollm2-135m-dolly-lora/README.md) | Quickstart, what makes this project's before/after comparison different from `tinyllama-1.1b-lora`'s |
| [`docs/APPROACH.md`](../fine_tuning/smollm2-135m-dolly-lora/docs/APPROACH.md) | Why this model, dataset, and technique — each chosen to maximize the visible before/after contrast |
| [`docs/TRAINING_RESULTS.md`](../fine_tuning/smollm2-135m-dolly-lora/docs/TRAINING_RESULTS.md) | The real training run: loss curve, timing |
| [`docs/BEFORE_AFTER_COMPARISON.md`](../fine_tuning/smollm2-135m-dolly-lora/docs/BEFORE_AFTER_COMPARISON.md) | Real, unedited generated output — base model vs. fine-tuned, same prompts |

## Base-models track

Serves each fine-tuning experiment's **original, unmodified author checkpoint** as its
own standalone FastAPI endpoint — no LoRA adapter, no training — so the base model is a
first-class, independently runnable thing in this repo, not only ever the "before" half
of a comparison baked into another project.

### [`tinyllama-1.1b-base-serving`](../base_models/tinyllama-1.1b-base-serving/)

Serves the **original** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint — no LoRA
adapter — as its own FastAPI endpoint (port `8002`), separate from `tinyllama-1.1b-lora`'s
adapter-loaded server (port `8001`).

| Doc | Covers |
|---|---|
| [`README.md`](../base_models/tinyllama-1.1b-base-serving/README.md) | Quickstart, original-author repo/paper/license details, why it's kept separate from the LoRA server |
| [`docs/MODEL_DETAILS.md`](../base_models/tinyllama-1.1b-base-serving/docs/MODEL_DETAILS.md) | Full architecture (verified against the checkpoint's own `config.json`), tokenizer/chat-template details, pretraining data & procedure, the SFT+DPO recipe behind `-Chat-v1.0`, reported benchmarks, known limitations |

### [`smollm2-135m-base-serving`](../base_models/smollm2-135m-base-serving/)

Serves the **original** `HuggingFaceTB/SmolLM2-135M` base checkpoint — no LoRA adapter,
no chat template, plain-text completion only — as its own FastAPI endpoint (port `8003`),
separate from `smollm2-135m-dolly-lora`'s fine-tuned server.

| Doc | Covers |
|---|---|
| [`README.md`](../base_models/smollm2-135m-base-serving/README.md) | Quickstart, original-author repo/paper/license details, why this endpoint has no chat template |
| [`docs/MODEL_DETAILS.md`](../base_models/smollm2-135m-base-serving/docs/MODEL_DETAILS.md) | Full architecture (verified against the checkpoint's own `config.json`), tokenizer/special-token details, pretraining data & procedure, why there's no chat template, reported benchmarks, known limitations |

## Comparing the two tracks directly

| | From-scratch (`custom-gpt-153m`) | Fine-tuning (`tinyllama-1.1b-lora`) |
|---|---|---|
| Starting point | Random weights | A pretrained, already-competent model |
| What you're training | Every parameter | A small set of LoRA adapter weights only |
| Compute needed | Meaningful, even at 153M params, to get anything coherent | Far less — LoRA trains a small fraction of total parameters |
| What you learn | The full stack: tokenization, attention, causal masking, the training loop itself | How adaptation/fine-tuning works, and how much a good base model already knows |
| Realistic output quality | Limited — small model, small dataset, no post-training | Meaningfully better — inherits the base model's pretraining |
| Where checkpoints matter most | Long-running local training, resumed across sessions/machines (`MIGRATION.md`) | Adapter checkpoints during a shorter, targeted training run |

**When to reach for which**: use the from-scratch track when the goal is understanding
*how* a language model is actually built — every piece is visible and editable. Use the
fine-tuning track when the goal is a genuinely more useful model for a specific behavior
or dataset, without re-deriving the architecture.

## Adding a new experiment

All three tracks are meant to hold more than one experiment over time. To add one:

1. Create a new subfolder under `from_scratch/`, `fine_tuning/`, or `base_models/`, named
   for the model/approach (matching the existing `custom-gpt-153m/`,
   `tinyllama-1.1b-lora/`, and `tinyllama-1.1b-base-serving/` convention).
2. Give it its own `README.md`, `requirements.txt`, and any scripts it needs — keep it
   self-contained rather than sharing code across experiments, so each one stays
   runnable independently.
3. Add a row to this page's per-track table, and to the top-level [`README.md`](../README.md)
   if it's significant enough to headline.
