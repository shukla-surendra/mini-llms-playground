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

## From-scratch track: [`custom-gpt-153m`](../from_scratch/custom-gpt-153m/)

A ~153M-parameter GPT-style model, architecture and training loop written from scratch,
trained on conversational data (LMSYS Chat-1M plus several public chat datasets).

| Doc | Covers |
|---|---|
| [`README.md`](../from_scratch/custom-gpt-153m/README.md) | Architecture, parameter-count breakdown, quickstart, dataset prep, training/resume flow, Colab compatibility |
| [`docs/LLM_DEV_GUIDE.md`](../from_scratch/custom-gpt-153m/docs/LLM_DEV_GUIDE.md) | Beginner-friendly, end-to-end walkthrough of every stage (data → tokenize → train → infer → evaluate) and why each piece exists |
| [`docs/API_SERVER.md`](../from_scratch/custom-gpt-153m/docs/API_SERVER.md) | Running and calling the FastAPI serving endpoint |
| [`docs/MIGRATION.md`](../from_scratch/custom-gpt-153m/docs/MIGRATION.md) | Moving a training run between a cloud GPU machine and a local Mac (checkpoint sync, resume) |

## Fine-tuning track: [`tinyllama-1.1b-lora`](../fine_tuning/tinyllama-1.1b-lora/)

LoRA fine-tuning of the pretrained `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model on
`HuggingFaceH4/ultrachat_200k`.

| Doc | Covers |
|---|---|
| [`README.md`](../fine_tuning/tinyllama-1.1b-lora/README.md) | Install, LoRA training (with resume/checkpoint knobs), serving, MacBook/MPS-specific notes |

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

Both tracks are meant to hold more than one experiment over time. To add one:

1. Create a new subfolder under `from_scratch/` or `fine_tuning/`, named for the model/
   approach (matching the existing `custom-gpt-153m/` and `tinyllama-1.1b-lora/`
   convention).
2. Give it its own `README.md`, `requirements.txt`, and any scripts it needs — keep it
   self-contained rather than sharing code across experiments, so each one stays
   runnable independently.
3. Add a row to this page's per-track table, and to the top-level [`README.md`](../README.md)
   if it's significant enough to headline.
