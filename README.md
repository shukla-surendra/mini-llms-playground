# mini-llms-playground

A hands-on lab for exploring small language models three ways: **training a GPT-style
model completely from scratch**, **fine-tuning an existing pretrained model**, and
**serving an original author checkpoint standalone**, unmodified, as its own baseline.
All three tracks are meant to grow over time as more experiments get added — this isn't
one fixed project, it's a home for many small, runnable LLM experiments.

## The three tracks

| Track | What it is | Start here |
|---|---|---|
| [`from_scratch/`](from_scratch/) | Custom architecture, custom training loop, trained on conversation data from zero | [`from_scratch/custom-gpt-153m/README.md`](from_scratch/custom-gpt-153m/README.md) |
| [`fine_tuning/`](fine_tuning/) | Adapting an existing pretrained model (LoRA) to a new dataset/behavior | [`fine_tuning/tinyllama-1.1b-lora/README.md`](fine_tuning/tinyllama-1.1b-lora/README.md) |
| [`base_models/`](base_models/) | Serving an original, unmodified author checkpoint on its own — no adapter, no training — as an independently runnable baseline | [`base_models/tinyllama-1.1b-base-serving/README.md`](base_models/tinyllama-1.1b-base-serving/README.md) |

Each experiment lives in its own subfolder under one of these three tracks, with its own
README, requirements, and scripts — self-contained enough to run independently of the
others.

## Standalone serving

[`serving/vllm-smollm2-135m/`](serving/vllm-smollm2-135m/) is a minimal
OpenAI-compatible local server for Hugging Face's 135M-parameter SmolLM2 Instruct
model. It automatically selects CUDA vLLM when available, uses the MLX/Metal route on
Apple Silicon, and provides a CPU-only Transformers fallback for a functional baseline.

[`serving/vllm-tinyllama-1.1b/`](serving/vllm-tinyllama-1.1b/) serves the original
TinyLlama-1.1B Chat checkpoint using the same CUDA/Metal/CPU selection, on port `8005`.

## Why these are separate tracks, not one project

Training from scratch and fine-tuning are genuinely different exercises, not two flavors
of the same task:

- **From scratch** means designing the architecture, writing the training loop, choosing
  the tokenizer, and learning everything — including how *little* a small model can
  learn from a small amount of compute and data. The payoff is understanding every layer
  of the stack, not state-of-the-art output quality.
- **Fine-tuning** starts from a model that already has real language competence (here,
  `TinyLlama-1.1B-Chat`) and adapts it — via LoRA, a parameter-efficient method that
  trains a small set of additional weights instead of the whole model — to a new dataset
  or behavior. The payoff is a genuinely more capable model for far less compute, at the
  cost of not building the underlying architecture yourself.

Keeping them structurally separate makes the comparison itself a learning tool: run the
same rough dataset through both, and the gap between them *is* the lesson.

- **Base-model serving** doesn't train anything at all — it loads the exact checkpoint
  published by the model's original authors (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`,
  `HuggingFaceTB/SmolLM2-135M`) and exposes it over its own API, with the original
  repo/paper/license details on record. It exists so that baseline is queryable on its
  own, live, side by side with its fine-tuned counterpart — not only ever visible inside
  an offline before/after comparison script.

## Learn the concepts, not just the commands

The **[LLM Engineering Curriculum](docs/llm-engineering/00_roadmap.md)** explains *why*
everything in this repo is built the way it is — history, tokenization, the Transformer
architecture, pretraining, fine-tuning, and serving, from layman's terms up to advanced
depth, with every concept grounded in this repo's actual code. Start there if you want to
understand LLMs, not just run this repo's scripts.

## Tools

[`tools/corpus-extractor`](tools/corpus-extractor/) — a standalone Rust CLI, not a
training track: turns an arbitrary local folder (`.pdf`/`.epub`/`.txt`/`.md`/`.rs`/
`.html`/`.js`/`.py`) into a GPT-2-token-chunked training dataset (JSONL + plain text,
train/test split) — the local-folder complement to the from-scratch track's
Hugging-Face-dataset pipeline.

## Infra

[`infra/aws-gpu-node`](infra/aws-gpu-node/) — Terraform for the AWS GPU box these tracks
train on (`g6.xlarge`, Deep Learning Base AMI, 100 GB gp3, S3 corpus/checkpoint bucket,
IAM instance role, and an automatic stop when the GPU goes idle). It is the executable
form of [`docs/AWS_RUNBOOK.md`](docs/AWS_RUNBOOK.md), which still holds the reasoning
behind each choice.

## Full docs

See [`docs/README.md`](docs/README.md) for a fuller comparison of the two tracks, when to
reach for which, and pointers into each track's own operational docs (training guides,
API server usage, cross-machine migration).

## Repo history note

This repo was previously named `tiny_llm` — the from-scratch custom GPT (`tiny_llm.py`)
was the original, single project here. It's been restructured to `from_scratch/custom-
gpt-153m/` to make room for the fine-tuning track and future experiments, without
changing any of its working code, checkpoint naming, or commands — only its location.
