# mini-llms-playground

A hands-on lab for exploring small language models two ways: **training a GPT-style
model completely from scratch**, and **fine-tuning an existing pretrained model**. Both
tracks are meant to grow over time as more experiments get added — this isn't one fixed
project, it's a home for many small, runnable LLM experiments.

## The two tracks

| Track | What it is | Start here |
|---|---|---|
| [`from_scratch/`](from_scratch/) | Custom architecture, custom training loop, trained on conversation data from zero | [`from_scratch/custom-gpt-153m/README.md`](from_scratch/custom-gpt-153m/README.md) |
| [`fine_tuning/`](fine_tuning/) | Adapting an existing pretrained model (LoRA) to a new dataset/behavior | [`fine_tuning/tinyllama-1.1b-lora/README.md`](fine_tuning/tinyllama-1.1b-lora/README.md) |

Each experiment lives in its own subfolder under one of these two tracks, with its own
README, requirements, and scripts — self-contained enough to run independently of the
others.

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

## Learn the concepts, not just the commands

The **[LLM Engineering Curriculum](docs/llm-engineering/00_roadmap.md)** explains *why*
everything in this repo is built the way it is — history, tokenization, the Transformer
architecture, pretraining, fine-tuning, and serving, from layman's terms up to advanced
depth, with every concept grounded in this repo's actual code. Start there if you want to
understand LLMs, not just run this repo's scripts.

## Full docs

See [`docs/README.md`](docs/README.md) for a fuller comparison of the two tracks, when to
reach for which, and pointers into each track's own operational docs (training guides,
API server usage, cross-machine migration).

## Repo history note

This repo was previously named `tiny_llm` — the from-scratch custom GPT (`tiny_llm.py`)
was the original, single project here. It's been restructured to `from_scratch/custom-
gpt-153m/` to make room for the fine-tuning track and future experiments, without
changing any of its working code, checkpoint naming, or commands — only its location.
