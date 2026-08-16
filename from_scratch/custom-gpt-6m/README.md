# TinyStories GPT (~5.85M params) — From Scratch, Trainable on a MacBook

Part of [mini-llms-playground](../../README.md)'s **from-scratch track**, sibling to
[`custom-gpt-153m`](../custom-gpt-153m/). See the
[top-level README](../../README.md) and [docs index](../../docs/README.md) for how this
relates to the rest of the repo, and this project's own [`docs/`](docs/) folder for the
full reasoning behind every tool and design decision below.

## The actual goal

Not a smart, general-purpose chatbot. Just: **a model small enough to train on a laptop
in minutes, that generates grammatically sensible, locally coherent short text — not
garbage.** That's a narrower, genuinely achievable target, and every design decision here
(dataset, tokenizer, model size) is chosen specifically to hit it. Full reasoning in
[`docs/DATASET_AND_TOKENIZER.md`](docs/DATASET_AND_TOKENIZER.md).

## What this project includes

This is an installable package (`src/gpt/`) with `pyproject.toml`-registered CLI commands
(`gpt-*`), the same convention every sibling `custom-gpt-*` project in
[`from_scratch/`](../) uses — `make <target>` wraps each one.

- **Dataset + tokenizer prep**: `gpt-data` (`src/gpt/data/prepare.py`) — downloads
  [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (short, simple
  children's stories with a deliberately restricted vocabulary), trains a small custom
  BPE tokenizer (`vocab_size=4096`) on it, and writes tokenized train/val splits.
- **Model**: `src/gpt/model.py` — a ~5.85M-parameter decoder-only Transformer, the same
  architecture family as [`../custom-gpt-153m/tiny_llm.py`](../custom-gpt-153m/tiny_llm.py),
  every dimension scaled down. Full sizing reasoning in
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- **Training**: `gpt-train` (`src/gpt/training/trainer.py`) — MPS-first (also runs on
  CUDA/CPU), with checkpointing, resume, and train/val loss tracking. Full mechanism and
  real, observed MPS performance numbers in [`docs/TRAINING.md`](docs/TRAINING.md).
- **Efficient training**: switchable naive/fused attention (`ATTN_IMPL=naive|sdpa`), mixed
  precision (`AMP=1`), and gradient checkpointing (`GRAD_CHECKPOINT=1`) — all real flags on
  `trainer.py`/`model.py`, benchmarked with actual measured numbers (throughput and memory)
  in [`docs/EFFICIENT_TRAINING.md`](docs/EFFICIENT_TRAINING.md).
- **Masked language modeling**: `src/gpt/model_mlm.py` + `gpt-train-mlm` — a second,
  bidirectional pretraining objective (BERT-style masked LM) reusing this project's blocks
  and tokenized data, trained separately from the causal-LM path above. Full explanation
  and real results in [`docs/MASKED_LM.md`](docs/MASKED_LM.md).
- **Contrastive self-supervised learning**: `src/gpt/model_contrastive.py` +
  `gpt-train-contrastive` — a third pretraining objective (SimCSE-style positive pairs,
  in-batch-negative InfoNCE loss), built on top of the *unchanged* causal backbone used as
  a sequence encoder. Full explanation and real results in
  [`docs/CONTRASTIVE_LEARNING.md`](docs/CONTRASTIVE_LEARNING.md).
- **Distributed training**: `gpt-train-ddp` + `gpt-train-fsdp` — DistributedDataParallel
  and FullyShardedDataParallel, wrapping the same causal-LM model/data, run for real as
  multi-process CPU jobs on this machine (`gloo` backend — a mechanism proof, not a
  GPU-cluster benchmark; two real environment-specific bugs hit and fixed along the way).
  Full explanation and real results in
  [`docs/DISTRIBUTED_TRAINING.md`](docs/DISTRIBUTED_TRAINING.md).
- **Inference**: `gpt-infer` (`src/gpt/inference/generate.py`) — command-line text
  generation from a checkpoint.
- **API server**: `gpt-serve` (`src/gpt/inference/server.py`) — FastAPI serving endpoint.
  Full explanation in [`docs/SERVING.md`](docs/SERVING.md).
- **Workflow script**: `scripts/workflow.sh` — one-command pipeline, now wrapping the
  `gpt-*` CLI commands (`uv run gpt-data`/`gpt-train`/`gpt-infer`/`gpt-serve`).

## Quickstart

This project is managed with [`uv`](https://docs.astral.sh/uv/) — no manual venv/pip
steps needed; `uv run` (used by every `make` target below) provisions `.venv` from
[`pyproject.toml`](pyproject.toml) automatically on first use.

```bash
cd from_scratch/custom-gpt-6m

# 1. Download data + train tokenizer + tokenize (~100k stories by default)
make data

# 2. Train (~12-15 min on Apple Silicon MPS for the default step count)
make train

# 3. Generate
make infer

# 4. Serve
make serve
```

See [Start, stop, and resume training](#start-stop-and-resume-training) below for the
full detail. Run `make help` for every available target, including dataset-round
snapshots and publishing to the Hugging Face Hub. `scripts/workflow.sh` (using `uv run`
directly) is also available as a plain-shell alternative to the Makefile.

## Start, stop, and resume training

```bash
make train                     # start (or resume, if a checkpoint already exists)
```

```bash
# stop: press Ctrl-C at any time
```
`trainer.py` catches the interrupt, saves `checkpoints/6m/causal/latest.pt`, and
exits — the checkpoint from the step you stopped at is safely on disk before the process
returns control to the shell.

```bash
make train                     # resume: re-run the same command, picks up from latest.pt
```
Resuming is the *default* behavior of `make train` — it happens automatically whenever
`checkpoints/6m/causal/latest.pt` exists, no flag needed. The resumed run also
verifies the checkpoint's saved hyperparameters match the current config before loading,
so resuming after an accidental config change fails loudly instead of silently
corrupting the run.

```bash
make train-fresh               # start over, ignoring any existing checkpoint
# or, equivalently:
uv run gpt-train --no-resume
```
Use this when you deliberately want to discard progress and retrain from step 0 —
otherwise `make train` always continues where the last run left off.

See [Chapter 27 — Checkpointing and Resuming Training](../../docs/llm-engineering/27_checkpointing_and_resuming_training.md)
for why this is safe (atomic saves, self-describing checkpoints), and
[docs/CONTINUAL_TRAINING_LOW_RESOURCE.md](docs/CONTINUAL_TRAINING_LOW_RESOURCE.md) for
snapshotting a checkpoint before a new training round so a bad round can be rolled back.

## Real results from this project's own training run

Actually run on a MacBook, Apple Silicon MPS, `GPT_STEPS=4000`, default 100k-story subset
(~22.4M training tokens). Total wall-clock time: **~15 minutes**, including dataset
download, tokenizer training, and all evaluation passes.

```
step     train_loss   val_loss   val_perplexity
0        8.372        8.368      4308.6   (near-random: ln(4096) ≈ 8.32)
200      4.378        4.395        81.1
1000     3.106        3.160        23.6
2000     2.674        2.728        15.3
3000     2.498        2.571        13.1
3999     2.442        2.471        11.8   (final)
```

Real, unedited generated output from this exact checkpoint (`temperature=0.8`):

> Once upon a time, there was a little girl named Lily. She loved to play with her toys
> and her toy cars. One day, she saw a small boy walking by the park. He was scared
> because he didn't know what to do.
>
> Lily said, "I don't want to go and see my toy car!" But then, she heard a loud noise.
> It was a scary dog that was running towards him. The dog did not want to hurt the dog
> anymore.
>
> Lily's mom saw...

Grammatically correct, locally coherent, stays on-topic — not garbage, and not claimed to
be anything smarter than that. Minor logical hiccups (a pronoun mismatch, "the dog did
not want to hurt the dog") are expected and honest at this model size and training
budget — see [`docs/TRAINING.md`](docs/TRAINING.md) for how to push training further if
better coherence is the next goal (more steps, a larger `--max-samples`, or scaling up
`embed_size`/`num_layers` in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)).

## Why this is a *different* project from `custom-gpt-153m`, not a smaller copy of it

| | `custom-gpt-153m` | `custom-gpt-6m` (this project) |
|---|---|---|
| Goal | Learn the full stack at meaningful scale | Fastest path to genuinely coherent (not garbage) output on a laptop |
| Dataset | LMSYS chat + public chat datasets (broad, varied) | TinyStories (deliberately narrow vocabulary/topics) |
| Tokenizer | GPT-2's, reused as-is (50,257 vocab) | Custom-trained BPE, 4,096 vocab, fit to this corpus |
| Parameters | ~152.8M | ~5.85M |
| Realistic output today | Often repetitive/degenerate (small model, huge varied dataset — a genuine mismatch) | Short, simple, but sensible sentences |

Both are legitimate, deliberately different choices — see
[`docs/DATASET_AND_TOKENIZER.md`](docs/DATASET_AND_TOKENIZER.md) for the full reasoning
on why narrowing the dataset (not just shrinking the model) is what actually makes
"small and coherent" achievable.

## Full docs

- [`docs/DATASET_AND_TOKENIZER.md`](docs/DATASET_AND_TOKENIZER.md) — why TinyStories,
  why a custom small vocabulary, the exact data pipeline.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — every sizing decision, with the full
  parameter-count derivation.
- [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md) — `src/gpt/model.py` and
  `src/gpt/training/trainer.py`, every class/function explained: the exact math, why each
  API/approach was chosen, and the real alternatives (flash attention, SwiGLU, RoPE,
  RMSNorm, and more).
- [`docs/TRAINING.md`](docs/TRAINING.md) — hyperparameters, real MPS throughput numbers,
  resume behavior, how to diagnose a bad run.
- [`docs/EFFICIENT_TRAINING.md`](docs/EFFICIENT_TRAINING.md) — naive vs. fused (SDPA)
  attention, mixed precision, and gradient checkpointing, with real measured
  throughput/memory numbers on this project's own hardware.
- [`docs/READING_TRAINING_OUTPUT.md`](docs/READING_TRAINING_OUTPUT.md) — the live
  progress-bar output (`loss=`, `lr=`, `train=`, `val=`, `step/s`), decoded term by term.
- [`docs/MASKED_LM.md`](docs/MASKED_LM.md) — the BERT-style masked-LM objective, why it
  needed bidirectional attention and a reserved `[MASK]` id, with real training numbers.
- [`docs/CONTRASTIVE_LEARNING.md`](docs/CONTRASTIVE_LEARNING.md) — SimCSE positive pairs,
  in-batch-negative InfoNCE, and an honest read of why this setup's task saturates fast.
- [`docs/DISTRIBUTED_TRAINING.md`](docs/DISTRIBUTED_TRAINING.md) — DDP vs. FSDP, two real
  environment-specific bugs hit and fixed, and a genuine side-by-side comparison run.
- [`docs/HOW_MUCH_TRAINING_IS_ENOUGH.md`](docs/HOW_MUCH_TRAINING_IS_ENOUGH.md) — what an
  epoch is, and how to actually decide when to stop training, using this project's real
  evaluation history as the worked example.
- [`docs/CONTINUING_TRAINING_ON_NEW_DATA.md`](docs/CONTINUING_TRAINING_ON_NEW_DATA.md) —
  continuing to train an existing checkpoint on a *different* dataset (continued
  pretraining), the tokenizer-mismatch pitfall that silently corrupts this if done wrong,
  and the `--reuse-tokenizer` fix.
- [`docs/CONTINUAL_TRAINING_LOW_RESOURCE.md`](docs/CONTINUAL_TRAINING_LOW_RESOURCE.md) —
  the full recommended workflow for repeatedly training on a *growing sequence* of
  datasets, no GPU required: replay mixing across all prior rounds
  (`src/gpt/data/replay.py`) and reversible checkpoint snapshots
  (`make snapshot`/`make restore-snapshot`).
- [`docs/PUBLISHING_TO_HUGGING_FACE.md`](docs/PUBLISHING_TO_HUGGING_FACE.md) — publishing
  the trained model, tokenizer, and code to the Hugging Face Hub
  (`scripts/upload_to_hf.py`), and the real model card ([`model_card.md`](model_card.md)).
- [`docs/SERVING.md`](docs/SERVING.md) — how generation and the API server work, and what
  production-serving concerns are deliberately out of scope at this size.
- [`docs/TEMPERATURE_AND_SAMPLING.md`](docs/TEMPERATURE_AND_SAMPLING.md) — this project's
  default sampling values and practical guidance for its `/generate` endpoint (the math
  itself is in the curriculum's [Chapter 21](../../docs/llm-engineering/21_inference_mechanics_decoding_sampling_and_kv_cache.md)).
- [`../../docs/llm-engineering/`](../../docs/llm-engineering/00_roadmap.md) — the
  from-first-principles curriculum every concept above links back to.
