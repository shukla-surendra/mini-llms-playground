# LLM Development Guide: Quickstart Map

The concepts behind every stage below — what a token is, why causal attention, why
cross-entropy, why AdamW, and the rest — are covered in the
[LLM Engineering Curriculum](../../../docs/llm-engineering/00_roadmap.md), grounded in
this repo's actual code, not generic examples. This page is just the map: which chapter
covers which stage of *this* project's pipeline, and the exact command to run it.

| Stage | Curriculum chapter | Command in this project |
|---|---|---|
| Collect/prepare data | [Ch. 12 — Pretraining Objective & Why Data Dominates](../../../docs/llm-engineering/12_the_pretraining_objective_and_why_data_dominates.md) | `make data-public` (see [`docs/DATASETS.md`](DATASETS.md) for the five sources and filters) |
| Tokenization | [Ch. 9 — Tokenization](../../../docs/llm-engineering/09_tokenization.md) | Handled inside data prep — GPT-2 tokenizer via `tiktoken` |
| Architecture (embeddings, causal attention, blocks) | [Ch. 10 — The Transformer Architecture](../../../docs/llm-engineering/10_transformer_architecture.md) | `make config` prints the exact parameter breakdown for the active preset |
| Training loop (forward/loss/backprop/AdamW) | [Ch. 3 — How Neural Networks Learn](../../../docs/llm-engineering/03_how_neural_networks_learn.md), [Ch. 13 — The Training Loop, Mechanism by Mechanism](../../../docs/llm-engineering/13_the_training_loop_mechanism_by_mechanism.md) | `make train` |
| Evaluating during training, when to stop | [Ch. 4 — Hyperparameter Tuning](../../../docs/llm-engineering/04_hyperparameter_tuning.md), [Ch. 15 — Evaluating a Model While It's Still Training](../../../docs/llm-engineering/15_evaluating_a_model_while_training.md) | `logs/train_eval_history_<label>.csv`, `make eval` |
| Choosing/interpreting step count & the LR schedule | [Ch. 3 — How Neural Networks Learn (LR schedule deep-dive)](../../../docs/llm-engineering/03_how_neural_networks_learn.md#deep-dive-what-the-learning-rate-schedule-is-actually-doing) | [`docs/TRAINING_SCHEDULE.md`](TRAINING_SCHEDULE.md) — what `steps` means here, how it drives the schedule, and how to judge whether a longer run still helps |
| Checkpointing, resuming, moving across machines | [Ch. 27 — Checkpointing and Resuming Training](../../../docs/llm-engineering/27_checkpointing_and_resuming_training.md) | `Ctrl-C` is safe; `make train` resumes. Cross-machine: [`docs/MIGRATION.md`](MIGRATION.md) |
| Training across multiple datasets/rounds | [Ch. 28 — Catastrophic Forgetting and Continual Training](../../../docs/llm-engineering/28_catastrophic_forgetting_and_continual_training.md) | — |
| Inference: decoding, sampling | [Ch. 8 — What Is a Language Model, Really](../../../docs/llm-engineering/08_what_is_a_language_model.md), [Ch. 21 — Inference Mechanics](../../../docs/llm-engineering/21_inference_mechanics_decoding_sampling_and_kv_cache.md) | `make infer` |
| Serving over HTTP | [Ch. 22 — From Script to API](../../../docs/llm-engineering/22_from_script_to_api_serving_a_model_for_real.md) | `make serve`; see [`docs/API_SERVER.md`](API_SERVER.md) |
| Common questions asked while training this project | — | [`docs/TRAINING_QA.md`](TRAINING_QA.md) — do multiple epochs help, does data arrangement matter, and more, answered against this project's actual numbers |

## What you built, in one line

An autoregressive, GPT-style Transformer decoder, trained from scratch with next-token
prediction, at a size (~10M parameters by default, configurable up to ~153M) chosen to
iterate fast on a laptop rather than to maximize output quality — see the top-level
[`README.md`](../README.md) for the full parameter-count breakdown and preset table.

## This project's own commands (not covered by the curriculum, since they're project-specific)

```bash
make setup && make config && make data-public && make train && make infer && make serve
```

`gpt-data --list` (dataset registry), `make audit` (corpus quality gate before a long
run), `make eval` (heuristic generation-quality scoring) — see the top-level
[`README.md`](../README.md)'s Quickstart section for the full command list and what each
one does.
