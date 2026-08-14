# LLM Development Guide: Quickstart Map

The concepts behind every stage below — what a token is, why causal attention, why
cross-entropy, why AdamW, and the rest — are covered in the
[LLM Engineering Curriculum](../../../docs/llm-engineering/00_roadmap.md), grounded in
this repo's actual code, not generic examples. This page is just the map: which chapter
covers which stage of *this* project's pipeline, and the exact command to run it.

| Stage | Curriculum chapter | Command in this project |
|---|---|---|
| Collect/prepare data | [Ch. 12 — Pretraining Objective & Why Data Dominates](../../../docs/llm-engineering/12_the_pretraining_objective_and_why_data_dominates.md) | `./scripts/prepare_all_datasets.sh` (or `./scripts/workflow.sh data`) |
| Tokenization | [Ch. 9 — Tokenization](../../../docs/llm-engineering/09_tokenization.md) | Handled inside data prep — GPT-2 tokenizer via `tiktoken` |
| Architecture (embeddings, causal attention, blocks) | [Ch. 10 — The Transformer Architecture](../../../docs/llm-engineering/10_transformer_architecture.md) | See this README's parameter-count derivation below |
| Training loop (forward/loss/backprop/AdamW) | [Ch. 3 — How Neural Networks Learn](../../../docs/llm-engineering/03_how_neural_networks_learn.md), [Ch. 13 — The Training Loop, Mechanism by Mechanism](../../../docs/llm-engineering/13_the_training_loop_mechanism_by_mechanism.md) | `./scripts/workflow.sh train` |
| Evaluating during training, when to stop | [Ch. 4 — Hyperparameter Tuning](../../../docs/llm-engineering/04_hyperparameter_tuning.md), [Ch. 15 — Evaluating a Model While It's Still Training](../../../docs/llm-engineering/15_evaluating_a_model_while_training.md) | `logs/train_eval_history.csv`, `./scripts/workflow.sh eval` |
| Checkpointing, resuming, moving across machines | [Ch. 27 — Checkpointing and Resuming Training](../../../docs/llm-engineering/27_checkpointing_and_resuming_training.md) | `Ctrl-C` is safe; re-run `./scripts/workflow.sh train` to resume. Cross-machine: [`docs/MIGRATION.md`](MIGRATION.md) |
| Training across multiple datasets/rounds | [Ch. 28 — Catastrophic Forgetting and Continual Training](../../../docs/llm-engineering/28_catastrophic_forgetting_and_continual_training.md) | — |
| Inference: decoding, sampling | [Ch. 8 — What Is a Language Model, Really](../../../docs/llm-engineering/08_what_is_a_language_model.md), [Ch. 21 — Inference Mechanics](../../../docs/llm-engineering/21_inference_mechanics_decoding_sampling_and_kv_cache.md) | `./scripts/workflow.sh infer` |
| Serving over HTTP | [Ch. 22 — From Script to API](../../../docs/llm-engineering/22_from_script_to_api_serving_a_model_for_real.md) | `./scripts/workflow.sh serve`; see [`docs/API_SERVER.md`](API_SERVER.md) |

## What you built, in one line

An autoregressive, GPT-style Transformer decoder, trained from scratch with next-token
prediction, ~152.8M parameters — see the top-level [`README.md`](../README.md) for the
full parameter-count derivation and architecture details.

## This project's own commands (not covered by the curriculum, since they're project-specific)

```bash
pip install -r requirements.txt
./scripts/workflow.sh data
./scripts/workflow.sh train
./scripts/workflow.sh infer
./scripts/workflow.sh eval
./scripts/workflow.sh serve
```

See the top-level [`README.md`](../README.md)'s Quickstart section for the full command
list, Colab compatibility, and dataset-access details.
