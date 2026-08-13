# TinyLlama-1.1B-Chat — Original Checkpoint, Served Standalone

Part of [mini-llms-playground](../../README.md)'s **base-models track**, the standalone
counterpart to [`fine_tuning/tinyllama-1.1b-lora`](../../fine_tuning/tinyllama-1.1b-lora/).
Serves the **original, unmodified** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint — no
LoRA adapter — as its own FastAPI endpoint, so it can be queried live and compared
directly against the fine-tuned server, rather than only appearing as one half of an
offline comparison script.

## Why this exists as a separate project

`tinyllama-1.1b-lora`'s server always loads the base model *with* the LoRA adapter
attached — there was previously no way to query the original checkpoint on its own
through a running endpoint. This project is exactly that: the same base model, no
adapter, its own port (`8002`, vs. `tinyllama-1.1b-lora`'s `8001`) — so both can run
**simultaneously** and be queried side by side in real time. It also lives in its own
top-level track (`base_models/`) rather than under `fine_tuning/`, since serving an
untouched author checkpoint isn't a fine-tuning exercise.

## Quickstart

```bash
cd base_models/tinyllama-1.1b-base-serving
uv run api_server.py
```

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain LoRA in simple terms."}'
```

`GET /health` reports the model ID and confirms no adapter is loaded.

## Original model details

- **Model**: [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Original authors**: the TinyLlama project (Zhang, Zeng, et al.) — see the model card
  and [the TinyLlama paper](https://arxiv.org/abs/2401.02385) for full training details,
  which this repo did not perform (only the LoRA fine-tuning in
  [`../../fine_tuning/tinyllama-1.1b-lora/`](../../fine_tuning/tinyllama-1.1b-lora/) is
  this repo's own work).
- **License**: Apache 2.0.
- **Already instruction-tuned** by its original authors — hence this server uses the
  model's own built-in chat template (`tokenizer.apply_chat_template`), same as
  [`../../fine_tuning/tinyllama-1.1b-lora/serve_tinyllama_lora.py`](../../fine_tuning/tinyllama-1.1b-lora/serve_tinyllama_lora.py)
  does. Contrast with [`../smollm2-135m-base-serving/`](../smollm2-135m-base-serving/),
  which serves a true base model with no chat template at all.

**Full writeup**: see [`docs/MODEL_DETAILS.md`](docs/MODEL_DETAILS.md) for the complete
architecture (verified against this checkpoint's own `config.json`), tokenizer/chat
template details, pretraining data and procedure, the SFT+DPO recipe that produced
`-Chat-v1.0`, reported benchmarks, and known limitations — all sourced from the official
paper/model card or verified directly against the actual files.

## Why keep this separate rather than adding a "no adapter" flag to the LoRA server

Running both as genuinely separate processes, on separate ports, means they can be
queried **at the same time**, with no risk of one server's state (which checkpoint is
currently loaded) affecting the other — a cleaner, more literal "before and after,
side by side" setup than a single server that has to be reconfigured to switch between
the two.
