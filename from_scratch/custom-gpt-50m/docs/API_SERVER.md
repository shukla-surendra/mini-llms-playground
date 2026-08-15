# FastAPI Serving Guide

## The mechanism (nothing new here)

`api_server.py` is a thin FastAPI wrapper around the exact same generation loop
`inference.py` uses — see
[Chapter 22 — From Script to API](../../../docs/llm-engineering/22_from_script_to_api_serving_a_model_for_real.md)
for why the wrapper is this thin, and
[Chapter 21 — Inference Mechanics](../../../docs/llm-engineering/21_inference_mechanics_decoding_sampling_and_kv_cache.md)
for what `temperature`/`top_k`/`top_p`/`repetition_penalty` below actually do. This doc
only covers what's specific to this project's serving setup.

## 1) Start server

```bash
uv run gpt-serve --port 8000 --reload
```

Or via make:
```bash
make serve
```

## 2) Health check

```bash
curl http://127.0.0.1:8000/health
```

## 3) Generate text

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "System: You are a helpful coding assistant for docker workflows.\nUser: How do I debug container startup failures?\nAssistant:",
    "max_new_tokens": 80,
    "do_sample": true,
    "temperature": 0.5,
    "top_k": 15,
    "top_p": 0.8,
    "repetition_penalty": 1.25
  }'
```

## 4) Python sample client

```bash
python examples/api_client.py
```

## Request fields specific to this model

- `trim_at_role_markers` (default `false`) — this is a **base** model trained on raw text,
  so by default the completion is returned verbatim, including any `User:` turn the model
  writes for itself. Set it to `true` to cut the completion at the next role marker when
  you want assistant-style output.

`/health` also reports `model_label`, `param_count`, `step`, and the checkpoint path, so
you can confirm which trained size is actually being served.
