# SmolLM2-135M — Original Base Checkpoint, Served Standalone

Part of [mini-llms-playground](../../README.md)'s **base-models track**, the standalone
counterpart to
[`fine_tuning/smollm2-135m-dolly-lora`](../../fine_tuning/smollm2-135m-dolly-lora/).
Serves the **original, unmodified** `HuggingFaceTB/SmolLM2-135M` base checkpoint — no
LoRA adapter — as its own FastAPI endpoint, so it can be queried live and compared
directly against the fine-tuned server, rather than only appearing as one half of
`compare_before_after.py`'s offline comparison.

## Why this exists as a separate project

Same reasoning as [`../tinyllama-1.1b-base-serving/`](../tinyllama-1.1b-base-serving/):
`smollm2-135m-dolly-lora`'s server always loads the base model *with* the LoRA adapter
attached. This project is the same base model, no adapter, its own port (`8003`) — so
this and the fine-tuned server can run **simultaneously** and be queried side by side. It
also lives in its own top-level track (`base_models/`) rather than under `fine_tuning/`,
since serving an untouched author checkpoint isn't a fine-tuning exercise.

## Important: plain-text completion only, no chat template

Unlike `tinyllama-1.1b-base-serving` (which serves an *already chat-tuned* model), this
one serves a genuine **base model** — confirmed directly: its `tokenizer_config.json` has
no `chat_template` field at all (see
[`../../fine_tuning/smollm2-135m-dolly-lora/docs/APPROACH.md`](../../fine_tuning/smollm2-135m-dolly-lora/docs/APPROACH.md)).
This means:

- Send it raw text to continue, **not** a question expecting an assistant-style answer.
- It will often start plausibly (e.g., correctly answer a factual question) and then
  keep going — continuing into unrelated, sometimes garbled text — because it has no
  concept of "stop, that was a complete answer." This is exactly the behavior
  `smollm2-135m-dolly-lora`'s fine-tuning is designed to fix — this endpoint's entire
  purpose is showing that starting point clearly.

## Quickstart

```bash
cd base_models/smollm2-135m-base-serving
uv run api_server.py
```

```bash
curl -X POST http://127.0.0.1:8003/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "max_new_tokens": 40}'
```

Real, actually-observed output from this exact endpoint:

```json
{
  "completion": "The capital of France is Paris.\n\nWhat is the capital of Spain?\n\nSpain is the capital of Spain.\n\nWhat is the capital of Canada?\n\nCanada is",
  "note": "Plain-text completion — this base model has no chat template / instruction-following behavior."
}
```

Notice it correctly answers the actual question ("Paris") — the base pretraining did
teach it real facts — but then keeps generating, inventing further unrelated Q&A pairs
(and even a wrong one — "Spain is the capital of Spain") rather than stopping. It has
no notion of "this was a complete, correct response, stop here" — that's a learned
*behavior*, not something pretraining alone produces, and it's exactly what
[`../../fine_tuning/smollm2-135m-dolly-lora/`](../../fine_tuning/smollm2-135m-dolly-lora/)'s
fine-tuning adds. Compare this same prompt against that project's fine-tuned server
directly — see
[`../../fine_tuning/smollm2-135m-dolly-lora/docs/BEFORE_AFTER_COMPARISON.md`](../../fine_tuning/smollm2-135m-dolly-lora/docs/BEFORE_AFTER_COMPARISON.md).

`GET /health` reports the model ID and confirms no adapter/chat template is present.

## Original model details

- **Model**: [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- **Original authors**: Hugging Face's SmolLM team — see the model card and the
  [SmolLM2 paper](https://arxiv.org/abs/2502.02737) for full pretraining details, which
  this repo did not perform (only the LoRA fine-tuning in
  [`../../fine_tuning/smollm2-135m-dolly-lora/`](../../fine_tuning/smollm2-135m-dolly-lora/)
  is this repo's own work).
- **License**: Apache 2.0.
- **Architecture**: `LlamaForCausalLM`, confirmed via its `config.json` — the same
  architecture family this curriculum's
  [Chapter 10](../../docs/llm-engineering/10_transformer_architecture.md) and
  [Chapter 17](../../docs/llm-engineering/17_lora_and_qlora.md) already cover.

**Full writeup**: see [`docs/MODEL_DETAILS.md`](docs/MODEL_DETAILS.md) for the complete
architecture (verified against this checkpoint's own `config.json`), tokenizer/special-
token details, pretraining data and procedure, why this checkpoint has no chat template,
reported benchmarks, and known limitations — all sourced from the official technical
report/blog post or verified directly against the actual files.
