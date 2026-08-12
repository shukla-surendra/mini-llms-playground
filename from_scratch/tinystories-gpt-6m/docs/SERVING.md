# Serving: From a Checkpoint to an HTTP API

## The mechanism (nothing new here — same as `custom-gpt-153m`)

`inference.py` and `api_server.py` implement exactly the autoregressive generation loop
from
[`../../../docs/llm-engineering/08_what_is_a_language_model.md`](../../../docs/llm-engineering/08_what_is_a_language_model.md) —
repeatedly: predict a probability distribution over the next token, sample (or take the
argmax of) one token, append it, repeat. If any of `temperature`, `top-k`, `top-p`, or
`repetition_penalty` are unfamiliar, that chapter covers each precisely; this doc only
covers what's specific to this project's serving setup.

## `inference.py`: command-line generation

```bash
python inference.py --prompt "Once upon a time," --max-new-tokens 150
python inference.py --prompt "The little dog" --greedy      # deterministic
python inference.py --prompt "One day" --temperature 1.1     # more varied/random
```

One difference from `custom-gpt-153m`'s inference script worth noting: generation stops
early if the model produces the `<|endoftext|>` token (the document-separator introduced
in [`DATASET_AND_TOKENIZER.md`](DATASET_AND_TOKENIZER.md)) — since this model was trained
to predict that token at the end of a story, a well-trained model naturally learns to
"know when to stop," rather than always generating the full `max_new_tokens` budget.

## `api_server.py`: the FastAPI wrapper

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8010 --reload
```

```bash
curl -X POST http://127.0.0.1:8010/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time,", "max_new_tokens": 150, "temperature": 0.8}'
```

`GET /health` reports the loaded checkpoint's training step, parameter count, and
device — useful for confirming which checkpoint (and how well-trained a model) is
actually being served, directly analogous to
[`../../custom-gpt-153m/docs/API_SERVER.md`](../../custom-gpt-153m/docs/API_SERVER.md)'s
server.

## Why this is a legitimate (if simple) serving setup, and what it deliberately skips

This is a real HTTP API a client can call — but it's explicitly **not** the production
serving architecture covered in the sibling `platform-lab` repo's
`system_design_foundation/01_ml_system_design/13_large_model_multi_gpu_inference/` folder
or its `gpu_infrastructure/` serving chapters. At this model's
scale (~5.85M parameters, comfortably fits on a laptop's CPU or MPS), none of the concerns
those tracks cover — batching multiple concurrent requests efficiently, KV cache memory
budgeting, multi-GPU tensor parallelism, quantization — are actually load-bearing here.
This server handles one request at a time, synchronously, and that's a legitimate,
correct choice at this scale, not a shortcut. Continuous batching and the rest become
relevant once request volume or model size grow past what a single synchronous process
comfortably handles — see those tracks when that's the actual problem you have.
