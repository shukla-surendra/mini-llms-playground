# vLLM HTTP API reference

This document maps the REST APIs of a current `vllm serve` process. vLLM implements
OpenAI-compatible APIs plus several task-specific endpoints. Routes available on a
particular server depend on its vLLM version, model task, installed extras, and launch
flags. The running server is the final source of truth:

```bash
curl http://127.0.0.1:8004/openapi.json
# Or browse http://127.0.0.1:8004/docs
```

## This project's API surface

| Runtime selected by `serve.py` | APIs available |
| --- | --- |
| CUDA vLLM or vLLM-Metal | vLLM APIs applicable to the model and installed extras |
| CPU fallback on this WSL host | `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` |

The CPU fallback deliberately implements only basic chat serving. The rest of this guide
describes what a vLLM server exposes when CUDA/Metal vLLM is active.

## Shared request conventions

- Send JSON with `Content-Type: application/json` unless an endpoint says multipart.
- Use `GET /v1/models` to obtain the valid `model` ID; do not guess it.
- `stream: true` produces Server-Sent Events for generation endpoints. Read events until
  the terminal event rather than expecting one JSON object.
- `temperature` controls randomness; `0` is greedy-style decoding. `top_p` is nucleus
  sampling. `max_tokens` or `max_output_tokens` caps generated text.
- `stop` is one or more stop strings; `seed` requests repeatable sampling.
- For vLLM-only fields in an OpenAI SDK, use `extra_body`; raw HTTP clients put them in
  the top-level JSON body.

## Operational and discovery endpoints

| Method and path | Purpose | Notes |
| --- | --- | --- |
| `GET /v1/models` | Discover model IDs | OpenAI-style `{object: "list", data: [...]}`. |
| `GET /health` | Basic engine health | Use a small inference check for full readiness. |
| `GET /metrics` | Prometheus metrics | Keep private; names vary by vLLM version. |
| `GET /version` | vLLM version where enabled | Useful when debugging compatibility. |
| `GET /docs`, `GET /openapi.json` | Interactive/schema documentation | Best version-specific route and schema list. |

## Generation APIs

### `POST /v1/chat/completions`

Use this for a chat/instruction model with a chat template, such as SmolLM2 Instruct.

```json
{
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "messages": [
    {"role": "system", "content": "Answer concisely."},
    {"role": "user", "content": "What is a KV cache?"}
  ],
  "max_tokens": 100,
  "temperature": 0.2
}
```

| Parameter | Type | Meaning |
| --- | --- | --- |
| `model` | string | Served model ID. |
| `messages` | array | Ordered role/content messages; multimodal models may accept content parts. |
| `max_tokens` | integer | Output-token cap. Newer clients may use `max_completion_tokens`. |
| `temperature`, `top_p` | number | Sampling controls. |
| `n` | integer | Number of candidates; each costs compute. |
| `stream` | boolean | Enable SSE token chunks. |
| `stop` | string or array | Stop sequences. |
| `seed` | integer | Sampling seed. |
| `logprobs`, `top_logprobs` | boolean/integer | Return token probability data. |
| `presence_penalty`, `frequency_penalty` | number | Discourage prior token reuse. |
| `response_format`, `structured_outputs` | object | JSON/schema/grammar constrained output where configured. |
| `tools`, `tool_choice` | array/string/object | Tool definitions and selection. A compatible model/parser is required. |
| `parallel_tool_calls` | boolean | `false` limits output to at most one tool call; `true` permits model-dependent parallel calls. |
| `user` | string | Accepted for compatibility but ignored by vLLM. |

The non-streaming response contains `choices`, assistant `message`, `finish_reason`, and
usually token `usage`. Streaming chunks carry incremental `delta` data.

### `POST /v1/completions`

Use raw text completion when the application owns exact prompt formatting. It does not
apply a chat template.

```json
{
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "prompt": "The KV cache is useful because",
  "max_tokens": 80,
  "temperature": 0.3,
  "stop": ["\n\n"]
}
```

The principal fields are `model`, `prompt` (one string or a batch), `max_tokens`,
`temperature`, `top_p`, `n`, `stream`, `stop`, `seed`, `logprobs`, `echo`, and
penalties. The response stores generated text under `choices[*].text`. OpenAI's
`suffix` field is not supported by vLLM.

### `POST /v1/chat/completions/batch`

Submit multiple independent Chat Completion-shaped requests in one HTTP operation. It is
appropriate for evaluation or offline work, not a normal interactive turn. Cap total
tokens so one client does not monopolize scheduler capacity.

### Responses API: `POST /v1/responses`

Use this newer OpenAI-style API for new agent/tool-oriented applications.

```json
{
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "input": "Explain continuous batching in two sentences.",
  "max_output_tokens": 100,
  "temperature": 0.2,
  "stream": false
}
```

| Parameter | Meaning |
| --- | --- |
| `input` | Required text or ordered input-item array. |
| `instructions` | Request-level system instructions. |
| `model` | Served model ID. |
| `max_output_tokens` | Generation cap. |
| `stream` | Stream response events using SSE. |
| `previous_response_id` | Continue from a prior response where supported. |
| `tools`, `tool_choice`, `parallel_tool_calls`, `max_tool_calls` | Tool definitions and controls. |
| `text` | Text-formatting/structured-output configuration. |
| `temperature`, `top_p`, `top_k`, `top_logprobs`, `logit_bias` | Sampling/logit controls. |
| `truncation` | `auto` or `disabled` handling for too-long input. |
| `store`, `metadata`, `user` | Compatibility/bookkeeping fields; exact behavior varies. |
| `request_id`, `session_id` | vLLM correlation identifiers. |
| `priority`, `cache_salt` | vLLM scheduling and prefix-cache isolation controls. |

Related routes are `GET /v1/responses/{response_id}` to retrieve a response and
`POST /v1/responses/{response_id}/cancel` to cancel one.

## vLLM generation additions

These commonly apply to both generation APIs.

| Parameter | Purpose |
| --- | --- |
| `top_k` | Sample only from the K highest-probability tokens. |
| `min_p` | Filter tokens below probability relative to the best token. |
| `repetition_penalty` | Penalize repeated tokens. |
| `min_tokens` | Do not stop before this many new tokens. |
| `ignore_eos` | Continue after EOS; always retain another output cap. |
| `stop_token_ids` | Stop on specified token IDs. |
| `include_stop_str_in_output` | Include stop text instead of removing it. |
| `truncate_prompt_tokens` | Retain only a bounded number of prompt tokens. |
| `prompt_logprobs` | Return log probabilities for input tokens. |
| `priority` | Lower integer runs earlier; non-zero requires priority scheduling. |
| `cache_salt` | Secret per-tenant salt for prefix-cache isolation. |

## Embedding, reranking, and scoring APIs

These require an embedding, pooling, or reranker model. A text-generation SmolLM2 server
does not support them just because the endpoint name exists.

| Method and path | Model task | Main request fields | Result |
| --- | --- | --- | --- |
| `POST /v1/embeddings` | Embedding/pooling | `model`, `input`, optional `encoding_format`, `dimensions`, `user` | Vector per input item. |
| `POST /pooling` | Pooling | Embedding-style input plus pooling options | General pooled hidden-state output. |
| `POST /score` | Cross-encoder/compatible embedding | `model`, `text_1`, `text_2`, optional `encoding_format` | Paired relevance/similarity score. |
| `POST /v1/rerank` | Reranker/cross-encoder | `model`, `query`, `documents`, optional `top_n` | Documents sorted by relevance. |

Embeddings retrieve candidate passages quickly; reranking scores a smaller candidate set
more accurately; a generation model then answers using selected passages.

## Audio, multimodal, and tokenizer APIs

| Method and path | Task | Main request fields |
| --- | --- | --- |
| `POST /v1/audio/transcriptions` | Speech-to-text | multipart `file`, `model`; optional `language`, `prompt`, `response_format`, `temperature`, timestamps. |
| `POST /v1/audio/translations` | Speech translated to target language | multipart `file`, `model`; optional prompt/format/temperature. |
| `POST /tokenize` | Text to token IDs | Model plus prompt/text and tokenizer options. |
| `POST /detokenize` | Token IDs to text | Model plus token-ID array. |

Audio endpoints need an ASR model and audio dependencies. Image/audio content in chat or
Responses requires a compatible multimodal model. SmolLM2-135M-Instruct is text-only.

## Headers, authentication, and safety

| Header | Requirement | Effect |
| --- | --- | --- |
| `Authorization: Bearer <key>` | `--api-key` configured | Authenticates `/v1`, `/v2`, and `/inference` prefixes. |
| `X-Request-Id` | `--enable-request-id-headers` | Request/response correlation; may cost performance at high QPS. |
| `X-Vllm-Priority` | Priority scheduling for non-zero values | Overrides JSON `priority`. |

API-key authentication does not secure every non-`/v1` vLLM endpoint. Bind vLLM to a
private interface and place a reverse proxy with authentication, authorization, request
limits, and TLS in front of it for production.

## Endpoint selection

```text
Chat assistant                   → /v1/chat/completions
Raw text continuation            → /v1/completions
New tool-using agent             → /v1/responses
Semantic-search vectors          → /v1/embeddings with an embedding model
Reorder retrieved documents      → /v1/rerank or /score with a reranker
Speech recognition               → /v1/audio/transcriptions with an ASR model
```

## Compatibility notes

- vLLM is OpenAI-compatible, not identical: Chat `user` is ignored and Completions
  `suffix` is unsupported.
- A 400/404 may indicate a model-task mismatch rather than malformed JSON.
- A model's Hugging Face `generation_config.json` can override sampling defaults. Start
  with `--generation-config vllm` when you require vLLM defaults.
- Pin the vLLM version and inspect `/openapi.json` during upgrades; fields evolve.

## Authoritative sources

- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [vLLM Responses request schema](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/responses/protocol/)
- [OpenAI API reference](https://platform.openai.com/docs/api-reference)
