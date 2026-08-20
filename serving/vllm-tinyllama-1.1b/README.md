# TinyLlama-1.1B vLLM serving

Standalone, platform-aware server for
[`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
It complements the existing base-model FastAPI project by exposing the same original,
unmodified checkpoint through an OpenAI-compatible server.

| Hardware | Server and model |
| --- | --- |
| NVIDIA CUDA | vLLM + `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Apple Silicon | vLLM-Metal + `mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit` |
| No supported accelerator | Transformers CPU fallback + original model |

The default port is **8005**, so it can run alongside SmolLM2's port 8004 and the
existing base-model endpoints. The first start automatically downloads weights from
Hugging Face. TinyLlama is 1.1B parameters, so CPU fallback is functional but slower and
needs substantially more memory than SmolLM2.

## Run

```bash
cd serving/vllm-tinyllama-1.1b
make install
make check
make serve
```

`make install` chooses vLLM for CUDA, vLLM-Metal on native Apple Silicon, or the CPU
fallback otherwise. Use `make serve-cuda`, `make serve-mps`, or `make serve-cpu` to
override detection.

## Query

```bash
curl http://127.0.0.1:8005/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Explain a KV cache simply."}],
    "max_tokens": 80,
    "temperature": 0.2
  }'
```

On Apple Metal, obtain the active model ID using `GET /v1/models` and use that ID in the
request. For vLLM architecture, API details, and custom-model integration, see the
[SmolLM2 vLLM serving guide](../vllm-smollm2-135m/docs/VLLM_SERVING_GUIDE.md).

## Known issues

- **vLLM-Metal currently fails to install on every real Mac** — same root cause and
  same automatic CPU fallback as the sibling SmolLM2 project; see
  [its README's Known issues](../vllm-smollm2-135m/README.md#known-issues) for the
  full explanation (vLLM's PyPI package unconditionally depends on
  `nvidia-cudnn-frontend`/`nvidia-cutlass-dsl`, neither of which has a macOS wheel).
  `make install`/`serve.py --backend auto` detect this and fall back to CPU, so
  `make install && make serve` still works end-to-end on Apple Silicon today.
