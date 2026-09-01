# Qwen2.5-7B-Instruct — local serving

Serves the model this repo's `fine_tuning/qwen2.5-7b-instruct-l4-lora/` project
is planned to LoRA fine-tune, so a baseline (pre-fine-tune) and every later
checkpoint can be compared through the same OpenAI-compatible API. Mirrors the
`serving/vllm-smollm2-135m` / `serving/vllm-tinyllama-1.1b` sibling projects'
platform-detection pattern.

| Hardware detected | Server | Model |
| --- | --- | --- |
| NVIDIA CUDA | real vLLM | `Qwen/Qwen2.5-7B-Instruct` (bf16) |
| Apple Silicon / MPS | `mlx_lm.server` | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| No supported accelerator | Transformers CPU fallback | `Qwen/Qwen2.5-7B-Instruct` (bf16) |

## Why mlx-lm and not vLLM-Metal, unlike the two sibling projects

The sibling projects' `metal` extra installs `vllm-metal[vllm]`, vLLM's own
Apple-Silicon plugin. As of 2026-09, that install does not actually complete on
macOS: `vllm-metal[vllm]`'s pinned `vllm` version range still transitively
depends on `nvidia-cudnn-frontend`/`nvidia-cutlass-dsl`, neither of which has
ever shipped an arm64/macOS wheel — confirmed directly (`uv sync --extra metal`
fails with "doesn't have a source distribution or wheel for the current
platform"), not assumed. This is the exact failure `vllm-smollm2-135m/serve.py`
already documents and silently falls back from.

Rather than repeat that failed install and fall back to a slow CPU server, this
project's Apple-Silicon path uses `mlx-lm`'s own native `mlx_lm.server`
directly — genuinely MLX-accelerated, verified working 2026-09-01 (4-bit
quantized, ~4.4GB peak memory, well under this machine's 24GB, fast: 14 varied
prompts in 33s total). Same OpenAI-compatible `/v1/chat/completions` surface a
`vllm serve` would give you, just reached without the broken wrapper package in
between. `mlx_lm.server` also takes `--adapter-path`, so once the sibling
`fine_tuning/qwen2.5-7b-instruct-l4-lora/` project actually produces a LoRA
adapter, serving the fine-tuned model needs no code change here — just an added
flag.

## Run it

```bash
cd serving/vllm-qwen2.5-7b-instruct
make install   # auto-detects: mlx-lm on Apple Silicon, vllm on CUDA, else CPU fallback
make check     # print the backend/model it would use
make serve     # start it (default port 8005)
```

Or force a specific backend: `make serve-mps` / `make serve-cuda` / `make serve-cpu`.

Test it once it's up:

```bash
curl -s http://127.0.0.1:8005/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 30}'
```

## Known issues

- **`vllm-metal` unusable on macOS as of 2026-09** — see above. Revisit once
  upstream ships arm64 wheels for its full CUDA-tooling dependency chain, or
  drops that dependency for non-CUDA backends.
- The CUDA path (`Qwen/Qwen2.5-7B-Instruct`, full bf16) is untested here — this
  machine has no NVIDIA GPU. The command is written from the sibling projects'
  proven pattern, not verified end-to-end.
