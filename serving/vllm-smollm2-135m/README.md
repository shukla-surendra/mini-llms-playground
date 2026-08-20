# Small vLLM server: SmolLM2-135M-Instruct

This self-contained serving project exposes the small, instruction-tuned
[`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
checkpoint through an OpenAI-compatible API. It has 135M parameters, so it is a good
first model for a modest GPU or CPU. The first run downloads the model from Hugging
Face into the normal Hugging Face cache.

For a complete explanation of this launcher's role, vLLM internals, GPU memory and
scheduling, tuning, scaling, deployment, and troubleshooting, read
[`docs/VLLM_SERVING_GUIDE.md`](docs/VLLM_SERVING_GUIDE.md).

For endpoint-by-endpoint request parameters and model/task applicability, read
[`docs/VLLM_API_REFERENCE.md`](docs/VLLM_API_REFERENCE.md).

To serve a custom PyTorch checkpoint or architecture, read
[`docs/CUSTOM_PYTORCH_MODEL_INTEGRATION.md`](docs/CUSTOM_PYTORCH_MODEL_INTEGRATION.md).

For the technical differences between PyTorch checkpoints and safetensors, including a
safe serving-export workflow, read [`docs/CHECKPOINT_FORMATS.md`](docs/CHECKPOINT_FORMATS.md).

`serve.py` detects the fastest usable local backend:

| Hardware detected | Server | Model |
| --- | --- | --- |
| NVIDIA CUDA | vLLM | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Apple Silicon / MPS | vLLM-Metal | `mlx-community/SmolLM2-135M-Instruct` |
| No supported accelerator | Transformers CPU fallback | `HuggingFaceTB/SmolLM2-135M-Instruct` |

Apple MPS does not execute ordinary CUDA vLLM kernels. The MPS path uses
vLLM-Metal—the vLLM hardware plugin backed by MLX/Metal—and the official MLX Community
conversion of the same Hugging Face model; it still provides the same OpenAI-compatible
API.

The Apple dependency intentionally installs `vllm-metal[vllm]`: the `[vllm]` extra is
what brings in vLLM core and the `vllm` command-line executable, in addition to the
Metal plugin.

The regular PyPI `vllm` wheel is CUDA-oriented. Its CPU backend needs a separate
CPU-wheel/source-build procedure, so this project deliberately uses a small
Transformers server on CPU instead. This makes the project runnable on CPU-only Linux
and WSL while retaining vLLM where its accelerated backends are available.

## Run it

From this directory, install the matching optional dependency once.

```bash
cd serving/vllm-smollm2-135m

# The Make target detects CUDA, Apple Silicon, or CPU and installs the matching backend.
make install

# Or explicitly install CUDA vLLM
uv sync --extra vllm

# Apple Silicon macOS instead (native arm64 Python 3.12)
# vLLM-Metal requires native arm64 Python 3.12.
uv sync --extra metal

# CPU-only Linux / WSL instead
uv sync --extra cpu
```

Or use the included Make targets:

```bash
make install       # automatically selects CUDA, Metal, or CPU dependencies
make check         # print selected backend and model
make serve         # start with automatic detection
```

`make install` detects native Apple Silicon and installs vLLM-Metal; detects an NVIDIA
GPU and installs vLLM; otherwise it installs the CPU fallback. At runtime, `make serve`
then selects CUDA when PyTorch can see an NVIDIA GPU, MPS/Metal on Apple Silicon, or
CPU when neither accelerator is usable. `make help` lists every available target,
including `make serve-cuda`, `make serve-mps`, and `make serve-cpu`.

Check what the launcher will use, then start it:

```bash
uv run python serve.py --check
uv run python serve.py
```

The default address is `http://127.0.0.1:8004`. Override it with `--host` / `--port`
or the `HOST` / `PORT` environment variables. The endpoint is local-only by default;
use `--host 0.0.0.0` only when you deliberately need LAN access.

To force a backend for debugging:

```bash
uv run python serve.py --backend cuda
uv run python serve.py --backend mps
uv run python serve.py --backend cpu
```

## Query it

After the server says it is ready:

```bash
curl http://127.0.0.1:8004/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "messages": [{"role": "user", "content": "Give a one-sentence description of vLLM."}],
    "max_tokens": 80,
    "temperature": 0.2
  }'
```

On the Apple path, send `"model": "mlx-community/SmolLM2-135M-Instruct"` instead.
Both servers expose the standard `/v1/models` endpoint if you want to discover the
loaded model programmatically.

## Notes

- CUDA is selected only when `torch.cuda.is_available()` is true; the launcher will not
  accidentally fall back after a broken GPU setup.
- CPU mode is functional but slower than CUDA; it uses the project’s lightweight
  Transformers/OpenAI-compatible server rather than trying to run a CUDA vLLM wheel.
- The CUDA settings reserve only 55% of GPU memory and cap context at 2048 tokens. They
  are deliberately small defaults; raise them once the basic server is working.
