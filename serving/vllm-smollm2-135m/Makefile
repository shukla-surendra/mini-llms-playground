.DEFAULT_GOAL := help

# Keep uv's cache inside this standalone project by default. This also makes the
# commands work in restricted environments where ~/.cache is not writable.
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR

.PHONY: help install install-metal install-cpu check serve serve-cuda serve-mps serve-cpu test

help: ## Show available commands.
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Detect the platform and install the appropriate serving backend.
	@if [ "$(shell uname -s)" = "Darwin" ] && [ "$(shell uname -m)" = "arm64" ]; then \
		echo "Apple Silicon detected: installing vLLM-Metal"; \
		uv sync --extra metal; \
	elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then \
		echo "NVIDIA CUDA detected: installing vLLM"; \
		uv sync --extra vllm; \
	else \
		echo "No supported accelerator detected: installing the Transformers CPU fallback"; \
		uv sync --extra cpu; \
	fi

install-metal: ## Install vLLM-Metal for Apple Silicon (native arm64 Python 3.12).
	uv sync --extra metal

install-cpu: ## Install the portable Transformers CPU fallback.
	uv sync --extra cpu

check: ## Show the backend and model selected by automatic detection.
	uv run --no-sync python serve.py --check

serve: ## Start the automatically selected OpenAI-compatible server.
	uv run --no-sync python serve.py

serve-cuda: ## Force vLLM CUDA serving.
	uv run --no-sync python serve.py --backend cuda

serve-mps: ## Force vLLM-Metal serving on Apple Silicon.
	uv run --no-sync python serve.py --backend mps

serve-cpu: ## Force the portable Transformers CPU fallback.
	uv run --no-sync python serve.py --backend cpu

test: ## Verify launcher syntax and automatic backend selection.
	uv run --no-sync python -m py_compile serve.py cpu_server.py
	uv run --no-sync python serve.py --check
