.DEFAULT_GOAL := help
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
.PHONY: help install install-metal install-cpu check serve serve-cuda serve-mps serve-cpu test

help: ## Show available commands.
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Detect the platform and install its serving backend.
	@if [ "$(shell uname -s)" = "Darwin" ] && [ "$(shell uname -m)" = "arm64" ]; then \
		uv sync --extra metal; \
	elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then \
		uv sync --extra vllm; \
	else \
		uv sync --extra cpu; \
	fi

install-metal: ## Install vLLM-Metal (native arm64 Python 3.12).
	uv sync --extra metal
install-cpu: ## Install the Transformers CPU fallback.
	uv sync --extra cpu
check: ## Print detected backend and model.
	uv run --no-sync python serve.py --check
serve: ## Start on port 8005 using automatic detection.
	uv run --no-sync python serve.py
serve-cuda: ## Force CUDA vLLM.
	uv run --no-sync python serve.py --backend cuda
serve-mps: ## Force Apple Metal vLLM.
	uv run --no-sync python serve.py --backend mps
serve-cpu: ## Force CPU fallback.
	uv run --no-sync python serve.py --backend cpu
test: ## Check launcher and CPU server syntax.
	uv run --no-sync python -m py_compile serve.py cpu_server.py
	uv run --no-sync python serve.py --check
