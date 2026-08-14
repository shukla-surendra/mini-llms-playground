.PHONY: help setup config presets data data-public audit train train-fresh infer serve eval test clean

help:
	@echo "custom-gpt — targets:"
	@echo ""
	@echo "  make setup         # uv sync: create .venv, install the package + deps"
	@echo "  make config        # show the active model config and exact parameter count"
	@echo "  make presets       # list every size preset with its parameter count"
	@echo ""
	@echo "  make data          # download all datasets + build the corpus (needs HF_TOKEN for LMSYS)"
	@echo "  make data-public   # same, but skip the gated LMSYS dataset (no token needed)"
	@echo "  make audit         # corpus quality gate, run before a long training run"
	@echo ""
	@echo "  make train         # train (auto-resumes from the latest checkpoint)"
	@echo "  make train-fresh   # train from scratch, ignoring existing checkpoints"
	@echo "  make infer         # generate from the trained checkpoint"
	@echo "  make serve         # FastAPI server on :8000"
	@echo "  make eval          # heuristic generation-quality report"
	@echo "  make test          # run the curated QA prompt set, render an HTML report"
	@echo "  make clean         # remove checkpoints and logs (data/ is left alone)"
	@echo ""
	@echo "Change model size (no code edits):"
	@echo "  GPT_PRESET=30m make train"
	@echo "  GPT_EMBED_SIZE=192 GPT_NUM_LAYERS=8 make train"
	@echo ""
	@echo "Attention kernel (same math, different memory-access pattern — see"
	@echo "docs/llm-engineering/25_efficient_attention_flash_and_sdpa.md):"
	@echo "  ATTN_IMPL=sdpa make train    # F.scaled_dot_product_attention (default: naive)"
	@echo "  Resuming across a different ATTN_IMPL than the checkpoint was saved with"
	@echo "  remaps weights automatically — no progress lost, see checkpoint.remap_attn_impl."

setup:
	uv sync

config: setup
	uv run gpt-config

presets: setup
	uv run gpt-config --list

data: setup
	uv run gpt-data

data-public: setup
	uv run gpt-data --no-gated

audit: setup
	uv run gpt-audit

train: setup
	uv run gpt-train

train-fresh: setup
	uv run gpt-train --no-resume

infer: setup
	uv run gpt-infer

serve: setup
	uv run gpt-serve --reload

eval: setup
	uv run gpt-eval

test: setup
	uv run gpt-qa-report

clean:
	rm -rf checkpoints logs
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
