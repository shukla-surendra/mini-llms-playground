.PHONY: help setup config presets data data-public tokenize tokenize-force benchmark benchmark-quick benchmark-sweep audit train train-fresh train-bg train-status train-stop train-logs infer serve eval test clean

GPT_TRAIN_BIN := $(CURDIR)/.venv/bin/gpt-train

# Refuses to start a second run against the same checkpoints/<label>/ — two concurrent
# gpt-train processes silently race on latest.pt/best.pt and roughly halve each other's
# throughput (a real, measured incident on this project) rather than failing loudly.
define guard_not_running
	@if pgrep -f "$(GPT_TRAIN_BIN)" > /dev/null 2>&1; then \
		echo "Error: gpt-train is already running (PID(s): $$(pgrep -f "$(GPT_TRAIN_BIN)" | tr '\n' ' '))." >&2; \
		echo "Two concurrent runs silently race on checkpoints/<label>/ and roughly halve" >&2; \
		echo "each other's throughput instead of failing loudly." >&2; \
		echo "Run 'make train-status' to check, or 'make train-stop' to stop it first." >&2; \
		exit 1; \
	fi
endef

help:
	@echo "custom-gpt — targets:"
	@echo ""
	@echo "  make setup         # uv sync: create .venv, install the package + deps"
	@echo "  make config        # show the active model config and exact parameter count"
	@echo "  make presets       # list every size preset with its parameter count"
	@echo ""
	@echo "  make data          # download all datasets + build the corpus (needs HF_TOKEN for LMSYS)"
	@echo "  make data-public   # same, but skip the gated LMSYS dataset (no token needed)"
	@echo "  make tokenize      # build data/*.bin token files (run once per corpus change)"
	@echo "  make benchmark     # measure throughput + project GPU hours/cost"
	@echo "  make audit         # corpus quality gate, run before a long training run"
	@echo ""
	@echo "  make train         # train in the foreground (auto-resumes from the latest checkpoint)"
	@echo "  make train-fresh   # train from scratch, ignoring existing checkpoints"
	@echo "  make train-bg      # train detached in the background — see below"
	@echo "  make train-status  # is a background run active? PID + last few log lines"
	@echo "  make train-stop    # stop a background run gracefully (SIGINT, saves latest.pt)"
	@echo "  make infer         # generate from the trained checkpoint"
	@echo "  make serve         # FastAPI server on :8000"
	@echo "  make eval          # heuristic generation-quality report"
	@echo "  make test          # run the curated QA prompt set, render an HTML report"
	@echo "  make clean         # remove checkpoints and logs (data/ is left alone)"
	@echo ""
	@echo "Optional parameters — same env vars work on train, train-fresh, AND train-bg:"
	@echo "  GPT_PRESET=30m make train                      # a named model-size preset"
	@echo "  GPT_EMBED_SIZE=192 GPT_NUM_LAYERS=8 make train  # or override individual fields"
	@echo "  ATTN_IMPL=naive make train                      # explicit nn.MultiheadAttention (default: sdpa)"
	@echo "  GPT_PRESET=30m ATTN_IMPL=sdpa make train-bg     # combine freely, same as foreground"
	@echo ""
	@echo "Attention kernel (same math, different memory-access pattern — see"
	@echo "docs/llm-engineering/25_efficient_attention_flash_and_sdpa.md):"
	@echo "  Resuming across a different ATTN_IMPL than the checkpoint was saved with"
	@echo "  remaps weights automatically — no progress lost, see checkpoint.remap_attn_impl."
	@echo ""
	@echo "Which checkpoint to eval/test (default: best.pt, falling back to latest/final):"
	@echo "  CHECKPOINT=latest make test    # best.pt can go stale after a non-regressive"
	@echo "  CHECKPOINT=latest make eval    # change (e.g. a corpus rebuild) — see checkpoint.select_checkpoint"

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

# Build data/*.bin (flat uint16 token files) once, so training never re-tokenizes
# the corpus on a GPU billed by the hour. Skips files whose .bin is already current.
tokenize: setup
	uv run gpt-tokenize

tokenize-force: setup
	uv run gpt-tokenize --force

# Measure real throughput and price the run BEFORE committing GPU hours.
# Defaults to 10 min warm-up + 50 min measured; --synthetic needs no corpus.
benchmark: setup
	uv run gpt-benchmark

benchmark-quick: setup
	uv run gpt-benchmark --warmup-min 2 --measure-min 5 --synthetic

benchmark-sweep: setup
	uv run gpt-benchmark --sweep-batch 8,16,24,32 --warmup-min 2 --measure-min 5 --synthetic

audit: setup
	uv run gpt-audit

train: setup tokenize
	$(guard_not_running)
	uv run gpt-train

train-fresh: setup tokenize
	$(guard_not_running)
	uv run gpt-train --no-resume

# Detached background run — survives closing this terminal. Forwards whatever env vars
# you set the same way `make train` does (GPT_PRESET, ATTN_IMPL, GPT_EMBED_SIZE, ...).
train-bg: setup
	$(guard_not_running)
	@mkdir -p logs
	@nohup uv run gpt-train > logs/train_stdout.log 2>&1 & \
	echo "Started in background — PID $$!  Log: logs/train_stdout.log"
	@echo "Check on it:   make train-status   /   tail -f logs/train_stdout.log"
	@echo "Stop cleanly:  make train-stop"

train-status:
	@if pgrep -f "$(GPT_TRAIN_BIN)" > /dev/null 2>&1; then \
		echo "Running — PID(s): $$(pgrep -f "$(GPT_TRAIN_BIN)" | tr '\n' ' ')"; \
		echo "Last log line:"; \
		tail -c 2000 logs/train_stdout.log 2>/dev/null | tr '\r' '\n' | grep . | tail -1; \
	else \
		echo "Not running."; \
	fi

train-stop:
	@PID="$$(pgrep -f "$(GPT_TRAIN_BIN)")"; \
	if [ -z "$$PID" ]; then \
		echo "Not running."; \
	else \
		mkdir -p checkpoints; \
		touch checkpoints/STOP_TRAINING; \
		echo "Sending SIGINT to PID $$PID and dropping checkpoints/STOP_TRAINING as a"; \
		echo "fallback (SIGINT isn't always caught — see Paths.stop_file's docstring in"; \
		echo "config.py). Either way, saves checkpoints/<label>/latest.pt before exiting."; \
		kill -INT $$PID; \
	fi

train-logs:
	@tail -f logs/train_stdout.log

infer: setup
	uv run gpt-infer

serve: setup
	uv run gpt-serve --reload

eval: setup
	uv run gpt-eval $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

test: setup
	uv run gpt-qa-report $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

clean:
	rm -rf checkpoints logs
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
