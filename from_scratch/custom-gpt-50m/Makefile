.PHONY: help setup config presets data data-public tokenize tokenize-force data-pretrain data-posttrain use-pretrain use-posttrain benchmark benchmark-quick score audit train train-fresh train-bg train-status train-stop train-logs infer serve export-vllm serve-vllm eval test clean

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
	@echo "  make audit         # corpus quality gate, run before a long training run"
	@echo ""
	@echo "  Two-phase training (see docs/TWO_PHASE_TRAINING.md):"
	@echo "  make data-pretrain  # build the pretrain-profile corpus (153m-sourced raw prose)"
	@echo "  make data-posttrain # build the posttrain-profile corpus (pure chat, no extra docs)"
	@echo "  make use-pretrain   # switch active corpus to pretrain, re-tokenize"
	@echo "  make use-posttrain  # switch active corpus to posttrain, re-tokenize"
	@echo ""
	@echo "  make train         # train in the foreground (auto-resumes from the latest checkpoint)"
	@echo "  make train-fresh   # train from scratch, ignoring existing checkpoints"
	@echo "  make train-bg      # train detached in the background — see below"
	@echo "  make train-status  # is a background run active? PID + last few log lines"
	@echo "  make train-stop    # stop a background run gracefully (SIGINT, saves latest.pt)"
	@echo "  make infer         # generate from the trained checkpoint"
	@echo "  make serve         # FastAPI server on :8000"
	@echo "  make export-vllm   # export a checkpoint as an equivalent vLLM/HF GPT-2 model"
	@echo "  make serve-vllm VLLM_MODEL_DIR=exports/vllm/50m/best-step-<N>"
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

# Build data/*.bin token files once. Training memmaps them instead of re-tokenizing
# the 1.19 GB corpus on every start (and keeps the token stream out of GPU memory).
tokenize: setup
	uv run gpt-tokenize

tokenize-force: setup
	uv run gpt-tokenize --force

# --- Two-phase training: pretrain (raw prose) then post-train (pure chat) -----------
#
# data/profiles/<name>/{train,test}.txt hold each corpus variant. `make use-<name>`
# copies one into the canonical data/train.txt + test.txt (what gpt-train actually
# reads) and re-tokenizes. Only one profile is "active" at a time — switching is one
# command, nothing is silently mixed. See docs/TWO_PHASE_TRAINING.md.
#
#   make use-pretrain  && make train-fresh              # phase 1: base LM, from step 0
#   make use-posttrain && GPT_STEPS=<higher> make train # phase 2: resumes phase-1
#                                                        # weights, now on chat data
#
# data/profiles/pretrain/{train,test}.txt: built by scripts/build_pretrain_split.py
# from the 5 sources copied over from custom-gpt-153m's enriched raw pool (see that
# script's docstring — same held-out-last-1%-per-source method 153m itself used).
data-pretrain: setup
	uv run python scripts/build_pretrain_split.py
	@mkdir -p data/profiles/pretrain
	@mv data/train.txt data/profiles/pretrain/train.txt
	@mv data/test.txt data/profiles/pretrain/test.txt

# data/profiles/posttrain/{train,test}.txt: pure chat, no books/repos/wikipedia —
# gpt-data with no --extra-jsonl, reusing the 7 chat sources already in data/raw/.
data-posttrain: setup
	uv run gpt-data --skip-download
	@mkdir -p data/profiles/posttrain
	@mv data/train.txt data/profiles/posttrain/train.txt
	@mv data/test.txt data/profiles/posttrain/test.txt
	@mv data/test_prompts.txt data/profiles/posttrain/test_prompts.txt

use-pretrain:
	@test -f data/profiles/pretrain/train.txt || (echo "Run 'make data-pretrain' first." >&2; exit 1)
	cp data/profiles/pretrain/train.txt data/train.txt
	cp data/profiles/pretrain/test.txt data/test.txt
	$(MAKE) tokenize-force
	@echo "Active corpus: pretrain (data/profiles/pretrain/)"

use-posttrain:
	@test -f data/profiles/posttrain/train.txt || (echo "Run 'make data-posttrain' first." >&2; exit 1)
	cp data/profiles/posttrain/train.txt data/train.txt
	cp data/profiles/posttrain/test.txt data/test.txt
	-cp data/profiles/posttrain/test_prompts.txt data/test_prompts.txt
	$(MAKE) tokenize-force
	@echo "Active corpus: posttrain (data/profiles/posttrain/)"

# Measure real throughput and project GPU hours/cost before renting anything.
benchmark: setup
	uv run gpt-benchmark

benchmark-quick: setup
	uv run gpt-benchmark --warmup-min 2 --measure-min 5 --synthetic

# Objective, comparable score for a checkpoint (accuracy / constraints / MC).
score: setup
	uv run gpt-score --checkpoint latest

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
train-bg: setup tokenize
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

# Reads a checkpoint and writes exports/vllm/<label>/<checkpoint>-step-<N>/.
# It never changes checkpoints/<label>/, so native `make train` resume remains intact.
export-vllm:
	uv sync --extra vllm
	uv run --extra vllm gpt-export-vllm

# `make export-vllm` prints the immutable directory to pass as VLLM_MODEL_DIR.
# Extra flags such as --port 8001 can be supplied through VLLM_ARGS.
serve-vllm:
	@test -n "$(VLLM_MODEL_DIR)" || (echo "Set VLLM_MODEL_DIR to an export directory." >&2; exit 2)
	uv sync --extra vllm
	uv run --extra vllm vllm serve "$(VLLM_MODEL_DIR)" --model-impl transformers $(VLLM_ARGS)

eval: setup
	uv run gpt-eval

test: setup
	uv run gpt-qa-report

clean:
	rm -rf checkpoints logs
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
