HOST ?= 127.0.0.1
PORT ?= 8010

GPT_TRAIN_BINS := $(CURDIR)/.venv/bin/gpt-train $(CURDIR)/.venv/bin/gpt-train-mlm \
                   $(CURDIR)/.venv/bin/gpt-train-contrastive $(CURDIR)/.venv/bin/gpt-train-ddp \
                   $(CURDIR)/.venv/bin/gpt-train-fsdp
GPT_TRAIN_PATTERN := gpt-train|gpt-train-mlm|gpt-train-contrastive|gpt-train-ddp|gpt-train-fsdp

.PHONY: help setup config presets data replay-mix \
        train train-fresh train-bg train-status train-stop train-logs \
        train-mlm train-contrastive train-ddp train-fsdp \
        bench-attention train-amp train-checkpointed \
        infer serve test snapshot list-snapshots restore-snapshot publish clean

# Refuses to start a second training run of ANY objective (causal/mlm/contrastive/
# ddp/fsdp) while one is already active — they all contend for the same CPU/MPS device,
# so two concurrent runs would silently race and roughly halve each other's throughput
# rather than failing loudly. Same guard pattern as the sibling custom-gpt-* projects,
# extended here to cover all five gpt-train-* binaries instead of just one.
define guard_not_running
	@if pgrep -f "$(GPT_TRAIN_PATTERN)" > /dev/null 2>&1; then \
		echo "Error: a gpt-train-* process is already running (PID(s): $$(pgrep -f "$(GPT_TRAIN_PATTERN)" | tr '\n' ' '))." >&2; \
		echo "Two concurrent runs silently race on checkpoints/<label>/<objective>/ and" >&2; \
		echo "roughly halve each other's throughput instead of failing loudly." >&2; \
		echo "Run 'make train-status' to check, or 'make train-stop' to stop it first." >&2; \
		exit 1; \
	fi
endef

help:
	@echo "custom-gpt-6m — targets:"
	@echo ""
	@echo "  make setup           # uv sync: create .venv, install the package + deps"
	@echo "  make config          # show the active model config and exact parameter count"
	@echo "  make presets         # list the size preset(s)"
	@echo ""
	@echo "  make data            # download TinyStories, train tokenizer, tokenize"
	@echo "  make replay-mix NEW=... REPLAY_FROM=... OUT=...  # mix in older tokenized data"
	@echo ""
	@echo "  make train           # causal LM (resumes automatically if a checkpoint exists)"
	@echo "  make train-fresh     # causal LM, ignoring any checkpoint"
	@echo "  make train-bg        # causal LM, detached in the background — see below"
	@echo "  make train-status    # is a background causal-LM run active? PID + last log line"
	@echo "  make train-stop      # stop a background causal-LM run gracefully (SIGINT)"
	@echo "  make train-logs      # tail the background causal-LM run's stdout log"
	@echo "  make train-mlm       # train the masked-LM (BERT-style) objective"
	@echo "  make train-contrastive  # train the SimCSE-style contrastive/InfoNCE objective"
	@echo "  make train-ddp WORLD_SIZE=2   # DistributedDataParallel demo (multi-process CPU, gloo)"
	@echo "  make train-fsdp WORLD_SIZE=2  # FullyShardedDataParallel demo (multi-process CPU, gloo)"
	@echo "  make bench-attention # short fresh runs comparing ATTN_IMPL=naive vs sdpa throughput"
	@echo "  make train-amp       # fresh causal-LM run with ATTN_IMPL=sdpa AMP=1"
	@echo "  make train-checkpointed  # fresh causal-LM run with ATTN_IMPL=sdpa AMP=1 GRAD_CHECKPOINT=1"
	@echo ""
	@echo "  make infer PROMPT=...  # generate a sample from the best causal-LM checkpoint"
	@echo "  make serve            # start the FastAPI server (HOST=... PORT=... to override)"
	@echo "  make test             # run the curated story-prompt set, render an HTML report"
	@echo ""
	@echo "  make snapshot NAME=round2-shakespeare    # archive current checkpoints"
	@echo "  make list-snapshots                       # list archived snapshots"
	@echo "  make restore-snapshot NAME=...            # restore an archived snapshot as active"
	@echo "  make publish REPO_ID=user/model-name      # upload checkpoint+tokenizer+code to the HF Hub"
	@echo "                                               (reads HF_TOKEN from .env — see .env.example)"
	@echo "  make clean            # remove checkpoints, logs, and .venv"
	@echo ""
	@echo "Optional parameters — env vars work the same way on every train-* target:"
	@echo "  GPT_EMBED_SIZE=384 GPT_NUM_LAYERS=8 make train  # override an architecture field"
	@echo "  ATTN_IMPL=sdpa make train                       # explicit sdpa kernel (default: naive)"
	@echo "  AMP=1 GRAD_CHECKPOINT=1 make train              # mixed precision / activation checkpointing"
	@echo "  GPT_STEPS=200 GPT_EVAL_INTERVAL=50 make train   # quick smoke run"

setup:
	uv sync

config: setup
	uv run gpt-config

presets: setup
	uv run gpt-config --list

data: setup
	uv run gpt-data --max-samples $${MAX_SAMPLES:-100000} --vocab-size $${VOCAB_SIZE:-4096}

replay-mix: setup
	uv run gpt-replay-mix --new "$(NEW)" --replay-from "$(REPLAY_FROM)" --out "$(OUT)"

train: setup
	$(guard_not_running)
	uv run gpt-train

train-fresh: setup
	$(guard_not_running)
	uv run gpt-train --no-resume

# Detached background run — survives closing this terminal.
train-bg: setup
	$(guard_not_running)
	@mkdir -p logs
	@nohup uv run gpt-train > logs/train_stdout.log 2>&1 & \
	echo "Started in background — PID $$!  Log: logs/train_stdout.log"
	@echo "Check on it:   make train-status   /   tail -f logs/train_stdout.log"
	@echo "Stop cleanly:  make train-stop"

train-status:
	@if pgrep -f "$(GPT_TRAIN_PATTERN)" > /dev/null 2>&1; then \
		echo "Running — PID(s): $$(pgrep -f "$(GPT_TRAIN_PATTERN)" | tr '\n' ' ')"; \
		echo "Last log line:"; \
		tail -c 2000 logs/train_stdout.log 2>/dev/null | tr '\r' '\n' | grep . | tail -1; \
	else \
		echo "Not running."; \
	fi

train-stop:
	@PID="$$(pgrep -f "$(GPT_TRAIN_PATTERN)")"; \
	if [ -z "$$PID" ]; then \
		echo "Not running."; \
	else \
		echo "Sending SIGINT to PID $$PID — saves checkpoints/<label>/<objective>/latest.pt before exiting."; \
		kill -INT $$PID; \
	fi

train-logs:
	@tail -f logs/train_stdout.log

train-mlm: setup
	$(guard_not_running)
	uv run gpt-train-mlm

train-contrastive: setup
	$(guard_not_running)
	uv run gpt-train-contrastive

# See docs/DISTRIBUTED_TRAINING.md — multi-process CPU (gloo backend) mechanism demos,
# not GPU-cluster speed benchmarks. Launched directly (mp.spawn internally, WORLD_SIZE
# env var), not via torchrun — see that doc's gotcha section for why.
train-ddp: setup
	$(guard_not_running)
	uv run gpt-train-ddp

train-fsdp: setup
	$(guard_not_running)
	uv run gpt-train-fsdp

# See docs/EFFICIENT_TRAINING.md for the real measured numbers these produce.
bench-attention: setup
	@echo "--- ATTN_IMPL=naive ---"
	STEPS=100 GPT_EVAL_INTERVAL=1000 GPT_SAVE_EVERY=1000 ATTN_IMPL=naive uv run gpt-train --no-resume
	@echo "--- ATTN_IMPL=sdpa ---"
	STEPS=100 GPT_EVAL_INTERVAL=1000 GPT_SAVE_EVERY=1000 ATTN_IMPL=sdpa uv run gpt-train --no-resume

train-amp: setup
	ATTN_IMPL=sdpa AMP=1 uv run gpt-train --no-resume

train-checkpointed: setup
	ATTN_IMPL=sdpa AMP=1 GRAD_CHECKPOINT=1 uv run gpt-train --no-resume

infer: setup
	uv run gpt-infer --prompt "$(if $(PROMPT),$(PROMPT),Once upon a time,)"

serve: setup
	uv run gpt-serve --host $(HOST) --port $(PORT) --reload

test: setup
	uv run gpt-test

# Archive the current checkpoints + eval logs before starting a new dataset round, so you
# can roll back if the new round hurts quality more than it helps. See
# docs/CONTINUAL_TRAINING_LOW_RESOURCE.md.
snapshot:
	@if [ -z "$(NAME)" ]; then echo "Usage: make snapshot NAME=round2-shakespeare"; exit 1; fi
	@mkdir -p checkpoints_archive/$(NAME)
	@cp -r checkpoints checkpoints_archive/$(NAME)/checkpoints 2>/dev/null || true
	@cp -r logs checkpoints_archive/$(NAME)/logs 2>/dev/null || true
	@echo "Snapshotted current checkpoints to checkpoints_archive/$(NAME)/"

list-snapshots:
	@ls -1 checkpoints_archive/ 2>/dev/null || echo "(no snapshots yet)"

restore-snapshot:
	@if [ -z "$(NAME)" ]; then echo "Usage: make restore-snapshot NAME=round2-shakespeare"; exit 1; fi
	@if [ ! -d "checkpoints_archive/$(NAME)" ]; then echo "No snapshot named $(NAME)"; exit 1; fi
	@cp -r checkpoints_archive/$(NAME)/checkpoints/* checkpoints/ 2>/dev/null || true
	@echo "Restored checkpoints from checkpoints_archive/$(NAME)/ as the active checkpoints"

# Reads HF_TOKEN from .env automatically (scripts/upload_to_hf.py loads it via
# python-dotenv) — see .env.example and docs/PUBLISHING_TO_HUGGING_FACE.md.
publish: setup
	@if [ -z "$(REPO_ID)" ]; then echo "Usage: make publish REPO_ID=your-username/custom-gpt-6m"; exit 1; fi
	uv run scripts/upload_to_hf.py --repo-id "$(REPO_ID)"

clean:
	rm -rf checkpoints logs
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
