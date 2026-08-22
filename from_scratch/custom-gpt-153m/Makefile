.PHONY: help setup config presets books books-force data data-public data-legacy data-legacy-public data-all data-all-public curate-shared tokenize tokenize-force tokenize-legacy tokenize-legacy-force benchmark benchmark-quick benchmark-sweep audit train train-fresh train-bg train-status train-stop train-logs infer serve export-vllm serve-vllm eval test clean

GPT_TRAIN_BIN := $(CURDIR)/.venv/bin/gpt-train
GPT_TRAIN_SFT_BIN := $(CURDIR)/.venv/bin/gpt-train-sft
CORPUS_EXTRACTOR := $(CURDIR)/../../tools/corpus-extractor/target/release/corpus-extractor

# Local folder of source books (PDF/EPUB) to extract — override on the command line
# for a one-off, e.g. `make books BOOKS_DIR=~/some-other-folder`.
BOOKS_DIR ?= ~/Downloads/books

# gpt-data auto-picks this up (see the `data`/`data-public` targets below) the moment
# it exists — `make books` is what creates it. Nothing to configure by hand: delete
# data/books_staging/ to go back to an HF-only corpus, it's that simple to opt out.
# Recursive (=), not immediate (:=): must re-check on every recipe invocation, not
# once at Makefile-parse time, so `make books data` (books, then data, in one command)
# sees the file books just created rather than the pre-books "doesn't exist yet" state.
EXTRA_JSONL_FLAG = $(if $(wildcard data/books_staging/dataset.jsonl),--extra-jsonl data/books_staging/dataset.jsonl,)

# Refuses to build on top of a symlinked data/ — this project owns its data
# independently (see docs/DATA_PIPELINE.md); every sibling project's dataset is
# different now, so a symlink here would mean silently reading or writing into
# ANOTHER project's directory instead of this one's. Checked with a real filesystem
# test (`-L`), not just documentation, specifically because the previous symlink here
# almost certainly came from copy-pasting a sibling project's setup steps — a doc
# alone doesn't stop that from happening again.
define guard_not_symlinked
	@if [ -L data ]; then \
		echo "Error: data/ is a symlink ($$(readlink data)) into another project." >&2; \
		echo "This project must own its data independently — every sibling's dataset" >&2; \
		echo "is different now, so building on a shared symlink risks silently reading" >&2; \
		echo "or writing the wrong project's data." >&2; \
		echo "Remove the symlink (rm data), then re-run this target to build this" >&2; \
		echo "project's own copy from scratch — see docs/DATA_PIPELINE.md." >&2; \
		exit 1; \
	fi
endef

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

# Same protection as guard_not_running, keyed to gpt-train-sft instead — a base
# pretraining run and an SFT run write to different checkpoints/<label>/ dirs so they
# can't corrupt each other's checkpoints, but on a single-GPU machine (e.g. this Mac)
# running both at once still just makes each other slower for no benefit.
define guard_not_running_sft
	@if pgrep -f "$(GPT_TRAIN_SFT_BIN)" > /dev/null 2>&1; then \
		echo "Error: gpt-train-sft is already running (PID(s): $$(pgrep -f "$(GPT_TRAIN_SFT_BIN)" | tr '\n' ' '))." >&2; \
		echo "Run 'make train-sft-status' to check, or 'make train-sft-stop' to stop it first." >&2; \
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
	@echo "  make books           # extract local books (BOOKS_DIR, default ~/Downloads/books) into data/books_staging/"
	@echo "                       # BOOKS_THREADS (default 2) caps parallel PDF/EPUB decompression — see Makefile comment"
	@echo "  make data            # download all HF datasets + build the corpus (needs HF_TOKEN for LMSYS)."
	@echo "                       # auto-includes data/books_staging/ if 'make books' has already run"
	@echo "                       # Rust (corpus-extractor build-corpus) — see 'make data-legacy' for the Python path"
	@echo "  make data-public     # same, but skip the gated LMSYS dataset (no token needed)"
	@echo "  make data-legacy         # Python fallback (gpt-data) for 'make data' — same output shape"
	@echo "  make data-legacy-public  # Python fallback for 'make data-public'"
	@echo "  make data-all        # books, then data, then tokenize — the full reproducible pipeline, one command"
	@echo "  make data-all-public # same, HF-public-only"
	@echo "  make tokenize        # build data/*.bin token files (run once per corpus change)."
	@echo "                       # Rust (corpus-extractor tokenize) — see 'make tokenize-legacy' for the Python path"
	@echo "  make tokenize-legacy       # Python fallback (gpt-tokenize) for 'make tokenize'"
	@echo "  make tokenize-legacy-force # Python fallback for 'make tokenize-force'"
	@echo "  make benchmark     # measure throughput + project GPU hours/cost"
	@echo "  make audit         # corpus quality gate, run before a long training run"
	@echo "  make curate-shared # curate ../_shared_data into data/curated/ (does not replace training data)"
	@echo ""
	@echo "  make train         # train in the foreground (auto-resumes from the latest checkpoint)"
	@echo "  make train-fresh   # train from scratch, ignoring existing checkpoints"
	@echo "  make export-vllm   # export a checkpoint as a vLLM/HF GPT-2 artifact"
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

# --chunk-tokens matches this project's context_length (1024) exactly, so a "1024-token
# chunk" here really is one training window's worth once it reaches gpt-train, not an
# estimate that drifts once real tokenization happens. --no-split: this step is purely
# extraction — the train/test split and shuffle happen once, across ALL sources
# together, in `data`/`data-public` below (see docs/DATA_PIPELINE.md).
#
# BOOKS_THREADS caps corpus-extractor's rayon pool for this stage. Left at 0 (rayon's
# own default of one thread per logical CPU), a folder of large/scanned PDFs gets
# several of them fully decompressed into memory at once — each pdf-extract call reads
# the whole file (fs::read) and then unpacks page/font/image streams, which can easily
# run several times the file's on-disk size. On a laptop with limited swap, that many
# multi-GB extractions running concurrently exceeds physical RAM and takes down the
# whole session, not just this process — a real incident on this machine (24GB unified
# memory, ~3GB swap) that looked like "VSCode crashed" but was actually system-wide
# memory pressure. Default of 2 keeps peak concurrent decompression bounded regardless
# of core count; override per-run for a folder known to be small/text-only, e.g.
# `make books BOOKS_THREADS=0`.
BOOKS_THREADS ?= 2

books: setup
	$(guard_not_symlinked)
	@mkdir -p $(BOOKS_DIR)
	@start=$$(date +%s); \
	$(CORPUS_EXTRACTOR) extract \
		--input $(BOOKS_DIR) \
		--output data/books_staging \
		--extensions pdf,epub \
		--chunk-tokens 1024 \
		--chunk-overlap 100 \
		--no-split \
		--threads $(BOOKS_THREADS); \
	status=$$?; \
	elapsed=$$(( $$(date +%s) - start )); \
	printf "wall clock: %02d:%02d:%02d\n" $$((elapsed/3600)) $$((elapsed%3600/60)) $$((elapsed%60)); \
	exit $$status
	@echo "books staged: data/books_staging/dataset.jsonl — 'make data' will include it automatically"

books-force: setup
	$(guard_not_symlinked)
	rm -rf data/books_staging
	$(MAKE) books

# Rust path (default): corpus-extractor build-corpus downloads/parses/quality-filters
# the registered HF datasets directly — no Python `datasets`/`huggingface_hub` needed
# for this stage. See tools/corpus-extractor/README.md and docs/DATA_PIPELINE.md.
data: setup
	$(guard_not_symlinked)
	$(CORPUS_EXTRACTOR) build-corpus --data-dir data $(EXTRA_JSONL_FLAG)

data-public: setup
	$(guard_not_symlinked)
	$(CORPUS_EXTRACTOR) build-corpus --data-dir data --no-gated $(EXTRA_JSONL_FLAG)

# Python fallback path — kept for reference/cross-checking, not the default. Same
# output shape (data/train.txt, data/test.txt, data/test_prompts.txt) as the Rust path.
data-legacy: setup
	$(guard_not_symlinked)
	uv run gpt-data $(EXTRA_JSONL_FLAG)

data-legacy-public: setup
	$(guard_not_symlinked)
	uv run gpt-data --no-gated $(EXTRA_JSONL_FLAG)

# The SFT (post-training) corpus — same registered chat/instruction datasets as
# `data`/`data-legacy` above, but written to data/sft/{train,test}.jsonl with turn
# structure preserved (not collapsed to flat text), since masked loss needs to know
# exactly which tokens are an Assistant turn's. See data/sft_prepare.py.
sft-data: setup
	$(guard_not_symlinked)
	uv run gpt-sft-data

# The full reproducible pipeline in one command, in the order that matters: extract
# local sources first (nothing downloads yet), THEN build the merged/shuffled corpus
# (downloads all registered HF datasets, pools them with the books JSONL, writes
# data/train.txt + data/test.txt), THEN tokenize. Each stage is also independently
# re-runnable on its own — see docs/DATA_PIPELINE.md for why they're kept separate
# rather than one script, and what to check after each one.
#
# Recipe (not plain prerequisites) so the elapsed-time wrapper covers the whole
# books+data+tokenize run, not just whichever stage happens to run last — `target:
# deps` alone has no hook that fires before the first dependency starts. Each stage
# runs as its own $(MAKE) so a failure partway through still reports the elapsed
# time so far before propagating the error (set -e stops at the failing stage; the
# trap prints and re-raises).
data-all:
	@start=$$(date +%s); \
	trap 'elapsed=$$(( $$(date +%s) - start )); printf "data-all wall clock: %02d:%02d:%02d\n" $$((elapsed/3600)) $$((elapsed%3600/60)) $$((elapsed%60))' EXIT; \
	set -e; \
	$(MAKE) books; \
	$(MAKE) data; \
	$(MAKE) tokenize

data-all-public:
	@start=$$(date +%s); \
	trap 'elapsed=$$(( $$(date +%s) - start )); printf "data-all-public wall clock: %02d:%02d:%02d\n" $$((elapsed/3600)) $$((elapsed%3600/60)) $$((elapsed%60))' EXIT; \
	set -e; \
	$(MAKE) books; \
	$(MAKE) data-public; \
	$(MAKE) tokenize

curate-shared: setup
	uv run gpt-curate-data --input-dir ../_shared_data --output-dir data/curated

# Build data/*.bin (flat uint16 token files) once, so training never re-tokenizes
# the corpus on a GPU billed by the hour. Skips files whose .bin is already current.
tokenize: setup
	$(guard_not_symlinked)
	$(CORPUS_EXTRACTOR) tokenize --data-dir data

tokenize-force: setup
	$(guard_not_symlinked)
	$(CORPUS_EXTRACTOR) tokenize --data-dir data --force

tokenize-legacy: setup
	$(guard_not_symlinked)
	uv run gpt-tokenize

tokenize-legacy-force: setup
	$(guard_not_symlinked)
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
	$(guard_not_symlinked)
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

# Instruction/chat fine-tuning on top of a base checkpoint — see sft_trainer.py.
# Requires data/sft/{train,test}.jsonl (`make sft-data` first).
train-sft: setup
	$(guard_not_running_sft)
	uv run gpt-train-sft

train-sft-bg: setup
	$(guard_not_running_sft)
	@mkdir -p logs
	@nohup uv run gpt-train-sft > logs/train_sft_stdout.log 2>&1 & \
	echo "Started in background — PID $$!  Log: logs/train_sft_stdout.log"
	@echo "Check on it:   make train-sft-status   /   tail -f logs/train_sft_stdout.log"
	@echo "Stop cleanly:  make train-sft-stop"

train-sft-status:
	@if pgrep -f "$(GPT_TRAIN_SFT_BIN)" > /dev/null 2>&1; then \
		echo "Running — PID(s): $$(pgrep -f "$(GPT_TRAIN_SFT_BIN)" | tr '\n' ' ')"; \
		echo "Last log line:"; \
		tail -c 2000 logs/train_sft_stdout.log 2>/dev/null | tr '\r' '\n' | grep . | tail -1; \
	else \
		echo "Not running."; \
	fi

# $(or $(GPT_PRESET),153m)-sft matches train_sft.py's own sft_label derivation
# (base-preset + "-sft"). NOT checkpoints/STOP_TRAINING (train-stop's file) — SFT uses
# its own namespaced stop file (checkpoints/<label>-sft/STOP_TRAINING, see
# sft_trainer.py) specifically so this can never also stop an unrelated, concurrently
# running base `gpt-train` process.
train-sft-stop:
	@PID="$$(pgrep -f "$(GPT_TRAIN_SFT_BIN)")"; \
	if [ -z "$$PID" ]; then \
		echo "Not running."; \
	else \
		mkdir -p checkpoints/$(or $(GPT_PRESET),153m)-sft; \
		touch checkpoints/$(or $(GPT_PRESET),153m)-sft/STOP_TRAINING; \
		echo "Sending SIGINT to PID $$PID and dropping"; \
		echo "checkpoints/$(or $(GPT_PRESET),153m)-sft/STOP_TRAINING as a fallback."; \
		kill -INT $$PID; \
	fi

infer: setup
	uv run gpt-infer

serve: setup
	uv run gpt-serve --reload

export-vllm:
	uv sync --extra hf-export
	uv run --extra hf-export gpt-export-vllm

serve-vllm:
	@if [ "$$(uname -s)" = "Linux" ]; then uv sync --extra vllm; fi
	@test -n "$(VLLM_MODEL_DIR)" || (echo "Set VLLM_MODEL_DIR to an export directory." >&2; exit 2)
	@test -x "$(CURDIR)/.venv/bin/vllm" || (echo "vLLM is not installed in this project's .venv." >&2; exit 2)
	"$(CURDIR)/.venv/bin/vllm" serve "$(VLLM_MODEL_DIR)" --model-impl transformers $(VLLM_ARGS)

eval: setup
	uv run gpt-eval $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

test: setup
	uv run gpt-qa-report $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) $(QA_ARGS)

clean:
	rm -rf checkpoints logs
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
