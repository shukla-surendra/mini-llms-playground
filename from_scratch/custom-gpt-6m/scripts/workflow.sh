#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

# This project is uv-managed (see pyproject.toml) — `uv run` provisions/updates .venv
# automatically before every command, so no separate install step is needed here.
# Equivalent, and preferred, Makefile targets exist for all of this (`make data`,
# `make train`, etc., plus `make snapshot`/`make publish` which this script doesn't
# cover) — this script is kept for people who prefer a plain shell entrypoint.

MAX_SAMPLES="${MAX_SAMPLES:-100000}"
VOCAB_SIZE="${VOCAB_SIZE:-4096}"

usage() {
  cat <<'EOF'
Usage:
  scripts/workflow.sh data     # download TinyStories, train tokenizer, tokenize
  scripts/workflow.sh train    # train the model
  scripts/workflow.sh infer    # generate a sample from the best checkpoint
  scripts/workflow.sh serve    # start the FastAPI server
  scripts/workflow.sh pipeline # data + train + infer
EOF
}

run_data() {
  uv run gpt-data --max-samples "$MAX_SAMPLES" --vocab-size "$VOCAB_SIZE"
}

run_train() {
  uv run gpt-train
}

run_infer() {
  uv run gpt-infer --prompt "${PROMPT:-Once upon a time,}"
}

run_serve() {
  uv run gpt-serve --host 127.0.0.1 --port 8010 --reload
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  data) run_data ;;
  train) run_train ;;
  infer) run_infer ;;
  serve) run_serve ;;
  pipeline)
    run_data
    run_train
    run_infer
    ;;
  *)
    usage
    exit 1
    ;;
esac
