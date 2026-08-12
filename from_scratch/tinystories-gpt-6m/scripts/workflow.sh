#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${PY:-}" ]]; then
  PY_BIN="$PY"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
else
  PY_BIN="python3"
fi

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
  "$PY_BIN" prepare_dataset.py --max-samples "$MAX_SAMPLES" --vocab-size "$VOCAB_SIZE"
}

run_train() {
  "$PY_BIN" train.py
}

run_infer() {
  "$PY_BIN" inference.py --prompt "${PROMPT:-Once upon a time,}"
}

run_serve() {
  "$PY_BIN" -m uvicorn api_server:app --host 127.0.0.1 --port 8010 --reload
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
