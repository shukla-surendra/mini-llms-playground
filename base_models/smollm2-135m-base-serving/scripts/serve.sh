#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolLM2-135M}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8003}"

cd "${PROJECT_DIR}"
unset VIRTUAL_ENV
uv run api_server.py \
  --model-id "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT"
