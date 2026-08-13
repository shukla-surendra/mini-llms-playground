#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8002}"

cd "${PROJECT_DIR}"
unset VIRTUAL_ENV
uv run api_server.py \
  --model-id "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT"
