#!/usr/bin/env bash
#
# run.sh — start (or restart) the inference server. Use this during iteration:
# after `git pull`, just re-run this; deps/models are already in place.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VID_MODELS_DIR="${VID_MODELS_DIR:-/workspace/models}"
export HF_HOME="${HF_HOME:-${VID_MODELS_DIR}/hf}"
export VID_OUTPUT_DIR="${VID_OUTPUT_DIR:-/workspace/outputs}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PORT="${PORT:-8000}"

echo ">> uvicorn on 0.0.0.0:${PORT}  (model load happens at startup)"
exec uvicorn app:app --host 0.0.0.0 --port "${PORT}" --app-dir "$ROOT/server"
