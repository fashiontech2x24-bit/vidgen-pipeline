#!/usr/bin/env bash
# Container entrypoint: ensure models exist on the volume, then serve.
set -euo pipefail

export VID_MODELS_DIR="${VID_MODELS_DIR:-/workspace/models}"
export HF_HOME="${HF_HOME:-${VID_MODELS_DIR}/hf}"
export VID_OUTPUT_DIR="${VID_OUTPUT_DIR:-/workspace/outputs}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PORT="${PORT:-8000}"

mkdir -p "$VID_MODELS_DIR" "$VID_OUTPUT_DIR"

if [[ "${VID_SKIP_DOWNLOAD:-0}" != "1" ]]; then
  echo ">> Ensuring models are present..."
  python /app/scripts/download_models.py
fi

echo ">> Starting server on :${PORT}"
exec uvicorn app:app --host 0.0.0.0 --port "${PORT}" --app-dir /app/server
