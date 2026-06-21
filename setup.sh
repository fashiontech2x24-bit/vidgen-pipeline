#!/usr/bin/env bash
#
# setup.sh — one-shot bring-up on a RunPod PyTorch pod (A100 80GB).
#
# Assumes a base image that already ships torch + CUDA (see README for the
# recommended RunPod template). Idempotent: re-running skips installed deps and
# already-downloaded models. Pass SKIP_SERVE=1 to set up without launching.
#
#   bash setup.sh                 # install + download + serve (foreground)
#   SKIP_SERVE=1 bash setup.sh    # install + download only
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Models/outputs live on the persistent /workspace volume so they survive restarts.
export VID_MODELS_DIR="${VID_MODELS_DIR:-/workspace/models}"
export HF_HOME="${HF_HOME:-${VID_MODELS_DIR}/hf}"
export VID_OUTPUT_DIR="${VID_OUTPUT_DIR:-/workspace/outputs}"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "================================================================"
echo " vidgen-pipeline setup"
echo "   repo:       $ROOT"
echo "   models:     $VID_MODELS_DIR"
echo "   outputs:    $VID_OUTPUT_DIR"
echo "================================================================"

# ---- 1. system deps (best-effort; base image usually has git already) -------
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -y >/dev/null 2>&1 || true
  apt-get install -y --no-install-recommends ffmpeg git aria2 >/dev/null 2>&1 || true
fi

# ---- 2. python deps (torch comes from the base image — not reinstalled) -----
echo ">> Installing Python dependencies..."
pip install --no-cache-dir -r "$ROOT/server/requirements.txt"

# ---- 3. models -> persistent volume -----------------------------------------
mkdir -p "$VID_MODELS_DIR" "$VID_OUTPUT_DIR"
echo ">> Downloading / verifying models (VACE 14B + CausVid LoRA)..."
python "$ROOT/scripts/download_models.py"

# ---- 4. serve ---------------------------------------------------------------
if [[ "${SKIP_SERVE:-0}" == "1" ]]; then
  echo ">> Setup complete. Start the server later with:  bash run.sh"
  exit 0
fi

echo ">> Launching server..."
exec bash "$ROOT/run.sh"
