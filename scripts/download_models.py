"""
Fetch the diffusers-format VACE 14B model + CausVid LoRA into the RunPod volume.

Run at container start (idempotent — skips what's already present). Mirrors the
model set in `download_v2v.sh` but for the diffusers layout instead of ComfyUI.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

# Mirror server/config.py defaults (kept standalone so this runs without the app).
BASE_MODEL = os.environ.get("VID_BASE_MODEL", "Wan-AI/Wan2.1-VACE-14B-diffusers")
CAUSVID_REPO = os.environ.get("VID_CAUSVID_REPO", "Kijai/WanVideo_comfy")
CAUSVID_FILE = os.environ.get(
    "VID_CAUSVID_FILE", "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
)
MODELS_DIR = os.environ.get("VID_MODELS_DIR", "/workspace/models")
HF_HOME = os.environ.get("HF_HOME", os.path.join(MODELS_DIR, "hf"))
TOKEN = os.environ.get("HF_TOKEN")

os.makedirs(HF_HOME, exist_ok=True)
LORA_DIR = os.path.join(MODELS_DIR, "loras")
os.makedirs(LORA_DIR, exist_ok=True)


def main() -> int:
    print(f">> Base model:   {BASE_MODEL}")
    print(f">> CausVid LoRA: {CAUSVID_REPO}/{CAUSVID_FILE}")
    print(f">> HF cache:     {HF_HOME}")
    print(f">> LoRA dir:     {LORA_DIR}")

    # Full diffusers repo (transformer / vae / text_encoder / tokenizer / scheduler)
    # into the HF cache on the persistent volume; from_pretrained reuses it.
    snapshot_download(
        repo_id=BASE_MODEL,
        cache_dir=HF_HOME,
        token=TOKEN,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.spm"],
    )

    lora_path = os.path.join(LORA_DIR, CAUSVID_FILE)
    if not os.path.exists(lora_path):
        got = hf_hub_download(
            repo_id=CAUSVID_REPO, filename=CAUSVID_FILE, token=TOKEN, cache_dir=HF_HOME
        )
        # Materialize a stable path the engine reads from.
        if os.path.realpath(got) != os.path.realpath(lora_path):
            import shutil

            shutil.copy(got, lora_path)
    print(f">> [ok] LoRA ready: {lora_path}")
    print(">> All models present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
