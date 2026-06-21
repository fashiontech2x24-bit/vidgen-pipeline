"""
Central configuration for the Wan2.1-VACE-14B video-generation server.

Every default here is a 1:1 port of the validated ComfyUI graph in
`workflow_api.json` (Wan VACE 14B V2V + CausVid 4-step). Keep them in sync.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


# --- Prompts (verbatim from workflow_api.json nodes 6 / 7) -------------------
POSITIVE_PROMPT = (
    "High-fidelity, cinematic fashion video of the model from the reference image. "
    "The camera remains completely static and fixed in place.The fabric of the garment "
    "drapes naturally, displaying realistic physical movement, soft folds, and consistent "
    "texture as the torso rotates. The soft, diffused key lighting highlights the fabric "
    "weave, preserving the garment’s logos and structural integrity.static locked-off "
    "camera, fixed tripod shot, no camera movement, no panning, no tilting, no zooming, "
    "no dolly, no trucking, no handheld motion, no camera shake, subject motion only"
)

NEGATIVE_PROMPT = (
    "过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走,"
    "camera movement, pan, tilt, zoom, dolly, truck, orbit, crane shot, handheld, "
    "camera shake, motion blur from camera movement, changing perspective"
)


@dataclass
class GenConfig:
    """Generation params mirroring the ComfyUI nodes."""

    # WanVaceToVideo (node 49)
    width: int = field(default_factory=lambda: _env_int("VID_WIDTH", 768))
    height: int = field(default_factory=lambda: _env_int("VID_HEIGHT", 1024))
    num_frames: int = field(default_factory=lambda: _env_int("VID_FRAMES", 29))
    conditioning_scale: float = field(  # ComfyUI "strength" = 1
        default_factory=lambda: _env_float("VID_COND_SCALE", 1.0)
    )

    # KSampler (node 3) — CausVid distilled
    steps: int = field(default_factory=lambda: _env_int("VID_STEPS", 4))
    guidance_scale: float = field(  # ComfyUI cfg = 1 -> CFG disabled
        default_factory=lambda: _env_float("VID_CFG", 1.0)
    )
    seed: int = field(default_factory=lambda: _env_int("VID_SEED", 806029486951597))

    # ModelSamplingSD3 (node 48) shift -> scheduler flow_shift
    flow_shift: float = field(default_factory=lambda: _env_float("VID_FLOW_SHIFT", 4.0))

    # CreateVideo (node 68)
    fps: int = field(default_factory=lambda: _env_int("VID_FPS", 12))

    # VACE control-video mask convention. The diffusers VACE PR has no example
    # combining a control video WITH a reference image (our exact case), so this
    # is the first knob to flip during quality validation against ComfyUI.
    #   "generate_all"  -> mask all-white (255): generate appearance everywhere,
    #                      control signal injected via VACE control branch.
    #   "preserve_all"  -> mask all-black (0): preserve/condition on control video.
    mask_mode: str = field(default_factory=lambda: _env("VID_MASK_MODE", "generate_all"))

    prompt: str = POSITIVE_PROMPT
    negative_prompt: str = NEGATIVE_PROMPT


@dataclass
class ModelConfig:
    """Model identifiers and on-disk locations."""

    # Diffusers-format VACE 14B (== wan2.1_vace_14B_fp16 in ComfyUI)
    base_model_id: str = field(
        default_factory=lambda: _env("VID_BASE_MODEL", "Wan-AI/Wan2.1-VACE-14B-diffusers")
    )
    # CausVid LoRA (Kijai/ComfyUI format) + strength 0.3 (node 107 strength_model)
    causvid_lora_repo: str = field(
        default_factory=lambda: _env("VID_CAUSVID_REPO", "Kijai/WanVideo_comfy")
    )
    causvid_lora_file: str = field(
        default_factory=lambda: _env(
            "VID_CAUSVID_FILE", "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
        )
    )
    causvid_lora_scale: float = field(
        default_factory=lambda: _env_float("VID_CAUSVID_SCALE", 0.3)
    )

    # Where models live on the RunPod volume (persisted across restarts).
    models_dir: str = field(default_factory=lambda: _env("VID_MODELS_DIR", "/workspace/models"))
    hf_home: str = field(default_factory=lambda: _env("HF_HOME", "/workspace/models/hf"))

    torch_dtype: str = field(default_factory=lambda: _env("VID_DTYPE", "bfloat16"))
    # "none" | "model" (sequential cpu offload). A100 80GB needs none.
    offload: str = field(default_factory=lambda: _env("VID_OFFLOAD", "none"))

    @property
    def lora_local_path(self) -> str:
        return os.path.join(self.models_dir, "loras", self.causvid_lora_file)


# Singletons used across the app.
GEN = GenConfig()
MODELS = ModelConfig()
