"""
I/O + preprocessing helpers for the VACE pipeline.

Turns raw user inputs (reference image bytes, control-video bytes/path) into the
exact tensors/PIL lists `WanVACEPipeline.__call__` expects, mirroring the
ComfyUI VHS_LoadVideo (force_rate=12) + LoadImage + WanVaceToVideo behaviour.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
from typing import List

import imageio.v3 as iio
import numpy as np
import PIL.Image
from PIL import Image


def load_reference_image(data: bytes | str, width: int, height: int) -> PIL.Image.Image:
    """Load the reference (identity) image and resize to the generation size."""
    if isinstance(data, (bytes, bytearray)):
        img = Image.open(io.BytesIO(data))
    else:
        img = Image.open(data)
    return img.convert("RGB").resize((width, height), Image.LANCZOS)


def _read_video_frames(src: bytes | str) -> List[np.ndarray]:
    """Decode a video to a list of HxWx3 uint8 RGB frames."""
    if isinstance(src, (bytes, bytearray)):
        # imageio needs a real container; write to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            frames = list(iio.imiter(path, plugin="pyav"))
        finally:
            os.unlink(path)
    else:
        frames = list(iio.imiter(src, plugin="pyav"))
    return [np.asarray(fr)[..., :3] for fr in frames]


def _resample_fps(frames: List[np.ndarray], src_fps: float, target_fps: int) -> List[np.ndarray]:
    """Nearest-neighbour temporal resample to target_fps (matches VHS force_rate)."""
    if src_fps <= 0 or abs(src_fps - target_fps) < 1e-3:
        return frames
    n_out = max(1, round(len(frames) * target_fps / src_fps))
    idx = np.linspace(0, len(frames) - 1, num=n_out).round().astype(int)
    return [frames[i] for i in idx]


def _probe_fps(src: bytes | str) -> float:
    try:
        if isinstance(src, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(src)
                path = f.name
            try:
                meta = iio.immeta(path, plugin="pyav")
            finally:
                os.unlink(path)
        else:
            meta = iio.immeta(src, plugin="pyav")
        return float(meta.get("fps", 0) or 0)
    except Exception:
        return 0.0


def load_control_video(
    src: bytes | str, width: int, height: int, num_frames: int, target_fps: int
) -> List[PIL.Image.Image]:
    """
    Decode the control (pose/motion) video, resample to target_fps, resize, and
    pad/trim to exactly num_frames — mirroring VHS_LoadVideo(force_rate=12) feeding
    WanVaceToVideo(length=29).
    """
    raw = _read_video_frames(src)
    if not raw:
        raise ValueError("control video decoded to 0 frames")
    raw = _resample_fps(raw, _probe_fps(src), target_fps)

    pil = [Image.fromarray(fr).convert("RGB").resize((width, height), Image.LANCZOS) for fr in raw]

    if len(pil) >= num_frames:
        return pil[:num_frames]
    # Pad by repeating the last frame (rare; control video should be >= 29 frames).
    pil.extend([pil[-1]] * (num_frames - len(pil)))
    return pil


def build_mask(width: int, height: int, num_frames: int, mode: str) -> List[PIL.Image.Image]:
    """
    VACE mask: white (255) = generate, black (0) = preserve/condition.
    See GenConfig.mask_mode — the convention to validate against ComfyUI.
    """
    value = 255 if mode == "generate_all" else 0
    frame = Image.new("L", (width, height), value)
    return [frame] * num_frames


def export_video(frames, out_path: str, fps: int) -> str:
    """Write decoded frames (np uint8 list / array) to an mp4 at the given fps."""
    arr = [np.asarray(f) for f in frames]
    arr = [(f * 255).astype(np.uint8) if f.dtype != np.uint8 else f for f in arr]
    iio.imwrite(out_path, np.stack(arr), plugin="pyav", codec="libx264", fps=fps)
    return out_path
