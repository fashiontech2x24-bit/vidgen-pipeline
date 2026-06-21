"""
VACE inference engine.

Loads `Wan-AI/Wan2.1-VACE-14B-diffusers` + the CausVid LoRA once, then serves
4-step generations that reproduce the ComfyUI `workflow_api.json` graph.

The prompt is constant across requests, so its text embeddings are encoded once
and reused — every request then only pays for the 4 DiT denoising steps + VAE.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import PIL.Image
import torch

from config import GEN, MODELS, GenConfig

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class GenResult:
    frames: list
    timings: dict  # seconds, per stage


class VaceEngine:
    def __init__(self) -> None:
        self.pipe = None
        self._text_cache: dict = {}
        self._lock = threading.Lock()  # one GPU -> serialize denoising
        self._loaded = False

    # -- loading ------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        dtype = _DTYPES[MODELS.torch_dtype]
        t0 = time.time()

        # VAE stays fp32 (per Wan diffusers guidance) for decode fidelity.
        vae = AutoencoderKLWan.from_pretrained(
            MODELS.base_model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanVACEPipeline.from_pretrained(
            MODELS.base_model_id, vae=vae, torch_dtype=dtype
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=GEN.flow_shift
        )

        self._apply_lora(pipe)

        if MODELS.offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        self.pipe = pipe
        self._loaded = True
        print(f"[engine] loaded in {time.time() - t0:.1f}s "
              f"(model={MODELS.base_model_id}, dtype={MODELS.torch_dtype}, "
              f"offload={MODELS.offload})", flush=True)

    def _apply_lora(self, pipe) -> None:
        """Load the CausVid LoRA at the validated 0.3 strength (ComfyUI node 107).

        The LoRA is in Kijai/ComfyUI format; recent diffusers converts Wan LoRAs
        on load. If conversion ever fails this raises loudly — it's the #1
        validation risk flagged for the first RunPod run.
        """
        path = MODELS.lora_local_path
        try:
            pipe.load_lora_weights(path, adapter_name="causvid")
            pipe.set_adapters(["causvid"], adapter_weights=[MODELS.causvid_lora_scale])
            print(f"[engine] CausVid LoRA applied @ {MODELS.causvid_lora_scale} ({path})",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load CausVid LoRA from {path}. The Kijai/ComfyUI LoRA "
                f"format may need conversion for this diffusers version. Original: {e}"
            ) from e

    # -- text encoding (cached) --------------------------------------------
    def _text_embeds(self, cfg: GenConfig):
        do_cfg = cfg.guidance_scale > 1.0
        key = (cfg.prompt, cfg.negative_prompt if do_cfg else None)
        if key in self._text_cache:
            return self._text_cache[key]
        with torch.no_grad():
            pos, neg = self.pipe.encode_prompt(
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt if do_cfg else None,
                do_classifier_free_guidance=do_cfg,
                num_videos_per_prompt=1,
                device=self.pipe._execution_device,
            )
        self._text_cache[key] = (pos, neg)
        return pos, neg

    # -- generation ---------------------------------------------------------
    def generate(
        self,
        reference_image: PIL.Image.Image,
        control_frames: List[PIL.Image.Image],
        mask: List[PIL.Image.Image],
        cfg: Optional[GenConfig] = None,
        seed: Optional[int] = None,
    ) -> GenResult:
        if not self._loaded:
            self.load()
        cfg = cfg or GEN
        seed = GEN.seed if seed is None else seed
        timings: dict = {}

        t = time.time()
        try:
            pos, neg = self._text_embeds(cfg)
            text_kwargs = {"prompt_embeds": pos, "negative_prompt_embeds": neg}
        except Exception as e:  # noqa: BLE001 — fall back to raw prompt if signature differs
            print(f"[engine] prompt-embed cache disabled ({e}); using raw prompt", flush=True)
            text_kwargs = {"prompt": cfg.prompt, "negative_prompt": cfg.negative_prompt}
        timings["text_encode"] = time.time() - t

        generator = torch.Generator(device="cuda").manual_seed(seed)

        with self._lock:  # single GPU: serialize the two pair jobs cleanly
            t = time.time()
            out = self.pipe(
                video=control_frames,
                mask=mask,
                reference_images=[reference_image],
                height=cfg.height,
                width=cfg.width,
                num_frames=cfg.num_frames,
                num_inference_steps=cfg.steps,
                guidance_scale=cfg.guidance_scale,
                conditioning_scale=cfg.conditioning_scale,
                generator=generator,
                output_type="np",
                **text_kwargs,
            )
            timings["denoise_decode"] = time.time() - t

        return GenResult(frames=out.frames[0], timings=timings)


# Module-level singleton.
ENGINE = VaceEngine()
