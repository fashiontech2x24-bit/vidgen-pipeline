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

        # Order matters: materialize the model on-device BEFORE loading the LoRA.
        # The CausVid T2V LoRA has no weights for VACE's vace_blocks / bias terms,
        # so PEFT creates those adapter slots empty. If the LoRA is applied while
        # the model is still on CPU/meta, those empty params stay on the `meta`
        # device and `.to("cuda")` then fails ("Cannot copy out of meta tensor").
        if MODELS.offload == "model":
            # CPU-offload hooks need the LoRA injected first, then offload.
            self._apply_lora(pipe)
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
            self._apply_lora(pipe)

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
            # low_cpu_mem_usage=False: force adapter params to be created as real
            # tensors (not on `meta`) so empty/missing-key slots don't break a
            # later device move.
            pipe.load_lora_weights(path, adapter_name="causvid", low_cpu_mem_usage=False)
            pipe.set_adapters(["causvid"], adapter_weights=[MODELS.causvid_lora_scale])
            self._materialize_meta_params(pipe.transformer)
            print(f"[engine] CausVid LoRA applied @ {MODELS.causvid_lora_scale} ({path})",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load CausVid LoRA from {path}. The Kijai/ComfyUI LoRA "
                f"format may need conversion for this diffusers version. Original: {e}"
            ) from e

    @staticmethod
    def _materialize_meta_params(module) -> None:
        """Replace any params still on the `meta` device with real zero tensors.

        LoRA adapter slots with no matching weights in the file (CausVid has none
        for vace_blocks / biases) can land on `meta`. Zero is the correct value:
        a zero lora_B / bias is an identity contribution, so this changes nothing
        numerically — it just makes the params movable/usable.
        """
        device = next(
            (p.device for p in module.parameters() if p.device.type != "meta"), None
        )
        if device is None:
            device = torch.device("cuda")
        n_fixed = 0
        for name, p in list(module.named_parameters()):
            if p.device.type != "meta":
                continue
            parent = module
            *parents, leaf = name.split(".")
            for attr in parents:
                parent = getattr(parent, attr)
            new_p = torch.nn.Parameter(
                torch.zeros(p.shape, dtype=p.dtype, device=device),
                requires_grad=p.requires_grad,
            )
            setattr(parent, leaf, new_p)
            n_fixed += 1
        if n_fixed:
            print(f"[engine] materialized {n_fixed} empty (meta) LoRA params to zeros",
                  flush=True)

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

        # Pass the prompt as a string (the documented path). The 0.34 VACE pipeline
        # rejects prompt=None, so we don't pre-cache prompt_embeds here; text
        # encoding is a small fixed cost inside pipe() and a candidate for later
        # optimization once the batched path is built.
        generator = torch.Generator(device="cuda").manual_seed(seed)

        with self._lock:  # single GPU: serialize the two pair jobs cleanly
            t = time.time()
            out = self.pipe(
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt,
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
            )
            timings["denoise_decode"] = time.time() - t

        return GenResult(frames=out.frames[0], timings=timings)


# Module-level singleton.
ENGINE = VaceEngine()
