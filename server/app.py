"""
FastAPI server for the Wan2.1-VACE-14B 2-pair video pipeline.

POST /generate
    multipart: user_image, control_video_1, control_video_2
    -> runs two VACE generations (image+control1, image+control2) and returns
       both output videos plus per-stage timing + end-to-end latency.

Single A100 80GB: the model is loaded once and resident; the two jobs share the
warm pipeline and cached prompt embeddings. The GPU-bound denoise is serialized
(one device), while decode/encode/IO of the two jobs overlap.
"""
from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import GEN, MODELS
from inference import ENGINE
from vace_io import build_mask, export_video, load_control_video, load_reference_image

OUTPUT_DIR = os.environ.get("VID_OUTPUT_DIR", "/workspace/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Wan2.1-VACE-14B Video Server")

# The web app runs locally (file:// -> "null" origin, or localhost) and calls the
# pod cross-origin, so allow all origins. No cookies/credentials are used.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
_pool = ThreadPoolExecutor(max_workers=2)


@app.on_event("startup")
def _startup() -> None:
    # Warm the model so the first request isn't penalised with load time.
    if os.environ.get("VID_EAGER_LOAD", "1") == "1":
        ENGINE.load()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": ENGINE._loaded, "model": MODELS.base_model_id}


def _run_one(job_id: str, ref_img, control_bytes: bytes, tag: str) -> dict:
    t0 = time.time()
    control = load_control_video(
        control_bytes, GEN.width, GEN.height, GEN.num_frames, GEN.fps
    )
    mask = build_mask(GEN.width, GEN.height, GEN.num_frames, GEN.mask_mode)
    preprocess_s = time.time() - t0

    result = ENGINE.generate(ref_img, control, mask)

    t = time.time()
    out_name = f"{job_id}_{tag}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    export_video(result.frames, out_path, GEN.fps)
    encode_s = time.time() - t

    return {
        "tag": tag,
        "video_url": f"/outputs/{out_name}",
        "timings_s": {
            "preprocess": round(preprocess_s, 3),
            **{k: round(v, 3) for k, v in result.timings.items()},
            "export": round(encode_s, 3),
        },
    }


@app.post("/generate")
async def generate(
    user_image: UploadFile = File(...),
    control_video_1: UploadFile = File(...),
    control_video_2: UploadFile = File(...),
) -> JSONResponse:
    req_start = time.time()
    job_id = uuid.uuid4().hex[:12]

    img_bytes = await user_image.read()
    c1 = await control_video_1.read()
    c2 = await control_video_2.read()

    ref_img = load_reference_image(img_bytes, GEN.width, GEN.height)

    # Two logical-parallel jobs; GPU denoise serializes inside the engine lock.
    infer_start = time.time()
    futures = [
        _pool.submit(_run_one, job_id, ref_img, c1, "pair1"),
        _pool.submit(_run_one, job_id, ref_img, c2, "pair2"),
    ]
    results = [f.result() for f in futures]
    inference_wall_s = time.time() - infer_start

    return JSONResponse(
        {
            "job_id": job_id,
            "results": results,
            "config": {
                "model": MODELS.base_model_id,
                "lora_scale": MODELS.causvid_lora_scale,
                "steps": GEN.steps,
                "size": f"{GEN.width}x{GEN.height}",
                "num_frames": GEN.num_frames,
                "fps": GEN.fps,
                "mask_mode": GEN.mask_mode,
            },
            "metrics_s": {
                "inference_wall": round(inference_wall_s, 3),
                "end_to_end_latency": round(time.time() - req_start, 3),
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
