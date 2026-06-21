# vidgen-pipeline

GPU inference server for the **Wan2.1-VACE-14B** fashion video pipeline — a
faithful port of the validated ComfyUI graph (`workflow_api.json`) to a
Diffusers `WanVACEPipeline`, served over HTTP for a local web app.

Given one **reference image** + two **control videos**, it produces two
**29-frame** videos (image+control₁, image+control₂) and reports inference
time + end-to-end latency.

## Why Diffusers (not LightX2V)

LightX2V was evaluated and **ruled out**: it has zero VACE support (accelerates
Wan2.1/2.2 T2V/I2V only), so it can't run the `WanVaceToVideo` reference-image +
control-video path this workflow depends on. Diffusers `WanVACEPipeline` natively
supports VACE 14B with reference images, control video, `UniPCMultistepScheduler`
(matches `uni_pc`), and LoRA loading for CausVid.

## Workflow → Diffusers mapping

| ComfyUI (`workflow_api.json`)        | Diffusers                                              |
| ------------------------------------ | ----------------------------------------------------- |
| `wan2.1_vace_14B_fp16`               | `Wan-AI/Wan2.1-VACE-14B-diffusers` (bf16)             |
| `WanVaceToVideo` (ref + control)     | `pipe(reference_images=[img], video=control, mask=…)` |
| `Wan21_CausVid_14B_T2V_lora` @ 0.3   | `load_lora_weights(...)` + `set_adapters(.., [0.3])`  |
| 4 steps · cfg 1 · uni_pc · shift 4   | `num_inference_steps=4, guidance_scale=1.0, flow_shift=4.0` |
| 768×1024 · 29 frames · 12 fps        | `width=768, height=1024, num_frames=29`, export fps 12 |

Two things to **validate on the first GPU run** (flagged in code):
1. **CausVid LoRA conversion** — the Kijai/ComfyUI LoRA must convert cleanly in
   this diffusers version (`inference.py::_apply_lora`).
2. **VACE mask convention** for control-video + reference together — toggle
   `VID_MASK_MODE` (`generate_all` ↔ `preserve_all`) and compare to ComfyUI.

## RunPod setup

**Recommended template:** a RunPod **PyTorch 2.x** pod (e.g.
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) on **1× A100 80GB**.
Attach a **persistent volume mounted at `/workspace`** (≥80 GB) so the ~40 GB of
models download once and survive restarts. Expose **HTTP port 8000**.

```bash
# in the pod terminal (clone into the persistent volume)
cd /workspace
git clone https://<TOKEN>@github.com/fashiontech2x24-bit/vidgen-pipeline.git
cd vidgen-pipeline

# install deps + download models + start server (foreground)
bash setup.sh
```

The server is then reachable at the pod's proxy URL:
`https://<POD_ID>-8000.proxy.runpod.net`

### Iteration loop

```bash
# Ctrl-C the server, then:
git pull
bash run.sh          # deps + models already in place — just restarts
```

## API

`POST /generate` (multipart):

| field             | type  | meaning                       |
| ----------------- | ----- | ----------------------------- |
| `user_image`      | file  | reference image (identity)    |
| `control_video_1` | file  | control/pose video for pair 1 |
| `control_video_2` | file  | control/pose video for pair 2 |

Response:

```json
{
  "job_id": "…",
  "results": [
    {"tag": "pair1", "video_url": "/outputs/…_pair1.mp4", "timings_s": {…}},
    {"tag": "pair2", "video_url": "/outputs/…_pair2.mp4", "timings_s": {…}}
  ],
  "config": {…},
  "metrics_s": {"inference_wall": 0.0, "end_to_end_latency": 0.0}
}
```

Videos are served from `/outputs/...`. `GET /health` reports load status.

Quick local test (from your machine, against the pod URL):

```bash
python scripts/test_client.py \
  --url https://<POD_ID>-8000.proxy.runpod.net \
  --image ref.png --c1 control1.mp4 --c2 control2.mp4
```

## Configuration (env vars)

All generation params are overridable without code changes — see `server/config.py`.
Common ones: `VID_MASK_MODE`, `VID_CAUSVID_SCALE` (0.3), `VID_FLOW_SHIFT` (4.0),
`VID_STEPS` (4), `VID_FRAMES` (29), `VID_WIDTH`/`VID_HEIGHT`, `VID_FPS`,
`VID_OFFLOAD` (`none` on A100 80GB), `HF_TOKEN` (only if a repo is gated).

## Layout

```
server/        config.py · vace_io.py · inference.py · app.py · requirements.txt
scripts/       download_models.py · test_client.py
docker/        Dockerfile · entrypoint.sh · build_and_push.sh   (optional path)
setup.sh       one-shot: install + download + serve
run.sh         start/restart server
workflow_api.json · download_v2v.sh   (ComfyUI reference)
```

## Docker (optional)

A `docker/Dockerfile` is provided if you later want a baked image instead of
git-clone. `DOCKER_USER=you bash docker/build_and_push.sh`.
