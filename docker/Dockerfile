# Wan2.1-VACE-14B video server — RunPod A100 80GB target.
# Base ships torch 2.5.1 + CUDA 12.4 (Ampere/sm_80 ok). Models are NOT baked in;
# they download to the persistent volume at first start (see entrypoint).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VID_MODELS_DIR=/workspace/models \
    HF_HOME=/workspace/models/hf \
    VID_OUTPUT_DIR=/workspace/outputs

# git: pip install diffusers from source. ffmpeg: belt-and-suspenders for media io.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

COPY server /app/server
COPY scripts /app/scripts
COPY docker /app/docker
RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker/entrypoint.sh"]
