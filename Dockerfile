# syntax=docker/dockerfile:1.7
#
# MiniMax H3 (image-to-video / FL2VA) baked into a RunPod-serverless-ready image.
# Everything the model needs — ComfyUI, weights, text encoder, VAEs — is
# downloaded at BUILD time so cold starts don't pay any download cost.
#
# Build (locally, needs a lot of disk — see README for the CI variant that
# frees GitHub Actions runner disk first):
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=HF_TOKEN,env=HF_TOKEN \
#     --secret id=CIVITAI_TOKEN,env=CIVITAI_TOKEN \
#     -t myrepo/minimax-h3-i2v:latest .

FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ARG COMFYUI_REF=v0.36.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFYUI_ROOT=/workspace/comfyui \
    COMFYUI_MODELS_DIR=/workspace/comfyui/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git wget ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

RUN python -m pip install --upgrade pip

# --- ComfyUI, pinned past the MiniMax H3 native-node merge (>= 0.33.0) ---
WORKDIR /workspace
RUN git clone --depth 1 --branch ${COMFYUI_REF} \
    https://github.com/Comfy-Org/ComfyUI.git comfyui

WORKDIR /workspace/comfyui

# Torch: ComfyUI (>= 0.33, and definitely v0.36+) requires cu130 or newer on
# modern NVIDIA GPUs — comfy_kitchen (its compiled quantization kernel
# package, needed for int8_convrot checkpoints like ours) relies on
# torch.library schema-inference behavior that only landed in newer torch
# releases. cu124's torch build predates it and fails at import with a
# "Parameter stride has unsupported type list[int]" error. Don't downgrade
# this back to cu124.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN pip install -r requirements.txt

# Handler-side deps.
COPY requirements.txt /workspace/handler-requirements.txt
RUN pip install -r /workspace/handler-requirements.txt

RUN mkdir -p models/diffusion_models models/text_encoders models/vae models/loras \
    input output

# --- Bake the model weights in. Token is mounted as a BuildKit secret so it
# never gets written into an image layer or the build history. ---
COPY scripts/download_models.py /workspace/download_models.py
RUN --mount=type=secret,id=HF_TOKEN \
    --mount=type=secret,id=CIVITAI_TOKEN \
    HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || true)" \
    CIVITAI_TOKEN="$(cat /run/secrets/CIVITAI_TOKEN 2>/dev/null || true)" \
    python /workspace/download_models.py

# --- Handler code ---
COPY src/ /workspace/src/
COPY test_input.json /workspace/test_input.json

ENV COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188

CMD ["python", "-u", "/workspace/src/handler.py"]
