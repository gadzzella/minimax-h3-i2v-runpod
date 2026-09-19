# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG COMFYUI_REF=v0.36.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFYUI_ROOT=/workspace/comfyui \
    COMFYUI_MODELS_DIR=/workspace/comfyui/models \
    COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188 \
    COMFY_EXTRA_ARGS="--dont-upcast-attention"

# Correct Python 3.11 setup via deadsnakes PPA to prevent sys library collisions
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        git wget ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
RUN git clone --depth 1 --branch ${COMFYUI_REF} \
    https://github.com/Comfy-Org/ComfyUI.git comfyui

WORKDIR /workspace/comfyui

# Install PyTorch with explicit index preventing pip from downgrading torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

# Handler-side deps
COPY requirements.txt /workspace/handler-requirements.txt
RUN pip install -r /workspace/handler-requirements.txt

RUN mkdir -p models/diffusion_models models/text_encoders models/vae models/loras input output

# Bake model weights into image layer
COPY scripts/download_models.py /workspace/download_models.py
RUN --mount=type=secret,id=HF_TOKEN \
    --mount=type=secret,id=CIVITAI_TOKEN \
    HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || true)" \
    CIVITAI_TOKEN="$(cat /run/secrets/CIVITAI_TOKEN 2>/dev/null || true)" \
    python /workspace/download_models.py

# Handler code
COPY src/ /workspace/src/
COPY test_input.json /workspace/test_input.json

CMD ["python", "-u", "/workspace/src/handler.py"]