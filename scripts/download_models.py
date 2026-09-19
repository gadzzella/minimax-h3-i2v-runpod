#!/usr/bin/env python3
"""
Bakes the MiniMax H3 (FL2VA / image-to-video) model files into the image at
build time, straight into ComfyUI's model folders.
"""
import gc
import os
import sys
import requests

# Disable hf_transfer to prevent GitHub Actions RAM OOM during massive downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import hf_hub_download

MODELS_DIR = os.environ.get("COMFYUI_MODELS_DIR", "/workspace/comfyui/models")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN") or None

FILES = [
    ("TenStrip/10Eros-Max",
     "10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors",
     "diffusion_models"),

    ("Comfy-Org/MiniMax-H3",
     "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     "text_encoders"),

    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", "vae"),
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", "vae"),
]

TURBO_LORA = (
    "lightx2v/Minimax-h3-Turbo",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "loras",
)

EXTRA_LORAS: list[tuple[str, str, str]] = []

DIRECT_URL_LORAS: list[tuple[str, str]] = [
    ("https://civitai.red/api/download/models/3266628?fileId=3150341",
     "helper_v4_fl2va_ref2va.safetensors"),
]


def fetch_lora(repo_id: str, filename: str, local_filename: str) -> None:
    dest_dir = os.path.join(MODELS_DIR, "loras")
    os.makedirs(dest_dir, exist_ok=True)
    print(f"==> LoRA {repo_id}/{filename} -> {dest_dir}/{local_filename}", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
    downloaded_path = os.path.join(dest_dir, filename)
    target_path = os.path.join(dest_dir, local_filename)
    if downloaded_path != target_path:
        os.replace(downloaded_path, target_path)
    gc.collect()


def fetch_lora_from_url(url: str, local_filename: str) -> None:
    dest_dir = os.path.join(MODELS_DIR, "loras")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, local_filename)
    print(f"==> LoRA {url} -> {dest_path}", flush=True)

    headers = {}
    if CIVITAI_TOKEN and "civitai." in url:
        headers["Authorization"] = f"Bearer {CIVITAI_TOKEN}"

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    gc.collect()


def fetch(repo_id: str, filename: str, subdir: str) -> None:
    dest_dir = os.path.join(MODELS_DIR, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    print(f"==> {repo_id}/{filename} -> {dest_dir}/", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
    gc.collect()


def main() -> None:
    for repo_id, filename, subdir in FILES:
        fetch(repo_id, filename, subdir)

    fetch(TURBO_LORA[0], TURBO_LORA[1], TURBO_LORA[2])

    for repo_id, filename, local_filename in EXTRA_LORAS:
        fetch_lora(repo_id, filename, local_filename)

    for url, local_filename in DIRECT_URL_LORAS:
        fetch_lora_from_url(url, local_filename)

    print("All MiniMax H3 model files baked into the image successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        sys.exit(1)