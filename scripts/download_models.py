#!/usr/bin/env python3
"""
Bakes the MiniMax H3 (FL2VA / image-to-video) model files into the image at
build time, straight into ComfyUI's model folders.

Files match exactly what ComfyUI's official native template
(video_minimax_h3_i2v.json, ComfyUI >= 0.33) expects to find, so the graph
in src/workflow.py resolves every checkpoint by filename with no extra config.

Controlled by env vars (all optional):
  HF_TOKEN              - HuggingFace token, needed for gated repos (e.g. TenStrip/10Eros-Max)
  CIVITAI_TOKEN          - Civitai API key, needed for some gated/mature Civitai downloads
  COMFYUI_MODELS_DIR     - defaults to /workspace/comfyui/models
"""
import os
import sys
import requests
from huggingface_hub import hf_hub_download

MODELS_DIR = os.environ.get("COMFYUI_MODELS_DIR", "/workspace/comfyui/models")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN") or None

# (repo_id, path_in_repo, local_subdir)
#
# Diffusion model swapped to TenStrip/10Eros-Max, a fine-tune of MiniMax H3
# (same architecture, same UNETLoader/int8-convrot format as the base model
# — that's why this is a one-entry swap and nothing else in this script
# changes). Gated repo: you must accept its terms on huggingface.co while
# logged in, and HF_TOKEN is REQUIRED for this one (not optional like the
# public Comfy-Org files below).
#
# To go back to the vanilla base model, swap this tuple back to:
#   ("Comfy-Org/MiniMax-H3",
#    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
#    "diffusion_models"),
FILES = [
    ("TenStrip/10Eros-Max",
     "10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors",
     "diffusion_models"),

    # Text encoder (Qwen3-VL-32B based), NVFP4+AWQ quantized: ~15.7 GB
    # instead of ~48 GB bf16. Unchanged by the fine-tune swap above — same
    # architecture means the same encoder still applies.
    ("Comfy-Org/MiniMax-H3",
     "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     "text_encoders"),

    # VAEs (small, unchanged).
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", "vae"),
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", "vae"),
]

TURBO_LORA = (
    "lightx2v/Minimax-h3-Turbo",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "loras",
)

# ---------------------------------------------------------------------------
# EXTRA LoRAs — add yours here.
#
# This is the ONE place to list LoRAs you want baked into the image. Each
# entry is (repo_id, filename_in_repo, local_filename). local_filename is
# what you'll reference from job input / workflow.py — it's usually just
# the same as filename_in_repo, but lets you rename on the way in if you
# want a short/consistent name across different source repos.
#
# These all land in models/loras/. Nothing here is wired into the graph
# automatically — see src/workflow.py's LORA_CATALOG for that half.
#
# Example:
# EXTRA_LORAS = [
#     ("someuser/some-h3-lora-repo", "cool_style_v2.safetensors", "cool_style_v2.safetensors"),
#     ("anotheruser/another-repo", "character_lora.safetensors", "character_lora.safetensors"),
# ]
EXTRA_LORAS: list[tuple[str, str, str]] = [
]

# ---------------------------------------------------------------------------
# EXTRA LoRAs from a direct URL (Civitai, or anywhere else that isn't the HF
# Hub). Each entry is (url, local_filename). Civitai's download endpoint
# (civitai.com or civitai.red — same backend, split by content rating) works
# straight off its "Copy download link" URL; CIVITAI_TOKEN is only needed
# for early-access or more heavily gated models, but it's sent whenever set.
#
# Example (the URL shape you get from Civitai's download button):
# DIRECT_URL_LORAS = [
#     ("https://civitai.red/api/download/models/3266628?fileId=3150341",
#      "helper_v4_fl2va_ref2va.safetensors"),
# ]
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


def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    for repo_id, filename, subdir in FILES:
        fetch(repo_id, filename, subdir)

    for repo_id, filename, local_filename in EXTRA_LORAS:
        fetch_lora(repo_id, filename, local_filename)

    for url, local_filename in DIRECT_URL_LORAS:
        fetch_lora_from_url(url, local_filename)

    if not EXTRA_LORAS and not DIRECT_URL_LORAS:
        print("No extra LoRAs configured — see EXTRA_LORAS / DIRECT_URL_LORAS near the top of this file.")

    print("All MiniMax H3 model files baked into the image.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Model download failed: {exc}", file=sys.stderr)
        sys.exit(1)
