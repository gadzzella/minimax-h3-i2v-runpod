#!/usr/bin/env python3
"""
Bakes the MiniMax H3 (FL2VA / image-to-video) model files into the image at
build time, straight into ComfyUI's model folders.

Files match exactly what ComfyUI's official native template
(video_minimax_h3_i2v.json, ComfyUI >= 0.33) expects to find, so the graph
in src/workflow.py resolves every checkpoint by filename with no extra config.

Controlled by env vars (all optional):
  HF_TOKEN              - HuggingFace token, only needed if a source repo is gated
  INCLUDE_TURBO_LORA     - "1" (default) also bakes the 8-step turbo LoRA
  COMFYUI_MODELS_DIR     - defaults to /workspace/comfyui/models
"""
import os
import sys
from huggingface_hub import hf_hub_download

MODELS_DIR = os.environ.get("COMFYUI_MODELS_DIR", "/workspace/comfyui/models")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
# Default OFF: the active diffusion model (TenStrip/10Eros-Max TURBO-hybrid)
# already has turbo distillation baked in, so the separate base-model turbo
# LoRA below would double up on it. Flip this back to "1" only if you swap
# the diffusion model back to a non-turbo base/fine-tune checkpoint.
INCLUDE_TURBO_LORA = os.environ.get("INCLUDE_TURBO_LORA", "0") == "1"

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

    if INCLUDE_TURBO_LORA:
        fetch(*TURBO_LORA)
    else:
        print("Skipping turbo LoRA (INCLUDE_TURBO_LORA=0)")

    print("All MiniMax H3 model files baked into the image.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Model download failed: {exc}", file=sys.stderr)
        sys.exit(1)
