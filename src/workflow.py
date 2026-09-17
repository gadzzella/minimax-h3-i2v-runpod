"""
Constructs the MiniMax H3 image-to-video graph in ComfyUI's API (/prompt)
format.

This mirrors ComfyUI's own native template `video_minimax_h3_i2v.json`
(Comfy-Org/workflow_templates), flattened out of its subgraph form:

  LoadImage -> MiniMaxH3ImageToVideo (clip, vae, first_frame, prompt, w, h, length)
             -> BasicGuider -> SamplerCustomAdvanced -> VAEDecode -> \
             -> VAEDecodeAudio -----------------------------------> CreateVideo -> SaveVideo

MiniMaxH3ImageToVideo is a native ComfyUI core node (added in ComfyUI PR
#15224, ComfyUI >= 0.33) that builds both the positive conditioning and the
initial A/V latent in one step, so no separate CLIPTextEncode is needed.
"""
from __future__ import annotations

# Active diffusion model: TenStrip/10Eros-Max, a MiniMax H3 fine-tune
# (same architecture/UNETLoader format as the base model). This is a
# "TURBO-hybrid" build — turbo distillation is already baked into the
# weights, so USE_SEPARATE_TURBO_LORA stays False: stacking the base
# model's separate turbo LoRA on top of an already-turbo checkpoint would
# double up the distillation and likely degrade or break output.
#
# To revert to the vanilla base model, set DIFFUSION_MODEL back to
# "minimax_h3_fl2va_pruned_int8_convrot.safetensors" and USE_SEPARATE_TURBO_LORA
# back to True (and re-enable INCLUDE_TURBO_LORA in download_models.py).
DIFFUSION_MODEL = "10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors"
USE_SEPARATE_TURBO_LORA = False

TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

# Friendly-name -> actual filename in models/loras/. Add one entry here for
# every LoRA you baked in via EXTRA_LORAS in scripts/download_models.py, so
# job requests can say {"name": "my_style"} instead of the full filename.
# (An unrecognized name is still tried as a literal filename, so this is
# convenience, not a hard requirement.)
LORA_CATALOG: dict[str, str] = {
    "helper_v4": "helper_v4_fl2va_ref2va.safetensors",
    # "my_style": "cool_style_v2.safetensors",
    # "character_x": "character_lora.safetensors",
}

# With a turbo-hybrid checkpoint, "turbo" just means "use fewer steps" —
# there's no LoRA to attach. Community reports for this checkpoint's turbo
# mode land around 4-8 steps; tune via the `steps` job field if results look
# under/over-cooked.
DEFAULT_STEPS = 20
TURBO_STEPS = 8
FPS = 24


def snap_length(duration_seconds: float) -> int:
    """
    Reproduce the official template's Math Expression node exactly:
        max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17
    H3 generates in blocks of 17 frames + 5, at 24 fps.
    """
    frames = max(5, round(duration_seconds * FPS))
    return frames + (5 - (frames % 17)) % 17


def build_workflow(
    image_filename: str,
    prompt: str,
    width: int = 864,
    height: int = 480,
    duration_seconds: float = 4.0,
    seed: int = 0,
    last_frame_filename: str | None = None,
    turbo: bool = True,
    steps: int | None = None,
    loras: list[dict] | None = None,
    output_prefix: str = "video/MiniMax_H3",
) -> tuple[dict, str]:
    """
    Returns (prompt_graph, save_node_id) ready to POST to ComfyUI's /prompt.

    image_filename / last_frame_filename must already exist in ComfyUI's
    `input/` directory (see handler.py, which saves uploaded images there).

    loras: optional list of {"name": str, "strength": float=1.0}, applied in
    order, each stacking on the previous one's output. `name` is looked up
    in LORA_CATALOG first, then tried as a literal filename in models/loras/.
    """
    length = snap_length(duration_seconds)
    g: dict = {}

    g["load_image"] = {
        "class_type": "LoadImage",
        "inputs": {"image": image_filename},
    }

    g["unet_loader"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
    }
    g["clip_loader"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"},
    }
    g["video_vae_loader"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": VIDEO_VAE},
    }
    g["audio_vae_loader"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": AUDIO_VAE},
    }

    model_ref = ["unet_loader", 0]
    if turbo and USE_SEPARATE_TURBO_LORA:
        g["turbo_lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": TURBO_LORA,
                "strength_model": 1.0,
            },
        }
        model_ref = ["turbo_lora", 0]

    for i, lora in enumerate(loras or []):
        lora_name = LORA_CATALOG.get(lora["name"], lora["name"])
        strength = float(lora.get("strength", 1.0))
        node_id = f"lora_{i}"
        g[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": lora_name,
                "strength_model": strength,
            },
        }
        model_ref = [node_id, 0]

    resolved_steps = steps or (TURBO_STEPS if turbo else DEFAULT_STEPS)

    h3_inputs = {
        "clip": ["clip_loader", 0],
        "vae": ["video_vae_loader", 0],
        "first_frame": ["load_image", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
    }
    if last_frame_filename:
        g["load_last_frame"] = {
            "class_type": "LoadImage",
            "inputs": {"image": last_frame_filename},
        }
        h3_inputs["last_frame"] = ["load_last_frame", 0]

    g["h3_i2v"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": h3_inputs}

    g["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["sampler_select"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "res_multistep"},
    }
    g["scheduler"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "model": model_ref,
            "scheduler": "simple",
            "steps": resolved_steps,
            "denoise": 1.0,
        },
    }
    g["guider"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": model_ref, "conditioning": ["h3_i2v", 0]},
    }
    g["sample"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise", 0],
            "guider": ["guider", 0],
            "sampler": ["sampler_select", 0],
            "sigmas": ["scheduler", 0],
            "latent_image": ["h3_i2v", 1],
        },
    }
    g["decode_video"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sample", 0], "vae": ["video_vae_loader", 0]},
    }
    g["decode_audio"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": ["sample", 0], "vae": ["audio_vae_loader", 0]},
    }
    g["create_video"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["decode_video", 0],
            "audio": ["decode_audio", 0],
            "fps": FPS,
        },
    }
    g["save_video"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["create_video", 0],
            "filename_prefix": output_prefix,
            "format": "auto",
            "codec": "auto",
        },
    }

    return g, "save_video"
