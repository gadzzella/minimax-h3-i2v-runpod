"""
RunPod serverless handler for MiniMax H3 image-to-video.

Expected job input:
{
  "input": {
    "image": "<base64 png/jpg OR https:// URL>",
    "prompt": "text description of the shot(s) and audio",
    "width": 864,               # optional, multiple of 32, see ResolutionSelector table in README
    "height": 480,              # optional
    "duration_seconds": 4,      # optional, 1-15ish
    "seed": 0,                  # optional
    "turbo": true,              # optional, 8-step turbo LoRA vs 20-step base
    "last_frame": "<base64 or URL>"   # optional, enables first+last-frame mode
  }
}

Returns:
{
  "video_base64": "<mp4 bytes, base64>",
  "seed": <int used>,
  "length_frames": <int>
}
"""
from __future__ import annotations

import base64
import binascii
import os
import subprocess
import sys
import tempfile
import time
import uuid

import requests
import runpod

sys.path.insert(0, os.path.dirname(__file__))
import comfy_client  # noqa: E402
import workflow  # noqa: E402

COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT", "/workspace/comfyui")
COMFY_PORT = os.environ.get("COMFY_PORT", "8188")

_comfy_process: subprocess.Popen | None = None


def _start_comfyui() -> None:
    global _comfy_process
    if _comfy_process is not None and _comfy_process.poll() is None:
        return  # already running

    extra_args = os.environ.get("COMFY_EXTRA_ARGS", "").split()
    cmd = [
        sys.executable,
        os.path.join(COMFYUI_ROOT, "main.py"),
        "--listen", "0.0.0.0",
        "--port", COMFY_PORT,
        "--disable-auto-launch",
    ] + extra_args

    print(f"Launching ComfyUI: {' '.join(cmd)}", flush=True)
    _comfy_process = subprocess.Popen(cmd, cwd=COMFYUI_ROOT)
    comfy_client.wait_for_server(timeout=int(os.environ.get("COMFY_BOOT_TIMEOUT", "900")))
    print("ComfyUI is ready.", flush=True)


def _material_to_local_file(value: str, workdir: str, name: str) -> str:
    """Accepts a base64 string or an http(s) URL, writes it to disk, returns
    the local path."""
    path = os.path.join(workdir, name)
    if value.startswith("http://") or value.startswith("https://"):
        r = requests.get(value, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return path

    # assume base64, tolerate a data: URI prefix
    if "," in value and value.strip().startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        raw = base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise ValueError("`image`/`last_frame` must be a base64 string or an http(s) URL") from exc
    with open(path, "wb") as f:
        f.write(raw)
    return path


def handler(job: dict) -> dict:
    _start_comfyui()
    job_input = job.get("input", {})

    if "image" not in job_input or "prompt" not in job_input:
        return {"error": "`image` and `prompt` are required fields"}

    with tempfile.TemporaryDirectory() as tmp:
        local_image = _material_to_local_file(job_input["image"], tmp, "first_frame.png")
        uploaded_name = comfy_client.upload_image(local_image)

        last_frame_name = None
        if job_input.get("last_frame"):
            local_last = _material_to_local_file(job_input["last_frame"], tmp, "last_frame.png")
            last_frame_name = comfy_client.upload_image(local_last)

        seed = job_input.get("seed", int.from_bytes(os.urandom(4), "big"))

        graph, save_node = workflow.build_workflow(
            image_filename=uploaded_name,
            prompt=job_input["prompt"],
            width=int(job_input.get("width", 864)),
            height=int(job_input.get("height", 480)),
            duration_seconds=float(job_input.get("duration_seconds", 4.0)),
            seed=seed,
            last_frame_filename=last_frame_name,
            turbo=bool(job_input.get("turbo", True)),
            steps=job_input.get("steps"),
            output_prefix=f"video/{uuid.uuid4().hex}",
        )

        prompt_id = comfy_client.queue_prompt(graph)
        started = time.time()
        history = comfy_client.wait_for_completion(
            prompt_id, timeout=int(os.environ.get("JOB_TIMEOUT", "1800"))
        )
        video_path = comfy_client.find_output_video(history, save_node)

        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        try:
            os.remove(video_path)
        except OSError:
            pass

        return {
            "video_base64": video_b64,
            "seed": seed,
            "length_frames": workflow.snap_length(float(job_input.get("duration_seconds", 4.0))),
            "generation_seconds": round(time.time() - started, 1),
        }


if __name__ == "__main__":
    _start_comfyui()
    runpod.serverless.start({"handler": handler})
