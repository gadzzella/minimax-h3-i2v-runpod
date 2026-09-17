"""Minimal client for a locally-running ComfyUI server (127.0.0.1)."""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import requests

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = os.environ.get("COMFY_PORT", "8188")
BASE_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"


def wait_for_server(timeout: int = 600) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1)
    raise RuntimeError(f"ComfyUI server never became ready: {last_err}")


def upload_image(local_path: str, subfolder: str = "") -> str:
    """Uploads a file into ComfyUI's input/ dir, returns the filename to
    reference in a LoadImage node."""
    with open(local_path, "rb") as f:
        files = {"image": (os.path.basename(local_path), f)}
        data = {"overwrite": "true"}
        if subfolder:
            data["subfolder"] = subfolder
        r = requests.post(f"{BASE_URL}/upload/image", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()["name"]


def queue_prompt(graph: dict) -> str:
    client_id = str(uuid.uuid4())
    payload = {"prompt": graph, "client_id": client_id}
    r = requests.post(f"{BASE_URL}/prompt", json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected the prompt: {r.status_code} {r.text}")
    return r.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 1800) -> dict[str, Any]:
    """Polls /history/{id} until the job finishes; returns the history entry."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE_URL}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        hist = r.json()
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI job failed: {json.dumps(status)}")
        time.sleep(2)
    raise RuntimeError("Timed out waiting for ComfyUI to finish the job")


def find_output_video(history_entry: dict, save_node_id: str) -> str:
    """Returns the absolute path to the produced video file."""
    outputs = history_entry.get("outputs", {})
    node_out = outputs.get(save_node_id)
    if not node_out:
        raise RuntimeError(f"No outputs found for node {save_node_id}: {outputs}")

    for key in ("videos", "gifs", "images"):
        if key in node_out and node_out[key]:
            item = node_out[key][0]
            comfy_root = os.environ.get("COMFYUI_ROOT", "/workspace/comfyui")
            out_dir = os.path.join(comfy_root, "output")
            return os.path.join(out_dir, item.get("subfolder", ""), item["filename"])

    raise RuntimeError(f"Could not locate video in node output: {node_out}")
