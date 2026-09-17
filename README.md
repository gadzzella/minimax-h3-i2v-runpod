# MiniMax H3 — Image-to-Video, baked into a RunPod serverless image

Bakes MiniMax's open-weight **H3** model (FL2VA / image-to-video checkpoint,
released 2026-08-03) and ComfyUI into a single Docker image at **build
time**, so a RunPod serverless worker has zero cold-start download — it just
loads weights already on disk.

## ⚠️ Read this before you build or deploy

**MiniMax H3's weights are released under MiniMax's own Community License,
not Apache/MIT.** That license's territory clause has been widely reported
as excluding the **US, EU, UK, and South Korea**. GitHub (Actions runners),
Docker Hub, and RunPod are all US-domiciled infrastructure providers.

This is a factual note, not legal advice — I'm not a lawyer and this isn't a
substitute for one. Before you build, push, or deploy this:

- Read the actual `LICENSE` file at
  [huggingface.co/MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  yourself — license terms and reporting on them can both be out of date by
  the time you read this.
- Consider where you (and your GitHub org / RunPod account / Docker Hub
  account) are actually located, and whether the license's territory and
  revenue clauses apply to you.
- If you're at all unsure, talk to someone qualified before shipping this
  publicly or commercially.

Nothing below changes or works around that license — it's just infrastructure.

## What's in the image

| Component | File | Size |
|---|---|---|
| Diffusion model — **TenStrip/10Eros-Max**, a NSFW-oriented MiniMax H3 fine-tune (TURBO-hybrid, int8-convrot) | `10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors` | ~21 GB |
| Text encoder (Qwen3-VL-32B based, NVFP4+AWQ) | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | ~15.7 GB |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | small |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | small |
| Turbo LoRA | not used — this checkpoint already bakes turbo distillation in | — |
| Engine | ComfyUI (pinned `v0.36.0`+, native `MiniMaxH3ImageToVideo` node) | — |

**This diffusion model is gated on Hugging Face** ("marked as containing
sensitive content"). Log into huggingface.co, open
[TenStrip/10Eros-Max](https://huggingface.co/TenStrip/10Eros-Max), and accept
its terms with your account *before* building — otherwise the download in
CI will fail with a 401/403. `HF_TOKEN` is now **required**, not optional:
generate one at huggingface.co/settings/tokens and set it as a repo secret.

Its model card states it's a fine-tune built for NSFW use on top of
MiniMax H3, distributed under MiniMax's own community license (`license:
other`, inherited — same territory caveat as the base model applies). To
switch back to the vanilla base model, see the revert notes in
`scripts/download_models.py` and `src/workflow.py`.

These are the exact files ComfyUI's own official `video_minimax_h3_i2v`
template uses — see `src/workflow.py`, which reconstructs that same node
graph in the `/prompt` API format by hand (subgraphs in the UI template
don't submit directly to the API).

Total baked weight is ~35–40 GB before the CUDA/PyTorch/ComfyUI layers.
Full BF16 weights exist too but roughly triple the size for a quality gain
that mostly matters at higher resolutions; swap the filenames in
`scripts/download_models.py` and `src/workflow.py` if you want them instead.

## Building

You said not to worry about GitHub Actions runner disk — `.github/workflows/docker-build.yml`
carries over your `free-disk-space` + manual-cleanup steps unchanged. On top of that:

- **HF_TOKEN is passed as a BuildKit secret**, not a build-arg, via
  `docker/build-push-action`'s `secrets:` input. That keeps it out of image
  layers and build history entirely — build-args get baked into history,
  secrets don't. It's optional: the source repos
  (`Comfy-Org/MiniMax-H3`, `lightx2v/Minimax-h3-Turbo`) are public as of
  writing, so an unset/empty `HF_TOKEN` secret works fine unless that changes.
- Add repo secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, and (optionally) `HF_TOKEN`.
- Push to `main`, or run the workflow manually.

To build locally (needs BuildKit and ~60GB+ free disk):

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=HF_TOKEN,env=HF_TOKEN \
  -t yourrepo/minimax-h3-i2v:latest .
```

## Deploying to RunPod Serverless

1. Push the built image to Docker Hub (the workflow does this for you).
2. In the RunPod console: **Serverless → New Endpoint → Custom Container**,
   point it at `yourrepo/minimax-h3-i2v:latest`.
3. **GPU**: pick something with real headroom. The diffusion model alone is
   ~20 GB and the text encoder ~16 GB; ComfyUI loads/offloads them
   sequentially, but you still want margin for activations and the video/audio
   VAEs. An **L40S (48GB)** or **A100 80GB** is a safe starting point; a 24GB
   card (4090/3090) is worth testing but may need `COMFY_EXTRA_ARGS=--lowvram`
   and could still OOM at higher resolutions or longer durations.
4. Set a generous **container disk** size (40–50 GB+ isn't needed since
   weights are baked into the image layer, not a volume — but leave room for
   ComfyUI's temp/output files per job).
5. No environment variables are required to run. Optional:
   - `COMFY_EXTRA_ARGS` — extra ComfyUI CLI flags, e.g. `--lowvram`
   - `JOB_TIMEOUT` — seconds before a job is considered stuck (default 1800)

## Calling the endpoint

```json
{
  "input": {
    "image": "https://example.com/your-first-frame.png",
    "prompt": "Describe the shot, camera motion, and audio (voice/SFX/music) in one block.",
    "width": 864,
    "height": 480,
    "duration_seconds": 4,
    "turbo": true,
    "seed": 42
  }
}
```

`image` accepts a base64 string (with or without a `data:` prefix) or an
`http(s)` URL. Optional `last_frame` (same formats) enables first+last-frame
mode. See `test_input.json` for a ready-to-use RunPod test payload borrowed
from ComfyUI's own example asset.

**Resolution**: H3's native canvas is a 768px short edge, capped at
768×1344, always a multiple of 32. `864×480` (~0.4 MP, 16:9) is a fast
default; see the size table in the official template
([video_minimax_h3_i2v.json](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json))
for the full range up to `1920×1088`.

**Response**:

```json
{
  "video_base64": "...",
  "seed": 42,
  "length_frames": 101,
  "generation_seconds": 47.3
}
```

Decode `video_base64` to get an MP4 with synced audio.

## Repo layout

```
Dockerfile                       # bakes ComfyUI + weights at build time
scripts/download_models.py       # pulls the exact files the template needs
src/workflow.py                  # hand-built API-format graph (mirrors the official template)
src/comfy_client.py              # talks to the local ComfyUI server
src/handler.py                   # RunPod serverless entrypoint
requirements.txt                 # handler-side Python deps
test_input.json                  # sample RunPod test payload
.github/workflows/docker-build.yml
```

## Notes / known limitations

- The turbo LoRA cuts sampling to ~8 steps at some quality cost; set
  `"turbo": false` in a job for the full 20-step base model.
- `src/workflow.py` was built by hand from ComfyUI's official template graph
  (fetched and inspected node-by-node), not exported via ComfyUI's own
  "Save (API Format)" — there was no running ComfyUI instance available to
  export from directly. It matches the template's node types, widget
  defaults (`res_multistep` / `simple` scheduler / 20 steps / the exact
  `17k+5`-frame length-snapping formula), and wiring. Still worth a real
  test run (`test_input.json`) before you rely on it.
- If MiniMax ships a newer H3 checkpoint or ComfyUI changes the native node's
  inputs, update `DIFFUSION_MODEL` / `TEXT_ENCODER` in `src/workflow.py` and
  the matching entries in `scripts/download_models.py`.
