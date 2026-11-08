"""OpenAI-compatible vLLM endpoint on Modal.

PREEXISTING: adapted from modal_vllm/vllm_serve.py in Sai's October AI Collusion
sprint repo, written before this sprint. Changes made during this sprint: newer
vLLM, tool-calling flags, one-container cap, a CPU download step, and a secret
that holds only the endpoint token.

Steps:
  SERVE_MODEL=openai/gpt-oss-120b GPU=H100 modal run modal_vllm/vllm_serve.py::download
  SERVE_MODEL=openai/gpt-oss-120b GPU=H100 modal deploy modal_vllm/vllm_serve.py
The URL from `modal deploy` plus /v1 goes in VLLM_BASE_URL in .env.

Cost control: scales to zero after SCALEDOWN idle seconds and never runs more
than one GPU container.
"""
import os
import shlex
import subprocess
import threading
from pathlib import Path

import modal

try:  # local deploy only, the container has no dotenv and gets the token from the secret
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

# Baked into the image, so the container never depends on local env vars.
MODEL = os.environ.get("SERVE_MODEL", "openai/gpt-oss-120b")
GPU = os.environ.get("GPU", "H100")
ARGS = os.environ.get(
    "VLLM_ARGS",
    "--enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss",
)
MAX_LEN = os.environ.get("MAX_MODEL_LEN", "40000")
SCALEDOWN = 300
PORT = 8000
APP_NAME = "swarm-vllm-" + MODEL.split("/")[-1].lower().replace(".", "-") + "-" + GPU.lower()

# The official image carries the CUDA toolkit. vLLM compiles kernels at startup and
# fails on a slim image with "Could not find nvcc".
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.33.0", add_python="3.12")
    .entrypoint([])
    .env({"SERVE_MODEL": MODEL, "VLLM_ARGS": ARGS, "MAX_MODEL_LEN": MAX_LEN})
)
fetch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "SERVE_MODEL": MODEL})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
volumes = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache}
token = modal.Secret.from_dict({"VLLM_API_KEY": os.environ.get("VLLM_API_KEY", "")})

app = modal.App(APP_NAME)


@app.function(image=fetch_image, volumes=volumes, timeout=60 * 60, cpu=4)
def download():
    """Fetch weights on a CPU container, so no GPU time goes to the download."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(os.environ["SERVE_MODEL"], ignore_patterns=["original/*", "metal/*"])
    hf_cache.commit()
    print("downloaded to", path)


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN,
    timeout=60 * 60 * 6,
    max_containers=1,
    volumes=volumes,
    secrets=[token],
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=PORT, startup_timeout=25 * 60)
def serve():
    model = os.environ["SERVE_MODEL"]
    cmd = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", os.environ["MAX_MODEL_LEN"],
        "--served-model-name", model,
        "--enable-prefix-caching",
        "--gpu-memory-utilization", "0.92",
    ] + shlex.split(os.environ["VLLM_ARGS"])
    proc = subprocess.Popen(cmd)

    def watch() -> None:
        # A dead server must end the container, or the GPU bills until startup_timeout.
        print("vllm exited with", proc.wait(), flush=True)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()
