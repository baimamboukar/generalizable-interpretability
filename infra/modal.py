from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import modal

from infra.env import job_environment, load_project_env

ROOT = Path(__file__).resolve().parents[1]
load_project_env()
REMOTE_ROOT = Path("/workspace")
REMOTE_CACHE = Path("/cache")
EXTERNAL = tuple(filter(None, os.environ.get("GI_EXTERNAL", "").split(",")))

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")
    .pip_install("uv")
    .add_local_dir(
        ROOT,
        str(REMOTE_ROOT),
        copy=True,
        ignore=[
            ".git",
            ".venv",
            ".env",
            ".cache",
            "artifacts",
            "external",
            "outputs",
            "papers-base",
            "tmp",
            "**/__pycache__",
            "*.pyc",
        ],
    )
)
for name in EXTERNAL:
    image = image.add_local_dir(
        ROOT / "external" / name,
        str(REMOTE_ROOT / "external" / name),
        copy=True,
    )

app = modal.App("generalizable-interpretability")
volume = modal.Volume.from_name("gi-cache", create_if_missing=True)
secrets = []
if name := os.environ.get("GI_MODAL_SECRET", "mup-env"):
    secrets.append(modal.Secret.from_name(name))
if values := job_environment():
    secrets.append(modal.Secret.from_dict(values))


@app.function(
    image=image,
    gpu=os.environ.get("GI_GPU", "A100-80GB"),
    secrets=secrets,
    volumes={str(REMOTE_CACHE): volume},
    memory=65536,
    ephemeral_disk=524288,
    timeout=int(os.environ.get("GI_TIMEOUT", "21600")),
)
def execute(command: str) -> None:
    env = os.environ | {
        "GI_ARTIFACT_DIR": str(REMOTE_CACHE / "artifacts"),
        "HF_HOME": str(REMOTE_CACHE / "huggingface"),
        "UV_CACHE_DIR": str(REMOTE_CACHE / "uv"),
    }
    subprocess.run(
        ["uv", "run", "--frozen", "--", *json.loads(command)],
        cwd=REMOTE_ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "--",
            "python",
            "-m",
            "infra.artifacts",
            "sync",
            "--path",
            str(REMOTE_CACHE / "artifacts"),
        ],
        cwd=REMOTE_ROOT,
        env=env,
        check=True,
    )
    volume.commit()


@app.local_entrypoint()
def main(command: str) -> None:
    execute.remote(command)
