from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
JOB_KEYS = (
    "HF_TOKEN",
    "HF_BUCKET",
    "HF_SYNC_ENABLED",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_ACTIVATIONS_BUCKET",
)


def load_project_env() -> None:
    for key, value in dotenv_values(ENV_FILE).items():
        if value:
            os.environ.setdefault(key, value)


def job_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in JOB_KEYS if os.environ.get(key)}
