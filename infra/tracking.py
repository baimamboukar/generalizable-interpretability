from __future__ import annotations

import os
from typing import Any

from infra.env import load_project_env


def init_run(name: str, config: dict[str, Any] | None = None):
    load_project_env()
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb

    return wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "generalizable-interpretability"),
        name=name,
        config=config,
    )
