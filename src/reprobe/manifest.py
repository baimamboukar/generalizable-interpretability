"""Everything needed to reproduce or audit one run, saved as manifest.json beside its outputs."""

import inspect_ai
from inspect_ai.event import ModelEvent
from inspect_ai.util import sandbox

from constants import COMPOSE, DOCKER_IMAGE_ARCH, DOCKER_IMAGE_ID, DOCKER_VERSION, HERE, ROOT
from geninterp import benchmark_task
from helpers import load_yaml, now, read_checksums, sh, sha256, started


async def build_manifest(state, events, efficiency: dict, scores: dict, reexecution: dict) -> dict:
    config = state.metadata
    model_calls = [e for e in events if isinstance(e, ModelEvent)]

    return {
        "config": config,
        "environment": await environment(config["arm"]),
        "inputs": inputs(config),
        "harness": harness(state, model_calls),
        "model": model(state, model_calls),
        "outcome": outcome(state, events, efficiency, scores, reexecution),
    }


async def environment(arm: str) -> dict:
    image = load_yaml(COMPOSE[arm])["services"]["default"]["image"]
    packages = await sandbox().exec(["pip", "freeze"])

    return {
        "image": image,
        "image_id": sh("docker", "image", "inspect", image, "--format", DOCKER_IMAGE_ID),
        "architecture": sh("docker", "image", "inspect", image, "--format", DOCKER_IMAGE_ARCH),
        "docker": sh("docker", "version", "--format", DOCKER_VERSION),
        "packages": packages.stdout.split(),
    }


def inputs(config: dict) -> dict:
    return {
        "bucket_folder": f"reprobe/{config['study']}",
        "sha256": read_checksums(config["study"]),
        "repository_commit": benchmark_task(config["task_id"])["commit"],
    }


def harness(state, model_calls: list) -> dict:
    changes = sh("git", "-C", str(ROOT), "status", "--porcelain", str(HERE))
    tools = model_calls[0].tools if model_calls else []

    return {
        "commit": sh("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "uncommitted_changes": bool(changes),
        "inspect": inspect_ai.__version__,
        "prompt_sha256": sha256(state.input_text),
        "tools": [tool.name for tool in tools],
    }


def model(state, model_calls: list) -> dict:
    responses = [call.call.response for call in model_calls if call.call]
    providers = {r.get("provider") for r in responses if isinstance(r, dict)} - {None}
    settings = model_calls[0].config.model_dump(exclude_none=True) if model_calls else {}

    return {
        "name": str(state.model),
        "settings_sent": settings,
        "providers": sorted(providers),
    }


def outcome(state, events, efficiency: dict, scores: dict, reexecution: dict) -> dict:
    return {
        "repeat": state.epoch,
        "started": started(events).isoformat(),
        "ended": now().isoformat(),
        "tokens": state.token_usage,
        **efficiency,
        **scores,
        "reexecution": reexecution,
    }
