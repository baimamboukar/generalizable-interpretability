import hashlib
import io
import json
import subprocess
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml
from inspect_ai.event import ModelEvent
from inspect_ai.model import ModelCost, ModelInfo, set_model_cost, set_model_info
from inspect_ai.util import sandbox

from constants import HERE, OPENROUTER_MODELS_URL, TASKS


def load_yaml(name: str) -> dict:
    return yaml.safe_load((HERE / name).read_text())


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2))


def sh(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True).stdout.strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def now() -> datetime:
    return datetime.now(timezone.utc)


def started(events: list) -> datetime:
    # the first model call, so time spent waiting for a free sandbox is not counted
    model_calls = [e for e in events if isinstance(e, ModelEvent)]
    return (model_calls or events)[0].timestamp


def read_checksums(study: str) -> dict:
    checksums = {}
    for line in (TASKS / study / "inputs.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        checksums[path] = digest
    return checksums


def per_million(price: str) -> float:
    return float(price) * 1_000_000


def openrouter_cost(model_id: str) -> ModelCost:
    with urllib.request.urlopen(OPENROUTER_MODELS_URL) as response:
        models = json.load(response)["data"]
    price = next(m["pricing"] for m in models if m["id"] == model_id)

    return ModelCost(
        input=per_million(price["prompt"]),
        output=per_million(price["completion"]),
        input_cache_read=per_million(price.get("input_cache_read") or price["prompt"]),
        input_cache_write=per_million(price.get("input_cache_write") or price["prompt"]),
    )


def register_cost(model: str, cost: ModelCost) -> None:
    try:
        set_model_cost(model, cost)
    except ValueError:  # a model Inspect does not know yet
        set_model_info(model, ModelInfo(cost=cost))


def parse_submission(text: str) -> dict:
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


async def copy_outputs(run: Path) -> Path:
    await sandbox().exec(["tar", "czf", "/tmp/output.tgz", "-C", "/work", "output"])
    archive = await sandbox().read_file("/tmp/output.tgz", text=False)
    tarfile.open(fileobj=io.BytesIO(archive)).extractall(run, filter="data")
    return run / "output"
