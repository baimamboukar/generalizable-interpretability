"""Rerun the agent's own code in a fresh container and check that it re-derives the submitted results."""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from constants import CODE_SUFFIXES, COMPOSE, REEXECUTION_TOLERANCE, REPRODUCE_SCRIPT
from helpers import load_yaml


def copy_code(output: Path) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="reexecute-"))
    for path in output.rglob("*"):
        if path.suffix in CODE_SUFFIXES:
            target = workdir / path.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target)
    return workdir


def docker_command(config: dict, workdir: Path) -> list[str]:
    image = load_yaml(COMPOSE[config["arm"]])["services"]["default"]["image"]

    return [
        "docker", "run", "--rm", "--network", "none",
        "--cpus", "4", "--memory", config["memory"],
        "--env", f"SEED={config['seed']}",
        "--volume", f"{config['task_dir']}:/work/task:ro",
        "--volume", f"{workdir}:/work/output",
        image, "bash", f"/work/output/{REPRODUCE_SCRIPT}",
    ]


def read_results(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def count_matches(submitted: dict, rederived: dict) -> int:
    matched = 0
    for result_id, value in submitted.items():
        other = rederived.get(result_id)
        if isinstance(other, (int, float)) and abs(other - value) <= REEXECUTION_TOLERANCE:
            matched += 1
    return matched


def reexecute(output: Path, run: Path, config: dict, submitted: dict) -> dict:
    if not (output / REPRODUCE_SCRIPT).exists():
        return {"status": f"no {REPRODUCE_SCRIPT}", "matched": 0, "total": len(submitted)}

    workdir = copy_code(output)
    started = time.time()
    try:
        process = subprocess.run(docker_command(config, workdir), capture_output=True, text=True, timeout=config["time_limit"])
        (run / "reexecution.log").write_text(process.stdout + process.stderr)
    except subprocess.TimeoutExpired:
        (run / "reexecution.log").write_text(f"timed out after {config['time_limit']} seconds")

    rederived = read_results(workdir / "results.json")
    (run / "reexecution.json").write_text(json.dumps(rederived, indent=2))
    shutil.rmtree(workdir, ignore_errors=True)
    matched = count_matches(submitted, rederived)

    return {
        "status": "passed" if matched == len(submitted) else "failed",
        "matched": matched,
        "total": len(submitted),
        "seconds": round(time.time() - started, 1),
    }
