from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from vastai import VastAI

from infra.env import load_project_env


ACTIVE_STATES = {"running", "loading", "created"}
TERMINAL_STATES = {"exited", "stopped", "destroyed"}
TERMINAL_ARTIFACT_STAGES = {"complete", "evaluated", "finished", "trained"}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(path: Path, message: str) -> None:
    line = f"{timestamp()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(line + "\n")


def normalize_logs(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, default=str)


def phase(logs: str) -> str:
    layer_markers = [
        line.strip()
        for line in logs.splitlines()
        if "train-all-layers:" in line or "test-all-layers:" in line
    ]
    if layer_markers:
        return layer_markers[-1]
    markers = [line for line in logs.splitlines() if "extracting:" in line]
    if markers:
        return markers[-1].strip()
    if "evaluate_probe" in logs:
        return "evaluating probes"
    for marker, label in (
        ("##### Training the early exit MLP", "training early-exit MLP"),
        ("##### Static MLPs", "training static MLPs"),
        ("##### Linear probe", "evaluating linear probe"),
        ("##### Bilinear probe", "training bilinear probe"),
        ("##### Poly probe", "training polynomial probe"),
        ("activation batches", "extracting activations"),
    ):
        if marker in logs:
            return label
    return "starting"


def is_complete(info: dict, logs: str) -> bool:
    state = str(info.get("actual_status", info.get("cur_state", ""))).lower()
    return state not in ACTIVE_STATES and "artifacts:" in logs


def failed(logs: str) -> bool:
    return "Traceback (most recent call last):" in logs or "No solution found" in logs


def terminal(logs: str) -> bool:
    return "Sync completed." in logs


def artifact_ready(artifact: str) -> bool:
    bucket = os.environ["HF_BUCKET"].removeprefix("hf://buckets/")
    target = f"hf://buckets/{bucket}/models-under-pressure/{artifact}"
    result = subprocess.run(
        ["hf", "buckets", "list", target, "-R", "--quiet"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "run.json" in result.stdout


def hf_artifact_ready(artifact: str) -> bool:
    bucket = os.environ["HF_BUCKET"].rstrip("/")
    relative = artifact.lstrip("/")
    if relative.startswith("artifacts/"):
        relative = relative.removeprefix("artifacts/")
    target = f"{bucket}/{relative}"
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / Path(relative).name
        result = subprocess.run(
            ["hf", "buckets", "cp", target, str(destination)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not destination.exists():
            return False
        if destination.name != "run.json":
            return True
        state = json.loads(destination.read_text())
        return state.get("stage") in TERMINAL_ARTIFACT_STAGES


def destroy(client: VastAI, instance_id: int, log: Path) -> bool:
    for _ in range(3):
        client.destroy_instance(instance_id)
        for _ in range(6):
            time.sleep(10)
            try:
                info = client.show_instance(instance_id)
            except Exception:
                append(log, f"instance={instance_id} destroyed=true")
                return True
            if not info or "id" not in info:
                append(log, f"instance={instance_id} destroyed=true")
                return True
    append(log, f"instance={instance_id} destroyed=false")
    return False


def monitor(args: argparse.Namespace) -> None:
    load_project_env()
    client = VastAI(api_key=os.environ["VAST_API_KEY"])
    artifacts = dict(spec.split("=", 1) for spec in args.artifact)
    hf_artifacts = dict(spec.split("=", 1) for spec in args.hf_artifact)
    remaining = set(args.instance)
    last_log_hash: dict[int, str] = {}
    idle_checks: dict[int, int] = {}
    checks = 0
    while remaining:
        checks += 1
        for instance_id in tuple(remaining):
            try:
                info = client.show_instance(instance_id)
                if not info:
                    append(args.log, f"instance={instance_id} destroyed=true")
                    remaining.remove(instance_id)
                    continue
                logs = normalize_logs(client.logs(instance_id, tail="12000"))
                state = str(
                    info.get("actual_status", info.get("cur_state", "unknown"))
                )
                append(
                    args.log,
                    " ".join(
                        [
                            f"instance={instance_id}",
                            f"state={state}",
                            f"gpu_util={info.get('gpu_util')}",
                            f"cpu_util={info.get('cpu_util')}",
                            f"phase={phase(logs)}",
                        ]
                    ),
                )
                log_hash = hashlib.sha256(logs.encode()).hexdigest()
                if last_log_hash.get(instance_id) == log_hash:
                    idle_checks[instance_id] = idle_checks.get(instance_id, 0) + 1
                else:
                    idle_checks[instance_id] = 0
                    last_log_hash[instance_id] = log_hash
                artifact = artifacts.get(str(instance_id))
                if artifact is not None and artifact_ready(artifact):
                    append(args.log, f"instance={instance_id} artifact_ready=true")
                    if destroy(client, instance_id, args.log):
                        remaining.remove(instance_id)
                    continue
                hf_artifact = hf_artifacts.get(str(instance_id))
                if hf_artifact is not None and hf_artifact_ready(hf_artifact):
                    append(args.log, f"instance={instance_id} artifact_ready=true")
                    if destroy(client, instance_id, args.log):
                        remaining.remove(instance_id)
                    continue
                if terminal(logs):
                    append(
                        args.log,
                        f"instance={instance_id} terminal_sync=true success={not failed(logs)}",
                    )
                    if destroy(client, instance_id, args.log):
                        remaining.remove(instance_id)
                    continue
                if (
                    args.max_idle_checks is not None
                    and state.lower() in ACTIVE_STATES
                    and idle_checks[instance_id] >= args.max_idle_checks
                    and destroy(client, instance_id, args.log)
                ):
                    append(args.log, f"instance={instance_id} destroyed=idle")
                    remaining.remove(instance_id)
                    continue
                if state.lower() in TERMINAL_STATES:
                    completion = is_complete(info, logs)
                    append(
                        args.log,
                        " ".join(
                            [
                                f"instance={instance_id}",
                                f"completion_marker={completion}",
                                "artifact_ready=false",
                            ]
                        ),
                    )
                    if destroy(client, instance_id, args.log):
                        remaining.remove(instance_id)
            except Exception as error:
                append(
                    args.log,
                    f"instance={instance_id} watch_error={type(error).__name__}",
                )
        if args.once or not remaining or (
            args.max_checks is not None and checks >= args.max_checks
        ):
            return
        time.sleep(args.interval_minutes * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, action="append", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--hf-artifact", action="append", default=[])
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-checks", type=int)
    parser.add_argument("--max-idle-checks", type=int)
    monitor(parser.parse_args())


if __name__ == "__main__":
    main()
