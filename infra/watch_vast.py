from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

from vastai import VastAI

from infra.env import load_project_env


ACTIVE_STATES = {"running", "loading", "created"}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(path: Path, message: str) -> None:
    line = f"{timestamp()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(line + "\n")


def phase(logs: str) -> str:
    markers = [line for line in logs.splitlines() if "extracting:" in line]
    if markers:
        return markers[-1].strip()
    if "evaluate_probe" in logs:
        return "evaluating probes"
    return "starting"


def is_complete(info: dict, logs: str) -> bool:
    state = str(info.get("actual_status", info.get("cur_state", ""))).lower()
    return state not in ACTIVE_STATES and "artifacts:" in logs


def artifact_ready(artifact: str) -> bool:
    bucket = os.environ["HF_BUCKET"].removeprefix("hf://buckets/")
    target = f"hf://buckets/{bucket}/models-under-pressure/{artifact}"
    result = subprocess.run(
        ["hf", "buckets", "list", target, "-R", "--quiet"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "run.json" in result.stdout


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
            if not info or info.get("instances") is None or "id" not in info:
                append(log, f"instance={instance_id} destroyed=true")
                return True
    append(log, f"instance={instance_id} destroyed=false")
    return False


def monitor(args: argparse.Namespace) -> None:
    load_project_env()
    client = VastAI(api_key=os.environ["VAST_API_KEY"])
    artifacts = dict(spec.split("=", 1) for spec in args.artifact)
    remaining = set(args.instance)
    checks = 0
    while remaining:
        checks += 1
        for instance_id in tuple(remaining):
            try:
                info = client.show_instance(instance_id)
                logs = client.logs(instance_id, tail="12000")
                state = str(
                    info.get("actual_status", info.get("cur_state", "unknown"))
                )
                append(
                    args.log,
                    f"instance={instance_id} state={state} phase={phase(logs)}",
                )
                if state.lower() not in ACTIVE_STATES:
                    completion = is_complete(info, logs)
                    artifact = artifacts.get(str(instance_id))
                    ready = artifact is not None and artifact_ready(artifact)
                    append(
                        args.log,
                        " ".join(
                            [
                                f"instance={instance_id}",
                                f"completion_marker={completion}",
                                f"artifact_ready={ready}",
                            ]
                        ),
                    )
                    if ready and destroy(client, instance_id, args.log):
                        remaining.remove(instance_id)
            except Exception as error:
                append(args.log, f"instance={instance_id} watch_error={error!r}")
        if args.once or not remaining or (
            args.max_checks is not None and checks >= args.max_checks
        ):
            return
        for _ in range(args.interval_minutes * 15):
            time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, action="append", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-checks", type=int)
    monitor(parser.parse_args())


if __name__ == "__main__":
    main()
