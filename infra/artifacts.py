from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from infra.env import load_project_env


load_project_env()


def enabled() -> bool:
    return os.environ.get("HF_SYNC_ENABLED", "0").lower() in {"1", "true", "yes"}


def bucket() -> str:
    value = os.environ.get("HF_BUCKET")
    if not value:
        raise RuntimeError("HF_BUCKET is not configured")
    return value


def bucket_id() -> str:
    return bucket().removeprefix("hf://buckets/")


def check() -> None:
    subprocess.run(["hf", "buckets", "info", bucket_id()], check=True)


def create() -> None:
    subprocess.run(
        ["hf", "buckets", "create", bucket_id(), "--exist-ok"], check=True
    )


def sync(path: Path) -> None:
    if enabled() and path.exists():
        subprocess.run(["hf", "buckets", "sync", str(path), bucket()], check=True)


def pull(path: Path, prefix: str = "") -> None:
    source = bucket().rstrip("/")
    if prefix:
        source = f"{source}/{prefix.strip('/')}"
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["hf", "buckets", "sync", source, str(path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "check", "sync", "pull"])
    parser.add_argument("--path", type=Path)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    if args.action == "create":
        create()
    elif args.action == "check":
        check()
    elif args.action == "sync":
        path = args.path or Path(os.environ.get("GI_ARTIFACT_DIR", "artifacts"))
        sync(path)
    else:
        path = args.path or Path(os.environ.get("GI_ARTIFACT_DIR", "artifacts"))
        pull(path, args.prefix)


if __name__ == "__main__":
    main()
