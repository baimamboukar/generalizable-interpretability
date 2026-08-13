from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

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


def api() -> HfApi:
    return HfApi(token=os.environ.get("HF_TOKEN"))


def check() -> None:
    print(api().bucket_info(bucket_id()))


def create() -> None:
    print(api().create_bucket(bucket_id(), private=False, exist_ok=True))


def sync(path: Path) -> None:
    if enabled() and path.exists():
        api().sync_bucket(source=str(path), dest=bucket(), quiet=False)


def pull(path: Path, prefix: str = "") -> None:
    source = bucket().rstrip("/")
    if prefix:
        source = f"{source}/{prefix.strip('/')}"
    path.mkdir(parents=True, exist_ok=True)
    api().sync_bucket(source=source, dest=str(path), quiet=False)


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
