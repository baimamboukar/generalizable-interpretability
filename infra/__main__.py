from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from infra.env import load_project_env

ROOT = Path(__file__).resolve().parents[1]
load_project_env()


def call(command: list[str], **kwargs) -> None:
    print(shlex.join(command))
    subprocess.run(command, check=True, **kwargs)


def command(args: argparse.Namespace) -> list[str]:
    value = args.command
    return value[1:] if value[:1] == ["--"] else value


def runtime_env(root: Path) -> dict[str, str]:
    cache = root / ".cache"
    return {
        "GI_ARTIFACT_DIR": str(root / "artifacts"),
        "HF_HOME": str(cache / "huggingface"),
        "UV_CACHE_DIR": str(cache / "uv"),
    }


def run_local(args: argparse.Namespace) -> None:
    env = os.environ | runtime_env(ROOT)
    call(command(args), cwd=ROOT, env=env)
    if args.sync_artifacts:
        from infra.artifacts import sync

        sync(Path(env["GI_ARTIFACT_DIR"]))


def run_modal(args: argparse.Namespace) -> None:
    os.environ.update(
        GI_GPU=args.gpu,
        GI_TIMEOUT=str(args.timeout),
        GI_EXTERNAL=",".join(args.external),
    )
    import modal
    from infra.modal import app, execute

    with modal.enable_output():
        with app.run():
            execute.remote(json.dumps(command(args)))


def run_vast(args: argparse.Namespace) -> None:
    from infra.vast import VastRunner

    result = VastRunner().launch(
        root=ROOT,
        command=command(args),
        external=args.external,
        gpu=args.gpu,
        min_vram=args.min_vram,
        min_cpu=args.min_cpu,
        min_ram=args.min_ram,
        min_cuda=args.min_cuda,
        max_price=args.max_price,
        disk=args.disk,
        image=args.image,
        label=args.label,
        env_keys=args.env_key,
    )
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m infra")
    providers = root.add_subparsers(dest="provider", required=True)

    local = providers.add_parser("local")
    local.add_argument("--sync-artifacts", action="store_true")
    local.add_argument("command", nargs=argparse.REMAINDER)

    modal = providers.add_parser("modal")
    modal.add_argument("--gpu", default="A100-80GB")
    modal.add_argument("--timeout", type=int, default=21600)
    modal.add_argument("--external", action="append", default=[])
    modal.add_argument("command", nargs=argparse.REMAINDER)

    vast = providers.add_parser("vast")
    vast.add_argument("--gpu", default="A100_PCIE")
    vast.add_argument("--min-vram", type=float, default=0)
    vast.add_argument("--min-cpu", type=int, default=8)
    vast.add_argument("--min-ram", type=float, default=0)
    vast.add_argument("--min-cuda", type=float, default=0)
    vast.add_argument("--max-price", type=float, default=2)
    vast.add_argument("--disk", type=float, default=100)
    vast.add_argument(
        "--image",
        default="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime",
    )
    vast.add_argument("--label", required=True)
    vast.add_argument("--external", action="append", default=[])
    vast.add_argument("--env-key", action="append", default=[])
    vast.add_argument("command", nargs=argparse.REMAINDER)
    return root


def main() -> None:
    args = parser().parse_args()
    if not command(args):
        raise SystemExit("A command is required after --")
    {"local": run_local, "modal": run_modal, "vast": run_vast}[args.provider](args)


if __name__ == "__main__":
    main()
