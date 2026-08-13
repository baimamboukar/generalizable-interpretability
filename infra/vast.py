from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
from uuid import uuid4

from vastai import VastAI


EXCLUDED = {
    ".git",
    ".venv",
    ".cache",
    ".env",
    "artifacts",
    "data",
    "docs",
    "figures",
    "outputs",
    "papers-base",
    "reports",
    "tmp",
    "__pycache__",
}


class VastRunner:
    def __init__(self) -> None:
        self.client = VastAI(api_key=os.environ.get("VAST_API_KEY"))

    def launch(
        self,
        root: Path,
        command: list[str],
        external: list[str],
        gpu: str,
        min_vram: float,
        min_cpu: int,
        min_ram: float,
        max_price: float,
        disk: float,
        image: str,
        label: str,
        env_keys: list[str],
    ) -> dict:
        bucket = os.environ["HF_BUCKET"].removeprefix("hf://buckets/")
        bundle_path = f"infra/bundles/{uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "project.tar.gz"
            self._archive(root, archive, external)
            subprocess.run(
                [
                    "hf",
                    "buckets",
                    "cp",
                    str(archive),
                    f"hf://buckets/{bucket}/{bundle_path}",
                ],
                check=True,
            )

        offers = self.client.search_offers(
            query=(
                f"gpu_name={gpu} num_gpus=1 rentable=true "
                f"reliability>0.98 disk_space>={disk} "
                f"cpu_cores_effective>={min_cpu} cpu_ram>={min_ram} "
                f"gpu_ram>={min_vram}"
            ),
            order="dph_total",
            limit=20,
            storage=disk,
        )
        script = self._script(bucket, bundle_path, command, label)
        selected = {
            key: os.environ[key]
            for key in env_keys
            if key in os.environ
        }
        errors = []
        for offer in offers:
            if float(offer["dph_total"]) > max_price:
                continue
            try:
                result = self.client.create_instance(
                    id=int(offer["id"]),
                    image=image,
                    disk=disk,
                    label=label,
                    env=selected,
                    runtype="args",
                    args=["bash", "-lc", script],
                    cancel_unavail=True,
                )
                break
            except Exception as error:
                errors.append(f"offer {offer['id']}: {type(error).__name__}")
        else:
            detail = ", ".join(errors) or "no offer within price cap"
            raise RuntimeError(f"Unable to create Vast instance: {detail}")
        result["offer"] = {
            "id": offer["id"],
            "gpu": offer["gpu_name"],
            "price_per_hour": offer["dph_total"],
        }
        result["bundle"] = f"hf://buckets/{bucket}/{bundle_path}"
        return result

    @staticmethod
    def _script(
        bucket: str,
        bundle_path: str,
        command: list[str],
        label: str,
    ) -> str:
        run = shlex.join(command)
        log = f"artifacts/logs/{label}.log"
        return "\n".join(
            [
                "set -euo pipefail",
                "export HF_HUB_DISABLE_XET=1",
                "python -m pip install -q uv",
                "python -m pip uninstall -y torchvision torchaudio >/dev/null 2>&1 || true",
                "uvx --from huggingface_hub hf buckets cp "
                f"hf://buckets/{bucket}/{bundle_path} "
                "/workspace/project.tar.gz",
                "mkdir -p /workspace/generalizable-interpretability",
                "tar -xzf /workspace/project.tar.gz -C /workspace/generalizable-interpretability",
                "cd /workspace/generalizable-interpretability",
                f"mkdir -p {shlex.quote(str(Path(log).parent))}",
                "set +e",
                f"{run} 2>&1 | tee {shlex.quote(log)}",
                "status=${PIPESTATUS[0]}",
                "set -e",
                "uvx --from huggingface_hub hf buckets sync "
                f"artifacts hf://buckets/{bucket} || true",
                "exit $status",
            ]
        )

    @staticmethod
    def _archive(
        root: Path,
        output: Path,
        external: list[str],
    ) -> None:
        def clean(member: tarfile.TarInfo):
            parts = Path(member.name).parts
            is_external = parts[:1] == ("external",)
            excluded = EXCLUDED - {"data"} if is_external else EXCLUDED
            if excluded.intersection(parts) or member.name.endswith(".pyc"):
                return None
            return member

        with tarfile.open(output, "w:gz") as bundle:
            for path in root.iterdir():
                if path.name not in EXCLUDED | {"external"}:
                    bundle.add(path, arcname=path.name, filter=clean)
            for name in external:
                path = root / "external" / name
                if not path.is_dir():
                    raise FileNotFoundError(path)
                bundle.add(path, arcname=f"external/{name}", filter=clean)
