from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from compat import prepare


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "external" / "sycophancy"
ARTIFACTS = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts")) / "sycophancy"
COMMIT = "09d51c4474d9a43be0d22bebae37c9fadd38b699"
MODELS = ("gemma-3", "llama-3.2")
ACTIVATIONS = ("mha", "mlp", "residual")
SEEDS = (3407, 3408, 3409)
SOURCE_SUFFIXES = {".py", ".toml", ".md", ".lock"}


def digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    }


def commit() -> str:
    if not (SOURCE / ".git").exists():
        manifest = Path(__file__).parent / "upstream.json"
        return json.loads(manifest.read_text())["commit"]
    return subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()


def worktree(model: str, seed: int) -> Path:
    target = ARTIFACTS / "runs" / model / f"seed-{seed}" / "source"
    if not target.exists():
        shutil.copytree(SOURCE, target, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    marker = target / ".gi-compat.json"
    if not marker.exists() and digest(target) != digest(SOURCE):
        raise RuntimeError("Artifact worktree differs from the pinned release")
    prepare(target, model, COMMIT)
    return target


def inspect() -> None:
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "commit": commit(),
                "expected_commit": COMMIT,
                "models": {
                    "gemma-3": "google/gemma-3-4b-it",
                    "llama-3.2": "meta-llama/Llama-3.2-3B-Instruct",
                },
                "targets": list(ACTIVATIONS),
                "dataset": "TruthfulQA generation split",
                "paper": {
                    "optimizer": "Adam",
                    "batch_size": 32,
                    "learning_rate": 1e-5,
                    "epochs": 16,
                    "concept": "correct-to-incorrect after user disagreement",
                },
                "replication_seeds": list(SEEDS),
                "release_defaults": {
                    "batch_size": 25,
                    "learning_rate": 1e-3,
                    "epochs": 25,
                },
                "unresolved": [
                    "release dependency file is empty",
                    "released code contains no published checkpoints or result tables",
                ],
            },
            indent=2,
        )
    )


def train(model: str, activation: str, seed: int) -> None:
    if commit() != COMMIT:
        raise RuntimeError(f"Expected upstream commit {COMMIT}")
    tree = worktree(model, seed)
    command = [
        sys.executable,
        str(Path(__file__).parent / "seeded.py"),
        "--seed",
        str(seed),
        "--script",
        "train.py",
        "--",
        "--model_id",
        model,
        "--activation_type",
        activation,
        "--batch_size",
        "32",
        "--lr",
        "1e-5",
        "--epochs",
        "16",
        "--concept",
        "sycophancy_challenged",
        "--probe_type",
        "linear",
    ]
    subprocess.run(command, cwd=tree / "probe", check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inspect", "train", "baseline"), required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--activation", choices=ACTIVATIONS)
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    if args.stage == "inspect":
        inspect()
        return
    if not args.model:
        parser.error("--model is required")
    targets = (args.activation,) if args.activation else ACTIVATIONS
    seeds = args.seed or (SEEDS if args.stage == "baseline" else SEEDS[:1])
    if args.stage == "train" and not args.activation:
        parser.error("--activation is required for --stage train")
    for seed in seeds:
        for activation in targets:
            train(args.model, activation, seed)


if __name__ == "__main__":
    main()
