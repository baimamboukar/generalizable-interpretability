from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "external" / "tpc"
MANIFEST = json.loads((HERE / "upstream.json").read_text())
ARTIFACTS = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts")) / "beyond-linear-probes"
UPSTREAM_CACHE = Path("/data/scratch/acw663/poly-monitor/activation-cache")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.env import load_project_env

MODELS = MANIFEST["models"]
DATASETS = tuple(MANIFEST["datasets"])
REQUIRED_FILES = (
    "model.py",
    "train.py",
    "sweep_monitors.py",
    "utils.py",
    "sweep.sh",
)

def load() -> None:
    load_project_env()


def source_commit() -> str:
    git_dir = SOURCE / ".git"
    if git_dir.exists():
        try:
            return subprocess.check_output(
                ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except FileNotFoundError:
            pass
    return MANIFEST["commit"]


def verify_source() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(
            f"Missing upstream checkout {SOURCE}; clone the repository from "
            f"{MANIFEST['repository']}"
        )
    if source_commit() != MANIFEST["commit"]:
        raise RuntimeError(
            f"Expected upstream commit {MANIFEST['commit']}, found {source_commit()}"
        )
    missing = [path for path in REQUIRED_FILES if not (SOURCE / path).is_file()]
    if missing:
        raise RuntimeError(f"Missing upstream files: {missing}")


def inspect() -> None:
    source_files = {name: (SOURCE / name).exists() for name in REQUIRED_FILES}
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "repository": MANIFEST["repository"],
                "commit": source_commit() if SOURCE.exists() else None,
                "expected_commit": MANIFEST["commit"],
                "models": MANIFEST["models"],
                "datasets": MANIFEST["datasets"],
                "paper_settings": MANIFEST["paper_settings"],
                "baseline_targets": MANIFEST["baseline_targets"],
                "required_source_files": source_files,
                "activation_cache": str(UPSTREAM_CACHE),
                "released_results": False,
                "released_activations": False,
            },
            indent=2,
        )
    )


def preflight() -> None:
    verify_source()
    packages = (
        "datasets",
        "einops",
        "numpy",
        "sklearn",
        "torch",
        "torchmetrics",
        "tqdm",
        "transformers",
    )
    print(
        json.dumps(
            {
                "commit": source_commit(),
                "python": sys.version.split()[0],
                "imports": {
                    package: importlib.util.find_spec(package) is not None
                    for package in packages
                },
                "hf_token_configured": bool(os.environ.get("HF_TOKEN")),
                "source_dependency_lock": False,
                "model_revisions_pinned": False,
                "activation_cache_parent_exists": UPSTREAM_CACHE.parent.exists(),
            },
            indent=2,
        )
    )


def extractor(model: str, dataset: str) -> Path:
    suffix = "_beaver" if dataset == "BeaverTrails" else ""
    name = MODELS[model]["extractor"]
    return SOURCE / Path(name).with_name(Path(name).stem + suffix + ".py")


def cache_dataset(dataset: str) -> Path:
    return UPSTREAM_CACHE / dataset


def ensure_cache(dataset: str) -> None:
    try:
        cache_dataset(dataset).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "The author scripts require /data/scratch/acw663/poly-monitor/activation-cache; "
            "run this wrapper on the configured Vast image or provide that path."
        ) from exc


def run_upstream(command: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    print("$", " ".join(command))
    subprocess.run(command, cwd=SOURCE, env=env, check=True)


def copy_tree(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)


def persist(dataset: str | None = None) -> None:
    if dataset:
        copy_tree(cache_dataset(dataset), ARTIFACTS / "activation-cache" / dataset)
    copy_tree(SOURCE / "results", ARTIFACTS / "results")
    metadata = {
        "source_commit": source_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper": MANIFEST["paper"],
    }
    (ARTIFACTS / "run_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def expected_files(model: str, dataset: str, layer: int) -> list[Path]:
    model_name = MODELS[model]["id"].split("/")[-1]
    if dataset == "WildGuard":
        suffix = "mean-"
        splits = (
            ("wildguard_train", "wildguard_test")
            if model == "qwen3"
            else ("train", "test")
        )
    else:
        suffix = "mean-330k_"
        splits = ("train", "test")
    return [
        cache_dataset(dataset) / f"{model_name}-L{layer}-{suffix}{split}-X.npy"
        for split in splits
    ] + [
        cache_dataset(dataset) / f"{model_name}-{suffix}{split}-y.npy"
        for split in splits
    ]


def training_files(model: str, dataset: str, layer: int) -> list[Path]:
    """Names consumed by the upstream monitor script."""
    model_name = MODELS[model]["id"].split("/")[-1]
    suffix = "mean-" if dataset == "WildGuard" else "mean-330k_"
    return [
        UPSTREAM_CACHE / dataset / f"{model_name}-L{layer}-{suffix}{split}-X.npy"
        for split in ("train", "test")
    ] + [
        UPSTREAM_CACHE / dataset / f"{model_name}-{suffix}{split}-y.npy"
        for split in ("train", "test")
    ]


def cache_complete(model: str, dataset: str) -> bool:
    return all(
        path.exists()
        for layer in MODELS[model]["layers"]
        for path in expected_files(model, dataset, layer)
    )


def extract(model: str, dataset: str) -> None:
    verify_source()
    ensure_cache(dataset)
    run_upstream([sys.executable, str(extractor(model, dataset))])
    persist(dataset)


def train(
    model: str,
    dataset: str,
    layer: int,
    rank: int,
    max_order: int,
    max_epochs: int,
    pool_type: str,
    train_mode: str,
    batch_size: int,
) -> None:
    verify_source()
    if layer not in MODELS[model]["layers"]:
        raise ValueError(f"Layer {layer} is not in the released layer set for {model}")
    missing = [path for path in training_files(model, dataset, layer) if not path.exists()]
    if missing:
        extracted = [path for path in expected_files(model, dataset, layer) if path.exists()]
        detail = (
            " The released Qwen3 extractor writes wildguard_train/test while the "
            "released monitor expects train/test; this source mismatch is not "
            "silently rewritten."
            if model == "qwen3" and dataset == "WildGuard" and extracted
            else ""
        )
        raise FileNotFoundError(
            "Missing activation files; run extract first: "
            + ", ".join(str(path) for path in missing)
            + detail
        )
    run_upstream(
        [
            sys.executable,
            "sweep_monitors.py",
            "--rank",
            str(rank),
            "--max_order",
            str(max_order),
            "--max_epochs",
            str(max_epochs),
            "--layer_idx",
            str(layer),
            "--pool_type",
            pool_type,
            "--train_mode",
            train_mode,
            "--batch_size",
            str(batch_size),
            "--model_name",
            MODELS[model]["id"],
            "--dataset",
            dataset,
        ]
    )
    persist(dataset)


def baseline(args: argparse.Namespace) -> None:
    models = [args.model] if args.model else list(MODELS)
    for model in models:
        if not cache_complete(model, args.dataset):
            extract(model, args.dataset)
        for layer in MODELS[model]["layers"]:
            train(
                model,
                args.dataset,
                layer,
                args.rank,
                args.max_order,
                args.max_epochs,
                args.pool_type,
                args.train_mode,
                args.batch_size,
            )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inspect", "preflight", "extract", "train", "baseline"), required=True)
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--dataset", choices=DATASETS, default="WildGuard")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max-order", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--pool-type", default="mean")
    parser.add_argument("--train-mode", default="progressive")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser


def main() -> None:
    load()
    args = parser().parse_args()
    if args.stage == "inspect":
        inspect()
    elif args.stage == "preflight":
        preflight()
    elif args.stage == "extract":
        if not args.model:
            raise SystemExit("--model is required for extract")
        extract(args.model, args.dataset)
    elif args.stage == "train":
        if not args.model or args.layer is None:
            raise SystemExit("--model and --layer are required for train")
        train(
            args.model,
            args.dataset,
            args.layer,
            args.rank,
            args.max_order,
            args.max_epochs,
            args.pool_type,
            args.train_mode,
            args.batch_size,
        )
    else:
        baseline(args)


if __name__ == "__main__":
    main()
