from __future__ import annotations

import argparse
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
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

from infra.env import load_project_env  # noqa: E402

MODELS = MANIFEST["models"]
DATASETS = tuple(MANIFEST["datasets"])
REQUIRED_FILES = (
    "model.py",
    "train.py",
    "sweep_monitors.py",
    "utils.py",
    "sweep.sh",
)
RESULT_FILES = (
    "poly-vals.pkl",
    "bi-vals.pkl",
    "linear-vals.pkl",
    "mlp-vals.pkl",
    "eemlp-matched-vals.pkl",
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest() -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in SOURCE.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and "results" not in path.relative_to(SOURCE).parts
        and path.suffix != ".pyc"
    )
    for path in files:
        digest.update(path.relative_to(SOURCE).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
    if source_digest() != MANIFEST["source_digest"]:
        raise RuntimeError(
            "The bundled upstream source does not match the audited release tree"
        )
    if (SOURCE / ".git").exists():
        tracked_diff = subprocess.run(
            ["git", "-C", str(SOURCE), "diff", "--quiet", "HEAD", "--"],
            check=False,
        ).returncode
        if tracked_diff:
            raise RuntimeError("The upstream checkout has tracked source changes")
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
                "source_digest": source_digest() if SOURCE.exists() else None,
                "expected_source_digest": MANIFEST["source_digest"],
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


def package_versions() -> dict[str, str | None]:
    packages = (
        "accelerate",
        "datasets",
        "einops",
        "huggingface-hub",
        "numpy",
        "scikit-learn",
        "torch",
        "torchmetrics",
        "tqdm",
        "transformers",
    )
    resolved = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = None
    return resolved


def resolved_sources(model: str, dataset: str) -> dict[str, str]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    return {
        "model": MODELS[model]["id"],
        "model_revision": api.model_info(MODELS[model]["id"]).sha,
        "dataset": MANIFEST["datasets"][dataset]["source"],
        "dataset_revision": api.dataset_info(
            MANIFEST["datasets"][dataset]["source"]
        ).sha,
    }


def compile_smoke_test() -> dict[str, str]:
    import torch

    compiler = shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        raise RuntimeError("A C compiler is required by the released torch.compile extractor")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for activation extraction")
    compiled = torch.compile(lambda value: value.square() + 1, mode="reduce-overhead")
    value = compiled(torch.ones(4, device="cuda"))
    torch.cuda.synchronize()
    if not torch.allclose(value, torch.full_like(value, 2)):
        raise RuntimeError("torch.compile smoke test returned an invalid value")
    return {
        "compiler": compiler,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda or "unknown",
        "torchdynamo_disabled": os.environ.get("TORCHDYNAMO_DISABLE", "0"),
    }


def preflight(model: str, dataset: str) -> None:
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
    imports = {
        package: importlib.util.find_spec(package) is not None
        for package in packages
    }
    missing = [package for package, available in imports.items() if not available]
    missing_env = [key for key in ("HF_TOKEN", "HF_BUCKET") if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Missing runtime packages: {', '.join(missing)}")
    if missing_env:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing_env)}")
    print(
        json.dumps(
            {
                "commit": source_commit(),
                "source_digest": source_digest(),
                "python": sys.version.split()[0],
                "imports": imports,
                "packages": package_versions(),
                "sources": resolved_sources(model, dataset),
                "compile_smoke_test": compile_smoke_test(),
                "hf_token_configured": True,
                "hf_bucket_configured": True,
                "environment_basis": "replication pin; no author lockfile released",
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


def persist(dataset: str | None = None, model: str | None = None) -> None:
    if dataset:
        copy_tree(cache_dataset(dataset), ARTIFACTS / "activation-cache" / dataset)
    copy_tree(SOURCE / "results", ARTIFACTS / "results")
    metadata = {
        "source_commit": source_commit(),
        "source_digest": source_digest(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper": MANIFEST["paper"],
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "sources": resolved_sources(model, dataset) if model and dataset else None,
        "environment_basis": "replication pin; no author lockfile released",
        "execution_overrides": {
            "TORCHDYNAMO_DISABLE": os.environ.get("TORCHDYNAMO_DISABLE", "0")
        },
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
    persist(dataset, model)


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
    persist(dataset, model)


def result_directory(model: str, dataset: str, layer: int) -> Path:
    model_name = MODELS[model]["id"].split("/")[-1]
    dataset_dir = "wildguard" if dataset == "WildGuard" else "beaver"
    pool_type = "mean-" if dataset == "WildGuard" else "mean-330k_"
    return (
        SOURCE
        / "results"
        / "numbers"
        / dataset_dir
        / f"{model_name}-L{layer}-{pool_type}-progressive-maxorder-5_rank-64"
    )


def result_values(value: object) -> list[float]:
    if hasattr(value, "tolist"):
        return result_values(value.tolist())
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in result_values(child)]
    return [float(value)]


def verify_results(model: str, dataset: str, layer: int) -> None:
    directory = result_directory(model, dataset, layer)
    missing = [name for name in RESULT_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing result files: {', '.join(missing)}")
    files = {}
    for name in RESULT_FILES:
        path = directory / name
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        values = payload.get("y_vals_test")
        if values is None:
            raise RuntimeError(f"{name} has no y_vals_test")
        flattened = result_values(values)
        if not flattened or not all(math.isfinite(value) for value in flattened):
            raise RuntimeError(f"{name} has invalid y_vals_test")
        files[name] = {
            "sha256": file_sha256(path),
            "test_values": flattened,
        }
    manifest = {
        "model": MODELS[model]["id"],
        "dataset": dataset,
        "layer": layer,
        "source_commit": source_commit(),
        "source_digest": source_digest(),
        "files": files,
    }
    output = ARTIFACTS / "verification" / f"{model}_{dataset}_L{layer}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


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
    parser.add_argument(
        "--stage",
        choices=("inspect", "preflight", "extract", "train", "verify", "baseline"),
        required=True,
    )
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
        if not args.model:
            raise SystemExit("--model is required for preflight")
        preflight(args.model, args.dataset)
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
    elif args.stage == "verify":
        if not args.model or args.layer is None:
            raise SystemExit("--model and --layer are required for verify")
        verify_results(args.model, args.dataset, args.layer)
    else:
        baseline(args)


if __name__ == "__main__":
    main()
