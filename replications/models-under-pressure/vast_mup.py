from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "external" / "models-under-pressure"
RUNTIME = ROOT / "replications" / "models-under-pressure" / "runtime"
TRAIN = PROJECT / "data/training/prompts_4x/train.jsonl"
VALIDATION = PROJECT / "data/training/prompts_4x/test.jsonl"
DEV_EVALS = tuple(
    PROJECT / "data/evals/dev" / name
    for name in (
        "anthropic_hh_balanced.jsonl",
        "toolace_balanced.jsonl",
        "mt_balanced.jsonl",
        "mts_balanced.jsonl",
    )
)
TEST_EVALS = tuple(
    PROJECT / "data/evals/test" / name
    for name in (
        "anthropic_hh_balanced.jsonl",
        "toolace_balanced.jsonl",
        "mt_balanced.jsonl",
        "mts_balanced.jsonl",
        "mental_health_balanced.jsonl",
        "aya_redteaming_balanced.jsonl",
    )
)
EVALS = DEV_EVALS + TEST_EVALS
DEFAULT_ACTIVATION_SITE = (
    "input to decoder block input_layernorm "
    "(pre-normalization residual stream, as specified in the paper)"
)
ACTIVATION_SITE = DEFAULT_ACTIVATION_SITE
os.environ.setdefault("MUP_LOCAL_ACTIVATIONS", "1")


def command(parts: tuple[str, ...]) -> list[str]:
    if parts[0] == "exp":
        script = PROJECT / "src/models_under_pressure/scripts/run_experiment.py"
        return [sys.executable, str(script), *parts[1:]]
    return [str(Path(sys.executable).with_name("mup")), *parts]


def run(args: argparse.Namespace, *parts: str) -> None:
    bin_dir = str(Path(sys.executable).parent)
    env = os.environ | {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": os.pathsep.join(
            part
            for part in (str(RUNTIME), os.environ.get("PYTHONPATH"))
            if part
        ),
        "DOUBLE_CHECK_CONFIG": "false",
        "WANDB_MODE": "disabled",
    }
    subprocess.run(command(parts), cwd=PROJECT, env=env, check=True)


def remove_temporary_tensors() -> None:
    for path in (PROJECT / "data/activations").rglob("*.pt"):
        path.unlink(missing_ok=True)


def pre_layer_norm_hook(cache: dict, layer: int):
    def hook_fn(_module, inputs, _output):
        if not inputs:
            raise RuntimeError("input_layernorm hook received no input")
        cache[layer] = inputs[0].detach().cpu()

    return hook_fn


def artifact_uri(args: argparse.Namespace) -> str:
    bucket = os.environ["HF_BUCKET"]
    if not bucket.startswith("hf://buckets/"):
        bucket = f"hf://buckets/{bucket}"
    return f"{bucket.rstrip('/')}/models-under-pressure/{args.artifact_name}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_digest(root: Path) -> dict[str, str | int]:
    paths = [root / "pyproject.toml", root / "uv.lock"]
    source_suffixes = {".py", ".json", ".toml", ".yaml", ".yml"}
    for directory in (root / "src", root / "config"):
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in source_suffixes
        )
    paths = sorted(path for path in paths if path.is_file())
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return {"sha256": digest.hexdigest(), "file_count": len(paths)}


def validate_frozen_inputs(args: argparse.Namespace) -> None:
    protocol = json.loads(Path(args.protocol).read_text())
    model = protocol["models"][args.model_key]
    expected = {
        "model": model["model_id"],
        "model_revision": model["revision"],
        "model_config": model["model_config"],
        "layers": tuple(model["candidate_layers"]),
        "seeds": tuple(protocol["seeds"]),
    }
    mismatches = [
        field
        for field, value in expected.items()
        if getattr(args, field) != value
    ]
    if source_tree_digest(PROJECT) != protocol["upstream_source_tree"]:
        mismatches.append("upstream_source_tree")
    observed_hashes = {
        str(path.relative_to(PROJECT)): sha256_file(path)
        for path in (TRAIN, VALIDATION, *EVALS)
    }
    if observed_hashes != protocol["dataset_sha256"]:
        mismatches.append("dataset_sha256")
    if mismatches:
        raise ValueError(f"frozen input mismatch: {', '.join(mismatches)}")


def activation_provenance_row(
    args: argparse.Namespace, dataset_path: Path, layer: int
) -> dict[str, str | int]:
    return {
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset": str(dataset_path.relative_to(PROJECT)),
        "dataset_sha256": sha256_file(dataset_path),
        "layer": layer,
        "activation_site": args.activation_site,
    }


def register_architecture_adapter(
    args: argparse.Namespace,
    architecture_registry=None,
    model_architecture=None,
) -> None:
    if args.architecture_adapter != "olmo2_decoder_block_input":
        return
    if architecture_registry is None or model_architecture is None:
        from models_under_pressure.model import (
            ArchitectureRegistry,
            ModelArchitecture,
        )

        architecture_registry = ArchitectureRegistry
        model_architecture = ModelArchitecture

    class Olmo2Arch(model_architecture):
        def get_layer_norm(self, model, layer_idx):
            return model.model.layers[layer_idx]

        def get_layers(self, model):
            return model.model.layers

        def set_layers(self, model, layers):
            model.model.layers = layers

    architecture_registry._architectures["olmo2"] = Olmo2Arch


def load_activation_provenance(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get("rows", []) if path.exists() else []


def save_activation_provenance(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n")


def expected_activation_provenance(args: argparse.Namespace) -> list[dict]:
    return [
        activation_provenance_row(args, dataset_path, layer)
        for dataset_path in (TRAIN, VALIDATION, *EVALS)
        for layer in args.layers
    ]


def validate_activation_provenance(args: argparse.Namespace, activation_root: Path) -> None:
    provenance_path = activation_root / "submission_provenance.json"
    manifest = activation_root / "manifest.json"
    if not manifest.is_file() or not provenance_path.is_file():
        raise ValueError("activation manifest or submission provenance is missing")
    observed = load_activation_provenance(provenance_path)
    missing = [row for row in expected_activation_provenance(args) if row not in observed]
    if missing:
        preview = ", ".join(
            f"{row['dataset']}@L{row['layer']}" for row in missing[:5]
        )
        raise ValueError(f"activation provenance is incomplete or stale: {preview}")


def checkpoint_activations(args: argparse.Namespace, activation_root: Path) -> None:
    subprocess.run(
        [
            "uvx",
            "--from",
            "huggingface_hub",
            "hf",
            "buckets",
            "sync",
            str(activation_root),
            f"{artifact_uri(args)}/activations",
        ],
        check=True,
    )


def restore_activations(args: argparse.Namespace, activation_root: Path) -> None:
    try:
        validate_activation_provenance(args, activation_root)
        return
    except ValueError:
        subprocess.run(
            [
                "uvx",
                "--from",
                "huggingface_hub",
                "hf",
                "buckets",
                "sync",
                f"{artifact_uri(args)}/activations",
                str(activation_root),
            ],
            check=True,
        )
    validate_activation_provenance(args, activation_root)


def result_is_complete(
    path: Path,
    *,
    model: str,
    layer: int,
    probe: str,
    config_id: str,
) -> bool:
    if not path.exists():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (json.JSONDecodeError, OSError):
        return False
    required = {"config", "dataset_name", "dataset_path", "output_scores", "ground_truth_labels", "ids"}
    if any(not required.issubset(row) for row in rows):
        return False
    observed = [str(row["dataset_path"]) for row in rows]
    expected = [str(path.relative_to(PROJECT)) for path in EVALS]
    if len(observed) != len(expected) or sorted(observed) != sorted(expected):
        return False
    expected_probe = "linear_then_softmax" if probe == "softmax" else probe
    for row in rows:
        config = row["config"]
        if (
            config.get("model_name") != model
            or int(config.get("layer", -1)) != layer
            or config.get("probe_spec", {}).get("name") != expected_probe
            or config.get("id") != config_id
        ):
            return False
        lengths = (len(row["ids"]), len(row["ground_truth_labels"]), len(row["output_scores"]))
        if not lengths[0] or len(set(lengths)) != 1:
            return False
    return True


def extract(args: argparse.Namespace) -> None:
    from models_under_pressure import r2

    r2.download_file = lambda _bucket, _key, path: Path(path).exists()
    r2.upload_file = lambda _bucket, _key, _path: True
    r2.list_bucket_files = lambda _bucket: []
    r2.delete_file = lambda _bucket, _key: True

    from models_under_pressure.activation_store import ActivationStore, ActivationsSpec
    from models_under_pressure.interfaces.dataset import LabelledDataset
    from models_under_pressure.model import HookedModel, LLMModel

    register_architecture_adapter(args)
    HookedModel.make_hook = lambda self, layer: pre_layer_norm_hook(  # type: ignore[method-assign]
        self.cache, layer
    )

    activation_root = PROJECT / "data/activations"
    for name in ("activations", "input_ids", "attention_masks", "_hf"):
        (activation_root / name).mkdir(parents=True, exist_ok=True)
    manifest = activation_root / "manifest.json"
    if not manifest.exists():
        manifest.write_text('{"rows": []}\n')
    run(args, "datasets", "download")
    validate_frozen_inputs(args)
    model = LLMModel.load(
        args.model,
        batch_size=args.batch_size,
        model_kwargs={"revision": args.model_revision},
        tokenizer_kwargs={"revision": args.model_revision},
    )
    store = ActivationStore()
    provenance_path = activation_root / "submission_provenance.json"
    provenance = load_activation_provenance(provenance_path)
    for dataset_path in (TRAIN, VALIDATION, *EVALS):
        dataset = LabelledDataset.load_from(dataset_path)
        for layer in args.layers:
            spec = ActivationsSpec(
                model_name=args.model, dataset_path=dataset_path, layer=layer
            )
            expected_provenance = activation_provenance_row(args, dataset_path, layer)
            if store.exists(spec) and expected_provenance in provenance:
                print(f"exists: {dataset_path.name} L{layer}")
                continue
            print(f"extracting: {dataset_path.name} L{layer}")
            activations, inputs = model.get_batched_activations_for_layers(
                dataset=dataset, layers=[layer]
            )
            store.save(args.model, dataset_path, [layer], activations, inputs)
            relative_dataset = str(dataset_path.relative_to(PROJECT))
            provenance = [
                row
                for row in provenance
                if not (
                    row.get("model") == args.model
                    and row.get("dataset") == relative_dataset
                    and row.get("layer") == layer
                )
            ]
            provenance.append(expected_provenance)
            save_activation_provenance(provenance_path, provenance)
            del activations, inputs
            remove_temporary_tensors()
            checkpoint_activations(args, activation_root)
            gc.collect()


def evaluate(args: argparse.Namespace) -> None:
    run(args, "datasets", "download")
    validate_frozen_inputs(args)
    restore_activations(args, PROJECT / "data/activations")
    eval_items = ", ".join(
        f"{path.parent.name}_{path.stem}: {path.relative_to(PROJECT)}" for path in EVALS
    )
    for seed in args.seeds:
        for probe in ("softmax", "attention"):
            for layer in args.layers:
                result_path = (
                    PROJECT
                    / "data/results/evaluate_probes"
                    / f"results_{args.run_id}_{probe}_L{layer}_seed{seed}.jsonl"
                )
                if result_is_complete(
                    result_path,
                    model=args.model,
                    layer=layer,
                    probe=probe,
                    config_id=f"{args.run_id}_{probe}_L{layer}_seed{seed}",
                ):
                    print(f"exists: {probe} L{layer} seed {seed}")
                    continue
                result_path.unlink(missing_ok=True)
                run(
                    args,
                    "exp",
                    "+experiment=evaluate_probe",
                    f"probe={probe}",
                    f"model={args.model_config}",
                    f"model.name={args.model}",
                    f"++model.layer={layer}",
                    f"eval_datasets={{{eval_items}}}",
                    f"+id={args.run_id}_{probe}_L{layer}_seed{seed}",
                    "validation_dataset=true",
                    "compute_activations=false",
                    f"random_seed={seed}",
                )


def collect(
    args: argparse.Namespace,
    include_activations: bool,
    include_results: bool,
) -> None:
    output = ROOT / "artifacts/models-under-pressure" / args.artifact_name
    output.mkdir(parents=True, exist_ok=True)
    if include_results:
        source = PROJECT / "data/results/evaluate_probes"
        destination = output / "results_evaluate_probes"
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()
        for seed in args.seeds:
            for probe in ("softmax", "attention"):
                for layer in args.layers:
                    result = source / f"results_{args.run_id}_{probe}_L{layer}_seed{seed}.jsonl"
                    if not result.exists():
                        raise FileNotFoundError(result)
                    shutil.copy2(result, destination / result.name)
    if include_activations:
        checkpoint_activations(args, PROJECT / "data/activations")
    dataset_paths = (TRAIN, VALIDATION, *EVALS)
    protocol = Path(args.protocol)
    shutil.copy2(protocol, output / "submission_protocol.json")
    activation_manifest = PROJECT / "data/activations/manifest.json"
    summary = {
        "schema_version": 2,
        "model": args.model,
        "model_revision": args.model_revision,
        "model_config": args.model_config,
        "run_id": args.run_id,
        "layers": args.layers,
        "seeds": args.seeds,
        "probes": ["attention", "softmax"],
        "activation_site": args.activation_site,
        "architecture_adapter": args.architecture_adapter,
        "train_dataset": str(TRAIN.relative_to(PROJECT)),
        "validation_dataset": str(VALIDATION.relative_to(PROJECT)),
        "dev_datasets": [str(path.relative_to(PROJECT)) for path in DEV_EVALS],
        "test_datasets": [str(path.relative_to(PROJECT)) for path in TEST_EVALS],
        "dataset_sha256": {
            str(path.relative_to(PROJECT)): sha256_file(path) for path in dataset_paths
        },
        "upstream_source_tree": source_tree_digest(PROJECT),
        "wrapper_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(protocol),
        "activation_manifest_sha256": (
            sha256_file(activation_manifest) if activation_manifest.exists() else None
        ),
        "activation_provenance_sha256": (
            sha256_file(PROJECT / "data/activations/submission_provenance.json")
            if (PROJECT / "data/activations/submission_provenance.json").exists()
            else None
        ),
        "activation_provenance_rows": len(expected_activation_provenance(args)),
        "runtime": {
            "python": sys.version.split()[0],
            "models_under_pressure": importlib.metadata.version(
                "models-under-pressure"
            ),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "activations": f"{artifact_uri(args)}/activations",
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"artifacts: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key")
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--model-config")
    parser.add_argument("--activation-site", default=DEFAULT_ACTIVATION_SITE)
    parser.add_argument("--architecture-adapter")
    parser.add_argument("--layers")
    parser.add_argument("--artifact-name")
    parser.add_argument("--run-id")
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--protocol", default=str(RUNTIME.parent / "submission_protocol.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--stage", choices=("all", "extract", "evaluate", "plan"), default="all"
    )
    args = parser.parse_args()
    if not args.model_key:
        parser.error("submission runs require --model-key from the frozen protocol")
    if args.model_key:
        protocol = json.loads(Path(args.protocol).read_text())
        try:
            model = protocol["models"][args.model_key]
        except KeyError:
            parser.error(f"unknown model key: {args.model_key}")
        args.model = model["model_id"]
        args.model_revision = model["revision"]
        args.model_config = model["model_config"]
        args.activation_site = model.get(
            "activation_site", protocol.get("activation_site", DEFAULT_ACTIVATION_SITE)
        )
        args.architecture_adapter = model.get("architecture_adapter")
        args.layers = ",".join(str(layer) for layer in model["candidate_layers"])
        args.seeds = ",".join(str(seed) for seed in protocol["seeds"])
        args.artifact_name = args.artifact_name or f"{protocol['analysis_id']}-{args.model_key}"
        args.run_id = args.run_id or f"{protocol['analysis_id']}_{args.model_key}".replace(
            "-", "_"
        )
    required = ("model", "model_revision", "model_config", "layers", "artifact_name", "run_id")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"missing explicit arguments without --model-key: {', '.join(missing)}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        parser.error("--model-revision must be a pinned 40-character commit")
    args.layers = tuple(int(layer) for layer in args.layers.split(","))
    args.seeds = tuple(int(seed) for seed in args.seeds.split(","))
    if len(args.seeds) < 3:
        parser.error("submission runs require at least three seeds")
    if args.stage == "plan":
        print(
            json.dumps(
                {
                    "model_key": args.model_key,
                    "model": args.model,
                    "model_revision": args.model_revision,
                    "model_config": args.model_config,
                    "layers": args.layers,
                    "seeds": args.seeds,
                    "artifact_name": args.artifact_name,
                    "run_id": args.run_id,
                    "activation_site": args.activation_site,
                    "architecture_adapter": args.architecture_adapter,
                },
                indent=2,
            )
        )
        return
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
    if args.stage in ("all", "extract"):
        extract(args)
    if args.stage in ("all", "evaluate"):
        evaluate(args)
    collect(
        args,
        include_activations=args.stage != "evaluate",
        include_results=args.stage != "extract",
    )


if __name__ == "__main__":
    main()
