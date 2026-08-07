from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "external" / "models-under-pressure"
RUNTIME = ROOT / "replications" / "models-under-pressure" / "runtime"
TRAIN = PROJECT / "data/training/prompts_4x/train.jsonl"
VALIDATION = PROJECT / "data/training/prompts_4x/test.jsonl"
EVALS = tuple(
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
    }
    subprocess.run(command(parts), cwd=PROJECT, env=env, check=True)


def remove_temporary_tensors() -> None:
    for path in (PROJECT / "data/activations").rglob("*.pt"):
        path.unlink(missing_ok=True)


def extract(args: argparse.Namespace) -> None:
    from models_under_pressure.activation_store import ActivationStore, ActivationsSpec
    from models_under_pressure.interfaces.dataset import LabelledDataset
    from models_under_pressure.model import LLMModel

    activation_root = PROJECT / "data/activations"
    for name in ("activations", "input_ids", "attention_masks", "_hf"):
        (activation_root / name).mkdir(parents=True, exist_ok=True)
    run(args, "datasets", "download")
    model = LLMModel.load(args.model, batch_size=args.batch_size)
    store = ActivationStore()
    for dataset_path in (TRAIN, VALIDATION, *EVALS):
        dataset = LabelledDataset.load_from(dataset_path)
        for layer in args.layers:
            spec = ActivationsSpec(
                model_name=args.model, dataset_path=dataset_path, layer=layer
            )
            if store.exists(spec):
                print(f"exists: {dataset_path.name} L{layer}")
                continue
            print(f"extracting: {dataset_path.name} L{layer}")
            activations, inputs = model.get_batched_activations_for_layers(
                dataset=dataset, layers=[layer]
            )
            store.save(args.model, dataset_path, [layer], activations, inputs)
            remove_temporary_tensors()
            gc.collect()


def evaluate(args: argparse.Namespace) -> None:
    run(args, "datasets", "download")
    for probe in ("softmax", "attention"):
        for layer in args.layers:
            run(
                args,
                "exp",
                "+experiment=evaluate_probe",
                f"probe={probe}",
                f"model={args.model_config}",
                f"model.name={args.model}",
                f"model.layer={layer}",
                "eval_datasets=test_balanced",
                f"+id={args.run_id}_{probe}_L{layer}",
                "validation_dataset=true",
                "compute_activations=false",
                "random_seed=42",
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
        for probe in ("softmax", "attention"):
            for layer in args.layers:
                result = source / f"results_{args.run_id}_{probe}_L{layer}.jsonl"
                if not result.exists():
                    raise FileNotFoundError(result)
                shutil.copy2(result, destination / result.name)
    if include_activations:
        source = PROJECT / "data/activations"
        destination = output / "activations"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    summary = {
        "model": args.model,
        "model_config": args.model_config,
        "layers": args.layers,
        "probes": ["attention", "softmax"],
        "train_dataset": str(TRAIN.relative_to(PROJECT)),
        "validation_dataset": str(VALIDATION.relative_to(PROJECT)),
        "eval_datasets": [str(path.relative_to(PROJECT)) for path in EVALS],
    }
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"artifacts: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--stage", choices=("all", "extract", "evaluate"), default="all"
    )
    args = parser.parse_args()
    args.layers = tuple(int(layer) for layer in args.layers.split(","))
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
