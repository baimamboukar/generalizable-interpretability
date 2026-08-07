from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "external" / "models-under-pressure"
MODEL = "Qwen/Qwen2.5-32B-Instruct"
MODEL_CONFIG = "qwen3-32b"
LAYERS = (23, 25, 33, 42, 50)
TRAIN = PROJECT / "data/training/prompts_4x/train.jsonl"
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
MUP = [str(Path(sys.executable).with_name("mup"))]
ACTIVATIONS = PROJECT / "data/activations"
RESULTS = PROJECT / "data/results/evaluate_probes"
ARTIFACTS = ROOT / "artifacts/models-under-pressure/qwen25-32b"


def run(*args: str) -> None:
    if args[0] != "mup":
        raise ValueError(f"unsupported command: {args[0]}")
    subprocess.run(MUP + list(args[1:]), cwd=PROJECT, check=True)


def remove_temporary_tensors() -> None:
    for path in ACTIVATIONS.rglob("*.pt"):
        path.unlink(missing_ok=True)


def extract() -> None:
    from models_under_pressure.activation_store import ActivationStore, ActivationsSpec
    from models_under_pressure.interfaces.dataset import LabelledDataset
    from models_under_pressure.model import LLMModel

    for name in ("activations", "input_ids", "attention_masks", "_hf"):
        (ACTIVATIONS / name).mkdir(parents=True, exist_ok=True)
    run("mup", "datasets", "download")
    model = LLMModel.load(MODEL, batch_size=4)
    store = ActivationStore()
    datasets = (TRAIN, *EVALS)

    for dataset_path in datasets:
        dataset = LabelledDataset.load_from(dataset_path)
        for layer in LAYERS:
            spec = ActivationsSpec(
                model_name=MODEL, dataset_path=dataset_path, layer=layer
            )
            if store.exists(spec):
                print(f"exists: {dataset_path.name} L{layer}")
                continue
            print(f"extracting: {dataset_path.name} L{layer}")
            activations, inputs = model.get_batched_activations_for_layers(
                dataset=dataset, layers=[layer]
            )
            store.save(MODEL, dataset_path, [layer], activations, inputs)
            remove_temporary_tensors()
            gc.collect()


def evaluate() -> None:
    run("mup", "datasets", "download")
    for probe in ("attention", "softmax"):
        for layer in LAYERS:
            run(
                "mup",
                "exp",
                "+experiment=evaluate_probe",
                f"probe={probe}",
                f"model={MODEL_CONFIG}",
                f"model.name={MODEL}",
                f"model.layer={layer}",
                "eval_datasets=test_balanced",
                f"+id=qwen25_{probe}_L{layer}",
                "validation_dataset=true",
                "compute_activations=false",
                "random_seed=42",
            )


def collect_artifacts() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (RESULTS, ARTIFACTS / "results"),
        (ACTIVATIONS, ARTIFACTS / "activations"),
    ):
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    summary = {
        "model": MODEL,
        "layers": list(LAYERS),
        "probes": ["attention", "softmax"],
        "train_dataset": str(TRAIN.relative_to(PROJECT)),
        "eval_datasets": [str(path.relative_to(PROJECT)) for path in EVALS],
    }
    (ARTIFACTS / "run.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
    extract()
    evaluate()
    collect_artifacts()
    print(f"artifacts: {ARTIFACTS}")
