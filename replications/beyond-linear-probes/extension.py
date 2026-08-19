from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "external" / "tpc"
CONFIG = json.loads((HERE / "extension_models.json").read_text())
ARTIFACTS = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts")) / "beyond-linear-probes" / "extensions"
UPSTREAM_CACHE = Path("/data/scratch/acw663/poly-monitor/activation-cache")
SUBLABELS = [
    "benign",
    "causing_material_harm_by_disseminating_misinformation",
    "copyright_violations",
    "cyberattack",
    "defamation_encouraging_unethical_or_unsafe_actions",
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns",
    "fraud_assisting_illegal_activities",
    "mental_health_over-reliance_crisis",
    "others",
    "private_information_individual",
    "sensitive_information_organization_government",
    "sexual_content",
    "social_stereotypes_and_unfair_discrimination",
    "toxic_language_hate_speech",
    "violence_and_physical_harm",
]
RESULT_FILES = (
    "poly-vals.pkl",
    "bi-vals.pkl",
    "linear-vals.pkl",
    "mlp-vals.pkl",
    "eemlp-matched-vals.pkl",
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.env import load_project_env  # noqa: E402

SPEC = importlib.util.spec_from_file_location("beyond_linear_probes_run", HERE / "run.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load the baseline wrapper")
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_config(key: str) -> dict:
    return CONFIG["models"][key]


def artifact_root(key: str) -> Path:
    return ARTIFACTS / key


def cache_root(key: str) -> Path:
    return artifact_root(key) / "activation-cache" / "WildGuard"


def model_name(key: str) -> str:
    return model_config(key)["id"].split("/")[-1]


def activation_path(key: str, layer: int, split: str) -> Path:
    return cache_root(key) / f"{model_name(key)}-L{layer}-mean-{split}-X.npy"


def label_path(key: str, split: str, kind: str = "y") -> Path:
    return cache_root(key) / f"{model_name(key)}-mean-{split}-{kind}.npy"


def state(key: str, stage: str, **extra) -> None:
    payload = {
        "stage": stage,
        "model_key": key,
        "model": model_config(key)["id"],
        "model_revision": model_config(key)["revision"],
        "dataset": CONFIG["dataset"]["id"],
        "dataset_revision": CONFIG["dataset"]["revision"],
        "layers": model_config(key)["layers"],
        "train_layer": model_config(key)["train_layer"],
        "timestamp": now(),
    } | extra
    path = artifact_root(key) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def verify_config(key: str) -> dict:
    from huggingface_hub import HfApi
    import torch
    from transformers import AutoConfig

    BASELINE.verify_source()
    cfg = model_config(key)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    current_model_revision = api.model_info(cfg["id"]).sha
    current_dataset_revision = api.dataset_info(CONFIG["dataset"]["id"]).sha
    architecture = AutoConfig.from_pretrained(
        cfg["id"], revision=cfg["revision"], trust_remote_code=True
    )
    layers = int(getattr(architecture, "num_hidden_layers"))
    hidden = int(getattr(architecture, "hidden_size"))
    if layers != cfg["num_hidden_layers"] or hidden != cfg["hidden_size"]:
        raise RuntimeError(
            f"Config mismatch for {key}: layers={layers}, hidden={hidden}"
        )
    if max(cfg["layers"]) >= layers:
        raise RuntimeError(f"Invalid layer selection for {key}")
    torch_version = torch.__version__.split("+")[0]
    if torch_version != "2.7.1":
        raise RuntimeError(f"Expected torch 2.7.1, found {torch.__version__}")
    compiler = shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        raise RuntimeError("A C compiler is required by the model attention kernels")
    return {
        "model": cfg["id"],
        "pinned_model_revision": cfg["revision"],
        "current_model_revision": current_model_revision,
        "dataset": CONFIG["dataset"]["id"],
        "pinned_dataset_revision": CONFIG["dataset"]["revision"],
        "current_dataset_revision": current_dataset_revision,
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "layers": cfg["layers"],
        "train_layer": cfg["train_layer"],
        "prompt_style": cfg["prompt_style"],
        "torch": torch.__version__,
        "compiler": compiler,
        "upstream_commit": BASELINE.source_commit(),
        "upstream_source_digest": BASELINE.source_digest(),
    }


def decoder_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "get_decoder"):
        decoder = model.get_decoder()
        if hasattr(decoder, "layers"):
            return decoder.layers
    for module in model.modules():
        if hasattr(module, "layers"):
            try:
                if len(module.layers):
                    return module.layers
            except TypeError:
                continue
    raise RuntimeError("Could not locate decoder blocks")


def prepare_dataset(tokenizer, split: str, cfg: dict):
    from datasets import load_dataset

    dataset_cfg = CONFIG["dataset"]
    source_config = dataset_cfg[f"{split}_config"]
    source_split = dataset_cfg[f"{split}_split"]
    dataset = load_dataset(
        dataset_cfg["id"],
        source_config,
        split=source_split,
        revision=dataset_cfg["revision"],
    )
    name_to_index = {name: index for index, name in enumerate(SUBLABELS)}
    max_length = CONFIG["extraction"]["max_length"]

    def tokenize(batch):
        if cfg["prompt_style"] == "chat":
            conversations = [
                [{"role": "user", "content": prompt}] for prompt in batch["prompt"]
            ]
            encoded = tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        else:
            encoded = tokenizer(
                batch["prompt"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": [
                int(label == "harmful") for label in batch["prompt_harm_label"]
            ],
            "y_multi": [name_to_index.get(name, name_to_index["others"]) for name in batch["subcategory"]],
            "adversarial": batch["adversarial"],
        }

    dataset = dataset.map(tokenize, batched=True, batch_size=128)
    return dataset.with_format(
        type="torch",
        columns=("input_ids", "attention_mask", "labels", "y_multi", "adversarial"),
    )


def write_split(model, loader, key: str, split: str) -> None:
    import torch
    from numpy.lib.format import open_memmap
    from tqdm import tqdm

    cfg = model_config(key)
    root = cache_root(key)
    root.mkdir(parents=True, exist_ok=True)
    rows = len(loader.dataset)
    final_paths = {
        layer: activation_path(key, layer, split) for layer in cfg["layers"]
    }
    partial_paths = {
        layer: path.with_suffix(".partial.npy") for layer, path in final_paths.items()
    }
    final_labels = {
        "y": label_path(key, split),
        "y_multi": label_path(key, split, "y_multi"),
        "y_adv": label_path(key, split, "y_adv"),
    }
    partial_labels = {
        name: path.with_suffix(".partial.npy") for name, path in final_labels.items()
    }
    for path in (*partial_paths.values(), *partial_labels.values()):
        path.unlink(missing_ok=True)
    maps = {
        layer: open_memmap(
            partial_paths[layer],
            mode="w+",
            dtype="float32",
            shape=(rows, cfg["hidden_size"]),
        )
        for layer in cfg["layers"]
    }
    labels = open_memmap(partial_labels["y"], "w+", dtype="int64", shape=(rows,))
    multi = open_memmap(partial_labels["y_multi"], "w+", dtype="int64", shape=(rows,))
    adversarial = open_memmap(partial_labels["y_adv"], "w+", dtype="int64", shape=(rows,))
    activations = {layer: None for layer in cfg["layers"]}
    handles = []
    blocks = decoder_blocks(model)
    for layer in cfg["layers"]:
        def hook(index):
            def capture(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                activations[index] = value.detach()
            return capture

        handles.append(blocks[layer].register_forward_hook(hook(layer)))

    try:
        offset = 0
        for batch in tqdm(loader, desc=f"{key}:{split}-all-layers"):
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention = batch["attention_mask"].cuda(non_blocking=True)
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention)
            size = input_ids.shape[0]
            mask = attention.unsqueeze(-1).cpu()
            for layer in cfg["layers"]:
                hidden = activations[layer].cpu().to(torch.float32)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                maps[layer][offset : offset + size] = pooled.numpy()
            end = offset + size
            labels[offset:end] = batch["labels"].numpy()
            multi[offset:end] = batch["y_multi"].numpy()
            adversarial[offset:end] = batch["adversarial"].long().numpy()
            offset = end
        if offset != rows:
            raise RuntimeError(f"Incomplete {key}:{split} extraction: {offset}/{rows}")
    finally:
        for handle in handles:
            handle.remove()
        for value in (*maps.values(), labels, multi, adversarial):
            value.flush()
    for layer, path in partial_paths.items():
        path.replace(final_paths[layer])
    for name, path in partial_labels.items():
        path.replace(final_labels[name])


def extract(key: str) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = model_config(key)
    state(key, "extracting", settings=CONFIG["extraction"])
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["id"], revision=cfg["revision"], trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["id"],
        revision=cfg["revision"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    for split in ("train", "test"):
        expected = [activation_path(key, layer, split) for layer in cfg["layers"]]
        expected += [label_path(key, split)]
        if all(path.exists() for path in expected):
            print(f"{key}:{split} already extracted")
            continue
        dataset = prepare_dataset(tokenizer, split, cfg)
        loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        write_split(model, loader, key, split)
        state(key, f"{split}_extracted", rows=len(dataset))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    state(key, "extracted", files=cache_manifest(key))


def link_cache(key: str) -> None:
    destination = UPSTREAM_CACHE / "WildGuard"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_symlink() and destination.resolve() == cache_root(key).resolve():
            return
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.symlink_to(cache_root(key), target_is_directory=True)


def result_directory(key: str, layer: int) -> Path:
    return SOURCE / "results" / "numbers" / "wildguard" / f"{model_name(key)}-L{layer}-mean--progressive-maxorder-5_rank-64"


def train(key: str) -> None:
    cfg = model_config(key)
    training = CONFIG["training"]
    layer = cfg["train_layer"]
    missing = [activation_path(key, layer, split) for split in ("train", "test") if not activation_path(key, layer, split).exists()]
    if missing:
        raise FileNotFoundError(f"Missing activations: {missing}")
    link_cache(key)
    output = artifact_root(key)
    metrics = output / "metrics.jsonl"
    predictions = output / "test_predictions.jsonl"
    state(key, "training", settings=training)
    command = [
        sys.executable,
        str(HERE / "instrument_sweep.py"),
        "--metrics-path", str(metrics),
        "--predictions-path", str(predictions),
        "--rank", str(training["rank"]),
        "--max_order", str(training["max_order"]),
        "--max_epochs", str(training["max_epochs"]),
        "--layer_idx", str(layer),
        "--pool_type", training["pool_type"],
        "--train_mode", training["train_mode"],
        "--batch_size", str(training["batch_size"]),
        "--model_name", cfg["id"],
        "--dataset", "WildGuard",
    ]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=SOURCE, check=True)
    destination = output / "results"
    shutil.copytree(result_directory(key, layer), destination, dirs_exist_ok=True)
    state(key, "trained", results=result_manifest(key))


def flatten(value) -> list[float]:
    if hasattr(value, "tolist"):
        return flatten(value.tolist())
    if isinstance(value, (list, tuple)):
        return [number for child in value for number in flatten(child)]
    return [float(value)]


def result_manifest(key: str) -> dict:
    directory = artifact_root(key) / "results"
    files = {}
    for name in RESULT_FILES:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        values = flatten(payload["y_vals_test"])
        files[name] = {
            "sha256": sha256(path),
            "test_f1_mean": float(np.mean(values)),
            "test_f1_std": float(np.std(values)),
            "values": values,
        }
    return files


def cache_manifest(key: str) -> dict:
    files = {}
    for path in sorted(cache_root(key).glob("*.npy")):
        array = np.load(path, mmap_mode="r")
        files[path.name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": path.stat().st_size,
        }
    return files


def verify(key: str) -> None:
    cfg = model_config(key)
    for split in ("train", "test"):
        for layer in cfg["layers"]:
            path = activation_path(key, layer, split)
            if not path.exists():
                raise FileNotFoundError(path)
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2 or array.shape[1] != cfg["hidden_size"]:
                raise RuntimeError(f"Invalid activation array: {path} {array.shape}")
    results = result_manifest(key)
    metrics = artifact_root(key) / "metrics.jsonl"
    predictions = artifact_root(key) / "test_predictions.jsonl"
    if not metrics.exists() or not predictions.exists():
        raise FileNotFoundError("Missing metric sidecars")
    state(
        key,
        "complete",
        extraction=cache_manifest(key),
        results=results,
        metrics_sha256=sha256(metrics),
        predictions_sha256=sha256(predictions),
    )
    print(json.dumps({"model": key, "stage": "complete", "results": results}, indent=2))


def pipeline(key: str) -> None:
    print(json.dumps(verify_config(key), indent=2))
    extract(key)
    train(key)
    verify(key)


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inspect", "preflight", "extract", "train", "verify", "pipeline"), required=True)
    parser.add_argument("--model", choices=tuple(CONFIG["models"]), required=True)
    args = parser.parse_args()
    if args.stage in {"inspect", "preflight"}:
        print(json.dumps(verify_config(args.model), indent=2))
    elif args.stage == "extract":
        extract(args.model)
    elif args.stage == "train":
        train(args.model)
    elif args.stage == "verify":
        verify(args.model)
    else:
        pipeline(args.model)


if __name__ == "__main__":
    main()
