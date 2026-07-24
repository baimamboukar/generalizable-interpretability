"""Modal entrypoints for the Among Us Phi-4 linear-probe replication.

Examples:
    modal run replications/among-us/modal_probes.py --stage prepare
    modal run replications/among-us/modal_probes.py --stage cache --dataset TruthfulQA
    modal run replications/among-us/modal_probes.py --stage cache --dataset all
    modal run replications/among-us/modal_probes.py --stage train
    modal run replications/among-us/modal_probes.py --stage evaluate
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys

import modal


HERE = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/cache")
REMOTE_UPSTREAM = Path("/upstream")
# During local image construction, resolve the pinned checkout from the project
# root. During Modal's remote module import, the same files already live at
# /upstream in the built image.
UPSTREAM = (
    HERE.parents[1] / "external" / "AmongUs"
    if len(HERE.parents) > 1
    else REMOTE_UPSTREAM
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.0,<2",
        "aiofiles==24.1.0",
        "aiohttp==3.11.11",
        "huggingface-hub>=0.35,<1",
        "httpx==0.28.1",
        "numpy==2.0.2",
        "pandas==2.2.3",
        "plotly==5.24.1",
        "PyYAML==6.0.2",
        "python-dotenv==1.0.1",
        "requests==2.32.3",
        "scikit-learn>=1.5,<2",
        "torch==2.5.1",
        "tqdm>=4.66,<5",
        "transformers",
    )
    # Modal mounts the entrypoint at /root/modal_probes.py. Keep the helper
    # beside it so `from baseline import ...` works in both local and remote
    # execution contexts.
    .add_local_file(HERE / "baseline.py", "/root/baseline.py", copy=True)
    .add_local_file(
        HERE / "extension.py", "/root/among_us_extension.py", copy=True
    )
    # These are copied byte-for-byte from the pinned upstream checkout. The
    # labeling stage executes eval.py directly; our wrapper does not reimplement
    # or patch its prompt, API request, parsing, or concurrency behavior.
    .add_local_dir(UPSTREAM / "evaluations", "/upstream/evaluations", copy=True)
    .add_local_dir(
        UPSTREAM / "linear-probes", "/upstream/linear-probes", copy=True
    )
    .add_local_file(UPSTREAM / "utils.py", "/upstream/utils.py", copy=True)
)

app = modal.App("among-us-probe-replication")
secret = modal.Secret.from_name("mup-env")
volume = modal.Volume.from_name("among-us-probes", create_if_missing=True)
mounts = {str(REMOTE_ROOT): volume}


@app.function(image=image, secrets=[secret], volumes=mounts, timeout=60 * 60)
def prepare() -> dict:
    from baseline import prepare_data

    result = prepare_data(REMOTE_ROOT)
    volume.commit()
    return result


@app.function(image=image, secrets=[secret], volumes=mounts, cpu=2, memory=8192,
              timeout=6 * 60 * 60)
def generate_labels() -> dict:
    """Run the official GPT-4o-mini labeling script without modifying it."""
    from baseline import AMONG_US_EXPERIMENT, AMONG_US_LABEL_FILE, prepare_data

    # The existing mup-env secret uses OPEN_ROUTER_API_KEY; upstream eval.py
    # reads OPENROUTER_API_KEY. Map the verified local name without changing
    # the official script or exposing the value.
    if not os.environ.get("OPENROUTER_API_KEY") and os.environ.get(
        "OPEN_ROUTER_API_KEY"
    ):
        os.environ["OPENROUTER_API_KEY"] = os.environ["OPEN_ROUTER_API_KEY"]

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "The mup-env Modal secret exposes neither OPEN_ROUTER_API_KEY nor "
            "OPENROUTER_API_KEY."
        )

    prepare_data(REMOTE_ROOT)
    log_source = REMOTE_ROOT / "data" / "agent-logs-compact.json"
    log_dir = REMOTE_UPSTREAM / "expt-logs" / AMONG_US_EXPERIMENT
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(log_source, log_dir / "agent-logs-compact.json")
    (REMOTE_UPSTREAM / "evaluations" / "results").mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "eval.py",
            "--expt_name",
            AMONG_US_EXPERIMENT,
            "--evaluator",
            "openai/gpt-4o-mini",
        ],
        cwd=REMOTE_UPSTREAM / "evaluations",
        check=True,
    )

    generated = REMOTE_UPSTREAM / "evaluations" / "results" / AMONG_US_LABEL_FILE
    destination = REMOTE_ROOT / "data" / AMONG_US_LABEL_FILE
    shutil.copy2(generated, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    rows = sum(1 for line in destination.open() if line.strip())
    volume.commit()
    result = {"path": str(destination), "rows": rows, "sha256": digest}
    print(result)
    return result


@app.function(image=image, volumes=mounts, cpu=2, memory=4096, timeout=30 * 60)
def inspect_labels() -> dict:
    """Read-only integrity check for the official evaluator output."""
    from baseline import AMONG_US_LABEL_FILE

    labels_path = REMOTE_ROOT / "data" / AMONG_US_LABEL_FILE
    logs_path = REMOTE_ROOT / "data" / "agent-logs-compact.json"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    with labels_path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    with logs_path.open() as handle:
        log_rows = [json.loads(line) for line in handle if line.strip()]

    score_fields = ("awareness", "lying", "deception", "planning")
    invalid = [
        row
        for row in rows
        if any(str(row.get(field)) == "-1" for field in score_fields)
    ]
    timestamps = [str(row.get("timestamp", "")) for row in rows]
    duplicate_timestamps = len(timestamps) - len(set(timestamps))
    by_timestamp = {str(row.get("timestamp", "")): row for row in rows}

    # The released Figure-7 code evaluates Among Us chunk 1: rows 500-999.
    eval_timestamps = [
        str(row.get("timestamp", "")) for row in log_rows[500:1000]
    ]
    missing_eval = [timestamp for timestamp in eval_timestamps if timestamp not in by_timestamp]
    invalid_eval = [
        timestamp
        for timestamp in eval_timestamps
        if timestamp in by_timestamp
        and any(
            str(by_timestamp[timestamp].get(field)) == "-1"
            for field in ("lying", "deception")
        )
    ]
    result = {
        "rows": len(rows),
        "invalid_rows": len(invalid),
        "duplicate_timestamps": duplicate_timestamps,
        "figure7_eval_rows": len(eval_timestamps),
        "figure7_missing_labels": len(missing_eval),
        "figure7_invalid_lying_or_deception": len(invalid_eval),
        "usable_for_figure7": not (
            duplicate_timestamps or missing_eval or invalid_eval
        ),
    }
    print(result)
    return result


@app.function(
    image=image,
    secrets=[secret],
    volumes=mounts,
    gpu="A100-80GB",
    memory=131072,
    # Modal currently enforces a 512 GiB minimum for explicit disk requests.
    ephemeral_disk=524288,
    timeout=24 * 60 * 60,
)
def cache_activations(
    dataset: str,
    profile: str = "paper",
    batch_size: int = 1,
    force: bool = False,
) -> str:
    """Cache with the released dataset classes and literal Phi-4 loader."""
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from baseline import (
        AMONG_US_EXPERIMENT,
        DATASET_NAMES,
        activation_path,
        get_profile,
        prepare_data,
    )

    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASET_NAMES}")
    if batch_size != 1:
        raise ValueError("The released activation code processes one row at a time")

    profile_config = get_profile(profile)
    destination = activation_path(REMOTE_ROOT, profile_config, dataset)
    if destination.exists() and not force:
        print(f"Already cached: {destination}")
        return str(destination)

    prepare_data(REMOTE_ROOT)
    probe_root = REMOTE_UPSTREAM / "linear-probes"

    # Put the public inputs where the released dataset classes expect them.
    truthful_dir = probe_root / "data" / "TruthfulQA"
    repeng_dir = probe_root / "data" / "RepE"
    truthful_dir.mkdir(parents=True, exist_ok=True)
    repeng_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        REMOTE_ROOT / "data" / "TruthfulQA.csv",
        truthful_dir / "TruthfulQA.csv",
    )
    shutil.copy2(
        REMOTE_ROOT / "data" / "true_false_facts.csv",
        repeng_dir / "true_false_facts.csv",
    )
    log_dir = REMOTE_UPSTREAM / "expt-logs" / AMONG_US_EXPERIMENT
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        REMOTE_ROOT / "data" / "agent-logs-compact.json",
        log_dir / "agent-logs-compact.json",
    )

    # Import and use the authors' dataset/preprocessing implementation unchanged.
    sys.path.insert(0, str(probe_root))
    from configs import config_phi4
    from probe_datasets import (
        AmongUsDataset,
        DishonestQADataset,
        RepEngDataset,
        TruthfulQADataset,
    )

    config = config_phi4.copy()
    config["layer"] = str(profile_config.layer)
    config["hook_component"] = f"model.layers[{profile_config.layer}]"
    model_name = config["model_name"]

    # These are the exact loader calls in all_layers_cache_train_eval.py.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto"
    )
    device = model.device

    classes = {
        "TruthfulQA": TruthfulQADataset,
        "DishonestQA": DishonestQADataset,
        "RepEng": RepEngDataset,
        "AmongUs": AmongUsDataset,
    }
    kwargs = {
        "config": config,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "test_split": 1.0,
    }
    if dataset == "AmongUs":
        kwargs["expt_name"] = config["expt_name"]
    official_dataset = classes[dataset](**kwargs)
    eval(f"model.{config['hook_component']}").register_forward_hook(
        official_dataset.activation_cache.hook_fn
    )
    official_dataset.populate_dataset(
        force_redo=True,
        just_load=False,
        max_rows=1000,
        seq_len=config["seq_len"],
        num_tokens=profile_config.num_tokens,
        chunk_size=500,
    )

    chunk_paths = sorted(
        (probe_root / official_dataset.activations_dir).glob("chunk_*.pkl")
    )
    if not chunk_paths:
        raise RuntimeError(f"The released cache code produced no chunks for {dataset}")
    rows = []
    for chunk_path in chunk_paths:
        with chunk_path.open("rb") as handle:
            rows.extend(pickle.load(handle))
    activations = torch.stack(
        [torch.from_numpy(np.stack(acts)).float() for acts, _ in rows]
    )
    labels = torch.tensor([int(label) for _, label in rows], dtype=torch.int64)

    metadata = [{} for _ in rows]
    if dataset == "AmongUs":
        metadata = [
            {
                "timestamp": str(row.get("timestamp", "")),
                "game_index": str(row.get("game_index", "")),
                "step": int(row.get("step", 0)),
                "player_name": str(row.get("player.name", "")),
                "player_identity": str(row.get("player.identity", "")),
            }
            for _, row in official_dataset.agent_logs_df.iloc[: len(rows)].iterrows()
        ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "activations": activations,
            "labels": labels,
            "metadata": metadata,
            "dataset": dataset,
            "model": model_name,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "transformers_version": transformers.__version__,
            "profile": profile_config.__dict__,
            "activation_source": (
                "external/AmongUs/linear-probes/probe_datasets.py"
            ),
        },
        destination,
    )
    volume.commit()
    print(f"Saved {tuple(activations.shape)} to {destination}")
    return str(destination)


@app.function(image=image, volumes=mounts, cpu=8, memory=32768, timeout=6 * 60 * 60)
def train(profile: str = "paper") -> list[str]:
    """Run the official train_probes.py unchanged on our cached activations."""
    import torch
    from baseline import (
        AMONG_US_EXPERIMENT,
        DATASET_NAMES,
        activation_path,
        checkpoint_path,
        get_profile,
        require_complete_cache,
    )

    require_complete_cache(REMOTE_ROOT, profile)
    profile_config = get_profile(profile)
    probe_root = REMOTE_UPSTREAM / "linear-probes"

    # Supply the exact on-disk layout consumed by the official dataset classes.
    truthful_dir = probe_root / "data" / "TruthfulQA"
    repeng_dir = probe_root / "data" / "RepE"
    truthful_dir.mkdir(parents=True, exist_ok=True)
    repeng_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REMOTE_ROOT / "data" / "TruthfulQA.csv", truthful_dir / "TruthfulQA.csv")
    shutil.copy2(
        REMOTE_ROOT / "data" / "true_false_facts.csv",
        repeng_dir / "true_false_facts.csv",
    )
    log_dir = REMOTE_UPSTREAM / "expt-logs" / AMONG_US_EXPERIMENT
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        REMOTE_ROOT / "data" / "agent-logs-compact.json",
        log_dir / "agent-logs-compact.json",
    )

    for dataset_name in DATASET_NAMES:
        payload = torch.load(
            activation_path(REMOTE_ROOT, profile_config, dataset_name),
            map_location="cpu",
            weights_only=False,
        )
        acts = payload["activations"][:, -profile_config.num_tokens :]
        labels = payload["labels"]
        chunks = (
            [
                (acts[: profile_config.among_us_chunk_size], labels[: profile_config.among_us_chunk_size]),
                (
                    acts[
                        profile_config.among_us_chunk_size : 2
                        * profile_config.among_us_chunk_size
                    ],
                    labels[
                        profile_config.among_us_chunk_size : 2
                        * profile_config.among_us_chunk_size
                    ],
                ),
            ]
            if dataset_name == "AmongUs"
            else [(acts, labels)]
        )
        output_dir = probe_root / "data" / f"{dataset_name}_phi4_acts_20"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        for chunk_idx, (chunk_acts, chunk_labels) in enumerate(chunks):
            upstream_chunk = [
                ([token.numpy() for token in row], int(label))
                for row, label in zip(chunk_acts, chunk_labels, strict=True)
            ]
            with (output_dir / f"chunk_{chunk_idx}.pkl").open("wb") as handle:
                pickle.dump(upstream_chunk, handle)

    (probe_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "train_probes.py"], cwd=probe_root, check=True
    )

    # Persist the exact upstream-trained weights in our stable checkpoint format.
    sys.path.insert(0, str(probe_root))
    paths = []
    upstream_names = {
        "TruthfulQA": "TruthfulQADataset",
        "DishonestQA": "DishonestQADataset",
        "AmongUs": "AmongUsDataset",
        "RepEng": "RepEngDataset",
    }
    for dataset_name in DATASET_NAMES:
        source = (
            probe_root
            / "checkpoints"
            / f"{upstream_names[dataset_name]}_probe_phi4.pkl"
        )
        with source.open("rb") as handle:
            upstream_probe = pickle.load(handle)
        destination = checkpoint_path(REMOTE_ROOT, profile_config, dataset_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": upstream_probe.model.state_dict(),
                "input_dim": 5120,
                "training_dataset": dataset_name,
                "profile": profile_config.__dict__,
                "trainer": "external/AmongUs/linear-probes/train_probes.py",
            },
            destination,
        )
        paths.append(destination)

    volume.commit()
    return [str(path) for path in paths]


@app.function(image=image, volumes=mounts, cpu=8, memory=32768, timeout=60 * 60)
def evaluate(profile: str = "paper") -> str:
    from baseline import evaluate_all, require_complete_cache

    require_complete_cache(REMOTE_ROOT, profile)
    path = evaluate_all(REMOTE_ROOT, profile)
    volume.commit()
    return str(path)


@app.function(image=image, volumes=mounts, cpu=8, memory=32768, timeout=60 * 60)
def evaluate_official(profile: str = "paper") -> str:
    """Evaluate the authors' released Phi-4 checkpoints on our activation cache."""
    import torch
    from baseline import DATASET_NAMES, evaluate_all, require_complete_cache

    require_complete_cache(REMOTE_ROOT, profile)
    upstream_names = {
        "TruthfulQA": "TruthfulQADataset",
        "DishonestQA": "DishonestQADataset",
        "AmongUs": "AmongUsDataset",
        "RepEng": "RepEngDataset",
    }
    checkpoint_overrides = {
        dataset_name: torch.load(
            REMOTE_UPSTREAM
            / "linear-probes"
            / "checkpoints"
            / f"{upstream_names[dataset_name]}_probe_phi4.pth",
            map_location="cpu",
            weights_only=True,
        )
        for dataset_name in DATASET_NAMES
    }
    path = evaluate_all(
        REMOTE_ROOT,
        profile,
        checkpoint_overrides=checkpoint_overrides,
        result_filename="phi4_L20_official_checkpoints.json",
    )
    volume.commit()
    return str(path)


@app.function(image=image, volumes=mounts, cpu=2, memory=4096, timeout=30 * 60)
def inspect_cache(profile: str = "paper") -> dict:
    from baseline import validate_cached_inputs

    return validate_cached_inputs(REMOTE_ROOT, profile)


@app.function(
    image=image,
    secrets=[secret],
    volumes=mounts,
    gpu="A100-80GB",
    memory=131072,
    ephemeral_disk=524288,
    timeout=24 * 60 * 60,
)
def cache_extension(model_key: str, dataset: str = "all", force: bool = False) -> list[str]:
    """Cache all three extension layers in one forward pass per example."""
    from baseline import DATASET_NAMES, prepare_data
    from among_us_extension import (
        cache_dataset_with_model,
        get_model_spec,
        load_model,
    )

    spec = get_model_spec(model_key)
    datasets = DATASET_NAMES if dataset == "all" else (dataset,)
    unknown = [name for name in datasets if name not in DATASET_NAMES]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; choose from {DATASET_NAMES}")
    prepare_data(REMOTE_ROOT)
    loaded_model, tokenizer = load_model(spec)
    paths = []
    for name in datasets:
        paths.extend(
            cache_dataset_with_model(
                REMOTE_ROOT,
                spec,
                loaded_model,
                tokenizer,
                name,
                force=force,
            )
        )
        # Preserve completed datasets if a later long extraction is preempted.
        volume.commit()
    return [str(path) for path in paths]


@app.function(image=image, volumes=mounts, cpu=8, memory=32768, timeout=12 * 60 * 60)
def train_extension(model_key: str) -> list[str]:
    from among_us_extension import train_model

    paths = train_model(REMOTE_ROOT, REMOTE_UPSTREAM / "linear-probes", model_key)
    volume.commit()
    return [str(path) for path in paths]


@app.function(image=image, volumes=mounts, cpu=8, memory=32768, timeout=3 * 60 * 60)
def evaluate_extension(model_key: str) -> str:
    from among_us_extension import evaluate_model

    path = evaluate_model(REMOTE_ROOT, model_key)
    volume.commit()
    return str(path)


@app.function(image=image, volumes=mounts, cpu=2, memory=4096, timeout=30 * 60)
def inspect_extension(model_key: str) -> dict:
    from among_us_extension import validate_cache

    result = validate_cache(REMOTE_ROOT, model_key)
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(
    stage: str = "inspect",
    dataset: str = "all",
    profile: str = "paper",
    model: str = "llama-8b",
    batch_size: int = 1,
    force: bool = False,
) -> None:
    from baseline import DATASET_NAMES, PROFILES
    # Remote functions import the image-mounted module name; the local
    # entrypoint reads the source file beside this runner.
    from extension import MODEL_SPECS

    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILES)}")

    if stage == "prepare":
        print(prepare.remote())
    elif stage == "labels":
        print(generate_labels.remote())
    elif stage == "inspect-labels":
        print(inspect_labels.remote())
    elif stage == "cache":
        datasets = DATASET_NAMES if dataset == "all" else (dataset,)
        unknown = [name for name in datasets if name not in DATASET_NAMES]
        if unknown:
            raise ValueError(f"Unknown datasets: {unknown}; choose from {DATASET_NAMES}")
        for name in datasets:
            print(
                cache_activations.remote(
                    dataset=name,
                    profile=profile,
                    batch_size=batch_size,
                    force=force,
                )
            )
    elif stage == "train":
        print(train.remote(profile))
    elif stage == "evaluate":
        print(evaluate.remote(profile))
    elif stage == "evaluate-official":
        print(evaluate_official.remote(profile))
    elif stage == "inspect":
        print(inspect_cache.remote(profile))
    elif stage == "extension-models":
        print(
            json.dumps(
                {key: value.__dict__ for key, value in MODEL_SPECS.items()},
                indent=2,
            )
        )
    elif stage == "extension-cache":
        print(cache_extension.remote(model, dataset=dataset, force=force))
    elif stage == "extension-inspect":
        print(inspect_extension.remote(model))
    elif stage == "extension-train":
        print(train_extension.remote(model))
    elif stage == "extension-evaluate":
        print(evaluate_extension.remote(model))
    else:
        raise ValueError(
            "stage must be one of: prepare, labels, inspect-labels, cache, "
            "train, evaluate, evaluate-official, inspect, extension-models, "
            "extension-cache, extension-inspect, extension-train, "
            "extension-evaluate"
        )
