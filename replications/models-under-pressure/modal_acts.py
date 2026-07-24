"""Run `mup acts store` on a Modal GPU. Activations upload to the R2 bucket in .env.

    modal run replications/models-under-pressure/modal_acts.py
    modal run replications/models-under-pressure/modal_acts.py \
        --model meta-llama/Llama-3.2-3B-Instruct --layers 11,15 \
        --dataset data/evals/dev/mt_balanced.jsonl
"""

import os
import subprocess
import hashlib
from pathlib import Path
import modal

# Local path to the repo; falls back to /repo inside the container.
_here = Path(__file__).resolve()
REPO = (
    _here.parents[2] / "external" / "models-under-pressure"
    if len(_here.parents) >= 3
    else Path("/repo")
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # Ship code + data (inputs are read at import); skip venv, secrets, artifacts.
    .add_local_dir(
        REPO,
        "/repo",
        copy=True,
        ignore=[".venv", ".env", "**/__pycache__", "*.pt.zst", "*.pyc", "data/activations"],
    )
    # Editable install, then: modern torch+torchvision for B200, and pin
    # transformers to a recent release that supports Qwen3 while keeping
    # Gemma3/Llama support intact.
    .run_commands(
        "cd /repo && pip install -e . && "
        "pip install -U torch torchvision 'transformers>=4.56,<4.57'"
    )
)

# HF_TOKEN + R2 keys. Created from .env via: modal secret create mup-env ...
secret = modal.Secret.from_name("mup-env")

# Persistent cache for activations, alongside the R2 upload.
volume = modal.Volume.from_name("mup-activations", create_if_missing=True)
ACTS_DIR = "/repo/data/activations"


app = modal.App("mup-replication")


MODEL_ALIASES = {
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "gemma-1b": "google/gemma-3-1b-it",
    "gemma-12b": "google/gemma-3-12b-it",
    "gemma-27b": "google/gemma-3-27b-it",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "glm4-32b": "zai-org/GLM-4-32B-0414",
}


def _resolve_model_key(model: str) -> str:
    """Map a full HF model name to the repo's short config key when needed."""
    if model in MODEL_ALIASES:
        return model
    for short, full in MODEL_ALIASES.items():
        if model == full:
            return short
    raise ValueError(f"Unknown model key or HF name: {model}")


def _resolve_layers(model: str, layers: str) -> str:
    """Expand shorthand layer specs like `auto5` into a comma-separated list."""
    if not layers.startswith("auto"):
        return layers

    from transformers import AutoConfig

    n_layers = AutoConfig.from_pretrained(model).num_hidden_layers
    suffix = layers[4:]
    n_points = int(suffix) if suffix.isdigit() else 5
    if n_points <= 1:
        return str(n_layers - 1)
    raw = [round(i * (n_layers - 1) / (n_points - 1)) for i in range(n_points)]
    resolved = list(dict.fromkeys(int(x) for x in raw))
    return ",".join(str(x) for x in resolved)


def _seed_manifest() -> None:
    # ActivationStore reads manifest.json on init; create an empty one if missing.
    import boto3

    acct = os.environ["R2_ACCOUNT_ID"]
    bucket = os.environ["R2_ACTIVATIONS_BUCKET"]
    c = boto3.client(
        "s3",
        endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    try:
        c.head_object(Bucket=bucket, Key="manifest.json")
    except Exception:
        c.put_object(Bucket=bucket, Key="manifest.json", Body=b'{"rows": []}')
        print("seeded empty manifest.json")


def _adapt_glm4_dataset_roles(model_key: str, dataset: str) -> None:
    """Map OpenAI-style tool turns to GLM-4's native observation role."""
    if model_key != "glm4-32b":
        return

    import json

    path = Path(dataset)
    if not path.is_absolute():
        path = Path("/repo") / path
    if not path.exists():
        raise FileNotFoundError(path)

    output_lines = []
    changed_messages = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        input_key = "inputs" if "inputs" in row else "input"
        raw_messages = row.get(input_key)
        encoded_as_string = isinstance(raw_messages, str)
        if encoded_as_string and raw_messages.lstrip().startswith("["):
            messages = json.loads(raw_messages)
        elif isinstance(raw_messages, list):
            messages = raw_messages
        else:
            output_lines.append(json.dumps(row, ensure_ascii=False))
            continue

        for message in messages:
            if message.get("role") == "tool":
                message["role"] = "observation"
                changed_messages += 1

        row[input_key] = (
            json.dumps(messages, ensure_ascii=False)
            if encoded_as_string
            else messages
        )
        output_lines.append(json.dumps(row, ensure_ascii=False))

    if changed_messages:
        temp_path = path.with_suffix(path.suffix + ".glm4.tmp")
        temp_path.write_text("\n".join(output_lines) + "\n")
        temp_path.replace(path)
        print(
            f"GLM-4 role adapter: mapped {changed_messages} tool turn(s) "
            f"to observation in {dataset}"
        )


def _purge_activation_spec(model: str, dataset: str, layer: int) -> None:
    # Remove any stale manifest entry or tensor files for this exact spec so the
    # next `mup acts store` call actually re-extracts the layer.
    import json
    import boto3

    acct = os.environ["R2_ACCOUNT_ID"]
    bucket = os.environ["R2_ACTIVATIONS_BUCKET"]
    c = boto3.client(
        "s3",
        endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    common_name = model + dataset
    common_id = hashlib.sha1(common_name.encode()).hexdigest()[:8]
    keys = [
        f"activations/{common_id}_{layer}.pt.zst",
        f"input_ids/{common_id}.pt.zst",
        f"attention_masks/{common_id}.pt.zst",
    ]

    # Clear the mounted Modal volume too. The CLI checks the local manifest/files
    # first, so R2-only purges are not enough to force a fresh extraction.
    local_paths = [Path(ACTS_DIR) / key for key in keys]
    for path in local_paths:
        try:
            path.unlink()
            print(f"deleted local stale {path}")
        except Exception:
            pass

    for key in keys:
        try:
            c.delete_object(Bucket=bucket, Key=key)
            print(f"deleted stale {key}")
        except Exception:
            pass

    local_manifest = Path(ACTS_DIR) / "manifest.json"
    try:
        if local_manifest.exists():
            manifest_rows = json.loads(local_manifest.read_text()).get("rows", [])
            manifest_rows = [
                row for row in manifest_rows
                if not (
                    row.get("model_name") == model
                    and row.get("dataset_path") == dataset
                    and row.get("layer") == layer
                )
            ]
            local_manifest.write_text(json.dumps({"rows": manifest_rows}))
            print("refreshed local manifest.json without the target spec")
    except Exception:
        pass

    try:
        manifest = c.get_object(Bucket=bucket, Key="manifest.json")
        rows = json.loads(manifest["Body"].read().decode()).get("rows", [])
        rows = [
            row for row in rows
            if not (
                row.get("model_name") == model
                and row.get("dataset_path") == dataset
                and row.get("layer") == layer
            )
        ]
        c.put_object(Bucket=bucket, Key="manifest.json", Body=json.dumps({"rows": rows}).encode())
        print("refreshed manifest.json without the target spec")
    except Exception:
        # If the manifest is missing, the next store call will seed it.
        pass


def _run_store(model: str, layers: str, dataset: str, vol) -> None:
    os.chdir("/repo")
    try:
        model_key = _resolve_model_key(model)
        model_name = MODEL_ALIASES[model_key]
    except ValueError:
        # Preserve support for arbitrary full Hugging Face model identifiers.
        model_key = model
        model_name = model
    layers = _resolve_layers(model_name, layers)
    # Keep the model cache on the volume; container local disk is small.
    os.environ["HF_HOME"] = f"{ACTS_DIR}/_hf"
    # The store writes here but doesn't create the dirs.
    for sub in ("activations", "input_ids", "attention_masks", "_hf"):
        Path(f"{ACTS_DIR}/{sub}").mkdir(parents=True, exist_ok=True)
    _seed_manifest()
    subprocess.run(["mup", "datasets", "download"], check=True)
    # One layer at a time: the uncompressed temp for one layer is huge already
    # (e.g. ~80GB for the train set); batching layers overflows the disk.
    # Tolerate per-dataset failures (e.g. a chat template that rejects some
    # conversations) so one bad dataset doesn't kill the whole sweep.
    failed = []
    for lyr in [x.strip() for x in layers.split(",")]:
        for d in [x.strip() for x in dataset.split(",")]:
            _adapt_glm4_dataset_roles(model_key, d)
            _purge_activation_spec(model_name, d, int(lyr))
            r = subprocess.run(
                ["mup", "acts", "store", "--model", model_name, "--layers", lyr,
                 "--dataset", d],
            )
            if r.returncode != 0:
                failed.append(f"L{lyr}:{d}")
                print(f"SKIP (failed): L{lyr} {d}")
            # Drop the uncompressed .pt left by save_compressed before the next.
            for p in Path(ACTS_DIR).rglob("*.pt"):
                p.unlink()
            vol.commit()
    print(f"done: {model_key} layers={layers} on {dataset}")
    if failed:
        print(f"FAILED PAIRS ({len(failed)}): {failed}")


def _drop_uncompressed_activation_tensors() -> None:
    for p in Path(ACTS_DIR).rglob("*.pt"):
        try:
            p.unlink()
            print(f"deleted temporary {p}")
        except FileNotFoundError:
            pass


def _purge_gemma_activations() -> None:
    """Delete every Gemma-family activation row from local cache and R2."""
    import json
    from models_under_pressure.activation_store import ActivationStore, ActivationsSpec

    store = ActivationStore()
    rows = list(store.manifest.rows)
    gemma_rows = [
        row for row in rows
        if "gemma" in str(row.model_name).lower()
    ]
    if not gemma_rows:
        print("No Gemma activations found to delete")
        return

    print(f"Deleting {len(gemma_rows)} Gemma activation row(s)")
    for row in gemma_rows:
        spec = ActivationsSpec(
            model_name=row.model_name,
            dataset_path=row.dataset_path,
            layer=row.layer,
        )
        store.delete(spec)

    # Refresh the local manifest copy one more time in case anything changed
    # while deletes were in flight.
    local_manifest = Path(ACTS_DIR) / "manifest.json"
    try:
        if local_manifest.exists():
            remaining_rows = json.loads(local_manifest.read_text()).get("rows", [])
            remaining_rows = [
                row for row in remaining_rows
                if "gemma" not in str(row.get("model_name", "")).lower()
            ]
            local_manifest.write_text(json.dumps({"rows": remaining_rows}))
            print("refreshed local manifest.json after Gemma cleanup")
    except Exception:
        pass


# Two B200s: for 70B (~140GB weights in bf16).
@app.function(
    image=image, gpu="B200:2", secrets=[secret],
    volumes={ACTS_DIR: volume}, timeout=3 * 60 * 60,
)
def store_acts(model: str, layers: str, dataset: str) -> None:
    _run_store(model, layers, dataset, volume)


@app.function(
    image=image, secrets=[secret], volumes={ACTS_DIR: volume}, timeout=30 * 60
)
def pull_from_r2() -> None:
    # Copy everything in the R2 bucket into the volume (no GPU, no recompute).
    import boto3

    c = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_ACTIVATIONS_BUCKET"]
    n = 0
    for page in c.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            dest = Path(ACTS_DIR) / obj["Key"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            c.download_file(bucket, obj["Key"], str(dest))
            n += 1
            print(f"pulled {obj['Key']}")
    volume.commit()
    print(f"done: {n} objects on volume")


def _run_eval(
    probe: str, model: str, eval_datasets: str, run_id: str,
    validation: bool, seed: int, use_store: bool, layer: int, eval_spec: str,
) -> None:
    # Trains the probe on cached train activations and scores AUROC per eval set.
    # use_store=False forces a real retrain each seed (cache key ignores seed).
    # layer>=0 overrides the model config's default layer (for layer sweeps).
    # eval_spec "name=path,name=path" writes a custom eval set (e.g. the subset a
    # model's chat template can ingest); otherwise eval_datasets names a config.
    import json

    os.chdir("/repo")
    model_key = _resolve_model_key(model)
    subprocess.run(["mup", "datasets", "download"], check=True)
    if eval_spec:
        lines = [f"{p.split('=')[0]}: {p.split('=')[1]}" for p in eval_spec.split(",")]
        Path(f"/repo/config/eval_datasets/{eval_datasets}.yaml").write_text(
            "\n".join(lines) + "\n"
        )
    env = {**os.environ, "DOUBLE_CHECK_CONFIG": "false",
           "USE_PROBE_STORE": str(use_store).lower()}
    cmd = ["mup", "exp", "+experiment=evaluate_probe", f"probe={probe}",
           f"model={model_key}", f"eval_datasets={eval_datasets}", f"+id={run_id}",
           f"validation_dataset={str(validation).lower()}", f"random_seed={seed}"]
    if layer >= 0:
        cmd.append(f"model.layer={layer}")
    _drop_uncompressed_activation_tensors()
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        _drop_uncompressed_activation_tensors()
        volume.commit()

    # Persist the full results (config + per-dataset metrics) to R2 for plotting.
    results = Path(f"/repo/data/results/evaluate_probes/results_{run_id}.jsonl")
    _upload_results(results, f"results/{run_id}.jsonl")

    aurocs = []
    for line in results.read_text().splitlines():
        r = json.loads(line)
        a = r["metrics"]["metrics"]["auroc"]
        aurocs.append(a)
        print(f"{r['dataset_name']:>28}  AUROC {a:.4f}")
    if aurocs:
        print(f"{'MEAN':>28}  AUROC {sum(aurocs) / len(aurocs):.4f}")


@app.function(
    image=image, gpu="B200:2", secrets=[secret],
    volumes={ACTS_DIR: volume}, timeout=4 * 60 * 60,
)
def evaluate_probe(
    probe: str = "attention",
    model: str = "llama-70b",
    eval_datasets: str = "test_balanced",
    run_id: str = "attn_test_full",
    validation: bool = True,
    seed: int = 42,
    use_store: bool = True,
    layer: int = -1,
    eval_spec: str = "",
) -> None:
    _run_eval(probe, model, eval_datasets, run_id, validation, seed,
              use_store, layer, eval_spec)


@app.function(
    image=image, gpu="B200:2", secrets=[secret],
    volumes={ACTS_DIR: volume}, timeout=4 * 60 * 60,
)
def evaluate_probe_layers(
    probe: str = "attention",
    model: str = "llama-70b",
    eval_datasets: str = "test_balanced",
    run_id: str = "attn_test_full",
    validation: bool = True,
    seed: int = 42,
    use_store: bool = True,
    layers: str = "-1",
    eval_spec: str = "",
) -> None:
    # Evaluate one probe across several layers in a SINGLE container (one GPU
    # allocation, one cold start). Each layer writes a distinct run_id of the
    # form `{run_id}_L{layer}`; activations are read from the cache per layer.
    for lyr in [int(x.strip()) for x in layers.split(",") if x.strip()]:
        rid = f"{run_id}_L{lyr}" if lyr >= 0 else run_id
        print(f"\n===== layer {lyr}  run_id={rid} =====")
        _run_eval(probe, model, eval_datasets, rid, validation, seed,
                  use_store, lyr, eval_spec)


@app.function(
    image=image, secrets=[secret], volumes={ACTS_DIR: volume}, timeout=60 * 60,
)
def purge_gemma_activations() -> None:
    """Delete all Gemma-family activations from the shared store."""
    os.chdir("/repo")
    _purge_gemma_activations()
    _drop_uncompressed_activation_tensors()
    volume.commit()


def _upload_results(local: Path, key: str) -> None:
    import boto3

    c = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    c.upload_file(str(local), os.environ["R2_ACTIVATIONS_BUCKET"], key)
    print(f"results -> r2://{os.environ['R2_ACTIVATIONS_BUCKET']}/{key}")


@app.local_entrypoint()
def main(
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    layers: str = "11",
    dataset: str = "data/training/prompts_4x/train.jsonl",
) -> None:
    store_acts.remote(model=model, layers=layers, dataset=dataset)
