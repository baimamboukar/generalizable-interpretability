"""Run `mup acts store` on a Modal GPU. Activations upload to the R2 bucket in .env.

    modal run replications/models-under-pressure/modal_acts.py
    modal run replications/models-under-pressure/modal_acts.py \
        --model meta-llama/Llama-3.2-3B-Instruct --layers 11,15 \
        --dataset data/evals/dev/mt_balanced.jsonl
"""

import os
import subprocess
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
    # transformers to 4.50 (what their code's Gemma3 layer paths target; the
    # lockfile's 5.12 moved them and breaks multimodal Gemma).
    .run_commands(
        "cd /repo && pip install -e . && "
        "pip install -U torch torchvision 'transformers==4.50.*'"
    )
)

# HF_TOKEN + R2 keys. Created from .env via: modal secret create mup-env ...
secret = modal.Secret.from_name("mup-env")

# Persistent cache for activations, alongside the R2 upload.
volume = modal.Volume.from_name("mup-activations", create_if_missing=True)
ACTS_DIR = "/repo/data/activations"

app = modal.App("mup-replication")


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


def _run_store(model: str, layers: str, dataset: str) -> None:
    os.chdir("/repo")
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
            r = subprocess.run(
                ["mup", "acts", "store", "--model", model, "--layers", lyr,
                 "--dataset", d],
            )
            if r.returncode != 0:
                failed.append(f"L{lyr}:{d}")
                print(f"SKIP (failed): L{lyr} {d}")
            # Drop the uncompressed .pt left by save_compressed before the next.
            for p in Path(ACTS_DIR).rglob("*.pt"):
                p.unlink()
            volume.commit()
    print(f"done: {model} layers={layers} on {dataset}")
    if failed:
        print(f"FAILED PAIRS ({len(failed)}): {failed}")


# Single GPU: fits up to ~12B in bf16.
@app.function(
    image=image, gpu="B200", secrets=[secret],
    volumes={ACTS_DIR: volume}, timeout=3 * 60 * 60,
)
def store_acts(model: str, layers: str, dataset: str) -> None:
    _run_store(model, layers, dataset)


# Two B200s: for 70B (~140GB weights in bf16).
@app.function(
    image=image, gpu="B200:2", secrets=[secret],
    volumes={ACTS_DIR: volume}, timeout=5 * 60 * 60,
)
def store_acts_big(model: str, layers: str, dataset: str) -> None:
    _run_store(model, layers, dataset)


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


@app.function(
    image=image, gpu="B200", secrets=[secret],
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
) -> None:
    # Trains the probe on cached train activations and scores AUROC per eval set.
    # use_store=False forces a real retrain each seed (cache key ignores seed).
    # layer>=0 overrides the model config's default layer (for layer sweeps).
    import json

    os.chdir("/repo")
    subprocess.run(["mup", "datasets", "download"], check=True)
    env = {**os.environ, "DOUBLE_CHECK_CONFIG": "false",
           "USE_PROBE_STORE": str(use_store).lower()}
    cmd = ["mup", "exp", "+experiment=evaluate_probe", f"probe={probe}",
           f"model={model}", f"eval_datasets={eval_datasets}", f"+id={run_id}",
           f"validation_dataset={str(validation).lower()}", f"random_seed={seed}"]
    if layer >= 0:
        cmd.append(f"model.layer={layer}")
    subprocess.run(cmd, check=True, env=env)

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
    big: bool = False,
) -> None:
    # --big routes to the 4-GPU function for large models (e.g. 70B).
    fn = store_acts_big if big else store_acts
    fn.remote(model=model, layers=layers, dataset=dataset)
