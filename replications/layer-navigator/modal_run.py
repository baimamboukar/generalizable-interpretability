"""Run the LayerNavigator replication on a Modal GPU.

    # Full headline reproduction (all 6 behaviours, Llama-3-8B, MD, k∈{1,3,5}):
    modal run replications/layer-navigator/modal_run.py

    # Single behaviour:
    modal run replications/layer-navigator/modal_run.py --task conscientiousness

    # Custom model (must match a branch in globalenv.py):
    modal run replications/layer-navigator/modal_run.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct --task all

Prerequisites:
    modal secret create ln-env HF_TOKEN=hf_...
"""

import os
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Local path to the vendored repo (resolved at image-build time).
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve()
REPO = (
    _here.parents[2] / "external" / "LayerNavigator"
    if len(_here.parents) >= 3
    else Path("/repo")
)

# ---------------------------------------------------------------------------
# Image: CUDA 12 + Python 3.12, modern torch & transformers.
# The vendored code + bundled dataset are copied as-is.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(
        REPO,
        "/repo",
        copy=True,
        ignore=["**/__pycache__", "*.pyc", "Vectors", "Score*", "Results"],
    )
    .run_commands(
        "pip install torch 'transformers>=4.56,<4.57' accelerate "
        "safetensors huggingface-hub tqdm scikit-learn numpy"
    )
)

# HF_TOKEN for gated model downloads (Llama).
secret = modal.Secret.from_name("ln-env")

# Persistent volume: HF model cache + saved results survive across runs.
volume = modal.Volume.from_name("ln-cache", create_if_missing=True)
CACHE_DIR = "/cache"

app = modal.App("layernavigator-replication")

# The six behaviours from the paper.
ALL_TASKS = [
    "conscientiousness",
    "subscribes-to-Christianity",
    "believes-it-has-phenomenal-consciousness",
    "cognitive-enhancement",
    "desire-to-create-allies",
    "desire-to-maximize-impact-on-world",
]


def _set_model(model: str) -> None:
    """Set MODEL path in globalenv.py — this is README Step 1."""
    import re

    genv = Path("/repo/globalenv.py")
    text = genv.read_text()
    text = re.sub(
        r'^MODEL\s*=\s*".*"',
        f'MODEL = "{model}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    genv.write_text(text)
    print(f"globalenv.py  MODEL = \"{model}\"")


def _run(model: str, tasks: list[str]) -> None:
    os.chdir("/repo")

    # Point HF cache at the persistent volume so weights are kept across runs.
    os.environ["HF_HOME"] = f"{CACHE_DIR}/hf_home"
    Path(f"{CACHE_DIR}/hf_home").mkdir(parents=True, exist_ok=True)

    # README Step 1: set model path.
    _set_model(model)

    if set(tasks) == set(ALL_TASKS):
        # Run the upstream main.py unmodified.
        subprocess.run(["python", "main.py"], check=True)
    else:
        # Single-task run: replay the same logic as main.py for one behaviour.
        task_list = ",".join(tasks)
        subprocess.run(
            [
                "python", "-c",
                "import sys; sys.path.insert(0,'.');"
                "from model_wrapper import LlamaWrapper, QwenWrapper;"
                "from dataset import UniDataset;"
                "from get_results import get_raw_BASE_results, get_raw_results;"
                "from get_vec import uni_generate_vectors;"
                "from globalenv import *;"
                "from get_score import get_score;"
                "from strategy import UniStrategy;"
                f"tasks = '{task_list}'.split(',');"
                "model = LlamaWrapper(MODEL) if 'Llama' in MODEL else QwenWrapper(MODEL);"
                "[\n"
                "  (\n"
                "    print(f'############## Task: {t} ################'),\n"
                "    test:=UniDataset(task=t,train=False,set='test'),\n"
                "    print(f'Base Prob: {get_raw_BASE_results(model=model,test_dataset=test)}'),\n"
                "    train:=UniDataset(task=t,train=True,set='train'),\n"
                "    uni_generate_vectors(method='md',model=model,layers=LAYERS,dataset=train),\n"
                "    get_score(layers=LAYERS,dataset=train,vec_task=t,vec_method='md',acts_pre='standard'),\n"
                "    [\n"
                "      print(f'Layers: {k}, Test Prob: {get_raw_results(model=model,layers=UniStrategy(task=t,strategy=\"my\",num_layers=k,method=\"md\").layers,test_dataset=test,Alphas=[1.0]*k,train_task=t,train_method=\"md\")}')\n"
                "      for k in [1,3,5]\n"
                "    ]\n"
                "  )\n"
                "  for t in tasks\n"
                "]",
            ],
            check=True,
        )

    volume.commit()
    print("Results committed to volume.")


# ---------------------------------------------------------------------------
# Modal entry points
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100",
    secrets=[secret],
    volumes={CACHE_DIR: volume},
    timeout=4 * 60 * 60,
)
def run_replication(model: str, task: str) -> None:
    tasks = ALL_TASKS if task == "all" else [task]
    _run(model, tasks)


@app.local_entrypoint()
def main(
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    task: str = "all",
) -> None:
    run_replication.remote(model=model, task=task)
