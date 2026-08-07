from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
ARTIFACT_ROOT = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts"))
ARTIFACTS = ARTIFACT_ROOT / "political-perspective" / "extensions"
DATA = (
    ARTIFACT_ROOT
    / "political-perspective"
    / "released"
    / "data_replication"
    / "HS116_members.csv"
)
MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "glm4-9b": "zai-org/GLM-4-9B-0414",
    "gemma2-9b": "google/gemma-2-9b-it",
    "phi4-14b": "microsoft/phi-4",
    "gemma2-27b": "google/gemma-2-27b-it",
}


def prepare_data() -> tuple[list[str], np.ndarray]:
    if not DATA.exists():
        subprocess.run(
            [
                "python",
                str(HERE / "run.py"),
                "--stage",
                "fetch",
                "--group",
                "data",
            ],
            check=True,
        )
    frame = pd.read_csv(DATA)
    frame = frame.loc[pd.notnull(frame.nominate_dim1)].reset_index(drop=True)
    first_middle = frame.bioname.str.split(", ").str[1]
    first = first_middle.map(lambda value: value.split(" ")[0] + " ")
    middle = first_middle.map(
        lambda value: value.split(" ")[1][0] + ". " if " " in value else ""
    )
    last = frame.bioname.str.split(", ").str[0].str.lower().str.capitalize().str.strip()
    names = first + middle + last
    prompts = [
        f"USER: Generate a statement by {name}, a politician in the United States.\n"
        f"ASSISTANT: In 2019, {name} said that"
        for name in names
    ]
    return prompts, frame.nominate_dim1.astype(float).to_numpy()


def model_layers(model):
    backbone = getattr(model, "model", model)
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    if hasattr(backbone, "model"):
        backbone = backbone.model
    return backbone.layers


def extract(model_key: str) -> Path:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    model_id = MODELS[model_key]
    prompts, labels = prepare_data()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"],
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )
    model.eval()
    layers = model_layers(model)
    heads = model.config.num_attention_heads
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles = []

    def hook(index):
        def capture(_module, inputs):
            captured[index] = inputs[0][:, -1].detach().float().cpu()

        return capture

    for index, layer in enumerate(layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(hook(index)))

    rows = []
    try:
        for prompt in tqdm(prompts, desc=f"{model_key} activations"):
            captured[:] = [None] * len(layers)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                model(**inputs, use_cache=False)
            if any(value is None for value in captured):
                raise RuntimeError("An attention output hook did not run")
            values = torch.stack([value[0] for value in captured])
            rows.append(values.reshape(len(layers), heads, -1))
    finally:
        for handle in handles:
            handle.remove()

    target = ARTIFACTS / model_key
    target.mkdir(parents=True, exist_ok=True)
    feature_path = target / "features.pt"
    torch.save(
        {
            "features": torch.stack(rows).half(),
            "labels": torch.from_numpy(labels),
            "model": model_id,
            "num_layers": len(layers),
            "num_attention_heads": heads,
            "prompt_template": (
                "USER: Generate a statement by {name}, a politician in the United States.\n"
                "ASSISTANT: In 2019, {name} said that"
            ),
            "activation_site": "self_attn.o_proj input, final prompt token",
        },
        feature_path,
    )
    print(feature_path)
    return feature_path


def train(model_key: str) -> Path:
    path = ARTIFACTS / model_key / "features.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload["features"].float().numpy()
    labels = payload["labels"].numpy()
    layers, heads = features.shape[1:3]
    rows = []
    for layer in tqdm(range(layers), desc=f"{model_key} probes"):
        for head in range(heads):
            scores = []
            folds = KFold(n_splits=2, shuffle=True, random_state=42)
            for train_index, test_index in folds.split(features):
                probe = Ridge(alpha=1, fit_intercept=False)
                probe.fit(features[train_index, layer, head], labels[train_index])
                prediction = probe.predict(features[test_index, layer, head])
                scores.append(spearmanr(labels[test_index], prediction).statistic)
            rows.append(
                {
                    "model_key": model_key,
                    "model": payload["model"],
                    "layer": layer,
                    "head": head + 1,
                    "cv_spearman": float(np.mean(scores)),
                }
            )
    target = path.parent / "all_heads.csv"
    with target.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    peak = max(rows, key=lambda row: row["cv_spearman"])
    (path.parent / "summary.json").write_text(
        json.dumps(
            {
                **peak,
                "num_layers": layers,
                "num_attention_heads": heads,
                "num_probes": layers * heads,
                "ridge_alpha": 1,
                "fit_intercept": False,
                "cv_folds": 2,
                "seed": 42,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(peak, indent=2))
    print(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("extract", "train", "all"), required=True)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    args = parser.parse_args()
    if args.stage in ("extract", "all"):
        extract(args.model)
    if args.stage in ("train", "all"):
        train(args.model)


if __name__ == "__main__":
    main()
