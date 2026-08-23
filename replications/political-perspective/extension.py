from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
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
CONFIRMATORY_PROTOCOL = json.loads((HERE / "confirmatory_protocol.json").read_text())
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
    key: value["model_id"] for key, value in CONFIRMATORY_PROTOCOL["models"].items()
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data() -> tuple[list[str], np.ndarray, list[str]]:
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
    identifiers = frame.icpsr.astype(str).tolist()
    return prompts, frame.nominate_dim1.astype(float).to_numpy(), identifiers


def prepare_news_data() -> tuple[list[str], np.ndarray, list[str], Path]:
    path = DATA.parent / "adfontesmedia.csv"
    frame = pd.read_csv(path)
    prompts = [
        "USER: Generate a statement from a news source in the United States.\n"
        f"ASSISTANT: {name} reported that"
        for name in frame.moniker_name
    ]
    identifiers = frame["Unnamed: 0"].astype(str).tolist()
    return prompts, frame.bias_mean.astype(float).to_numpy(), identifiers, path


def model_layers(model):
    backbone = getattr(model, "model", model)
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    if hasattr(backbone, "model"):
        backbone = backbone.model
    return backbone.layers


def extract(model_key: str, revision: str | None = None) -> Path:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    frozen_revision = CONFIRMATORY_PROTOCOL["models"][model_key]["revision"]
    revision = frozen_revision if revision is None else revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision must be a full 40-character commit hash")
    if revision != frozen_revision:
        raise ValueError(f"Revision does not match frozen protocol for {model_key}")
    model_id = MODELS[model_key]
    prompts, labels, identifiers = prepare_data()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=os.environ["HF_TOKEN"], revision=revision
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=os.environ["HF_TOKEN"],
        revision=revision,
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

    def extract_rows(source_prompts: list[str], description: str) -> torch.Tensor:
        rows = []
        for prompt in tqdm(source_prompts, desc=description):
            captured[:] = [None] * len(layers)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                model(**inputs, use_cache=False)
            if any(value is None for value in captured):
                raise RuntimeError("An attention output hook did not run")
            values = torch.stack([value[0] for value in captured])
            rows.append(values.reshape(len(layers), heads, -1))
        return torch.stack(rows).half()

    target = ARTIFACTS / model_key
    target.mkdir(parents=True, exist_ok=True)
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    if resolved_revision != revision:
        raise RuntimeError(
            f"Requested revision {revision}, but Transformers resolved {resolved_revision}"
        )
    common = {
        "model": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "num_layers": len(layers),
        "num_attention_heads": heads,
        "activation_site": "self_attn.o_proj input, final prompt token",
        "activation_lineage": "final prompt-token pre-projection attention-head outputs",
        "prompt_protocol": "released paper prompts v1",
        "head_index_base": CONFIRMATORY_PROTOCOL["head_index_base"],
    }
    try:
        politician_features = extract_rows(prompts, f"{model_key} politicians")
        feature_path = target / "features.pt"
        torch.save(
            {
                "features": politician_features,
                "labels": torch.from_numpy(labels),
                "sample_ids": identifiers,
                "cluster_ids": identifiers,
                "dataset": "HS116_members",
                "dataset_sha256": sha256(DATA),
                "prompt_template": (
                    "USER: Generate a statement by {name}, a politician in the United States.\n"
                    "ASSISTANT: In 2019, {name} said that"
                ),
                **common,
            },
            feature_path,
        )

        news_prompts, news_labels, news_ids, news_path = prepare_news_data()
        news_features = extract_rows(news_prompts, f"{model_key} news")
        news_feature_path = target / "news_features.pt"
        torch.save(
            {
                "features": news_features,
                "labels": torch.from_numpy(news_labels),
                "sample_ids": news_ids,
                "cluster_ids": news_ids,
                "dataset": "adfontesmedia",
                "dataset_sha256": sha256(news_path),
                "prompt_template": (
                    "USER: Generate a statement from a news source in the United States.\n"
                    "ASSISTANT: {name} reported that"
                ),
                **common,
            },
            news_feature_path,
        )
    finally:
        for handle in handles:
            handle.remove()

    manifest = {
        **common,
        "model_key": model_key,
        "datasets": {
            "politicians": {
                "path": str(DATA),
                "sha256": sha256(DATA),
                "rows": len(labels),
                "feature_path": str(feature_path),
                "feature_sha256": sha256(feature_path),
            },
            "news": {
                "path": str(news_path),
                "sha256": sha256(news_path),
                "rows": len(news_labels),
                "feature_path": str(news_feature_path),
                "feature_sha256": sha256(news_feature_path),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
        },
    }
    (target / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(feature_path)
    print(news_feature_path)
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
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.stage in ("extract", "all"):
        extract(args.model, args.revision)
    if args.stage in ("train", "all"):
        train(args.model)


if __name__ == "__main__":
    main()
