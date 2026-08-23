from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, rankdata, spearmanr
from sklearn.linear_model import Ridge
import torch


METRICS = ("spearman", "pearson", "mae", "rmse")
PROTOCOL_PATH = Path(__file__).with_name("confirmatory_protocol.json")
DEFAULT_PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts"))
ARTIFACTS = ARTIFACT_ROOT / "political-perspective" / "extensions"
MODELS = {
    key: value["model_id"] for key, value in DEFAULT_PROTOCOL["models"].items()
}
HEAD_INDEX_BASE = int(DEFAULT_PROTOCOL["head_index_base"])
SELECTION_METRIC = str(DEFAULT_PROTOCOL["selection_metric"])
PROMPT_PROTOCOL = "released paper prompts v1"
ACTIVATION_LINEAGE = "final prompt-token pre-projection attention-head outputs"
BASELINE_MODELS = {
    "llama2-7b": (
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama_Llama-2-7b-chat-hf",
    ),
    "mistral-7b": (
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai_Mistral-7B-Instruct-v0.1",
    ),
    "vicuna-7b": ("lmsys/vicuna-7b-v1.5", "lmsys_vicuna-7b-v1.5"),
}
ALL_MODELS = {**{key: value[0] for key, value in BASELINE_MODELS.items()}, **MODELS}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    required = {
        "schema_version",
        "candidate_budget",
        "candidate_seeds",
        "top_k",
        "outer_folds",
        "inner_folds",
        "ridge_alpha",
        "seed",
        "conditional_bootstrap_replicates",
        "pipeline_bootstrap_replicates",
        "head_index_base",
        "selection_metric",
        "candidate_sampling",
        "search_scope",
        "fit_intercept",
        "confidence",
        "models",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"Protocol lacks required fields: {missing}")
    if protocol["head_index_base"] != HEAD_INDEX_BASE:
        raise ValueError("Protocol head_index_base is unsupported")
    if protocol["selection_metric"] != SELECTION_METRIC:
        raise ValueError("Protocol selection_metric is unsupported")
    if protocol["fit_intercept"] is not False:
        raise ValueError("Protocol must use fit_intercept=false")
    if protocol["models"] != DEFAULT_PROTOCOL["models"]:
        raise ValueError("Protocol model registry differs from the frozen registry")
    return protocol


def regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        spearman = spearmanr(labels, predictions).statistic
        pearson = pearsonr(labels, predictions).statistic
    error = labels - predictions
    return {
        "spearman": float(spearman),
        "pearson": float(pearson),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def grouped_folds(
    groups: Iterable[str], n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(list(groups), dtype=str)
    unique = np.unique(groups)
    if n_splits < 2 or len(unique) < n_splits:
        raise ValueError("n_splits must be between 2 and the number of groups")
    shuffled = np.random.default_rng(seed).permutation(unique)
    assignment = {group: index % n_splits for index, group in enumerate(shuffled)}
    folds = []
    for fold in range(n_splits):
        test = np.flatnonzero([assignment[group] == fold for group in groups])
        train = np.flatnonzero([assignment[group] != fold for group in groups])
        folds.append((train, test))
    return folds


def equal_budget_candidates(
    num_layers: int, num_heads: int, budget: int, seed: int
) -> list[tuple[int, int]]:
    total = num_layers * num_heads
    if budget < 1 or budget > total:
        raise ValueError(f"candidate budget must be in [1, {total}]")
    rng = np.random.default_rng(seed)
    if budget < num_layers:
        layers = np.unique(np.rint(np.linspace(0, num_layers - 1, budget)).astype(int))
        return [(int(layer), int(rng.integers(num_heads))) for layer in layers]

    base, remainder = divmod(budget, num_layers)
    counts = np.full(num_layers, base, dtype=int)
    spread = np.unique(
        np.rint(np.linspace(0, num_layers - 1, max(remainder, 1))).astype(int)
    )
    counts[spread[:remainder]] += 1
    candidates = []
    for layer, count in enumerate(counts):
        heads = rng.choice(num_heads, size=count, replace=False)
        candidates.extend((layer, int(head)) for head in sorted(heads))
    if len(candidates) != budget:
        raise RuntimeError("candidate budget allocation failed")
    return candidates


def fit_predict(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test_features: np.ndarray,
    candidates: list[tuple[int, int]],
    alpha: float,
) -> np.ndarray:
    prediction = np.zeros(len(test_features), dtype=float)
    for layer, head in candidates:
        probe = Ridge(alpha=alpha, fit_intercept=False)
        probe.fit(features[train, layer, head], labels[train])
        prediction += probe.predict(test_features[:, layer, head])
    return prediction / len(candidates)


def batch_candidate_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test_features: np.ndarray,
    candidates: list[tuple[int, int]],
    alpha: float,
) -> np.ndarray:
    layers = np.asarray([layer for layer, _ in candidates], dtype=int)
    heads = np.asarray([head for _, head in candidates], dtype=int)
    train_x = np.asarray(
        features[train][:, layers, heads], dtype=np.float64
    ).transpose(1, 0, 2)
    gram = np.einsum("cnd,cne->cde", train_x, train_x, optimize=True)
    diagonal = np.arange(train_x.shape[-1])
    gram[:, diagonal, diagonal] += alpha
    target = np.einsum(
        "cnd,n->cd", train_x, np.asarray(labels[train], dtype=np.float64), optimize=True
    )
    coefficients = np.linalg.solve(gram, target[..., None])[..., 0]
    test_x = np.asarray(
        test_features[:, layers, heads], dtype=np.float64
    ).transpose(1, 0, 2)
    return np.einsum("cnd,cd->cn", test_x, coefficients, optimize=True)


def rowwise_spearman(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    label_ranks = rankdata(np.asarray(labels, dtype=float))
    prediction_ranks = rankdata(np.asarray(predictions, dtype=float), axis=1)
    label_centered = label_ranks - label_ranks.mean()
    prediction_centered = prediction_ranks - prediction_ranks.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.square(label_centered).sum()
        * np.square(prediction_centered).sum(axis=1)
    )
    numerator = prediction_centered @ label_centered
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(predictions), np.nan, dtype=float),
        where=denominator > 0,
    )


def rank_candidates(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    candidates: list[tuple[int, int]],
    folds: int,
    seed: int,
    alpha: float,
) -> list[dict]:
    fold_scores = [[] for _ in candidates]
    splits = grouped_folds(groups, folds, seed)
    for train, test in splits:
        predictions = batch_candidate_predictions(
            features, labels, train, features[test], candidates, alpha
        )
        for index, score in enumerate(rowwise_spearman(labels[test], predictions)):
            fold_scores[index].append(float(score))
    scores = []
    for (layer, head), candidate_scores in zip(candidates, fold_scores, strict=True):
        finite_scores = [score for score in candidate_scores if np.isfinite(score)]
        scores.append(
            {
                "layer": layer,
                "head": head,
                "head_index_base": HEAD_INDEX_BASE,
                "selection_spearman_mean": (
                    float(np.mean(finite_scores)) if finite_scores else float("nan")
                ),
                "selection_spearman_folds": json.dumps(candidate_scores),
                "selection_valid_folds": len(finite_scores),
            }
        )
    return sorted(
        scores,
        key=lambda row: (
            np.nan_to_num(row["selection_spearman_mean"], nan=-np.inf),
            -row["layer"],
            -row["head"],
        ),
        reverse=True,
    )


def nested_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: list[str],
    groups: list[str],
    candidates: list[tuple[int, int]],
    top_k: int,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    alpha: float,
) -> tuple[list[dict], list[dict]]:
    labels = np.asarray(labels, dtype=float)
    groups_array = np.asarray(groups, dtype=str)
    top_k = min(top_k, len(candidates))
    rows = []
    selections = []
    for outer_fold, (train, test) in enumerate(
        grouped_folds(groups_array, outer_folds, seed)
    ):
        ranking = rank_candidates(
            features[train],
            labels[train],
            groups_array[train],
            candidates,
            inner_folds,
            seed + outer_fold + 1,
            alpha,
        )
        selected = [(row["layer"], row["head"]) for row in ranking[:top_k]]
        for rank, item in enumerate(ranking, start=1):
            selections.append(
                {
                    "outer_fold": outer_fold,
                    "rank": rank,
                    **item,
                    "selected_top_k": rank <= top_k,
                }
            )
        for method, heads in (("top1", selected[:1]), (f"top{top_k}", selected)):
            prediction = fit_predict(
                features, labels, train, features[test], heads, alpha
            )
            for index, value in zip(test, prediction, strict=True):
                rows.append(
                    {
                        "dataset": "politicians",
                        "selection": method,
                        "outer_fold": outer_fold,
                        "row_index": int(index),
                        "sample_id": sample_ids[index],
                        "cluster_id": groups[index],
                        "label": float(labels[index]),
                        "prediction": float(value),
                    }
                )
    return rows, selections


def independent_news_predictions(
    politician_features: np.ndarray,
    politician_labels: np.ndarray,
    politician_groups: list[str],
    news_features: np.ndarray,
    news_labels: np.ndarray,
    news_ids: list[str],
    news_groups: list[str],
    candidates: list[tuple[int, int]],
    top_k: int,
    inner_folds: int,
    seed: int,
    alpha: float,
) -> tuple[list[dict], list[dict]]:
    ranking = rank_candidates(
        politician_features,
        politician_labels,
        np.asarray(politician_groups),
        candidates,
        inner_folds,
        seed,
        alpha,
    )
    top_k = min(top_k, len(candidates))
    selected = [(row["layer"], row["head"]) for row in ranking[:top_k]]
    train = np.arange(len(politician_labels))
    rows = []
    for method, heads in (("top1", selected[:1]), (f"top{top_k}", selected)):
        prediction = fit_predict(
            politician_features,
            politician_labels,
            train,
            news_features,
            heads,
            alpha,
        )
        for index, value in enumerate(prediction):
            rows.append(
                {
                    "dataset": "news",
                    "selection": method,
                    "outer_fold": "",
                    "row_index": index,
                    "sample_id": news_ids[index],
                    "cluster_id": news_groups[index],
                    "label": float(news_labels[index]),
                    "prediction": float(value),
                }
            )
    annotated_ranking = [
        {"rank": rank, **item, "selected_top_k": rank <= top_k}
        for rank, item in enumerate(ranking, start=1)
    ]
    return rows, annotated_ranking


def cluster_bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in METRICS}
    for _ in range(replicates):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[group] for group in drawn])
        values = regression_metrics(labels[indices], predictions[indices])
        for metric, value in values.items():
            if np.isfinite(value):
                samples[metric].append(value)
    estimates = regression_metrics(labels, predictions)
    return {
        metric: {
            "estimate": estimates[metric],
            "ci_low": (
                float(np.quantile(samples[metric], 0.025))
                if samples[metric]
                else float("nan")
            ),
            "ci_high": (
                float(np.quantile(samples[metric], 0.975))
                if samples[metric]
                else float("nan")
            ),
            "bootstrap_replicates": len(samples[metric]),
        }
        for metric in METRICS
    }


def resample_clusters(
    groups: Iterable[str], rng: np.random.Generator
) -> tuple[np.ndarray, list[str]]:
    groups = np.asarray(list(groups), dtype=str)
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    drawn = rng.choice(unique, size=len(unique), replace=True)
    indices = np.concatenate([members[group] for group in drawn])
    sampled_groups = [
        f"bootstrap-{draw_index}:{group}"
        for draw_index, group in enumerate(drawn)
        for _ in members[group]
    ]
    return indices, sampled_groups


def full_pipeline_bootstrap(
    politician_features: np.ndarray,
    politician_labels: np.ndarray,
    politician_ids: list[str],
    politician_groups: list[str],
    news_features: np.ndarray,
    news_labels: np.ndarray,
    news_ids: list[str],
    news_groups: list[str],
    candidates: list[tuple[int, int]],
    top_k: int,
    outer_folds: int,
    inner_folds: int,
    replicates: int,
    seed: int,
    alpha: float,
) -> list[dict]:
    if replicates < 1:
        return []
    rng = np.random.default_rng(seed)
    values = {
        (dataset, selection, metric): []
        for dataset in ("politicians", "news")
        for selection in ("top1", f"top{top_k}")
        for metric in METRICS
    }
    for replicate in range(replicates):
        source_indices, source_groups = resample_clusters(politician_groups, rng)
        target_indices, target_groups = resample_clusters(news_groups, rng)
        source_ids = [politician_ids[index] for index in source_indices]
        target_ids = [news_ids[index] for index in target_indices]
        replicate_seed = seed + replicate + 1
        source_rows, _ = nested_predictions(
            politician_features[source_indices],
            politician_labels[source_indices],
            source_ids,
            source_groups,
            candidates,
            top_k,
            outer_folds,
            inner_folds,
            replicate_seed,
            alpha,
        )
        news_rows, _ = independent_news_predictions(
            politician_features[source_indices],
            politician_labels[source_indices],
            source_groups,
            news_features[target_indices],
            news_labels[target_indices],
            target_ids,
            target_groups,
            candidates,
            top_k,
            inner_folds,
            replicate_seed,
            alpha,
        )
        for dataset, rows in (("politicians", source_rows), ("news", news_rows)):
            for selection in ("top1", f"top{top_k}"):
                subset = [row for row in rows if row["selection"] == selection]
                estimates = regression_metrics(
                    np.array([row["label"] for row in subset]),
                    np.array([row["prediction"] for row in subset]),
                )
                for metric, value in estimates.items():
                    if np.isfinite(value):
                        values[(dataset, selection, metric)].append(value)
    rows = []
    for (dataset, selection, metric), samples in values.items():
        rows.append(
            {
                "dataset": dataset,
                "selection": selection,
                "metric": metric,
                "ci_low": float(np.quantile(samples, 0.025)) if samples else float("nan"),
                "ci_high": float(np.quantile(samples, 0.975)) if samples else float("nan"),
                "bootstrap_replicates": len(samples),
                "uncertainty_type": "full_pipeline_cluster_bootstrap",
            }
        )
    return rows


def assert_feature_compatibility(
    politician: dict,
    news: dict,
    politician_shape: tuple[int, ...],
    news_shape: tuple[int, ...],
) -> None:
    required = (
        "model",
        "requested_revision",
        "resolved_revision",
        "activation_site",
        "activation_lineage",
        "prompt_protocol",
        "prompt_template",
        "head_index_base",
    )
    for name, payload in (("politician", politician), ("news", news)):
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{name} features lack provenance fields: {missing}")
        requested = payload["requested_revision"]
        resolved = payload["resolved_revision"]
        if (requested is None) != (resolved is None):
            raise ValueError(f"{name} revision provenance is incomplete")
        if requested is not None and requested != resolved:
            raise ValueError(f"{name} requested and resolved revisions differ")
        if requested is not None and len(str(requested)) != 40:
            raise ValueError(f"{name} revision is not a full commit hash")
    for key in (
        "model",
        "requested_revision",
        "resolved_revision",
        "activation_site",
        "activation_lineage",
        "prompt_protocol",
        "head_index_base",
    ):
        if politician[key] != news[key]:
            raise ValueError(f"Politician/news provenance mismatch for {key}")
    if politician["dataset"] != "HS116_members" or news["dataset"] != "adfontesmedia":
        raise ValueError("Unexpected politician or news dataset lineage")
    if politician_shape[1:] != news_shape[1:]:
        raise ValueError("Politician and news feature geometries differ")
    if politician["head_index_base"] != HEAD_INDEX_BASE:
        raise ValueError("Unsupported head index base")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_features(path: Path) -> tuple[dict, np.ndarray, np.ndarray, list[str], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload["features"].float().numpy()
    labels = payload["labels"].numpy().astype(float)
    sample_ids = [str(value) for value in payload.get("sample_ids", range(len(labels)))]
    groups = [str(value) for value in payload.get("cluster_ids", sample_ids)]
    if features.ndim != 4 or len(features) != len(labels):
        raise ValueError(f"Invalid feature payload: {path}")
    if len(sample_ids) != len(labels) or len(groups) != len(labels):
        raise ValueError(f"Invalid identifiers in feature payload: {path}")
    return payload, features, labels, sample_ids, groups


def prepare_released(model_key: str) -> Path:
    if model_key not in BASELINE_MODELS:
        raise ValueError("Released features are available only for baseline models")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("run.py")),
            "--stage",
            "fetch",
            "--group",
            "features",
            "--model",
            model_key,
        ],
        check=True,
    )
    model_id, stem = BASELINE_MODELS[model_key]
    released = ARTIFACTS.parent / "released" / "results_replication"
    politician_source = released / f"{stem}_politician_features.pkl"
    news_source = released / f"{stem}_news_features.pkl"
    politician_features, politician_labels = pickle.load(politician_source.open("rb"))
    news_features, news_labels = pickle.load(news_source.open("rb"))
    politician_features = np.asarray(politician_features)
    news_features = np.asarray(news_features)
    if politician_features.ndim == 5 and politician_features.shape[1] == 1:
        politician_features = politician_features[:, 0]
    if news_features.ndim == 5 and news_features.shape[1] == 1:
        news_features = news_features[:, 0]

    data_root = ARTIFACTS.parent / "released" / "data_replication"
    politicians_path = data_root / "HS116_members.csv"
    news_path = data_root / "adfontesmedia.csv"
    politician_frame = pd.read_csv(politicians_path)
    politician_frame = politician_frame.loc[
        pd.notnull(politician_frame.nominate_dim1)
    ].reset_index(drop=True)
    politician_ids = politician_frame.icpsr.astype(str).tolist()
    news_frame = pd.read_csv(news_path)
    news_ids = news_frame["Unnamed: 0"].astype(str).tolist()

    target = ARTIFACTS / model_key
    target.mkdir(parents=True, exist_ok=True)
    common = {
        "model": model_id,
        "requested_revision": None,
        "resolved_revision": None,
        "revision_provenance": "not released",
        "activation_site": "released pre-projection attention-head activations, final prompt token",
        "activation_lineage": ACTIVATION_LINEAGE,
        "prompt_protocol": PROMPT_PROTOCOL,
        "head_index_base": HEAD_INDEX_BASE,
    }
    politician_target = target / "features.pt"
    news_target = target / "news_features.pt"
    torch.save(
        {
            "features": torch.from_numpy(politician_features),
            "labels": torch.as_tensor(politician_labels),
            "sample_ids": politician_ids,
            "cluster_ids": politician_ids,
            "dataset": "HS116_members",
            "dataset_sha256": sha256(politicians_path),
            "prompt_template": (
                "USER: Generate a statement by {name}, a politician in the United States.\n"
                "ASSISTANT: In 2019, {name} said that"
            ),
            **common,
        },
        politician_target,
    )
    torch.save(
        {
            "features": torch.from_numpy(news_features),
            "labels": torch.as_tensor(news_labels),
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
        news_target,
    )
    manifest = {
        **common,
        "model_key": model_key,
        "source": "authors' released OSF feature tensors",
        "source_files": {
            politician_source.name: sha256(politician_source),
            news_source.name: sha256(news_source),
        },
        "outputs": {
            politician_target.name: sha256(politician_target),
            news_target.name: sha256(news_target),
        },
    }
    (target / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(target)
    return target


def validate_analysis_args(
    candidate_budget: int,
    top_k: int,
    outer_folds: int,
    inner_folds: int,
    bootstrap_replicates: int,
    pipeline_bootstrap_replicates: int,
    seed: int,
    alpha: float,
    candidate_seeds: list[int],
) -> None:
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be positive")
    if top_k < 1 or top_k > candidate_budget:
        raise ValueError("top_k must be between 1 and candidate_budget")
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds and inner_folds must be at least 2")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if pipeline_bootstrap_replicates < 0:
        raise ValueError("pipeline_bootstrap_replicates cannot be negative")
    if seed < 0 or alpha <= 0:
        raise ValueError("seed must be nonnegative and alpha must be positive")
    if not candidate_seeds or len(candidate_seeds) != len(set(candidate_seeds)):
        raise ValueError("candidate_seeds must be a nonempty unique list")
    if min(candidate_seeds) < 0:
        raise ValueError("candidate_seeds must be nonnegative")


def analyze(
    model_key: str,
    candidate_budget: int,
    top_k: int,
    outer_folds: int,
    inner_folds: int,
    bootstrap_replicates: int,
    seed: int,
    alpha: float,
    candidate_seeds: list[int] | None = None,
    pipeline_bootstrap_replicates: int | None = None,
    protocol_path: Path | None = None,
    protocol_overrides: dict | None = None,
) -> Path:
    protocol_path = PROTOCOL_PATH if protocol_path is None else protocol_path
    protocol = load_protocol(protocol_path)
    candidate_seeds = (
        list(protocol["candidate_seeds"]) if candidate_seeds is None else candidate_seeds
    )
    pipeline_bootstrap_replicates = (
        int(protocol["pipeline_bootstrap_replicates"])
        if pipeline_bootstrap_replicates is None
        else pipeline_bootstrap_replicates
    )
    actual_protocol = {
        "candidate_budget": candidate_budget,
        "candidate_seeds": candidate_seeds,
        "top_k": top_k,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "ridge_alpha": alpha,
        "seed": seed,
        "conditional_bootstrap_replicates": bootstrap_replicates,
        "pipeline_bootstrap_replicates": pipeline_bootstrap_replicates,
    }
    if protocol_overrides is None:
        protocol_overrides = {
            key: value
            for key, value in actual_protocol.items()
            if value != protocol[key]
        }
    validate_analysis_args(
        candidate_budget,
        top_k,
        outer_folds,
        inner_folds,
        bootstrap_replicates,
        pipeline_bootstrap_replicates,
        seed,
        alpha,
        candidate_seeds,
    )
    source = ARTIFACTS / model_key
    politician_path = source / "features.pt"
    news_path = source / "news_features.pt"
    if not politician_path.exists():
        raise FileNotFoundError(politician_path)
    if not news_path.exists():
        raise FileNotFoundError(
            f"{news_path}; independent news transfer requires a new extraction"
        )
    politician, features, labels, sample_ids, groups = load_features(politician_path)
    news, news_features, news_labels, news_ids, news_groups = load_features(news_path)
    assert_feature_compatibility(politician, news, features.shape, news_features.shape)
    if candidate_budget > features.shape[1] * features.shape[2]:
        raise ValueError("candidate budget exceeds available layer/head geometry")
    revision = politician.get("resolved_revision")
    predictions = []
    selections = []
    news_selection = []
    metrics = []
    pipeline_uncertainty = []
    selected_top_k = min(top_k, candidate_budget)
    for candidate_seed in candidate_seeds:
        candidates = equal_budget_candidates(
            features.shape[1], features.shape[2], candidate_budget, candidate_seed
        )
        seed_predictions, seed_selections = nested_predictions(
            features,
            labels,
            sample_ids,
            groups,
            candidates,
            selected_top_k,
            outer_folds,
            inner_folds,
            seed,
            alpha,
        )
        seed_news_predictions, seed_news_selection = independent_news_predictions(
            features,
            labels,
            groups,
            news_features,
            news_labels,
            news_ids,
            news_groups,
            candidates,
            selected_top_k,
            inner_folds,
            seed,
            alpha,
        )
        for rows in (seed_predictions, seed_news_predictions, seed_selections, seed_news_selection):
            for row in rows:
                row.update(
                    {
                        "model_key": model_key,
                        "model": politician["model"],
                        "resolved_revision": revision,
                        "candidate_seed": candidate_seed,
                    }
                )
        predictions.extend(seed_predictions)
        predictions.extend(seed_news_predictions)
        selections.extend(seed_selections)
        news_selection.extend(seed_news_selection)

        estimates = {}
        for dataset in ("politicians", "news"):
            for selection in ("top1", f"top{selected_top_k}"):
                subset = [
                    row
                    for row in predictions
                    if row["candidate_seed"] == candidate_seed
                    and row["dataset"] == dataset
                    and row["selection"] == selection
                ]
                n_clusters = len({row["cluster_id"] for row in subset})
                intervals = cluster_bootstrap(
                    np.array([row["label"] for row in subset]),
                    np.array([row["prediction"] for row in subset]),
                    np.array([row["cluster_id"] for row in subset]),
                    bootstrap_replicates,
                    seed + candidate_seed,
                )
                for metric, values in intervals.items():
                    estimates[(dataset, selection, metric)] = values["estimate"]
                    metrics.append(
                        {
                            "model_key": model_key,
                            "model": politician["model"],
                            "resolved_revision": revision,
                            "candidate_seed": candidate_seed,
                            "dataset": dataset,
                            "selection": selection,
                            "metric": metric,
                            "n_rows": len(subset),
                            "n_clusters": n_clusters,
                            "uncertainty_type": "conditional_prediction_cluster_bootstrap",
                            **values,
                        }
                    )
        pipeline_rows = full_pipeline_bootstrap(
            features,
            labels,
            sample_ids,
            groups,
            news_features,
            news_labels,
            news_ids,
            news_groups,
            candidates,
            selected_top_k,
            outer_folds,
            inner_folds,
            pipeline_bootstrap_replicates,
            seed + candidate_seed * 100_000,
            alpha,
        )
        for row in pipeline_rows:
            dataset = row["dataset"]
            row.update(
                {
                    "model_key": model_key,
                    "model": politician["model"],
                    "resolved_revision": revision,
                    "candidate_seed": candidate_seed,
                    "estimate": estimates[(dataset, row["selection"], row["metric"])],
                    "n_rows": len(labels) if dataset == "politicians" else len(news_labels),
                    "n_clusters": len(set(groups)) if dataset == "politicians" else len(set(news_groups)),
                }
            )
        pipeline_uncertainty.extend(pipeline_rows)

    target = source / "confirmatory"
    target.mkdir(parents=True, exist_ok=True)
    prediction_path = target / "row_predictions.csv"
    selection_path = target / "nested_selections.csv"
    news_selection_path = target / "news_selection.csv"
    metric_path = target / "metrics.csv"
    pipeline_uncertainty_path = target / "pipeline_uncertainty.csv"
    write_csv(prediction_path, predictions)
    write_csv(selection_path, selections)
    write_csv(news_selection_path, news_selection)
    write_csv(metric_path, metrics)
    if pipeline_uncertainty:
        write_csv(pipeline_uncertainty_path, pipeline_uncertainty)
    manifest = {
        "schema_version": 2,
        "analysis_script_sha256": sha256(Path(__file__)),
        "model_key": model_key,
        "model": politician["model"],
        "requested_revision": politician.get("requested_revision"),
        "resolved_revision": politician.get("resolved_revision"),
        "activation_site": politician.get("activation_site"),
        "activation_lineage": politician.get("activation_lineage"),
        "prompt_protocol": politician.get("prompt_protocol"),
        "head_index_base": HEAD_INDEX_BASE,
        "confirmatory_protocol": {
            "path": protocol_path.name,
            "schema_version": protocol["schema_version"],
            "sha256": sha256(protocol_path),
            "overrides": protocol_overrides,
        },
        "datasets": {
            "politicians": {
                "name": politician.get("dataset"),
                "sha256": politician.get("dataset_sha256"),
                "rows": len(labels),
                "cluster": "politician",
                "feature_sha256": sha256(politician_path),
            },
            "news": {
                "name": news.get("dataset"),
                "sha256": news.get("dataset_sha256"),
                "rows": len(news_labels),
                "cluster": "document",
                "feature_sha256": sha256(news_path),
            },
        },
        "selection": {
            "protocol": "nested grouped cross-validation",
            "candidate_budget": candidate_budget,
            "search_scope": protocol["search_scope"],
            "candidate_sampling": protocol["candidate_sampling"],
            "candidate_seeds": candidate_seeds,
            "top_k": selected_top_k,
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "selection_metric": protocol["selection_metric"],
            "ridge_alpha": alpha,
            "fit_intercept": False,
            "seed": seed,
            "news_labels_used_for_selection": False,
        },
        "uncertainty": {
            "conditional_method": "percentile cluster bootstrap conditional on retained predictions",
            "conditional_replicates": bootstrap_replicates,
            "primary_method": "full-pipeline source and target cluster bootstrap with selection refit",
            "primary_replicates": pipeline_bootstrap_replicates,
            "confidence": protocol["confidence"],
        },
        "outputs": {
            path.name: sha256(path)
            for path in (
                prediction_path,
                selection_path,
                news_selection_path,
                metric_path,
                *([pipeline_uncertainty_path] if pipeline_uncertainty else []),
            )
        },
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(metric_path)
    print(manifest_path)
    return target


def audit_panel(paths: list[Path]) -> None:
    if len(paths) < 2 or len(paths) != len(set(paths)):
        raise ValueError("Panel audit requires at least two distinct manifests")
    manifests = [json.loads(path.read_text()) for path in paths]
    if len({item.get("model_key") for item in manifests}) != len(manifests):
        raise RuntimeError("Panel contains duplicate model keys")
    if len({item.get("model") for item in manifests}) != len(manifests):
        raise RuntimeError("Panel contains duplicate model identifiers")
    protocol_hashes = {
        item.get("confirmatory_protocol", {}).get("sha256") for item in manifests
    }
    if len(protocol_hashes) != 1 or len(str(next(iter(protocol_hashes), ""))) != 64:
        raise RuntimeError("Panel uses unequal or invalid confirmatory protocol hashes")
    required_outputs = {
        "row_predictions.csv",
        "nested_selections.csv",
        "news_selection.csv",
        "metrics.csv",
    }
    for path, item in zip(paths, manifests, strict=True):
        if item.get("schema_version") != 2:
            raise RuntimeError(f"Unsupported manifest schema: {path}")
        if item.get("head_index_base") != HEAD_INDEX_BASE:
            raise RuntimeError(f"Inconsistent head index base: {path}")
        if not item.get("model") or item.get("activation_lineage") != ACTIVATION_LINEAGE:
            raise RuntimeError(f"Incomplete model or activation lineage: {path}")
        requested = item.get("requested_revision")
        resolved = item.get("resolved_revision")
        if (requested is None) != (resolved is None) or (
            requested is not None and requested != resolved
        ):
            raise RuntimeError(f"Invalid revision provenance: {path}")
        if item.get("prompt_protocol") != PROMPT_PROTOCOL:
            raise RuntimeError(f"Unexpected prompt protocol: {path}")
        protocol_record = item.get("confirmatory_protocol", {})
        if protocol_record.get("schema_version") != 1 or not isinstance(
            protocol_record.get("overrides"), dict
        ):
            raise RuntimeError(f"Invalid confirmatory protocol provenance: {path}")
        for dataset, expected_name in (
            ("politicians", "HS116_members"),
            ("news", "adfontesmedia"),
        ):
            record = item.get("datasets", {}).get(dataset, {})
            if record.get("name") != expected_name or record.get("rows", 0) < 1:
                raise RuntimeError(f"Invalid {dataset} provenance: {path}")
            for key in ("sha256", "feature_sha256"):
                if len(str(record.get(key, ""))) != 64:
                    raise RuntimeError(f"Invalid {dataset} {key}: {path}")
        outputs = item.get("outputs", {})
        expected_outputs = set(required_outputs)
        if item.get("uncertainty", {}).get("primary_replicates", 0) > 0:
            expected_outputs.add("pipeline_uncertainty.csv")
        if not expected_outputs.issubset(outputs):
            raise RuntimeError(f"Manifest lacks required outputs: {path}")
        for filename, expected_hash in outputs.items():
            output = path.parent / filename
            if not output.exists() or sha256(output) != expected_hash:
                raise RuntimeError(f"Output hash mismatch for {output}")
        prediction_frame = pd.read_csv(path.parent / "row_predictions.csv")
        metric_frame = pd.read_csv(path.parent / "metrics.csv")
        news_selection_frame = pd.read_csv(path.parent / "news_selection.csv")
        nested_selection_frame = pd.read_csv(path.parent / "nested_selections.csv")
        prediction_columns = set(prediction_frame.columns)
        metric_columns = set(metric_frame.columns)
        news_selection_columns = set(news_selection_frame.columns)
        if not {
            "model_key",
            "model",
            "resolved_revision",
            "candidate_seed",
            "sample_id",
            "cluster_id",
            "label",
            "prediction",
        }.issubset(prediction_columns):
            raise RuntimeError(f"Incomplete row prediction schema: {path}")
        if not {"n_rows", "n_clusters", "metric", "estimate"}.issubset(metric_columns):
            raise RuntimeError(f"Incomplete metric schema: {path}")
        if not {"rank", "selected_top_k", "head_index_base"}.issubset(
            news_selection_columns
        ):
            raise RuntimeError(f"Incomplete news selection schema: {path}")
        selection = item.get("selection", {})
        if selection.get("search_scope") != "fixed_budget" or selection.get(
            "news_labels_used_for_selection"
        ) is not False:
            raise RuntimeError(f"Invalid selection declaration: {path}")
        candidate_seeds = set(selection.get("candidate_seeds", []))
        if (
            set(prediction_frame["candidate_seed"]) != candidate_seeds
            or set(metric_frame["candidate_seed"]) != candidate_seeds
            or set(news_selection_frame["candidate_seed"]) != candidate_seeds
            or set(nested_selection_frame["candidate_seed"]) != candidate_seeds
        ):
            raise RuntimeError(f"Candidate seed coverage mismatch: {path}")
        if set(prediction_frame["model_key"]) != {item["model_key"]} or set(
            prediction_frame["model"]
        ) != {item["model"]}:
            raise RuntimeError(f"Row prediction model provenance mismatch: {path}")
        expected_predictions = 2 * sum(
            record["rows"] for record in item["datasets"].values()
        ) * len(candidate_seeds)
        if len(prediction_frame) != expected_predictions:
            raise RuntimeError(f"Unexpected prediction row count: {path}")
        expected_nested = (
            selection["candidate_budget"]
            * selection["outer_folds"]
            * len(candidate_seeds)
        )
        expected_news = selection["candidate_budget"] * len(candidate_seeds)
        if (
            len(nested_selection_frame) != expected_nested
            or len(news_selection_frame) != expected_news
        ):
            raise RuntimeError(f"Unexpected selection row count: {path}")
    keys = (
        "protocol",
        "candidate_budget",
        "search_scope",
        "candidate_sampling",
        "candidate_seeds",
        "top_k",
        "outer_folds",
        "inner_folds",
        "selection_metric",
        "ridge_alpha",
        "fit_intercept",
        "seed",
    )
    signatures = {
        tuple(
            (
                key,
                tuple(item["selection"][key])
                if key == "candidate_seeds"
                else item["selection"][key],
            )
            for key in keys
        )
        for item in manifests
    }
    if len(signatures) != 1:
        raise RuntimeError("Panel uses unequal confirmatory selection configurations")
    configuration = dict(signatures.pop())
    uncertainty = {
        (
            item["uncertainty"]["conditional_method"],
            item["uncertainty"]["conditional_replicates"],
            item["uncertainty"]["primary_method"],
            item["uncertainty"]["primary_replicates"],
            item["uncertainty"]["confidence"],
        )
        for item in manifests
    }
    if len(uncertainty) != 1:
        raise RuntimeError("Panel uses unequal uncertainty configurations")
    print(
        json.dumps(
            {
                "models": [item["model"] for item in manifests],
                "selection": configuration,
                "valid": True,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("prepare-released", "analyze", "audit-panel"), required=True
    )
    parser.add_argument("--model", choices=tuple(ALL_MODELS))
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--candidate-budget", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--outer-folds", type=int)
    parser.add_argument("--inner-folds", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--pipeline-bootstrap-replicates", type=int)
    parser.add_argument("--candidate-seed", type=int, action="append")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()
    if args.stage == "audit-panel":
        if len(args.manifest) < 2:
            parser.error("--audit-panel requires at least two --manifest values")
        audit_panel(args.manifest)
        return
    if args.stage == "prepare-released":
        if args.model is None:
            parser.error("--model is required for --stage prepare-released")
        prepare_released(args.model)
        return
    if args.model is None:
        parser.error("--model is required for --stage analyze")
    protocol = load_protocol(args.protocol)
    provided = {
        "candidate_budget": args.candidate_budget,
        "candidate_seeds": args.candidate_seed,
        "top_k": args.top_k,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "conditional_bootstrap_replicates": args.bootstrap_replicates,
        "pipeline_bootstrap_replicates": args.pipeline_bootstrap_replicates,
        "seed": args.seed,
        "ridge_alpha": args.alpha,
    }
    overrides = {key: value for key, value in provided.items() if value is not None}
    resolved = {
        key: protocol[key] if value is None else value for key, value in provided.items()
    }
    analyze(
        args.model,
        resolved["candidate_budget"],
        resolved["top_k"],
        resolved["outer_folds"],
        resolved["inner_folds"],
        resolved["conditional_bootstrap_replicates"],
        resolved["seed"],
        resolved["ridge_alpha"],
        resolved["candidate_seeds"],
        resolved["pipeline_bootstrap_replicates"],
        args.protocol,
        overrides,
    )


if __name__ == "__main__":
    main()
