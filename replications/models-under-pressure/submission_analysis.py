from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "submission_protocol.json"
METRIC_NAMES = (
    "auroc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "brier",
    "ece",
    "tn",
    "fp",
    "fn",
    "tp",
    "tpr_at_fpr_0_01",
    "tpr_at_fpr_0_05",
    "test_tpr_at_dev_fpr_0_01_threshold",
    "test_fpr_at_dev_fpr_0_01_threshold",
    "test_tpr_at_dev_fpr_0_05_threshold",
    "test_fpr_at_dev_fpr_0_05_threshold",
)
RATE_METRICS = tuple(name for name in METRIC_NAMES if name not in {"tn", "fp", "fn", "tp"})
FULL_PIPELINE_METRICS = RATE_METRICS
FILE_PATTERN = re.compile(
    r"_(?P<split>dev|test)_(?P<probe>attention|softmax)_L(?P<layer>\d+)_seed(?P<seed>\d+)\.jsonl$"
)


@dataclass(frozen=True)
class ResultRecord:
    split: str
    model: str
    probe: str
    layer: int
    seed: int | None
    dataset: str
    ids: tuple[str, ...]
    labels: tuple[int, ...]
    scores: tuple[float, ...]
    config_id: str
    source: Path


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported protocol schema")
    if len(protocol["seeds"]) < 3:
        raise ValueError("The confirmatory protocol requires at least three seeds")
    if not protocol["selection"].get("oracle_is_diagnostic_only"):
        raise ValueError("Oracle selection must be marked diagnostic-only")
    hashes = protocol.get("dataset_sha256", {})
    if not hashes or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()):
        raise ValueError("The confirmatory protocol must pin every dataset hash")
    for key, model in protocol["models"].items():
        revision = model.get("revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{key} does not pin a 40-character model revision")
        candidates = set(model["candidate_layers"])
        if not set(model["sparse_grid_layers"]).issubset(candidates):
            raise ValueError(f"{key} sparse grid is outside its candidate layers")
    return protocol


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_artifact_contract(
    artifact: Path,
    protocol: dict,
    protocol_path: Path,
    model_key: str,
    records: Sequence[ResultRecord] | None = None,
) -> dict:
    run_path = artifact / "run.json"
    copied_protocol = artifact / "submission_protocol.json"
    if not run_path.is_file() or not copied_protocol.is_file():
        raise ValueError("artifact is missing run.json or its frozen protocol copy")
    run = json.loads(run_path.read_text())
    model = protocol["models"][model_key]
    expected = {
        "model": model["model_id"],
        "model_revision": model["revision"],
        "model_config": model["model_config"],
        "activation_site": protocol["activation_site"],
    }
    mismatches = [
        f"{field}: expected {value!r}, observed {run.get(field)!r}"
        for field, value in expected.items()
        if run.get(field) != value
    ]
    for field, values in (
        ("layers", model["candidate_layers"]),
        ("seeds", protocol["seeds"]),
        ("probes", protocol["probes"]),
    ):
        if sorted(run.get(field, [])) != sorted(values):
            mismatches.append(f"{field}: expected {values!r}, observed {run.get(field)!r}")
    protocol_hash = _sha256(protocol_path)
    if run.get("protocol_sha256") != protocol_hash or _sha256(copied_protocol) != protocol_hash:
        mismatches.append("protocol hash does not match the frozen analysis protocol")
    if run.get("dataset_sha256") != protocol["dataset_sha256"]:
        mismatches.append("dataset hashes do not match the frozen analysis protocol")
    source = run.get("upstream_source_tree", {})
    if source != protocol.get("upstream_source_tree"):
        mismatches.append("upstream source-tree digest does not match the frozen snapshot")
    expected_provenance_rows = len(protocol["dataset_sha256"]) * len(model["candidate_layers"])
    if (
        run.get("activation_provenance_rows") != expected_provenance_rows
        or not re.fullmatch(r"[0-9a-f]{64}", str(run.get("activation_provenance_sha256", "")))
    ):
        mismatches.append("activation provenance coverage or digest is missing")
    if records is not None:
        wrong_models = sorted({record.model for record in records if record.model != model["model_id"]})
        if wrong_models:
            mismatches.append(f"result configs contain unexpected models: {wrong_models}")
        wrong_layers = sorted({record.layer for record in records} - set(model["candidate_layers"]))
        wrong_probes = sorted({record.probe for record in records} - set(protocol["probes"]))
        if wrong_layers:
            mismatches.append(f"result configs contain unexpected layers: {wrong_layers}")
        if wrong_probes:
            mismatches.append(f"result configs contain unexpected probes: {wrong_probes}")
        bad_ids = [
            record.config_id
            for record in records
            if record.seed is None
            or not record.config_id.endswith(
                f"_{record.probe}_L{record.layer}_seed{record.seed}"
            )
        ]
        if bad_ids:
            mismatches.append(
                f"result config IDs do not encode the expected run cell: {bad_ids[:3]}"
            )
    if mismatches:
        raise ValueError("artifact contract mismatch: " + "; ".join(mismatches))
    return run


def _split_from_record(path: Path, row: dict) -> str | None:
    match = FILE_PATTERN.search(path.name)
    if match:
        return match.group("split")
    dataset_path = str(row.get("dataset_path", "")).replace("\\", "/")
    for split in ("dev", "test"):
        if f"/evals/{split}/" in f"/{dataset_path}":
            return split
    return path.parent.name if path.parent.name in {"dev", "test"} else None


def _seed_from_path(path: Path) -> int | None:
    match = FILE_PATTERN.search(path.name)
    if match:
        return int(match.group("seed"))
    match = re.search(r"(?:^|_)seed(?P<seed>\d+)(?:_|\.)", path.name)
    return int(match.group("seed")) if match else None


def _probe_name(config: dict) -> str:
    name = config["probe_spec"]["name"]
    return "softmax" if name == "linear_then_softmax" else name


def load_results(root: Path) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    for path in sorted(root.rglob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            required = {"config", "dataset_name", "output_scores", "ground_truth_labels"}
            if not required.issubset(row):
                continue
            config = row["config"]
            split = _split_from_record(path, row)
            labels = tuple(int(value) for value in row["ground_truth_labels"])
            scores = tuple(float(value) for value in row["output_scores"])
            if row.get("ids") is None:
                raise ValueError(f"Row identifiers are missing in {path}")
            ids = tuple(str(value) for value in row["ids"])
            if not split:
                raise ValueError(f"Cannot establish dev/test provenance for {path}")
            if not (len(ids) == len(labels) == len(scores)):
                raise ValueError(f"Row-level arrays have different lengths in {path}")
            if any(label not in {0, 1} for label in labels):
                raise ValueError(f"Non-binary labels in {path}")
            if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
                raise ValueError(f"Scores outside [0, 1] in {path}")
            records.append(
                ResultRecord(
                    split=split,
                    model=config["model_name"],
                    probe=_probe_name(config),
                    layer=int(config["layer"]),
                    seed=_seed_from_path(path),
                    dataset=row["dataset_name"],
                    ids=ids,
                    labels=labels,
                    scores=scores,
                    config_id=str(config.get("id", "")),
                    source=path,
                )
            )
    return records


def _ranking_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives:
        return {
            "auroc": math.nan,
            "auprc": math.nan,
            "tpr_at_fpr_0_01": math.nan,
            "tpr_at_fpr_0_05": math.nan,
        }
    if not negatives:
        return {
            "auroc": math.nan,
            "auprc": 1.0,
            "tpr_at_fpr_0_01": math.nan,
            "tpr_at_fpr_0_05": math.nan,
        }
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positives = false_positives = 0
    previous_tpr = previous_fpr = 0.0
    auc = average_precision = 0.0
    previous_recall = 0.0
    low_fpr_tpr = {0.01: 0.0, 0.05: 0.0}
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        group_positive = sum(labels[order[index]] for index in range(start, end))
        true_positives += group_positive
        false_positives += end - start - group_positive
        tpr = true_positives / positives
        fpr = false_positives / negatives
        auc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2
        precision = true_positives / (true_positives + false_positives)
        average_precision += (tpr - previous_recall) * precision
        for target in low_fpr_tpr:
            if fpr <= target + 1e-12:
                low_fpr_tpr[target] = max(low_fpr_tpr[target], tpr)
        previous_tpr = tpr
        previous_fpr = fpr
        previous_recall = tpr
        start = end
    return {
        "auroc": auc,
        "auprc": average_precision,
        "tpr_at_fpr_0_01": low_fpr_tpr[0.01],
        "tpr_at_fpr_0_05": low_fpr_tpr[0.05],
    }


def _rank_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    return _ranking_metrics(labels, scores)["auroc"]


def _confusion(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> tuple[int, int, int, int]:
    tn = fp = fn = tp = 0
    for label, score in zip(labels, scores):
        prediction = int(score >= threshold)
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 1:
            fn += 1
        elif prediction == 1:
            fp += 1
        else:
            tn += 1
    return tn, fp, fn, tp


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _ece(labels: Sequence[int], scores: Sequence[float], bins: int = 10) -> float:
    total = len(labels)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, score in enumerate(scores)
            if lower <= score < upper or (bin_index == bins - 1 and score == 1.0)
        ]
        if not indices:
            continue
        confidence = sum(scores[index] for index in indices) / len(indices)
        frequency = sum(labels[index] for index in indices) / len(indices)
        error += len(indices) / total * abs(confidence - frequency)
    return error


def calculate_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, float | int]:
    if not labels or len(labels) != len(scores):
        raise ValueError("Metrics require equally sized non-empty labels and scores")
    tn, fp, fn, tp = _confusion(labels, scores, threshold)
    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    recall = sensitivity
    ranking = _ranking_metrics(labels, scores)
    return {
        "auroc": ranking["auroc"],
        "auprc": ranking["auprc"],
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2 * precision * recall, precision + recall),
        "brier": sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels),
        "ece": _ece(labels, scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr_at_fpr_0_01": ranking["tpr_at_fpr_0_01"],
        "tpr_at_fpr_0_05": ranking["tpr_at_fpr_0_05"],
    }


def choose_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Development threshold selection requires both classes")
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    candidates: list[tuple[float, float]] = [(math.nextafter(max(scores), math.inf), 0.5)]
    tp = fp = 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        group_positive = sum(labels[order[index]] for index in range(start, end))
        tp += group_positive
        fp += end - start - group_positive
        balanced_accuracy = (tp / positives + (negatives - fp) / negatives) / 2
        candidates.append((scores[order[start]], balanced_accuracy))
        start = end
    return max(candidates, key=lambda item: (item[1], -abs(item[0] - 0.5), item[0]))[0]


def choose_fpr_threshold(
    labels: Sequence[int], scores: Sequence[float], maximum_fpr: float
) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Development FPR threshold selection requires both classes")
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    candidates = [(math.nextafter(max(scores), math.inf), 0.0, 0.0)]
    tp = fp = 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        group_positive = sum(labels[order[index]] for index in range(start, end))
        tp += group_positive
        fp += end - start - group_positive
        fpr = fp / negatives
        if fpr <= maximum_fpr + 1e-12:
            candidates.append((scores[order[start]], tp / positives, fpr))
        start = end
    return max(candidates, key=lambda item: (item[1], -item[2], item[0]))[0]


def operating_rates(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> tuple[float, float]:
    tn, fp, fn, tp = _confusion(labels, scores, threshold)
    return _safe_ratio(tp, tp + fn), _safe_ratio(fp, fp + tn)


def development_thresholds(
    labels: Sequence[int], scores: Sequence[float]
) -> dict[str, float]:
    return {
        "balanced_accuracy": choose_threshold(labels, scores),
        "fpr_0_01": choose_fpr_threshold(labels, scores, 0.01),
        "fpr_0_05": choose_fpr_threshold(labels, scores, 0.05),
    }


def calculate_frozen_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    thresholds: dict[str, float],
) -> dict[str, float | int]:
    metrics = calculate_metrics(labels, scores, thresholds["balanced_accuracy"])
    for label, suffix in (("fpr_0_01", "0_01"), ("fpr_0_05", "0_05")):
        tpr, fpr = operating_rates(labels, scores, thresholds[label])
        metrics[f"test_tpr_at_dev_fpr_{suffix}_threshold"] = tpr
        metrics[f"test_fpr_at_dev_fpr_{suffix}_threshold"] = fpr
    return metrics


def _macro_auroc(records: Iterable[ResultRecord], datasets: set[str]) -> float:
    values = [
        _rank_auc(record.labels, record.scores)
        for record in records
        if record.dataset in datasets
    ]
    if len(values) != len(datasets):
        raise ValueError("Selection objective does not have exactly one record per dataset")
    return statistics.fmean(values)


def _closest_layer(layers: Sequence[int], num_hidden_layers: int, relative_depth: float) -> int:
    target = relative_depth * num_hidden_layers
    return min(layers, key=lambda layer: (abs(layer - target), layer))


def _index_records(
    records: Sequence[ResultRecord],
) -> dict[tuple[str, int, int, str, str], ResultRecord]:
    indexed: dict[tuple[str, int, int, str, str], ResultRecord] = {}
    for record in records:
        if record.seed is None:
            raise ValueError(f"Seed provenance is missing for {record.source}")
        key = (record.probe, record.seed, record.layer, record.split, record.dataset)
        if key in indexed:
            raise ValueError(f"Duplicate result for {key}")
        indexed[key] = record
    return indexed


def validate_coverage(records: Sequence[ResultRecord], protocol: dict, model_key: str) -> dict:
    model = protocol["models"][model_key]
    expected = {
        (probe, seed, layer, split, dataset)
        for probe in protocol["probes"]
        for seed in protocol["seeds"]
        for layer in model["candidate_layers"]
        for split, datasets in (
            ("dev", protocol["dev_datasets"]),
            ("test", protocol["test_datasets"]),
        )
        for dataset in datasets
    }
    actual = {
        (record.probe, record.seed, record.layer, record.split, record.dataset)
        for record in records
        if record.seed is not None
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return {
        "complete": not missing,
        "expected_records": len(expected),
        "observed_records": len(actual & expected),
        "missing_record_count": len(missing),
        "missing_records_preview": [list(item) for item in missing[:50]],
        "unexpected_records": [list(item) for item in unexpected],
        "missing_seed_files": sorted(
            {str(record.source) for record in records if record.seed is None}
        ),
    }


def _select_layers(
    records: Sequence[ResultRecord], protocol: dict, model_key: str
) -> list[dict[str, int | float | str | bool]]:
    model = protocol["models"][model_key]
    dev_datasets = set(protocol["dev_datasets"])
    test_datasets = set(protocol["test_datasets"])
    candidates = model["candidate_layers"]
    sparse = model["sparse_grid_layers"]
    relative = _closest_layer(
        candidates,
        model["num_hidden_layers"],
        protocol["selection"]["relative_depth"],
    )
    output = []
    for probe in protocol["probes"]:
        for seed in protocol["seeds"]:
            subset = [
                record
                for record in records
                if record.probe == probe and record.seed == seed
            ]
            dev_scores = {
                layer: _macro_auroc(
                    (
                        record
                        for record in subset
                        if record.split == "dev" and record.layer == layer
                    ),
                    dev_datasets,
                )
                for layer in candidates
            }
            test_scores = {
                layer: _macro_auroc(
                    (
                        record
                        for record in subset
                        if record.split == "test" and record.layer == layer
                    ),
                    test_datasets,
                )
                for layer in candidates
            }
            selected = {
                "relative_depth": relative,
                "sparse_grid": max(sparse, key=lambda layer: (dev_scores[layer], -layer)),
                "dev_selected": max(candidates, key=lambda layer: (dev_scores[layer], -layer)),
                "oracle_test": max(candidates, key=lambda layer: (test_scores[layer], -layer)),
            }
            for strategy, layer in selected.items():
                output.append(
                    {
                        "probe": probe,
                        "seed": seed,
                        "strategy": strategy,
                        "layer": layer,
                        "relative_depth": layer / model["num_hidden_layers"],
                        "dev_macro_auroc": dev_scores[layer],
                        "test_macro_auroc": test_scores[layer],
                        "diagnostic_only": strategy == "oracle_test",
                    }
                )
    return output


def _pooled(records: Iterable[ResultRecord]) -> tuple[list[str], list[int], list[float]]:
    ids: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    for record in records:
        ids.extend(f"{record.dataset}:{identifier}" for identifier in record.ids)
        labels.extend(record.labels)
        scores.extend(record.scores)
    return ids, labels, scores


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _cluster_bootstrap_cis(
    ids: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    metrics: set[str],
    replicates: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, tuple[float, float]]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, identifier in enumerate(ids):
        clusters[identifier].append(index)
    keys = sorted(clusters)
    rng = random.Random(random_seed)
    estimates: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sampled = [rng.choice(keys) for _ in keys]
        indices = [index for key in sampled for index in clusters[key]]
        sample_labels = [labels[index] for index in indices]
        if not sum(sample_labels) or sum(sample_labels) == len(sample_labels):
            continue
        sample_scores = [scores[index] for index in indices]
        sample_metrics = calculate_metrics(sample_labels, sample_scores, threshold)
        for metric in metrics:
            estimates[metric].append(float(sample_metrics[metric]))
    alpha = 1 - confidence_level
    return {
        metric: (
            _percentile(values, alpha / 2),
            _percentile(values, 1 - alpha / 2),
        )
        for metric, values in estimates.items()
    }


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _aligned_records(
    records: Sequence[ResultRecord],
    *,
    split: str,
    dataset: str,
    layers: Sequence[int],
) -> dict[int, ResultRecord]:
    selected = {
        record.layer: record
        for record in records
        if record.split == split and record.dataset == dataset and record.layer in layers
    }
    if set(selected) != set(layers):
        raise ValueError(f"missing paired {split} records for {dataset}")
    reference = selected[layers[0]]
    for record in selected.values():
        if record.ids != reference.ids or record.labels != reference.labels:
            raise ValueError(
                f"row IDs or labels are not aligned across layers for {split}/{dataset}"
            )
    return selected


def _full_pipeline_seed_bootstrap(
    records: Sequence[ResultRecord],
    protocol: dict,
    model_key: str,
    probe: str,
    seed: int,
) -> dict:
    candidates = protocol["models"][model_key]["candidate_layers"]
    subset = [record for record in records if record.probe == probe and record.seed == seed]
    dev = {
        dataset: _aligned_records(
            subset, split="dev", dataset=dataset, layers=candidates
        )
        for dataset in protocol["dev_datasets"]
    }
    test = {
        dataset: _aligned_records(
            subset, split="test", dataset=dataset, layers=candidates
        )
        for dataset in protocol["test_datasets"]
    }
    point_dev = {
        layer: statistics.fmean(
            _rank_auc(dev[dataset][layer].labels, dev[dataset][layer].scores)
            for dataset in protocol["dev_datasets"]
        )
        for layer in candidates
    }
    point_layer = max(candidates, key=lambda layer: (point_dev[layer], -layer))
    _, point_dev_labels, point_dev_scores = _pooled(
        dev[dataset][point_layer] for dataset in protocol["dev_datasets"]
    )
    point_thresholds = development_thresholds(point_dev_labels, point_dev_scores)
    point_metrics = {
        metric: statistics.fmean(
            float(
                calculate_frozen_metrics(
                    test[dataset][point_layer].labels,
                    test[dataset][point_layer].scores,
                    point_thresholds,
                )[metric]
            )
            for dataset in protocol["test_datasets"]
        )
        for metric in FULL_PIPELINE_METRICS
    }
    rng = random.Random(
        protocol["uncertainty"]["random_seed"] + _stable_seed("full_pipeline", seed)
    )
    replicates = []
    for _ in range(protocol["uncertainty"]["bootstrap_replicates"]):
        dev_indices = {
            dataset: [
                rng.randrange(len(dev[dataset][candidates[0]].labels))
                for _ in dev[dataset][candidates[0]].labels
            ]
            for dataset in protocol["dev_datasets"]
        }
        dev_scores = {}
        for layer in candidates:
            values = []
            for dataset in protocol["dev_datasets"]:
                record = dev[dataset][layer]
                indices = dev_indices[dataset]
                value = _rank_auc(
                    [record.labels[index] for index in indices],
                    [record.scores[index] for index in indices],
                )
                if not math.isfinite(value):
                    break
                values.append(value)
            if len(values) == len(protocol["dev_datasets"]):
                dev_scores[layer] = statistics.fmean(values)
        if not dev_scores:
            continue
        layer = max(dev_scores, key=lambda item: (dev_scores[item], -item))
        dev_labels: list[int] = []
        selected_dev_scores: list[float] = []
        for dataset in protocol["dev_datasets"]:
            record = dev[dataset][layer]
            indices = dev_indices[dataset]
            dev_labels.extend(record.labels[index] for index in indices)
            selected_dev_scores.extend(record.scores[index] for index in indices)
        try:
            thresholds = development_thresholds(dev_labels, selected_dev_scores)
        except ValueError:
            continue
        test_metrics = {metric: [] for metric in FULL_PIPELINE_METRICS}
        for dataset in protocol["test_datasets"]:
            record = test[dataset][layer]
            indices = [rng.randrange(len(record.labels)) for _ in record.labels]
            labels = [record.labels[index] for index in indices]
            scores = [record.scores[index] for index in indices]
            metrics = calculate_frozen_metrics(labels, scores, thresholds)
            for metric in FULL_PIPELINE_METRICS:
                test_metrics[metric].append(float(metrics[metric]))
        replicates.append(
            {
                "layer": layer,
                "threshold": thresholds["balanced_accuracy"],
                "dev_fpr_0_01_threshold": thresholds["fpr_0_01"],
                "dev_fpr_0_05_threshold": thresholds["fpr_0_05"],
                **{
                    metric: (
                        statistics.fmean(values)
                        if all(math.isfinite(value) for value in values)
                        else math.nan
                    )
                    for metric, values in test_metrics.items()
                },
            }
        )
    if not replicates:
        raise ValueError(f"no valid full-pipeline bootstrap replicates for {probe}/seed{seed}")
    return {
        "probe": probe,
        "seed": seed,
        "point_layer": point_layer,
        "point_threshold": point_thresholds["balanced_accuracy"],
        "point_thresholds": point_thresholds,
        "point_metrics": point_metrics,
        "replicates": replicates,
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in known:
                fieldnames.append(field)
                known.add(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(artifact: Path, output: Path, protocol_path: Path, model_key: str) -> dict:
    protocol = load_protocol(protocol_path)
    records = load_results(artifact)
    run = validate_artifact_contract(artifact, protocol, protocol_path, model_key, records)
    coverage = validate_coverage(records, protocol, model_key)
    if not coverage["complete"]:
        raise ValueError(
            f"Confirmatory coverage is incomplete: {coverage['observed_records']}/"
            f"{coverage['expected_records']} records; run audit for details"
        )
    indexed = _index_records(records)
    selections = _select_layers(records, protocol, model_key)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "selections.csv", selections)
    selection_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selections:
        selection_groups[(str(row["probe"]), str(row["strategy"]))].append(row)
    selection_aggregate = []
    for (probe, strategy), rows in sorted(selection_groups.items()):
        values = [float(row["test_macro_auroc"]) for row in rows]
        selection_aggregate.append(
            {
                "model_key": model_key,
                "probe": probe,
                "strategy": strategy,
                "diagnostic_only": strategy == "oracle_test",
                "selected_layers": ";".join(str(row["layer"]) for row in rows),
                "n_seeds": len(values),
                "test_macro_auroc_mean": statistics.fmean(values),
                "test_macro_auroc_standard_deviation": statistics.stdev(values),
                "test_macro_auroc_min": min(values),
                "test_macro_auroc_max": max(values),
                "uncertainty_note": "raw three-seed SD and range; no three-point percentile CI",
            }
        )
    _write_csv(output / "selection_aggregate.csv", selection_aggregate)

    metric_rows = []
    row_records = []
    uncertainty = protocol["uncertainty"]
    bootstrap_metrics = {
        "auroc",
        "auprc",
        "balanced_accuracy",
        "tpr_at_fpr_0_01",
        "tpr_at_fpr_0_05",
    }
    for selection in selections:
        probe = str(selection["probe"])
        seed = int(selection["seed"])
        strategy = str(selection["strategy"])
        layer = int(selection["layer"])
        dev = [
            indexed[(probe, seed, layer, "dev", dataset)]
            for dataset in protocol["dev_datasets"]
        ]
        _, dev_labels, dev_scores = _pooled(dev)
        thresholds = development_thresholds(dev_labels, dev_scores)
        threshold = thresholds["balanced_accuracy"]
        for dataset in protocol["test_datasets"]:
            record = indexed[(probe, seed, layer, "test", dataset)]
            metrics = calculate_frozen_metrics(record.labels, record.scores, thresholds)
            row = {
                "model_key": model_key,
                "probe": probe,
                "seed": seed,
                "strategy": strategy,
                "diagnostic_only": selection["diagnostic_only"],
                "layer": layer,
                "dataset": dataset,
                "threshold": threshold,
                "dev_fpr_0_01_threshold": thresholds["fpr_0_01"],
                "dev_fpr_0_05_threshold": thresholds["fpr_0_05"],
                "fpr_resolution": 1 / max(1, len(record.labels) - sum(record.labels)),
                **metrics,
            }
            if strategy == "dev_selected":
                cluster_ids = [f"{dataset}:{identifier}" for identifier in record.ids]
                intervals = _cluster_bootstrap_cis(
                    cluster_ids,
                    record.labels,
                    record.scores,
                    threshold,
                    bootstrap_metrics,
                    uncertainty["bootstrap_replicates"],
                    uncertainty["confidence_level"],
                    uncertainty["random_seed"]
                    + _stable_seed(probe, seed, layer, dataset),
                )
                for metric, (low, high) in intervals.items():
                    row[f"{metric}_conditional_ci_low"] = low
                    row[f"{metric}_conditional_ci_high"] = high
                row["conditional_ci_scope"] = "test-row bootstrap conditional on selected layer and dev threshold"
            metric_rows.append(row)
            for identifier, label, score in zip(record.ids, record.labels, record.scores):
                row_records.append(
                    {
                        "model_key": model_key,
                        "probe": probe,
                        "seed": seed,
                        "strategy": strategy,
                        "diagnostic_only": selection["diagnostic_only"],
                        "layer": layer,
                        "dataset": dataset,
                        "id": identifier,
                        "label": label,
                        "score": score,
                        "probability": score,
                        "threshold": threshold,
                        "threshold_source": "pooled_development_balanced_accuracy",
                        "prediction": int(score >= threshold),
                        "dev_fpr_0_01_threshold": thresholds["fpr_0_01"],
                        "prediction_at_dev_fpr_0_01": int(
                            score >= thresholds["fpr_0_01"]
                        ),
                        "dev_fpr_0_05_threshold": thresholds["fpr_0_05"],
                        "prediction_at_dev_fpr_0_05": int(
                            score >= thresholds["fpr_0_05"]
                        ),
                    }
                )
    _write_csv(output / "metrics.csv", metric_rows)
    with (output / "rows.jsonl").open("w") as handle:
        for row in row_records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    full_pipeline = {
        (probe, seed): _full_pipeline_seed_bootstrap(
            records, protocol, model_key, probe, seed
        )
        for probe in protocol["probes"]
        for seed in protocol["seeds"]
    }
    full_seed_rows = []
    full_selection_rows = []
    confidence = uncertainty["confidence_level"]
    alpha = 1 - confidence
    for (probe, seed), result in sorted(full_pipeline.items()):
        counts = Counter(int(row["layer"]) for row in result["replicates"])
        for layer in protocol["models"][model_key]["candidate_layers"]:
            full_selection_rows.append(
                {
                    "model_key": model_key,
                    "probe": probe,
                    "seed": seed,
                    "layer": layer,
                    "bootstrap_selection_count": counts[layer],
                    "bootstrap_selection_fraction": counts[layer] / len(result["replicates"]),
                }
            )
        for metric in FULL_PIPELINE_METRICS:
            values = [
                float(row[metric])
                for row in result["replicates"]
                if math.isfinite(float(row[metric]))
            ]
            full_seed_rows.append(
                {
                    "model_key": model_key,
                    "probe": probe,
                    "seed": seed,
                    "metric": metric,
                    "selected_layer": result["point_layer"],
                    "dev_threshold": result["point_threshold"],
                    "dev_fpr_0_01_threshold": result["point_thresholds"]["fpr_0_01"],
                    "dev_fpr_0_05_threshold": result["point_thresholds"]["fpr_0_05"],
                    "estimate": result["point_metrics"][metric],
                    "ci_low": _percentile(values, alpha / 2),
                    "ci_high": _percentile(values, 1 - alpha / 2),
                    "valid_bootstrap_replicates": len(values),
                    "invalid_bootstrap_replicates": len(result["replicates"]) - len(values),
                    "ci_scope": "joint dev-row/test-row bootstrap repeating layer and threshold selection",
                }
            )
    _write_csv(output / "full_pipeline_seed.csv", full_seed_rows)
    _write_csv(output / "full_pipeline_layer_selection.csv", full_selection_rows)

    hierarchy_rng = random.Random(uncertainty["random_seed"] + _stable_seed("hierarchy"))
    hierarchical_rows = []
    hierarchical_samples: dict[tuple[str, str], list[float]] = {}
    for probe in protocol["probes"]:
        for metric in FULL_PIPELINE_METRICS:
            values = [
                full_pipeline[(probe, seed)]["point_metrics"][metric]
                for seed in protocol["seeds"]
            ]
            boot = []
            for _ in range(5000):
                sampled_seeds = [hierarchy_rng.choice(protocol["seeds"]) for _ in protocol["seeds"]]
                sampled_values = []
                for sampled_seed in sampled_seeds:
                    rows = [
                        row
                        for row in full_pipeline[(probe, sampled_seed)]["replicates"]
                        if math.isfinite(float(row[metric]))
                    ]
                    if not rows:
                        sampled_values = []
                        break
                    sampled_values.append(float(hierarchy_rng.choice(rows)[metric]))
                if sampled_values:
                    boot.append(statistics.fmean(sampled_values))
            hierarchical_samples[(probe, metric)] = boot
            hierarchical_rows.append(
                {
                    "model_key": model_key,
                    "probe": probe,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": statistics.fmean(values),
                    "seed_standard_deviation": statistics.stdev(values),
                    "seed_min": min(values),
                    "seed_max": max(values),
                    "hierarchical_ci_low": _percentile(boot, alpha / 2),
                    "hierarchical_ci_high": _percentile(boot, 1 - alpha / 2),
                    "valid_hierarchical_replicates": len(boot),
                    "ci_scope": "seed resampling plus full dev-selection/test-row bootstrap",
                }
            )
    _write_csv(output / "full_pipeline_aggregate.csv", hierarchical_rows)

    if {"attention", "softmax"}.issubset(protocol["probes"]):
        paired_rows = []
        paired_rng = random.Random(uncertainty["random_seed"] + _stable_seed("paired"))
        for metric in FULL_PIPELINE_METRICS:
            seed_differences = [
                full_pipeline[("attention", seed)]["point_metrics"][metric]
                - full_pipeline[("softmax", seed)]["point_metrics"][metric]
                for seed in protocol["seeds"]
            ]
            boot = []
            for _ in range(5000):
                sampled_seeds = [paired_rng.choice(protocol["seeds"]) for _ in protocol["seeds"]]
                differences = []
                for sampled_seed in sampled_seeds:
                    attention_rows = full_pipeline[("attention", sampled_seed)]["replicates"]
                    softmax_rows = full_pipeline[("softmax", sampled_seed)]["replicates"]
                    valid_indices = [
                        index
                        for index in range(min(len(attention_rows), len(softmax_rows)))
                        if math.isfinite(float(attention_rows[index][metric]))
                        and math.isfinite(float(softmax_rows[index][metric]))
                    ]
                    if not valid_indices:
                        differences = []
                        break
                    index = paired_rng.choice(valid_indices)
                    differences.append(
                        float(attention_rows[index][metric]) - float(softmax_rows[index][metric])
                    )
                if differences:
                    boot.append(statistics.fmean(differences))
            paired_rows.append(
                {
                    "model_key": model_key,
                    "contrast": "attention_minus_softmax",
                    "metric": metric,
                    "n_paired_seeds": len(seed_differences),
                    "mean_difference": statistics.fmean(seed_differences),
                    "seed_difference_standard_deviation": statistics.stdev(seed_differences),
                    "hierarchical_ci_low": _percentile(boot, alpha / 2),
                    "hierarchical_ci_high": _percentile(boot, 1 - alpha / 2),
                    "valid_hierarchical_replicates": len(boot),
                    "ci_scope": "paired seed and paired row bootstrap with each probe reselected",
                }
            )
        _write_csv(output / "paired_probe_differences.csv", paired_rows)

    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in metric_rows:
        grouped[(row["model_key"], row["probe"], row["strategy"], row["dataset"])].append(row)
    aggregate_rows = []
    for key, rows in sorted(grouped.items()):
        for metric in RATE_METRICS:
            values = [float(row[metric]) for row in rows]
            aggregate_rows.append(
                {
                    "model_key": key[0],
                    "probe": key[1],
                    "strategy": key[2],
                    "diagnostic_only": key[2] == "oracle_test",
                    "dataset": key[3],
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": statistics.fmean(values),
                    "standard_deviation": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                    "uncertainty_note": "raw three-seed SD and range; use full_pipeline_aggregate.csv for hierarchical intervals",
                }
            )
    _write_csv(output / "seed_aggregate.csv", aggregate_rows)

    manifest = {
        "schema_version": 1,
        "analysis_id": protocol["analysis_id"],
        "model_key": model_key,
        "model": protocol["models"][model_key],
        "run_sha256": _sha256(artifact / "run.json"),
        "run_contract": {
            "model": run["model"],
            "model_revision": run["model_revision"],
            "activation_site": run["activation_site"],
            "dataset_sha256": run["dataset_sha256"],
            "upstream_source_tree": run["upstream_source_tree"],
        },
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "source_artifact": str(artifact.resolve()),
        "coverage": coverage,
        "selection_note": (
            "oracle_test is a diagnostic upper bound and is excluded from "
            "confirmatory claims"
        ),
        "outputs": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def audit(artifact: Path, protocol_path: Path, model_key: str) -> dict:
    protocol = load_protocol(protocol_path)
    records = load_results(artifact)
    validate_artifact_contract(artifact, protocol, protocol_path, model_key, records)
    coverage = validate_coverage(records, protocol, model_key)
    cell_counts = Counter(
        (record.split, record.probe, record.layer, record.dataset, record.seed)
        for record in records
    )
    duplicate_cells = [
        [*key, count] for key, count in sorted(cell_counts.items()) if count > 1
    ]
    return {
        "artifact": str(artifact.resolve()),
        "model_key": model_key,
        "files": len({record.source for record in records}),
        "result_records": len(records),
        "splits": sorted({record.split for record in records}),
        "seeds": sorted({record.seed for record in records if record.seed is not None}),
        "duplicate_result_cell_count": len(duplicate_cells),
        "duplicate_result_cells_preview": duplicate_cells[:20],
        "row_scores_retained": all(
            record.scores and record.labels and record.ids for record in records
        ),
        "coverage": coverage,
        "ready_for_confirmatory_analysis": (
            coverage["complete"] and not coverage["missing_seed_files"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "analyze"))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(args.artifact, args.protocol, args.model_key), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for analyze")
    print(json.dumps(analyze(args.artifact, args.output, args.protocol, args.model_key), indent=2))


if __name__ == "__main__":
    main()
