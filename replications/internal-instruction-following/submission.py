from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import compat

CONFIG = compat.CONFIG["submission_analysis"]
NORMALIZATION_MODES = tuple(CONFIG["normalization_modes"])
METRICS = tuple(CONFIG["metrics"])


def file_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_statistics(values):
    mean = values.mean(dim=0)
    scale = values.std(dim=0)
    scale = scale.where(scale > 0, scale.new_ones(scale.shape))
    return mean, scale


def apply_statistics(values, statistics_pair):
    mean, scale = statistics_pair
    normalized = (values - mean) / scale
    if not normalized.isfinite().all():
        raise RuntimeError("Non-finite values after normalization")
    return normalized


def normalize_partitions(train, development, test, mode: str):
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode: {mode}")
    if mode == "released_splitwise":
        return tuple(
            apply_statistics(values, fit_statistics(values))
            for values in (train, development, test)
        )
    train_statistics = fit_statistics(train)
    return tuple(
        apply_statistics(values, train_statistics)
        for values in (train, development, test)
    )


def frozen_threshold(labels, probabilities) -> float:
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    scores = [balanced_accuracy_score(y, p >= threshold) for threshold in candidates]
    best = max(scores)
    tied = [
        float(threshold)
        for threshold, score in zip(candidates, scores, strict=True)
        if math.isclose(score, best)
    ]
    return min(tied, key=lambda threshold: (abs(threshold - 0.5), threshold))


def frozen_fpr_threshold(labels, probabilities, maximum_fpr: float) -> float:
    import numpy as np

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError("Development FPR threshold selection requires both classes")
    order = np.argsort(-p, kind="stable")
    candidates = [(float(np.nextafter(p.max(), np.inf)), 0.0, 0.0)]
    tp = fp = start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and p[order[end]] == p[order[start]]:
            end += 1
        group = y[order[start:end]]
        tp += int(group.sum())
        fp += len(group) - int(group.sum())
        fpr = fp / negatives
        if fpr <= maximum_fpr + 1e-12:
            candidates.append((float(p[order[start]]), tp / positives, fpr))
        start = end
    return max(candidates, key=lambda item: (item[1], -item[2], item[0]))[0]


def frozen_thresholds(labels, probabilities) -> dict[str, float]:
    return {
        "balanced_accuracy": frozen_threshold(labels, probabilities),
        "fpr_0_01": frozen_fpr_threshold(labels, probabilities, 0.01),
        "fpr_0_05": frozen_fpr_threshold(labels, probabilities, 0.05),
    }


def expected_calibration_error(labels, probabilities, bins: int) -> float:
    import numpy as np

    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (p >= lower) & (p < upper if index < bins - 1 else p <= upper)
        if selected.any():
            error += selected.mean() * abs(y[selected].mean() - p[selected].mean())
    return float(error) if total else float("nan")


def tpr_at_fpr(labels, probabilities, maximum_fpr: float) -> float | str:
    import numpy as np
    from sklearn.metrics import roc_curve

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(np.unique(y)) < 2:
        return ""
    fpr, tpr, _ = roc_curve(y, p)
    eligible = tpr[fpr <= maximum_fpr]
    return float(eligible.max()) if len(eligible) else 0.0


def metric_values(
    labels, probabilities, thresholds: dict[str, float]
) -> dict[str, float | int | str]:
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    prediction = (p >= thresholds["balanced_accuracy"]).astype(int)
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    tp = int(((y == 1) & (prediction == 1)).sum())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, average="binary", zero_division=0
    )
    complete = len(np.unique(y)) == 2
    metrics = {
        "auroc": float(roc_auc_score(y, p)) if complete else "",
        "auprc": float(average_precision_score(y, p)) if complete else "",
        "accuracy": float((prediction == y).mean()),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, prediction)) if complete else ""
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p, int(CONFIG["calibration_bins"])),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr_at_fpr_0_01": tpr_at_fpr(y, p, 0.01),
        "tpr_at_fpr_0_05": tpr_at_fpr(y, p, 0.05),
    }
    for key, suffix in (("fpr_0_01", "0_01"), ("fpr_0_05", "0_05")):
        selected = (p >= thresholds[key]).astype(int)
        test_fp = int(((y == 0) & (selected == 1)).sum())
        test_tp = int(((y == 1) & (selected == 1)).sum())
        metrics[f"test_tpr_at_dev_fpr_{suffix}_threshold"] = (
            test_tp / int((y == 1).sum()) if (y == 1).any() else ""
        )
        metrics[f"test_fpr_at_dev_fpr_{suffix}_threshold"] = (
            test_fp / int((y == 0).sum()) if (y == 0).any() else ""
        )
    return metrics


def cluster_bootstrap(
    labels,
    probabilities,
    clusters,
    thresholds: dict[str, float],
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    import numpy as np

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    cluster_values = np.asarray(clusters)
    unique, inverse = np.unique(cluster_values, return_inverse=True)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(unique), size=(samples, len(unique)))
    counts = np.zeros((samples, len(unique)), dtype=np.int16)
    np.add.at(
        counts,
        (np.repeat(np.arange(samples), len(unique)), selected.reshape(-1)),
        1,
    )
    draws = weighted_metric_draws(y, p, counts[:, inverse], thresholds)
    alpha = (1.0 - float(CONFIG["bootstrap_confidence"])) / 2.0
    estimates = metric_values(y, p, thresholds)
    rows = []
    for metric in METRICS:
        values = draws[metric]
        values = values[np.isfinite(values)]
        estimate = estimates[metric]
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci_lower": float(np.quantile(values, alpha)) if len(values) else "",
                "ci_upper": (
                    float(np.quantile(values, 1.0 - alpha)) if len(values) else ""
                ),
                "valid_bootstrap_samples": len(values),
            }
        )
    return rows


def weighted_metric_draws(labels, probabilities, weights, thresholds):
    import numpy as np

    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = w.sum(axis=1)
    positives = w @ y
    negatives = total - positives

    prediction = p >= thresholds["balanced_accuracy"]
    tp = w @ (prediction * y)
    fp = w @ (prediction * (1 - y))
    fn = positives - tp
    tn = negatives - fp

    def divide(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator != 0,
        )

    recall = divide(tp, positives)
    specificity = divide(tn, negatives)
    precision = divide(tp, tp + fp)
    output = {
        "accuracy": divide(tp + tn, total),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": np.nan_to_num(precision, nan=0.0),
        "recall": np.nan_to_num(recall, nan=0.0),
        "f1": np.nan_to_num(divide(2 * tp, 2 * tp + fp + fn), nan=0.0),
        "brier": divide(w @ ((p - y) ** 2), total),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    positive_groups = np.add.reduceat(w[:, order] * y[order], starts, axis=1)
    negative_groups = np.add.reduceat(w[:, order] * (1 - y[order]), starts, axis=1)
    cumulative_positive = np.cumsum(positive_groups, axis=1)
    cumulative_negative = np.cumsum(negative_groups, axis=1)
    lower_negative = negatives[:, None] - cumulative_negative
    concordant = np.sum(
        positive_groups * (lower_negative + 0.5 * negative_groups), axis=1
    )
    output["auroc"] = divide(concordant, positives * negatives)
    group_precision = divide(
        cumulative_positive,
        cumulative_positive + cumulative_negative,
    )
    output["auprc"] = divide(
        np.nansum(positive_groups * group_precision, axis=1), positives
    )
    complete = (positives > 0) & (negatives > 0)
    output["auprc"] = np.where(complete, output["auprc"], np.nan)
    fpr = divide(cumulative_negative, negatives[:, None])
    tpr = divide(cumulative_positive, positives[:, None])
    for maximum, suffix in ((0.01, "0_01"), (0.05, "0_05")):
        eligible = np.where(fpr <= maximum + 1e-12, tpr, -np.inf)
        best = np.max(eligible, axis=1)
        output[f"tpr_at_fpr_{suffix}"] = np.where(
            complete, np.maximum(best, 0.0), np.nan
        )

    ece = np.zeros(len(w), dtype=float)
    edges = np.linspace(0.0, 1.0, int(CONFIG["calibration_bins"]) + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (p >= lower) & (p < upper if index < len(edges) - 2 else p <= upper)
        bin_weight = w @ mask
        bin_labels = w @ (mask * y)
        bin_probability = w @ (mask * p)
        observed = divide(bin_labels, bin_weight)
        expected = divide(bin_probability, bin_weight)
        ece += np.nan_to_num(bin_weight / total * np.abs(observed - expected))
    output["ece"] = ece

    for key, suffix in (("fpr_0_01", "0_01"), ("fpr_0_05", "0_05")):
        selected_at_threshold = p >= thresholds[key]
        selected_tp = w @ (selected_at_threshold * y)
        selected_fp = w @ (selected_at_threshold * (1 - y))
        output[f"test_tpr_at_dev_fpr_{suffix}_threshold"] = divide(
            selected_tp, positives
        )
        output[f"test_fpr_at_dev_fpr_{suffix}_threshold"] = divide(
            selected_fp, negatives
        )
    return output


def grouped_bootstrap(
    rows: list[dict[str, object]], samples: int, seed: int
) -> list[dict[str, object]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    alpha = (1.0 - float(CONFIG["bootstrap_confidence"])) / 2.0
    output = []
    for metric in METRICS:
        values = np.asarray(
            [float(row[metric]) for row in rows if row[metric] != ""], dtype=float
        )
        if not len(values):
            continue
        draws = np.asarray(
            [
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(samples)
            ]
        )
        output.append(
            {
                "metric": metric,
                "estimate": float(values.mean()),
                "ci_lower": float(np.quantile(draws, alpha)),
                "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
                "valid_bootstrap_samples": samples,
            }
        )
    return output


def development_mask(candidate_mask, task_ids, fraction: float, seed: int):
    import torch

    candidates = torch.unique(task_ids[candidate_mask])
    generator = torch.Generator().manual_seed(seed + 10_000)
    ordered = candidates[torch.randperm(len(candidates), generator=generator)]
    count = max(1, round(len(ordered) * fraction))
    selected = ordered[:count]
    return candidate_mask & torch.isin(task_ids, selected)


def output_paths(model: dict) -> dict[str, Path]:
    root = compat.paths(model)["root"]
    return {
        "runs": root / "submission_runs.csv",
        "predictions": root / "submission_predictions.csv",
        "uncertainty": root / "submission_uncertainty.csv",
        "contrasts": root / "submission_factorial_contrasts.csv",
        "contrast_summary": root / "submission_factorial_summary.csv",
        "manifest": root / "submission_manifest.json",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_controls(base_model: str) -> dict[str, object]:
    import numpy as np

    controls = [
        model
        for model in compat.CONFIG.get("controls", [])
        if model["base_model"] == base_model
    ]
    if len(controls) != 4:
        raise RuntimeError(f"Expected four crossed controls for {base_model}")
    conditions = {}
    for model in controls:
        path = output_paths(model)["runs"]
        if not path.exists():
            raise FileNotFoundError(path)
        policy = "fixed" if model["response_policy"].startswith("fixed") else "on_policy"
        with path.open(newline="") as handle:
            conditions[(model["prompt_mode"], policy)] = list(csv.DictReader(handle))

    index_keys = (
        "experiment",
        "token_position",
        "split",
        "heldout_instruction",
        "normalization",
    )
    indexed = {
        condition: {tuple(row[key] for key in index_keys): row for row in rows}
        for condition, rows in conditions.items()
    }
    contrasts = []

    def append_pair(
        name,
        context,
        left_condition,
        right_condition,
        key,
        metric,
        right_key=None,
    ):
        right_key = key if right_key is None else right_key
        left = indexed[left_condition][key]
        right = indexed[right_condition][right_key]
        if left[metric] == "" or right[metric] == "":
            return
        metadata = dict(zip(index_keys, key, strict=True))
        if key[-1] != right_key[-1]:
            metadata["normalization"] = f"{key[-1]}_to_{right_key[-1]}"
        contrasts.append(
            {
                "base_model": base_model,
                "contrast": name,
                "context": context,
                **metadata,
                "metric": metric,
                "left_condition": "/".join(left_condition),
                "right_condition": "/".join(right_condition),
                "left_value": float(left[metric]),
                "right_value": float(right[metric]),
                "delta": float(right[metric]) - float(left[metric]),
            }
        )

    shared = set.intersection(*(set(rows) for rows in indexed.values()))
    for key in sorted(shared):
        normalization = key[-1]
        for metric in METRICS:
            for prompt in ("released_llama2", "native"):
                append_pair(
                    "response_policy",
                    f"prompt={prompt}",
                    (prompt, "on_policy"),
                    (prompt, "fixed"),
                    key,
                    metric,
                )
            for policy in ("on_policy", "fixed"):
                append_pair(
                    "prompt_template",
                    f"response={policy}",
                    ("released_llama2", policy),
                    ("native", policy),
                    key,
                    metric,
                )
            if normalization == "train_fitted":
                released_key = (*key[:-1], "released_splitwise")
                for condition in indexed:
                    append_pair(
                        "normalization",
                        "/".join(condition),
                        condition,
                        condition,
                        released_key,
                        metric,
                        right_key=key,
                    )

    for key in sorted(shared):
        for metric in METRICS:
            rows = {condition: indexed[condition][key] for condition in indexed}
            if any(row[metric] == "" for row in rows.values()):
                continue
            released = float(rows[("released_llama2", "fixed")][metric]) - float(
                rows[("released_llama2", "on_policy")][metric]
            )
            native = float(rows[("native", "fixed")][metric]) - float(
                rows[("native", "on_policy")][metric]
            )
            contrasts.append(
                {
                    "base_model": base_model,
                    "contrast": "prompt_response_interaction",
                    "context": "difference_in_response_policy_effects",
                    **dict(zip(index_keys, key, strict=True)),
                    "metric": metric,
                    "left_condition": "released_response_effect",
                    "right_condition": "native_response_effect",
                    "left_value": released,
                    "right_value": native,
                    "delta": native - released,
                }
            )

    grouped = defaultdict(list)
    for row in contrasts:
        grouped[
            (
                row["contrast"],
                row["context"],
                row["experiment"],
                row["token_position"],
                row["normalization"],
                row["metric"],
            )
        ].append(row)
    rng = np.random.default_rng(int(CONFIG["factorial_bootstrap_seed"]))
    replicates = int(CONFIG["bootstrap_samples"])
    summary = []
    for key, rows in grouped.items():
        clusters = defaultdict(list)
        for row in rows:
            cluster = row["heldout_instruction"] or f"task_seed_{row['split']}"
            clusters[cluster].append(float(row["delta"]))
        cluster_means = np.asarray([np.mean(values) for values in clusters.values()])
        draws = rng.choice(
            cluster_means, size=(replicates, len(cluster_means)), replace=True
        ).mean(axis=1)
        summary.append(
            {
                "base_model": base_model,
                **dict(
                    zip(
                        (
                            "contrast",
                            "context",
                            "experiment",
                            "token_position",
                            "normalization",
                            "metric",
                        ),
                        key,
                        strict=True,
                    )
                ),
                "mean_delta": float(cluster_means.mean()),
                "ci_lower": float(np.quantile(draws, 0.025)),
                "ci_upper": float(np.quantile(draws, 0.975)),
                "n_clusters": len(cluster_means),
                "bootstrap_replicates": replicates,
            }
        )
    paths = output_paths(compat.MODELS[base_model])
    write_csv(paths["contrasts"], contrasts)
    write_csv(paths["contrast_summary"], summary)
    return {
        "base_model": base_model,
        "contrast_rows": len(contrasts),
        "summary_rows": len(summary),
        "contrasts": str(paths["contrasts"]),
        "summary": str(paths["contrast_summary"]),
    }


def analyze(
    model_key: str,
    device: str = "cpu",
    normalization: str = "both",
    bootstrap_samples: int | None = None,
) -> dict[str, object]:
    import torch

    model = compat.MODELS[model_key]
    source = compat.paths(model)
    output = output_paths(model)
    if not source["compact"].exists() or not source["labels"].exists():
        raise RuntimeError(
            f"GPU-dependent activations or labels are missing for {model_key}; "
            "run extraction and labeling before submission analysis"
        )
    modes = NORMALIZATION_MODES if normalization == "both" else (normalization,)
    if any(mode not in NORMALIZATION_MODES for mode in modes):
        raise ValueError(f"Unknown normalization selection: {normalization}")
    payload = torch.load(source["compact"], map_location="cpu", weights_only=False)
    activations = payload["activations"]
    label_rows = compat.read_jsonl(source["labels"])
    labels = torch.tensor(
        [row["follow_all_instructions"] for row in label_rows], dtype=torch.float32
    )
    instruction_names = list(
        dict.fromkeys(row["instruction_id_list"][0] for row in label_rows)
    )
    instruction_ids = torch.tensor(
        [instruction_names.index(row["instruction_id_list"][0]) for row in label_rows]
    )
    block = compat.CONFIG["dataset"]["released_rows_per_instruction"]
    task_ids = torch.tensor([index % block for index in range(len(labels))])
    epochs = compat.CONFIG["paper_settings"]["epochs"]
    dev_fraction = float(CONFIG["development_fraction"])
    bootstrap_samples = (
        int(CONFIG["bootstrap_samples"])
        if bootstrap_samples is None
        else bootstrap_samples
    )
    runs: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    uncertainty: list[dict[str, object]] = []

    split_specs = []
    for seed in compat.CONFIG["paper_settings"]["replication_seeds"]:
        generator = torch.Generator().manual_seed(seed)
        ordered = torch.randperm(block, generator=generator)
        train_count = int(
            compat.CONFIG["paper_settings"]["task_train_fraction"] * block
        )
        outer_train = torch.isin(task_ids, ordered[:train_count])
        test = ~outer_train
        development = development_mask(outer_train, task_ids, dev_fraction, seed)
        split_specs.append(
            (
                "task_generalization",
                seed,
                "",
                outer_train & ~development,
                development,
                test,
            )
        )
    instruction_seed = compat.CONFIG["paper_settings"]["instruction_probe_seed"]
    for heldout, name in enumerate(instruction_names):
        outer_train = instruction_ids != heldout
        test = ~outer_train
        development = development_mask(
            outer_train, task_ids, dev_fraction, instruction_seed + heldout
        )
        split_specs.append(
            (
                "instruction_type_generalization",
                heldout,
                name,
                outer_train & ~development,
                development,
                test,
            )
        )

    for layer_index, layer in enumerate(model["layers"]):
        for token_index, token in enumerate(compat.TOKENS):
            values = activations[:, token_index, layer_index]
            for experiment, split, heldout, train, development, test in split_specs:
                for mode in modes:
                    train_x, development_x, test_x = normalize_partitions(
                        values[train], values[development], values[test], mode
                    )
                    torch.manual_seed(int(split))
                    combined = torch.cat((development_x, test_x))
                    combined_probabilities, _ = compat.fit_probe(
                        train_x, labels[train], combined, device, epochs
                    )
                    dev_count = int(development.sum())
                    development_probabilities = combined_probabilities[:dev_count]
                    test_probabilities = combined_probabilities[dev_count:]
                    thresholds = frozen_thresholds(
                        labels[development].numpy(), development_probabilities.numpy()
                    )
                    metrics = metric_values(
                        labels[test].numpy(), test_probabilities.numpy(), thresholds
                    )
                    run_id = f"{model_key}|{experiment}|L{layer}|{token}|{split}|{mode}"
                    run = {
                        "run_id": run_id,
                        "model_key": model_key,
                        "model": model["paper_name"],
                        "model_revision": model["revision"],
                        "experiment": experiment,
                        "layer": layer,
                        "token_position": token,
                        "split": split,
                        "heldout_instruction": heldout,
                        "normalization": mode,
                        "threshold": thresholds["balanced_accuracy"],
                        "threshold_source": "development_balanced_accuracy",
                        "dev_fpr_0_01_threshold": thresholds["fpr_0_01"],
                        "dev_fpr_0_05_threshold": thresholds["fpr_0_05"],
                        "fpr_resolution": 1 / max(
                            1, int((labels[test] == 0).sum())
                        ),
                        "n_train": int(train.sum()),
                        "n_development": dev_count,
                        "n_test": int(test.sum()),
                        **metrics,
                        "status": "complete"
                        if metrics["auroc"] != ""
                        else "single_class",
                    }
                    runs.append(run)
                    test_indices = torch.where(test)[0]
                    test_labels = labels[test]
                    test_predictions = (
                        test_probabilities >= thresholds["balanced_accuracy"]
                    )
                    for index, label, probability, prediction in zip(
                        test_indices,
                        test_labels,
                        test_probabilities,
                        test_predictions,
                        strict=True,
                    ):
                        row = label_rows[int(index)]
                        predictions.append(
                            {
                                "run_id": run_id,
                                "model_key": model_key,
                                "model_revision": model["revision"],
                                "sample_index": int(index),
                                "task_id": int(task_ids[index]),
                                "instruction_type": row["instruction_id_list"][0],
                                "prompt_sha256": hashlib.sha256(
                                    row["prompt"].encode()
                                ).hexdigest(),
                                "response_sha256": hashlib.sha256(
                                    row.get("response", "").encode()
                                ).hexdigest(),
                                "label": int(label),
                                "probability": float(probability),
                                "prediction": int(prediction),
                                "threshold": thresholds["balanced_accuracy"],
                                "dev_fpr_0_01_threshold": thresholds["fpr_0_01"],
                                "prediction_at_dev_fpr_0_01": int(
                                    probability >= thresholds["fpr_0_01"]
                                ),
                                "dev_fpr_0_05_threshold": thresholds["fpr_0_05"],
                                "prediction_at_dev_fpr_0_05": int(
                                    probability >= thresholds["fpr_0_05"]
                                ),
                            }
                        )
                    if bootstrap_samples:
                        intervals = cluster_bootstrap(
                            test_labels.numpy(),
                            test_probabilities.numpy(),
                            task_ids[test].numpy(),
                            thresholds,
                            bootstrap_samples,
                            int(split) + layer * 100,
                        )
                        uncertainty.extend(
                            {
                                "run_id": run_id,
                                "scope": "within_run",
                                "cluster": "task_id",
                                **row,
                            }
                            for row in intervals
                        )

    if bootstrap_samples:
        instruction_groups: dict[tuple[int, str, str], list[dict[str, object]]] = (
            defaultdict(list)
        )
        for row in runs:
            if row["experiment"] == "instruction_type_generalization":
                instruction_groups[
                    (
                        int(row["layer"]),
                        str(row["token_position"]),
                        str(row["normalization"]),
                    )
                ].append(row)
        for (layer, token, mode), rows in instruction_groups.items():
            run_id = f"{model_key}|instruction_type_generalization|L{layer}|{token}|aggregate|{mode}"
            uncertainty.extend(
                {
                    "run_id": run_id,
                    "scope": "across_instruction_types",
                    "cluster": "instruction_type",
                    **row,
                }
                for row in grouped_bootstrap(
                    rows, bootstrap_samples, seed=layer * 100 + len(token)
                )
            )

    write_csv(output["runs"], runs)
    write_csv(output["predictions"], predictions)
    if uncertainty:
        write_csv(output["uncertainty"], uncertainty)
    provenance = {
        "analysis_script_sha256": file_digest(Path(__file__)),
        "compat_script_sha256": file_digest(compat.HERE / "compat.py"),
        "run_script_sha256": file_digest(compat.HERE / "run.py"),
        "configuration_sha256": compat.configuration_digest(),
        "configuration_files": [
            str(path.relative_to(compat.ROOT)) for path in compat.CONFIG_FILES
        ],
        "model": {
            "key": model_key,
            "id": model["id"],
            "revision": model["revision"],
            "architecture": model.get("architecture"),
            "layers": model["layers"],
        },
        "condition": {
            "scope": model.get("scope", "baseline"),
            "base_model": model.get("base_model", model_key),
            "prompt_mode": model.get("prompt_mode", "released_llama2"),
            "prompt_provenance": model.get(
                "prompt_provenance",
                "released hard-coded Llama-2 instruction template",
            ),
            "response_source": model.get("response_source", model_key),
            "response_policy": model.get("response_policy", f"on_policy_{model_key}"),
            "extraction_mode": model.get("extraction_mode", "released_generation"),
        },
        "analysis": {
            **CONFIG,
            "normalization_modes_run": list(modes),
            "bootstrap_samples_run": bootstrap_samples,
            "paper_comparison_outputs_unchanged": True,
        },
        "sources": {
            "apple_commit": compat.CONFIG["commit"],
            "apple_digest": compat.CONFIG["source_digest"],
            "ifeval_commit": compat.CONFIG["evaluator"]["commit"],
            "ifeval_digest": compat.CONFIG["evaluator"]["source_digest"],
        },
        "inputs": {
            "dataset_sha256": file_digest(
                compat.APPLE / "data" / "ifeval_simple.jsonl"
            ),
            "responses_sha256": file_digest(source["responses"]),
            "labels_sha256": file_digest(source["labels"]),
            "activations_sha256": file_digest(source["compact"]),
        },
        "outputs": {
            key: {"path": str(path), "sha256": file_digest(path)}
            for key, path in output.items()
            if key != "manifest" and path.exists()
        },
        "counts": {
            "runs": len(runs),
            "predictions": len(predictions),
            "uncertainty_rows": len(uncertainty),
        },
    }
    output["manifest"].write_text(json.dumps(provenance, indent=2) + "\n")
    compat.stamp(
        source["manifest"],
        "submission_analyzed",
        submission_manifest=str(output["manifest"]),
        submission_normalization_modes=list(modes),
        submission_threshold="development_balanced_accuracy",
        submission_bootstrap_samples=bootstrap_samples,
    )
    return provenance


def plan() -> list[dict[str, object]]:
    rows = []
    for model in compat.MODELS.values():
        source = compat.paths(model)
        ready = source["compact"].exists() and source["labels"].exists()
        rows.append(
            {
                "model_key": model["key"],
                "scope": model.get("scope", "baseline"),
                "base_model": model.get("base_model", model["key"]),
                "prompt_mode": model.get("prompt_mode", "released_llama2"),
                "response_source": model.get("response_source", model["key"]),
                "analysis_runnable_now": ready,
                "required_next_stage": "submission-analysis"
                if ready
                else "GPU extraction",
            }
        )
    return rows
