from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean


Number = int | float


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    tpr: float
    fpr: float
    tn: int
    fp: int
    fn: int
    tp: int
    n_positive: int
    n_negative: int

    @property
    def fpr_resolution(self) -> float:
        return 1 / self.n_negative


@dataclass(frozen=True)
class BinaryMetrics:
    n: int
    n_positive: int
    n_negative: int
    threshold: float
    threshold_space: str
    auroc: float | None
    auprc: float | None
    accuracy: float
    balanced_accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    brier: float | None
    ece: float | None
    tn: int
    fp: int
    fn: int
    tp: int
    roc_tpr_at_fpr_0_01: float | None
    roc_achieved_fpr_0_01: float | None
    roc_fp_at_fpr_0_01: int | None
    roc_threshold_at_fpr_0_01: float | None
    roc_tpr_at_fpr_0_05: float | None
    roc_achieved_fpr_0_05: float | None
    roc_fp_at_fpr_0_05: int | None
    roc_threshold_at_fpr_0_05: float | None

    def to_dict(self) -> dict[str, Number | str | None]:
        return asdict(self)


def _validated(labels: Sequence[int], values: Sequence[Number]) -> tuple[list[int], list[float]]:
    if len(labels) != len(values) or not labels:
        raise ValueError("labels and values must have equal non-zero length")
    clean_labels = [int(value) for value in labels]
    if set(clean_labels) - {0, 1}:
        raise ValueError("binary labels must be 0 or 1")
    clean_values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in clean_values):
        raise ValueError("values must be finite")
    return clean_labels, clean_values


def confusion(labels: Sequence[int], values: Sequence[Number], threshold: float) -> tuple[int, int, int, int]:
    labels, values = _validated(labels, values)
    predictions = [int(value >= threshold) for value in values]
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    return tn, fp, fn, tp


def auroc(labels: Sequence[int], scores: Sequence[Number]) -> float | None:
    labels, scores = _validated(labels, scores)
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        rank_sum += average_rank * sum(labels[index] for index in order[cursor:end])
        cursor = end
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def auprc(labels: Sequence[int], scores: Sequence[Number]) -> float | None:
    labels, scores = _validated(labels, scores)
    positive = sum(labels)
    if not positive:
        return None
    average_precision = 0.0
    previous_recall = 0.0
    for threshold in sorted(set(scores), reverse=True):
        selected = [index for index, score in enumerate(scores) if score >= threshold]
        true_positives = sum(labels[index] for index in selected)
        precision = true_positives / len(selected)
        recall = true_positives / positive
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def operating_point_at_fpr(
    labels: Sequence[int], scores: Sequence[Number], target_fpr: float
) -> OperatingPoint | None:
    labels, scores = _validated(labels, scores)
    if not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be in [0, 1]")
    if not sum(labels) or sum(label == 0 for label in labels) == 0:
        return None
    candidates = [math.inf, *sorted(set(scores), reverse=True)]
    feasible: list[OperatingPoint] = []
    for threshold in candidates:
        tn, fp, fn, tp = confusion(labels, scores, threshold)
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        if fpr <= target_fpr:
            feasible.append(
                OperatingPoint(
                    threshold=threshold,
                    tpr=tpr,
                    fpr=fpr,
                    tn=tn,
                    fp=fp,
                    fn=fn,
                    tp=tp,
                    n_positive=tp + fn,
                    n_negative=tn + fp,
                )
            )
    return max(feasible, key=lambda point: (point.tpr, -point.fpr, point.threshold))


def select_youden_threshold(labels: Sequence[int], probabilities: Sequence[Number]) -> float:
    labels, probabilities = _validated(labels, probabilities)
    if not sum(labels) or sum(label == 0 for label in labels) == 0:
        raise ValueError("Youden threshold requires both classes")
    candidates = [math.inf, *sorted(set(probabilities), reverse=True)]
    best = None
    for threshold in candidates:
        tn, fp, fn, tp = confusion(labels, probabilities, threshold)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        candidate = (tpr + tnr - 1, threshold)
        if best is None or candidate > best:
            best = candidate
    return float(best[1])


def apply_frozen_threshold(
    labels: Sequence[int], decision_values: Sequence[Number], threshold: float
) -> dict[str, Number | None]:
    labels, decision_values = _validated(labels, decision_values)
    tn, fp, fn, tp = confusion(labels, decision_values, threshold)
    n_positive = tp + fn
    n_negative = tn + fp
    return {
        "threshold": float(threshold),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr": tp / n_positive if n_positive else None,
        "fpr": fp / n_negative if n_negative else None,
        "fpr_resolution": 1 / n_negative if n_negative else None,
    }


def expected_calibration_error(
    labels: Sequence[int], probabilities: Sequence[Number], bins: int = 10
) -> float:
    labels, probabilities = _validated(labels, probabilities)
    if bins < 2 or any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("ECE requires probabilities in [0, 1] and at least two bins")
    total = len(labels)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper or (bin_index == bins - 1 and probability == 1)
        ]
        if members:
            confidence = fmean(probabilities[index] for index in members)
            accuracy = fmean(labels[index] for index in members)
            error += len(members) / total * abs(accuracy - confidence)
    return error


def binary_metrics(
    labels: Sequence[int],
    scores: Sequence[Number],
    *,
    probabilities: Sequence[Number] | None = None,
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> BinaryMetrics:
    labels, scores = _validated(labels, scores)
    if probabilities is None:
        probabilities = scores if all(0 <= value <= 1 for value in scores) else None
    probability_values = None
    if probabilities is not None:
        _, probability_values = _validated(labels, probabilities)
        if any(value < 0 or value > 1 for value in probability_values):
            raise ValueError("probabilities must be in [0, 1]")
    tn, fp, fn, tp = confusion(labels, scores, threshold)
    positive = tp + fn
    negative = tn + fp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positive if positive else None
    specificity = tn / negative if negative else None
    balanced_accuracy = (
        (recall + specificity) / 2 if recall is not None and specificity is not None else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    point_001 = operating_point_at_fpr(labels, scores, 0.01)
    point_005 = operating_point_at_fpr(labels, scores, 0.05)
    return BinaryMetrics(
        n=len(labels),
        n_positive=positive,
        n_negative=negative,
        threshold=float(threshold),
        threshold_space="decision_score",
        auroc=auroc(labels, scores),
        auprc=auprc(labels, scores),
        accuracy=(tp + tn) / len(labels),
        balanced_accuracy=balanced_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        brier=(
            fmean((probability - label) ** 2 for probability, label in zip(probability_values, labels))
            if probability_values is not None
            else None
        ),
        ece=(
            expected_calibration_error(labels, probability_values, ece_bins)
            if probability_values is not None
            else None
        ),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        roc_tpr_at_fpr_0_01=point_001.tpr if point_001 else None,
        roc_achieved_fpr_0_01=point_001.fpr if point_001 else None,
        roc_fp_at_fpr_0_01=point_001.fp if point_001 else None,
        roc_threshold_at_fpr_0_01=point_001.threshold if point_001 else None,
        roc_tpr_at_fpr_0_05=point_005.tpr if point_005 else None,
        roc_achieved_fpr_0_05=point_005.fpr if point_005 else None,
        roc_fp_at_fpr_0_05=point_005.fp if point_005 else None,
        roc_threshold_at_fpr_0_05=point_005.threshold if point_005 else None,
    )


def cluster_bootstrap(
    labels: Sequence[int],
    values: Sequence[Number],
    clusters: Sequence[Hashable],
    metric: Callable[[Sequence[int], Sequence[float]], float | None],
    *,
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Number | None]:
    labels, values = _validated(labels, values)
    if len(clusters) != len(labels) or not 0 < confidence < 1 or replicates < 1:
        raise ValueError("invalid cluster bootstrap inputs")
    members: dict[Hashable, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        members[cluster].append(index)
    keys = list(members)
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(keys) for _ in keys]
        indices = [index for key in sampled for index in members[key]]
        estimate = metric([labels[index] for index in indices], [values[index] for index in indices])
        if estimate is not None and math.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return {"estimate": metric(labels, values), "ci_low": None, "ci_high": None, "valid": 0}
    estimates.sort()
    alpha = (1 - confidence) / 2
    low = estimates[max(0, math.floor(alpha * len(estimates)))]
    high = estimates[min(len(estimates) - 1, math.ceil((1 - alpha) * len(estimates)) - 1)]
    return {
        "estimate": metric(labels, values),
        "ci_low": low,
        "ci_high": high,
        "valid": len(estimates),
    }
