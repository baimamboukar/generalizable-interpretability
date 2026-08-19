from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "external" / "tpc"


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def probe_name(call: int) -> str:
    if call == 1:
        return "tpc"
    if call == 2:
        return "bilinear"
    if call == 3:
        return "linear"
    if 4 <= call <= 8:
        return f"mlp_budget_{call - 3}"
    if call == 9:
        return "ee_mlp"
    return f"evaluation_{call}"


def predictions(classifier, loader, order: int, device) -> tuple[np.ndarray, np.ndarray]:
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    classifier.eval()
    with torch.no_grad():
        for features, target in loader:
            logits = classifier(
                features.to(device, non_blocking=True), test_time_order=order
            )[-1].squeeze(-1)
            scores.append(torch.sigmoid(logits).cpu())
            labels.append(target.cpu())
    return torch.cat(labels).numpy().astype(int), torch.cat(scores).numpy()


def metric_row(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    predicted = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": 0.5,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auroc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def install_instrumentation(metrics_path: Path, predictions_path: Path) -> None:
    sys.path.insert(0, str(SOURCE))
    import train as upstream_train

    original = upstream_train.ee_sweep
    counter = 0

    def measured(cls_all, train_loader, val_loader, test_loader, x_vals, metric, device=torch.device("cuda")):
        nonlocal counter
        counter += 1
        result = original(
            cls_all,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            x_vals=x_vals,
            metric=metric,
            device=device,
        )
        probe = probe_name(counter)
        metric_rows: list[dict] = []
        prediction_rows: list[dict] = []
        for seed, classifier in enumerate(cls_all):
            for order in x_vals:
                for split, loader in (("validation", val_loader), ("test", test_loader)):
                    labels, scores = predictions(classifier, loader, order, device)
                    row = {
                        "probe": probe,
                        "seed": seed,
                        "order": int(order),
                        "split": split,
                    } | metric_row(labels, scores)
                    metric_rows.append(row)
                    if split == "test":
                        prediction_rows.extend(
                            {
                                "probe": probe,
                                "seed": seed,
                                "order": int(order),
                                "row": index,
                                "label": int(label),
                                "score": float(score),
                                "prediction": int(score >= 0.5),
                            }
                            for index, (label, score) in enumerate(zip(labels, scores))
                        )
        append_jsonl(metrics_path, metric_rows)
        append_jsonl(predictions_path, prediction_rows)
        return result

    upstream_train.ee_sweep = measured


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--predictions-path", type=Path, required=True)
    known, upstream_args = parser.parse_known_args()
    for path in (known.metrics_path, known.predictions_path):
        path.unlink(missing_ok=True)
    install_instrumentation(known.metrics_path, known.predictions_path)
    sys.argv = [str(SOURCE / "sweep_monitors.py"), *upstream_args]
    runpy.run_path(str(SOURCE / "sweep_monitors.py"), run_name="__main__")


if __name__ == "__main__":
    main()
