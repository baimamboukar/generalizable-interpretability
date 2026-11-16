from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import time
from typing import Annotated

import numpy as np
from fastmcp import FastMCP


# The released feature files are pickles that torch >= 2.6 refuses by default; they are the authors' own release.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

mcp = FastMCP(name = "lookback_lens")


def load_source(repository: Path):
    script = repository / "step03_lookback_lens.py"
    if not script.is_file():
        raise ValueError(f"Missing released probe script: {script}")
    spec = importlib.util.spec_from_file_location("lookback_lens_source", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset(source, annotation, features, tokenizer, regime):
    predefined = regime == "predefined_span"
    values, labels = source.load_files(
        str(annotation),
        str(features),
        predefined_span = predefined,
        tokenizer_name = str(tokenizer),
    )
    if not predefined:
        values, labels = source.convert_to_token_level(
            values,
            labels,
            sliding_window = 8,
            sequential = True,
            min_pool_target = True,
        )
    matrix, _, _ = source.extract_time_series_features(values)
    return matrix, np.asarray(labels)  # window labels come back as a list, which cannot be indexed by fold


def write_predictions(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", newline = "") as handle:
        writer = csv.DictWriter(handle, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@mcp.tool()
def lookback_lens_fit_and_evaluate(
    repository_dir: Annotated[str, "Pinned Lookback Lens repository."],
    nq_annotations: Annotated[str, "Natural Questions annotation JSONL."],
    nq_features: Annotated[str, "Natural Questions lookback-ratio features."],
    cnndm_annotations: Annotated[str, "CNN/DailyMail annotation JSONL."],
    cnndm_features: Annotated[str, "CNN/DailyMail lookback-ratio features."],
    tokenizer_dir: Annotated[str, "Local Llama-2-7B tokenizer."],
    output_dir: Annotated[str, "Base output directory."],
) -> dict:
    """Fit Lookback Lens probes and evaluate both cross-dataset directions."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    started = time.time()
    repository = Path(repository_dir).expanduser().resolve()
    tokenizer = Path(tokenizer_dir).expanduser().resolve()
    paths = {
        "nq": (
            Path(nq_annotations).expanduser().resolve(),
            Path(nq_features).expanduser().resolve(),
        ),
        "cnndm": (
            Path(cnndm_annotations).expanduser().resolve(),
            Path(cnndm_features).expanduser().resolve(),
        ),
    }
    for annotation, features in paths.values():
        if not annotation.is_file() or not features.is_file():
            raise ValueError("An annotation or feature file is missing")
    if not tokenizer.is_dir():
        raise ValueError(f"Tokenizer directory is missing: {tokenizer}")

    output = Path(tempfile.mkdtemp(
        prefix = "lookback-lens-",
        dir = Path(output_dir).expanduser().resolve(),
    ))
    source = load_source(repository)
    results = []
    prediction_rows = []

    for regime in ("predefined_span", "sliding_window_8"):
        datasets = {
            name: load_dataset(source, annotation, features, tokenizer, regime)
            for name, (annotation, features) in paths.items()
        }
        for train_name, transfer_name in (("nq", "cnndm"), ("cnndm", "nq")):
            values, labels = datasets[train_name]
            transfer_values, transfer_labels = datasets[transfer_name]
            folds = train_test_split(
                np.arange(len(labels)),
                test_size = 0.5,
                random_state = 42,
            )
            train_scores = []
            test_scores = []
            for fold, (train, test) in enumerate((folds, folds[::-1])):
                probe = LogisticRegression(max_iter = 1000).fit(values[train], labels[train])
                for split, indices in (("train", train), ("test", test)):
                    probabilities = probe.predict_proba(values[indices])[:, 1]
                    score = float(roc_auc_score(labels[indices], probabilities))
                    if split == "train":
                        train_scores.append(score)
                    else:
                        test_scores.append(score)
                    for row, probability in zip(indices, probabilities):
                        prediction_rows.append({
                            "regime": regime,
                            "train_dataset": train_name,
                            "evaluation_dataset": train_name,
                            "split": split,
                            "fold": fold,
                            "row": int(row),
                            "label": int(labels[row]),
                            "probability": float(probability),
                        })

            final_probe = LogisticRegression(max_iter = 1000).fit(values, labels)
            train_probabilities = final_probe.predict_proba(values)[:, 1]
            transfer_probabilities = final_probe.predict_proba(transfer_values)[:, 1]
            result = {
                "regime": regime,
                "train_dataset": train_name,
                "transfer_dataset": transfer_name,
                "train_auroc": float(np.mean(train_scores)),
                "test_auroc": float(np.mean(test_scores)),
                "transfer_auroc": float(roc_auc_score(transfer_labels, transfer_probabilities)),
            }
            results.append(result)
            with (output / f"probe_{regime}_{train_name}.pkl").open("wb") as handle:
                pickle.dump(final_probe, handle)
            for row, probability in enumerate(transfer_probabilities):
                prediction_rows.append({
                    "regime": regime,
                    "train_dataset": train_name,
                    "evaluation_dataset": transfer_name,
                    "split": "transfer",
                    "fold": "",
                    "row": row,
                    "label": int(transfer_labels[row]),
                    "probability": float(probability),
                })

    with (output / "results.csv").open("w", newline = "") as handle:
        writer = csv.DictWriter(handle, fieldnames = list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_predictions(output / "predictions.csv.gz", prediction_rows)
    runtime = time.time() - started
    summary = {"auroc_value_count": 12, "probe_count": 4, "runtime_seconds": runtime}
    (output / "summary.json").write_text(json.dumps(summary, indent = 2) + "\n")
    (output / "run.log").write_text(
        f"fitted_probes=4\nruntime_seconds={runtime:.6f}\n"
    )
    (output / "REPORT.md").write_text(
        "# Lookback Lens MCP Run\n\n"
        "The MCP fitted four logistic probes from the released lookback-ratio features.\n\n"
        f"Runtime: {runtime:.1f} seconds.\n"
    )
    shutil.copy(__file__, output / "server.py")  # the code that produced these results
    return {
        "output_dir": str(output),
        "summary": summary,
        "artifacts": {
            "implementation": str(output / "server.py"),
            "probes": [str(path) for path in sorted(output.glob("probe_*.pkl"))],
            "results": str(output / "results.csv"),
            "predictions": str(output / "predictions.csv.gz"),
            "report": str(output / "REPORT.md"),
            "log": str(output / "run.log"),
        },
    }


if __name__ == "__main__":
    mcp.run(show_banner = False)
