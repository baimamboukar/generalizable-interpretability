from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
APPLE = ROOT / "external" / "internal-instruction-following"
GOOGLE = ROOT / "external" / "google-research-ifeval"
IFEVAL = GOOGLE / "instruction_following_eval"
CONFIG = json.loads((HERE / "upstream.json").read_text())
MODELS = {model["key"]: model for model in CONFIG["models"]}
TOKENS = ("first", "middle", "last")
THRESHOLD = float(CONFIG["diagnostic_metrics"]["threshold"])
METRICS = ("auroc", *CONFIG["diagnostic_metrics"]["derived"])


def upstream_saved_layers(num_hidden_layers: int) -> list[int]:
    count = num_hidden_layers + 1
    layer = count - 1
    leap = count // 5
    layers = []
    while layer > 10:
        layers.append(layer)
        layer -= leap
    return layers


def validate_extension_models() -> None:
    for model in MODELS.values():
        if model.get("scope") != "extension":
            continue
        layers = model["layers"]
        invalid = [
            layer
            for layer in layers
            if not isinstance(layer, int)
            or not 0 <= layer <= model["num_hidden_layers"]
        ]
        if len(layers) != 4 or len(set(layers)) != 4 or invalid:
            raise RuntimeError(f"Invalid extension layers for {model['key']}: {layers}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def tree_digest(root: Path) -> str | None:
    if not root.exists():
        return None
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".DS_Store"
    )
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def verify_sources() -> dict[str, str | None]:
    state = {
        "apple_commit": git_commit(APPLE),
        "apple_digest": tree_digest(APPLE),
        "ifeval_commit": git_commit(GOOGLE),
        "ifeval_digest": tree_digest(IFEVAL),
    }
    expected = {
        "apple_digest": CONFIG["source_digest"],
        "ifeval_digest": CONFIG["evaluator"]["source_digest"],
    }
    mismatches = [key for key, value in expected.items() if state[key] != value]
    if mismatches:
        values = ", ".join(f"{key}={state[key]}" for key in mismatches)
        raise RuntimeError(f"Pinned source mismatch: {values}")
    return state


def artifact_root() -> Path:
    base = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts"))
    return base / "internal-instruction-following"


def paths(model: dict) -> dict[str, Path]:
    root = artifact_root() / model["key"]
    work = root / "upstream_data"
    generated = work / model["id"].split("/")[-1] / "ifeval_simple"
    return {
        "root": root,
        "work": work,
        "input": root / "ifeval_input.jsonl",
        "responses": root / "input_response.jsonl",
        "raw_responses": generated / "response.jsonl",
        "raw_activations": generated / "activations",
        "labels": root / "eval_results_loose.jsonl",
        "compact": root / "activations.pt",
        "runs": root / "probe_runs.csv",
        "predictions": root / "probe_predictions.csv",
        "summary": root / "probe_summary.csv",
        "metrics_summary": root / "probe_metrics_summary.csv",
        "directions": root / "probe_directions.pt",
        "manifest": root / "run.json",
    }


def stamp(path: Path, stage: str, **values) -> None:
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(values)
    state["stage"] = stage
    state["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(state, indent=2) + "\n")


def validate_dataset(rows: list[dict]) -> dict[str, object]:
    counts = Counter(row["instruction_id_list"][0] for row in rows)
    expected = CONFIG["dataset"]
    if len(rows) != expected["released_rows"]:
        raise RuntimeError(
            f"Expected {expected['released_rows']} rows, found {len(rows)}"
        )
    if set(counts.values()) != {expected["released_rows_per_instruction"]}:
        raise RuntimeError(f"Unexpected instruction counts: {dict(counts)}")
    block = expected["released_rows_per_instruction"]
    sequence = [
        rows[index]["instruction_id_list"][0] for index in range(0, len(rows), block)
    ]
    if len(sequence) != expected["instruction_types"] or len(set(sequence)) != len(
        sequence
    ):
        raise RuntimeError(
            "IFEval-simple is not ordered as five contiguous task blocks"
        )
    return {
        "rows": len(rows),
        "instruction_counts": dict(counts),
        "block_order": sequence,
    }


def prepare(model_key: str) -> dict[str, object]:
    verify_sources()
    model = MODELS[model_key]
    target = paths(model)
    target["root"].mkdir(parents=True, exist_ok=True)
    target["work"].mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(APPLE / "data" / "ifeval_simple.jsonl")
    dataset = validate_dataset(rows)
    shutil.copy2(
        APPLE / "data" / "ifeval_simple.jsonl", target["work"] / "ifeval_simple.jsonl"
    )
    evaluator_rows = []
    for index, row in enumerate(rows):
        adapted = {"key": index, **row}
        if isinstance(adapted["kwargs"], dict):
            adapted["kwargs"] = [adapted["kwargs"]]
        evaluator_rows.append(adapted)
    write_jsonl(target["input"], evaluator_rows)
    stamp(
        target["manifest"],
        "prepared",
        model=model,
        paper=CONFIG["paper"],
        protocol=CONFIG["paper_settings"],
        dataset=dataset,
        source={
            "apple_commit": CONFIG["commit"],
            "apple_digest": CONFIG["source_digest"],
            "ifeval_commit": CONFIG["evaluator"]["commit"],
            "ifeval_digest": CONFIG["evaluator"]["source_digest"],
        },
        evaluator_schema_adapter="dictionary kwargs wrapped as a one-element list",
    )
    return dataset


def extraction_complete(target: dict[str, Path], rows: int) -> bool:
    if not target["raw_responses"].exists() or not target["raw_activations"].exists():
        return False
    responses = read_jsonl(target["raw_responses"])
    activations = list(target["raw_activations"].glob("sample_*.pkl"))
    return len(responses) == rows and len(activations) == rows


def training_complete(model: dict, target: dict[str, Path]) -> bool:
    required = (
        target["compact"],
        target["labels"],
        target["runs"],
        target["predictions"],
        target["summary"],
        target["metrics_summary"],
        target["directions"],
    )
    if any(not path.exists() or path.stat().st_size == 0 for path in required):
        return False
    expected_cells = len(model["layers"]) * len(TOKENS)
    expected_runs = expected_cells * (
        len(CONFIG["paper_settings"]["replication_seeds"])
        + CONFIG["dataset"]["instruction_types"]
    )
    with target["runs"].open() as handle:
        runs = list(csv.DictReader(handle))
    with target["summary"].open() as handle:
        summary = list(csv.DictReader(handle))
    return len(runs) == expected_runs and len(summary) == expected_cells * 2


def extract(model_key: str, force: bool = False) -> None:
    dataset = prepare(model_key)
    model = MODELS[model_key]
    target = paths(model)
    if extraction_complete(target, dataset["rows"]):
        print(f"Extraction already complete for {model_key}")
        return
    generated = target["raw_responses"].parent
    if generated.exists():
        if not force:
            raise RuntimeError(
                "Partial extraction exists; rerun with --force to replace it"
            )
        shutil.rmtree(generated)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    env = os.environ.copy()
    runtime = str(HERE / "runtime")
    env["PYTHONPATH"] = runtime + os.pathsep + env.get("PYTHONPATH", "")
    env["GI_MODEL_REVISION"] = model["revision"]
    env["TOKENIZERS_PARALLELISM"] = "false"
    entrypoint = APPLE / "run" / "save_LLMs_activations.py"
    if model.get("scope") == "extension":
        env["GI_TARGET_LAYERS"] = ",".join(map(str, model["layers"]))
        command = [
            sys.executable,
            str(HERE / "runtime" / "upstream_entrypoint.py"),
            str(entrypoint),
        ]
    else:
        command = [sys.executable, str(entrypoint)]
    subprocess.run(
        command
        + [
            "--data_path",
            str(target["work"]),
            "--model_name_hf",
            model["id"],
            "--task_type",
            "ifeval_simple",
        ],
        cwd=APPLE,
        env=env,
        check=True,
    )
    if not extraction_complete(target, dataset["rows"]):
        raise RuntimeError("Extraction finished without all expected artifacts")
    rows = read_jsonl(target["raw_responses"])
    write_jsonl(
        target["responses"],
        [{"prompt": row["prompt"], "response": row["output"]} for row in rows],
    )
    stamp(target["manifest"], "extracted", response_rows=len(rows))


def label(model_key: str) -> None:
    verify_sources()
    model = MODELS[model_key]
    target = paths(model)
    if not target["responses"].exists():
        raise RuntimeError("Run extraction first")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(GOOGLE) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "instruction_following_eval.evaluation_main",
            f"--input_data={target['input']}",
            f"--input_response_data={target['responses']}",
            f"--output_dir={target['root']}",
        ],
        cwd=GOOGLE,
        env=env,
        check=True,
    )
    rows = read_jsonl(target["labels"])
    if len(rows) != CONFIG["dataset"]["released_rows"]:
        raise RuntimeError(f"Expected 510 labels, found {len(rows)}")
    stamp(
        target["manifest"],
        "labeled",
        label_rows=len(rows),
        success_rate=sum(row["follow_all_instructions"] for row in rows) / len(rows),
    )


def compact(model_key: str, keep_raw: bool = False) -> None:
    import torch

    model = MODELS[model_key]
    target = paths(model)
    expected = CONFIG["dataset"]["released_rows"]
    if target["compact"].exists():
        print(f"Compact activations already exist for {model_key}")
        return
    files = [
        target["raw_activations"] / f"sample_{index}.pkl" for index in range(expected)
    ]
    missing = [path.name for path in files if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} activation files")
    samples = []
    for path in files:
        with path.open("rb") as handle:
            record = pickle.load(handle)
        sample = []
        for token in TOKENS:
            vectors = []
            for layer in model["layers"]:
                value = record[f"output_token_{token}"][f"layer_{layer}"]
                vectors.append(value[:, -1].squeeze(0).detach().float().cpu())
            sample.append(torch.stack(vectors))
        samples.append(torch.stack(sample))
    activations = torch.stack(samples)
    torch.save(
        {
            "model": model,
            "tokens": TOKENS,
            "layers": model["layers"],
            "activations": activations,
        },
        target["compact"],
    )
    if not keep_raw:
        shutil.rmtree(target["raw_activations"])
    stamp(
        target["manifest"],
        "compacted",
        activation_shape=list(activations.shape),
        raw_activations_retained=keep_raw,
    )


def normalize(values):
    values = values - values.mean(dim=0)
    values = values / values.std(dim=0)
    if not values.isfinite().all():
        raise RuntimeError("Non-finite values after released-notebook normalization")
    return values


def fit_probe(train_x, train_y, test_x, device: str, epochs: int):
    import torch

    class Probe(torch.nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(width, 1, bias=False),
                torch.nn.Sigmoid(),
            )

        def forward(self, values):
            return self.net(values).squeeze(-1)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    probe = Probe(train_x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=CONFIG["paper_settings"]["learning_rate"],
        weight_decay=CONFIG["paper_settings"]["weight_decay"],
    )
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.BCELoss()(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = probe(test_x.to(device)).cpu()
    return probabilities, probe.net[0].weight.detach().cpu()[0]


def classification_metrics(labels, probabilities) -> dict[str, object]:
    import torch

    labels = labels.detach().cpu().to(dtype=torch.int64)
    predictions = (probabilities.detach().cpu() >= THRESHOLD).to(dtype=torch.int64)
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())

    def rate(numerator: int, denominator: int) -> float | str:
        return numerator / denominator if denominator else ""

    accuracy = (tn + tp) / len(labels)
    precision = rate(tp, tp + fp)
    recall = rate(tp, tp + fn)
    specificity = rate(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision != "" and recall != "" and precision + recall
        else 0.0
    )
    return {
        "threshold": THRESHOLD,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support_negative": tn + fp,
        "support_positive": tp + fn,
        "predicted_negative": tn + fn,
        "predicted_positive": tp + fp,
        "accuracy": accuracy,
        "balanced_accuracy": (
            (recall + specificity) / 2 if recall != "" and specificity != "" else ""
        ),
        "precision": precision,
        "recall_tpr": recall,
        "specificity_tnr": specificity,
        "fpr": rate(fp, fp + tn),
        "fnr": rate(fn, fn + tp),
        "f1": f1,
    }


def prediction_rows(
    model: dict,
    experiment: str,
    layer: int,
    token: str,
    split: int,
    heldout: str,
    indices,
    labels,
    probabilities,
) -> list[dict[str, object]]:
    predictions = probabilities >= THRESHOLD
    return [
        {
            "model": model["paper_name"],
            "experiment": experiment,
            "layer": layer,
            "token_position": token,
            "split": split,
            "heldout_instruction": heldout,
            "sample_index": int(index),
            "label": int(label),
            "probability": float(probability),
            "prediction": int(prediction),
            "threshold": THRESHOLD,
        }
        for index, label, probability, prediction in zip(
            indices, labels, probabilities, predictions, strict=True
        )
    ]


def paper_target(
    model: str, experiment: str, layer: int, token: str
) -> tuple[str, str]:
    path = (
        ROOT
        / "data"
        / "results"
        / "internal-instruction-following"
        / "paper_targets.csv"
    )
    if not path.exists():
        return "", ""
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if (
                row["model"] == model
                and row["experiment"] == experiment
                and int(row["layer"]) == layer
                and row["token_position"] == token
            ):
                return row["paper_mean"], row["paper_std"]
    return "", ""


def train(model_key: str, device: str = "cuda") -> None:
    import torch
    from sklearn.metrics import roc_auc_score

    model = MODELS[model_key]
    target = paths(model)
    if not target["compact"].exists() or not target["labels"].exists():
        raise RuntimeError("Compact activations and labels are required")
    payload = torch.load(target["compact"], map_location="cpu", weights_only=False)
    acts = payload["activations"]
    labels_json = read_jsonl(target["labels"])
    labels = torch.tensor(
        [row["follow_all_instructions"] for row in labels_json], dtype=torch.float32
    )
    instructions = [row["instruction_id_list"][0] for row in labels_json]
    block = CONFIG["dataset"]["released_rows_per_instruction"]
    task_ids = torch.tensor([index % block for index in range(len(labels))])
    instruction_names = list(dict.fromkeys(instructions))
    instruction_ids = torch.tensor(
        [instruction_names.index(value) for value in instructions]
    )
    epochs = CONFIG["paper_settings"]["epochs"]
    runs: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    directions = {}

    for layer_index, layer in enumerate(model["layers"]):
        for token_index, token in enumerate(TOKENS):
            values = acts[:, token_index, layer_index]
            for seed in CONFIG["paper_settings"]["replication_seeds"]:
                torch.manual_seed(seed)
                split = torch.randperm(block) < int(
                    CONFIG["paper_settings"]["task_train_fraction"] * block
                )
                train_index = torch.isin(task_ids, torch.arange(block)[split])
                test_index = ~train_index
                train_x = normalize(values[train_index])
                test_x = normalize(values[test_index])
                probabilities, direction = fit_probe(
                    train_x, labels[train_index], test_x, device, epochs
                )
                test_labels = labels[test_index]
                complete = len(torch.unique(test_labels)) > 1
                auc = (
                    roc_auc_score(test_labels.numpy(), probabilities.numpy())
                    if complete
                    else ""
                )
                metrics = classification_metrics(test_labels, probabilities)
                runs.append(
                    {
                        "model": model["paper_name"],
                        "experiment": "task_generalization",
                        "layer": layer,
                        "token_position": token,
                        "split": seed,
                        "heldout_instruction": "",
                        "n_train": int(train_index.sum()),
                        "n_test": int(test_index.sum()),
                        "auroc": auc,
                        **metrics,
                        "status": "complete" if complete else "single_class",
                    }
                )
                predictions.extend(
                    prediction_rows(
                        model,
                        "task_generalization",
                        layer,
                        token,
                        seed,
                        "",
                        torch.where(test_index)[0],
                        test_labels,
                        probabilities,
                    )
                )
                directions[f"task|L{layer}|{token}|seed{seed}"] = direction

            for heldout, name in enumerate(instruction_names):
                train_index = instruction_ids != heldout
                test_index = ~train_index
                test_labels = labels[test_index]
                torch.manual_seed(CONFIG["paper_settings"]["instruction_probe_seed"])
                probabilities, direction = fit_probe(
                    normalize(values[train_index]),
                    labels[train_index],
                    normalize(values[test_index]),
                    device,
                    epochs,
                )
                complete = len(torch.unique(test_labels)) > 1
                auc = (
                    roc_auc_score(test_labels.numpy(), probabilities.numpy())
                    if complete
                    else ""
                )
                metrics = classification_metrics(test_labels, probabilities)
                runs.append(
                    {
                        "model": model["paper_name"],
                        "experiment": "instruction_type_generalization",
                        "layer": layer,
                        "token_position": token,
                        "split": heldout,
                        "heldout_instruction": name,
                        "n_train": int(train_index.sum()),
                        "n_test": int(test_index.sum()),
                        "auroc": auc,
                        **metrics,
                        "status": "complete" if complete else "single_class",
                    }
                )
                predictions.extend(
                    prediction_rows(
                        model,
                        "instruction_type_generalization",
                        layer,
                        token,
                        heldout,
                        name,
                        torch.where(test_index)[0],
                        test_labels,
                        probabilities,
                    )
                )
                directions[f"instruction|L{layer}|{token}|{name}"] = direction

    fields = list(runs[0])
    with target["runs"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runs)

    with target["predictions"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for row in runs:
        if row["status"] == "complete":
            for metric in METRICS:
                if row[metric] != "":
                    grouped[
                        (row["experiment"], row["layer"], row["token_position"], metric)
                    ].append(float(row[metric]))
    metrics_summary = []
    for (experiment, layer, token, metric), values in grouped.items():
        mean = statistics.fmean(values)
        std = statistics.pstdev(values)
        expected_mean, expected_std = (
            paper_target(model["paper_name"], experiment, layer, token)
            if metric == "auroc"
            else ("", "")
        )
        metrics_summary.append(
            {
                "model": model["paper_name"],
                "experiment": experiment,
                "layer": layer,
                "token_position": token,
                "metric": metric,
                "paper_mean": expected_mean,
                "paper_std": expected_std,
                "replication_mean": mean,
                "replication_std": std,
                "delta": mean - float(expected_mean) if expected_mean else "",
                "runs": len(values),
                "status": "complete",
            }
        )
    metrics_summary.sort(
        key=lambda row: (
            row["experiment"],
            row["layer"],
            row["token_position"],
            METRICS.index(row["metric"]),
        )
    )
    summary = [row for row in metrics_summary if row["metric"] == "auroc"]
    with target["summary"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with target["metrics_summary"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_summary[0]))
        writer.writeheader()
        writer.writerows(metrics_summary)
    torch.save(directions, target["directions"])
    stamp(
        target["manifest"],
        "trained",
        result_rows=len(runs),
        prediction_rows=len(predictions),
        summary_rows=len(summary),
        diagnostic_summary_rows=len(metrics_summary),
        diagnostic_threshold=THRESHOLD,
    )
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {target['root']}")


def aggregate() -> Path:
    output = artifact_root() / "probe_summary.csv"
    metrics_output = artifact_root() / "probe_metrics_summary.csv"
    rows = []
    for path in sorted(artifact_root().glob("*/probe_summary.csv")):
        with path.open() as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("No model summaries found")
    for row in rows:
        mean, std = paper_target(
            row["model"],
            row["experiment"],
            int(row["layer"]),
            row["token_position"],
        )
        row["paper_mean"] = mean
        row["paper_std"] = std
        row["delta"] = float(row["replication_mean"]) - float(mean) if mean else ""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_rows = []
    for path in sorted(artifact_root().glob("*/probe_metrics_summary.csv")):
        with path.open() as handle:
            metric_rows.extend(csv.DictReader(handle))
    if metric_rows:
        with metrics_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
    print(f"artifacts: {output}")
    if metric_rows:
        print(f"artifacts: {metrics_output}")
    return output


def baseline(model_key: str, device: str, force: bool, keep_raw: bool) -> None:
    model = MODELS[model_key]
    target = paths(model)
    if training_complete(model, target):
        stamp(target["manifest"], "trained")
        print(f"Baseline already complete for {model_key}")
        print(f"artifacts: {target['root']}")
        return
    extract(model_key, force=force)
    label(model_key)
    compact(model_key, keep_raw=keep_raw)
    train(model_key, device=device)
