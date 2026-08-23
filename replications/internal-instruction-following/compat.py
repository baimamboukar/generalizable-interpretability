from __future__ import annotations

import csv
import copy
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
BASE_CONFIG_PATH = HERE / "upstream.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_configuration(
    base_path: Path = BASE_CONFIG_PATH, overlay_path: Path | None = None
) -> tuple[dict, tuple[Path, ...]]:
    base = json.loads(base_path.read_text())
    if overlay_path is None:
        configured = os.environ.get("GI_IIF_CONFIG_OVERLAY")
        overlay_path = Path(configured) if configured else None
    if overlay_path is None:
        return base, (base_path,)
    overlay = json.loads(overlay_path.read_text())
    if overlay.get("inherits_sha256") != _sha256(base_path):
        raise RuntimeError("IIF configuration overlay does not inherit the pinned base")
    merged = copy.deepcopy(base)
    merged["models"].extend(overlay.get("models", []))
    merged.setdefault("controls", []).extend(overlay.get("controls", []))
    return merged, (base_path, overlay_path)


CONFIG, CONFIG_FILES = load_configuration()
MODELS = {
    model["key"]: model for model in [*CONFIG["models"], *CONFIG.get("controls", [])]
}
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


def validate_control_models() -> None:
    for model in CONFIG.get("controls", []):
        if model.get("base_model") not in MODELS:
            raise RuntimeError(f"Unknown base model for {model['key']}")
        if model.get("response_source") not in (None, *MODELS):
            raise RuntimeError(f"Unknown response source for {model['key']}")
        layers = model["layers"]
        if len(layers) != 1 or not 0 <= layers[0] <= model["num_hidden_layers"]:
            raise RuntimeError(f"Invalid control layer for {model['key']}: {layers}")


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


def file_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_digest() -> str:
    if len(CONFIG_FILES) == 1:
        return _sha256(CONFIG_FILES[0])
    digest = hashlib.sha256()
    for path in CONFIG_FILES:
        value = _sha256(path).encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
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
        "layer_selection_candidates": root / "layer_selection_candidates.csv",
        "layer_selection_runs": root / "layer_selection_runs.csv",
        "layer_selection_predictions": root / "layer_selection_predictions.csv",
        "layer_selection_summary": root / "layer_selection_summary.csv",
        "layer_selection_directions": root / "layer_selection_directions.pt",
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
        dataset={
            **dataset,
            "path": str(APPLE / "data" / "ifeval_simple.jsonl"),
            "sha256": file_digest(APPLE / "data" / "ifeval_simple.jsonl"),
        },
        condition={
            "prompt_mode": model.get("prompt_mode", "released_llama2"),
            "prompt_provenance": model.get(
                "prompt_provenance",
                "released hard-coded Llama-2 instruction template",
            ),
            "response_source": model.get("response_source", model_key),
            "response_policy": model.get("response_policy", f"on_policy_{model_key}"),
            "extraction_mode": model.get("extraction_mode", "released_generation"),
        },
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


def released_prompt(query: str) -> str:
    return (
        "<s>[INST] <<SYS>>\n You are a helpful assistant.\n \n<</SYS>>\n\n "
        f"{query} [/INST] "
    )


def teacher_forced_positions(
    prompt_tokens: int, response_tokens: int
) -> dict[str, int]:
    if prompt_tokens < 1 or response_tokens < 1:
        raise RuntimeError("Teacher-forced prompts and responses must be non-empty")
    steps = {
        "first": 0,
        "middle": response_tokens // 2,
        "last": response_tokens - 1,
    }
    return {name: prompt_tokens - 1 + step for name, step in steps.items()}


def control_prompt(model: dict, tokenizer, query: str) -> str:
    if model.get("prompt_mode") != "native":
        return released_prompt(query)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def ensure_response_source(model: dict) -> None:
    source_key = model.get("response_source")
    if not source_key:
        return
    source = paths(MODELS[source_key])
    if source["responses"].exists():
        return
    from infra.artifacts import pull

    pull(source["root"], f"internal-instruction-following/{source_key}")
    if not source["responses"].exists():
        raise RuntimeError(f"Missing response source after pull: {source_key}")


def extract_teacher_forced(model: dict, target: dict[str, Path]) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_response_source(model)
    prompts = read_jsonl(target["work"] / "ifeval_simple.jsonl")
    responses = read_jsonl(paths(MODELS[model["response_source"]])["responses"])
    if len(prompts) != len(responses) or any(
        prompt["prompt"] != response["prompt"]
        for prompt, response in zip(prompts, responses, strict=True)
    ):
        raise RuntimeError("Fixed-response prompts do not match IFEval-simple")
    token = os.environ["HF_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"], token=token
    )
    network = AutoModelForCausalLM.from_pretrained(
        model["id"],
        revision=model["revision"],
        token=token,
        device_map="auto",
        trust_remote_code=True,
    )
    network.eval()
    input_device = network.get_input_embeddings().weight.device
    target["raw_activations"].mkdir(parents=True, exist_ok=True)
    target["raw_responses"].parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for index, (prompt_row, response_row) in enumerate(
        tqdm(
            zip(prompts, responses, strict=True),
            total=len(prompts),
            desc=f"teacher-forcing {model['key']}",
        )
    ):
        prompt_ids = tokenizer(
            control_prompt(model, tokenizer, prompt_row["prompt"]),
            add_special_tokens=False,
        )["input_ids"]
        response_ids = tokenizer(response_row["response"], add_special_tokens=False)[
            "input_ids"
        ]
        positions = teacher_forced_positions(len(prompt_ids), len(response_ids))
        input_ids = torch.tensor(
            [prompt_ids + response_ids], device=input_device, dtype=torch.long
        )
        with torch.inference_mode():
            hidden = network(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            ).hidden_states
        record = {
            f"output_token_{name}": {
                f"layer_{layer}": hidden[layer][:, position : position + 1]
                .detach()
                .float()
                .cpu()
                for layer in model["layers"]
            }
            for name, position in positions.items()
        }
        with (target["raw_activations"] / f"sample_{index}.pkl").open("wb") as handle:
            pickle.dump(record, handle)
        output_rows.append({**prompt_row, "output": response_row["response"]})
        del hidden, input_ids
    write_jsonl(target["raw_responses"], output_rows)


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
    if model.get("extraction_mode") == "teacher_forced":
        extract_teacher_forced(model, target)
        if not extraction_complete(target, dataset["rows"]):
            raise RuntimeError("Teacher-forced extraction is incomplete")
        rows = read_jsonl(target["raw_responses"])
        write_jsonl(
            target["responses"],
            [{"prompt": row["prompt"], "response": row["output"]} for row in rows],
        )
        stamp(
            target["manifest"],
            "extracted",
            response_rows=len(rows),
            extraction_mode="teacher_forced",
            response_source=model["response_source"],
            response_policy=model.get("response_policy"),
            prompt_mode=model.get("prompt_mode", "released_llama2"),
            prompt_provenance=model.get("prompt_provenance"),
            response_tokenization="response text retokenized as causal continuation",
            responses_sha256=file_digest(target["responses"]),
        )
        return
    env = os.environ.copy()
    runtime = str(HERE / "runtime")
    env["PYTHONPATH"] = runtime + os.pathsep + env.get("PYTHONPATH", "")
    env["GI_MODEL_REVISION"] = model["revision"]
    env["TOKENIZERS_PARALLELISM"] = "false"
    entrypoint = APPLE / "run" / "save_LLMs_activations.py"
    if model.get("scope") in {"extension", "control"}:
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
    stamp(
        target["manifest"],
        "extracted",
        response_rows=len(rows),
        response_source=model.get("response_source", model_key),
        response_policy=model.get("response_policy", f"on_policy_{model_key}"),
        prompt_mode=model.get("prompt_mode", "released_llama2"),
        prompt_provenance=model.get(
            "prompt_provenance",
            "released hard-coded Llama-2 instruction template",
        ),
        responses_sha256=file_digest(target["responses"]),
    )


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
        labels_sha256=file_digest(target["labels"]),
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
        activations_sha256=file_digest(target["compact"]),
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


def select_layer(model_key: str, device: str = "cpu") -> None:
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
    names = list(dict.fromkeys(row["instruction_id_list"][0] for row in labels_json))
    instruction_ids = torch.tensor(
        [names.index(row["instruction_id_list"][0]) for row in labels_json]
    )
    seed = CONFIG["paper_settings"]["instruction_probe_seed"]
    epochs = CONFIG["paper_settings"]["epochs"]
    inner_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    directions = {}
    fold_summaries = []

    for outer, heldout_name in enumerate(names):
        candidate_scores: dict[tuple[int, str], list[float]] = defaultdict(list)
        for inner, validation_name in enumerate(names):
            if inner == outer:
                continue
            inner_train = (instruction_ids != outer) & (instruction_ids != inner)
            inner_validation = instruction_ids == inner
            validation_labels = labels[inner_validation]
            for layer_index, layer in enumerate(model["layers"]):
                for token_index, token in enumerate(TOKENS):
                    values = acts[:, token_index, layer_index]
                    torch.manual_seed(seed)
                    probabilities, _ = fit_probe(
                        normalize(values[inner_train]),
                        labels[inner_train],
                        normalize(values[inner_validation]),
                        device,
                        epochs,
                    )
                    complete = len(torch.unique(validation_labels)) > 1
                    auc = (
                        float(
                            roc_auc_score(
                                validation_labels.numpy(), probabilities.numpy()
                            )
                        )
                        if complete
                        else ""
                    )
                    if auc != "":
                        candidate_scores[(layer, token)].append(auc)
                    inner_rows.append(
                        {
                            "model": model["paper_name"],
                            "outer_heldout_instruction": heldout_name,
                            "inner_validation_instruction": validation_name,
                            "layer": layer,
                            "token_position": token,
                            "n_train": int(inner_train.sum()),
                            "n_validation": int(inner_validation.sum()),
                            "auroc": auc,
                            "status": "complete" if complete else "single_class",
                        }
                    )

        layer_scores = {
            layer: statistics.fmean(
                score
                for (candidate_layer, _), scores in candidate_scores.items()
                if candidate_layer == layer
                for score in scores
            )
            for layer in model["layers"]
        }
        selected_layer = max(model["layers"], key=layer_scores.__getitem__)
        selected_index = model["layers"].index(selected_layer)
        outer_train = instruction_ids != outer
        outer_test = ~outer_train
        test_labels = labels[outer_test]
        outer_aurocs = []
        for token_index, token in enumerate(TOKENS):
            values = acts[:, token_index, selected_index]
            torch.manual_seed(seed)
            probabilities, direction = fit_probe(
                normalize(values[outer_train]),
                labels[outer_train],
                normalize(values[outer_test]),
                device,
                epochs,
            )
            complete = len(torch.unique(test_labels)) > 1
            auc = (
                float(roc_auc_score(test_labels.numpy(), probabilities.numpy()))
                if complete
                else ""
            )
            if auc != "":
                outer_aurocs.append(auc)
            metrics = classification_metrics(test_labels, probabilities)
            outer_rows.append(
                {
                    "model": model["paper_name"],
                    "heldout_instruction": heldout_name,
                    "selected_layer": selected_layer,
                    "token_position": token,
                    "inner_mean_auroc": layer_scores[selected_layer],
                    "n_train": int(outer_train.sum()),
                    "n_test": int(outer_test.sum()),
                    "auroc": auc,
                    **metrics,
                    "status": "complete" if complete else "single_class",
                }
            )
            predictions.extend(
                prediction_rows(
                    model,
                    "nested_layer_selection",
                    selected_layer,
                    token,
                    outer,
                    heldout_name,
                    torch.where(outer_test)[0],
                    test_labels,
                    probabilities,
                )
            )
            directions[f"{heldout_name}|L{selected_layer}|{token}"] = direction
        fold_summaries.append(
            {
                "model": model["paper_name"],
                "heldout_instruction": heldout_name,
                "selected_layer": selected_layer,
                "inner_mean_auroc": layer_scores[selected_layer],
                "outer_mean_auroc": statistics.fmean(outer_aurocs),
                "valid_token_positions": len(outer_aurocs),
            }
        )

    selected_counts = Counter(row["selected_layer"] for row in fold_summaries)
    consensus_layer = max(
        model["layers"],
        key=lambda layer: (
            selected_counts[layer],
            statistics.fmean(
                row["inner_mean_auroc"]
                for row in fold_summaries
                if row["selected_layer"] == layer
            )
            if selected_counts[layer]
            else float("-inf"),
        ),
    )
    summary = [
        *fold_summaries,
        {
            "model": model["paper_name"],
            "heldout_instruction": "aggregate",
            "selected_layer": consensus_layer,
            "inner_mean_auroc": statistics.fmean(
                row["inner_mean_auroc"] for row in fold_summaries
            ),
            "outer_mean_auroc": statistics.fmean(
                row["outer_mean_auroc"] for row in fold_summaries
            ),
            "valid_token_positions": sum(
                row["valid_token_positions"] for row in fold_summaries
            ),
        },
    ]
    for path, rows in (
        (target["layer_selection_candidates"], inner_rows),
        (target["layer_selection_runs"], outer_rows),
        (target["layer_selection_predictions"], predictions),
        (target["layer_selection_summary"], summary),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    torch.save(directions, target["layer_selection_directions"])
    stamp(
        target["manifest"],
        "layer_selected",
        selected_layer=consensus_layer,
        nested_outer_mean_auroc=summary[-1]["outer_mean_auroc"],
        layer_selection="nested leave-one-instruction-type-out",
    )
    print(json.dumps(summary, indent=2))


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


def controls(base_model: str, device: str, force: bool, keep_raw: bool) -> None:
    selected = [
        model
        for model in CONFIG.get("controls", [])
        if model["base_model"] == base_model
    ]
    if not selected:
        raise RuntimeError(f"No controls configured for {base_model}")
    for model in selected:
        baseline(model["key"], device, force, keep_raw)
