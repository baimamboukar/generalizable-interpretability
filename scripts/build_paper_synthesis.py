#!/usr/bin/env python3
"""Build an artifact-only synthesis for the frozen corpus and later evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
CORE_STUDIES = {
    "Models Under Pressure",
    "Internal Instruction Following",
    "Political Perspective",
    "PRISM",
    "Lookback Lens",
    "Beyond Linear Probes",
}
SUPPORTING_STUDIES = {"Among Us", "Sycophancy", "Implicit Personalization"}
POST_FREEZE_STUDIES = {"LLMs Know"}
STUDY_ROLES = {
    **{study: "core" for study in CORE_STUDIES},
    "Among Us": "supporting_baseline_failure",
    "Sycophancy": "supporting_baseline_failure",
    "Implicit Personalization": "supporting_sanity_check",
    "LLMs Know": "postfreeze_positive_control",
}

OUTPUT_FIELDS = [
    "study",
    "manuscript_role",
    "claim_id",
    "phase",
    "comparison_unit",
    "model",
    "probe_or_method",
    "dataset_or_setting",
    "metric",
    "metric_scale",
    "paper_baseline",
    "replication_baseline",
    "delta_replication_vs_paper",
    "extension_result",
    "delta_extension_vs_replication",
    "delta_valid",
    "normalized_retention",
    "retention_definition",
    "retention_valid",
    "synthesis_grade",
    "outcome",
    "selection_regime",
    "source_path",
    "source_sha256",
    "caveat",
]


class Audit:
    def __init__(self, root: Path):
        self.root = root
        self.sources: dict[str, dict[str, object]] = {}
        self.notes: dict[str, object] = {}

    def path(self, relative: str) -> Path:
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required artifact is missing: {relative}")
        return path

    def csv(self, relative: str, required: set[str]) -> list[dict[str, str]]:
        path = self.path(relative)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            missing = required - set(fields)
            if missing:
                raise ValueError(f"{relative} is missing required fields: {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"{relative} has no rows")
        self._record(relative, fields, len(rows))
        return rows

    def text(self, relative: str) -> str:
        path = self.path(relative)
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"{relative} is empty")
        self._record(relative, None, None)
        return text

    def _record(self, relative: str, fields: list[str] | None, rows: int | None) -> None:
        path = self.root / relative
        payload: dict[str, object] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if fields is not None:
            payload.update({"columns": fields, "rows": rows})
        self.sources[relative] = payload

    def source_fields(self, *paths: str) -> tuple[str, str]:
        for path in paths:
            if path not in self.sources:
                raise ValueError(f"source was not audited before use: {path}")
        joined_paths = ";".join(paths)
        joined_hashes = ";".join(f"{path}={self.sources[path]['sha256']}" for path in paths)
        return joined_paths, joined_hashes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: str, *, field: str, source: str) -> float:
    if value == "":
        raise ValueError(f"{source}: empty numeric field {field}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{source}: invalid {field}={value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{source}: non-finite {field}={value!r}")
    return parsed


def format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def make_row(
    *,
    study: str,
    claim_id: str,
    phase: str,
    comparison_unit: str,
    model: str,
    probe_or_method: str,
    dataset_or_setting: str,
    metric: str,
    paper: float | None,
    replication: float,
    extension: float | None,
    delta_valid: bool,
    retention: float | None,
    retention_definition: str,
    retention_valid: bool,
    evidence_tier: str,
    outcome: str,
    selection_regime: str,
    source_path: str,
    source_sha256: str,
    caveat: str,
) -> dict[str, str]:
    baseline_delta = None if paper is None else replication - paper
    extension_delta = None if extension is None else extension - replication
    return {
        "study": study,
        "manuscript_role": STUDY_ROLES[study],
        "claim_id": claim_id,
        "phase": phase,
        "comparison_unit": comparison_unit,
        "model": model,
        "probe_or_method": probe_or_method,
        "dataset_or_setting": dataset_or_setting,
        "metric": metric,
        "metric_scale": "fraction",
        "paper_baseline": format_number(paper),
        "replication_baseline": format_number(replication),
        "delta_replication_vs_paper": format_number(baseline_delta),
        "extension_result": format_number(extension),
        "delta_extension_vs_replication": format_number(extension_delta),
        "delta_valid": str(delta_valid).lower(),
        "normalized_retention": format_number(retention),
        "retention_definition": retention_definition,
        "retention_valid": str(retention_valid).lower(),
        "synthesis_grade": evidence_tier,
        "outcome": outcome,
        "selection_regime": selection_regime,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "caveat": caveat,
    }


def build_mup(audit: Audit) -> list[dict[str, str]]:
    sweep_path = "data/results/model-under-pressure/model_layer_sweeps.csv"
    qwen_path = "data/results/model-under-pressure/qwen3_32b_layer_screen.csv"
    report_path = "reports/model-under-pressure.md"
    required = {"model", "probe", "layer", "dataset", "auroc"}
    rows = audit.csv(sweep_path, required)
    qwen = audit.csv(qwen_path, required | {"relative_depth", "sampling_phase", "passes_085"})
    report = audit.text(report_path)

    patterns = {
        "attention": r"\| Attention \| Llama-3\.3-70B\s+\| 31\s+\| baseline replication \|\s+0\.9500 \|\s+0\.9184",
        "softmax": r"\| Softmax\s+\| Llama-3\.3-70B\s+\| 31\s+\| baseline replication \|\s+0\.9432 \|\s+0\.9244",
    }
    for probe, pattern in patterns.items():
        if not re.search(pattern, report):
            raise ValueError(f"{report_path}: could not verify {probe} baseline headline")

    all_rows = [(sweep_path, row) for row in rows] + [(qwen_path, row) for row in qwen]
    grouped: dict[tuple[str, str, str], list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for source, row in all_rows:
        grouped[(row["model"], row["probe"], row["layer"])].append((source, row))

    candidates: dict[tuple[str, str], list[tuple[float, str, str]]] = defaultdict(list)
    for (model, probe, layer), members in grouped.items():
        means = [number(row["auroc"], field="auroc", source=source) for source, row in members if row["dataset"] == "mean"]
        details = [number(row["auroc"], field="auroc", source=source) for source, row in members if row["dataset"] != "mean"]
        if len(means) != 1 or len(details) != 6:
            raise ValueError(f"MUP {model}/{probe}/L{layer}: expected one mean and six datasets")
        if abs(means[0] - mean(details)) > 5e-8:
            raise ValueError(f"MUP {model}/{probe}/L{layer}: stored mean does not match datasets")
        candidates[(model, probe)].append((means[0], layer, members[0][0]))

    paper = {"attention": 0.917, "softmax": None}
    replication = {"attention": 0.9184, "softmax": 0.9244}
    output = []
    for (model, probe), values in sorted(candidates.items()):
        extension, layer, source = max(values)
        source_path, source_hash = audit.source_fields(report_path, source)
        output.append(
            make_row(
                study="Models Under Pressure",
                claim_id="six_dataset_sparse_layer_screen",
                phase="exploratory_extension",
                comparison_unit="best sampled layer over six released evaluation datasets",
                model=model,
                probe_or_method=probe,
                dataset_or_setting=f"six-dataset mean; layer {layer}",
                metric="AUROC",
                paper=paper[probe],
                replication=replication[probe],
                extension=extension,
                delta_valid=False,
                retention=None,
                retention_definition="",
                retention_valid=False,
                evidence_tier="C",
                outcome="target-model signal found; layer-selection portability unresolved",
                selection_regime="test-oracle maximum over a sparse layer screen",
                source_path=source_path,
                source_sha256=source_hash,
                caveat=(
                    "The extension layer is selected on evaluation AUROC. The attention paper value is an "
                    "approximate six-dataset headline; no comparable six-dataset paper softmax value is available."
                ),
            )
        )
    return output


def build_among_us(audit: Audit) -> list[dict[str, str]]:
    baseline_path = "data/results/among-us/figure7_baseline_comparison.csv"
    extension_path = "data/results/among-us/extension_probe_results.csv"
    baseline = audit.csv(
        baseline_path,
        {"train_dataset", "eval_dataset", "paper_auroc", "replication_auroc"},
    )
    extension = audit.csv(
        extension_path,
        {"model", "layer", "train_dataset", "eval_dataset", "auroc"},
    )
    if len(baseline) != 20:
        raise ValueError(f"{baseline_path}: expected 20 Figure 7 cells, found {len(baseline)}")
    keys = {(row["train_dataset"], row["eval_dataset"]) for row in baseline}
    if len(keys) != 20:
        raise ValueError(f"{baseline_path}: Figure 7 cells are not unique")
    paper = mean(number(row["paper_auroc"], field="paper_auroc", source=baseline_path) for row in baseline)
    replication = mean(
        number(row["replication_auroc"], field="replication_auroc", source=baseline_path) for row in baseline
    )

    aliases = {"Among Us (lying)": "Lying", "Among Us (deception)": "Deception"}
    grouped: dict[tuple[str, str], dict[tuple[str, str], float]] = defaultdict(dict)
    for row in extension:
        eval_dataset = aliases.get(row["eval_dataset"], row["eval_dataset"])
        key = (row["train_dataset"], eval_dataset)
        if key not in keys:
            continue
        group = (row["model"], row["layer"])
        if key in grouped[group]:
            raise ValueError(f"{extension_path}: duplicate {group}/{key}")
        grouped[group][key] = number(row["auroc"], field="auroc", source=extension_path)

    by_model: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (model, layer), values in grouped.items():
        if set(values) != keys:
            raise ValueError(f"{extension_path}: {model}/L{layer} does not cover the 20 Figure 7 cells")
        by_model[model].append((mean(values.values()), layer))

    source_path, source_hash = audit.source_fields(baseline_path, extension_path)
    output = []
    for model, values in sorted(by_model.items()):
        extension_value, layer = max(values)
        output.append(
            make_row(
                study="Among Us",
                claim_id="figure7_cross_dataset_mean",
                phase="exploratory_extension",
                comparison_unit="mean of the 20 released Figure 7 train-evaluation cells",
                model=model,
                probe_or_method="released linear probe pipeline",
                dataset_or_setting=f"Figure 7 grid; best sampled layer {layer}",
                metric="AUROC",
                paper=paper,
                replication=replication,
                extension=extension_value,
                delta_valid=False,
                retention=None,
                retention_definition="",
                retention_valid=False,
                evidence_tier="D",
                outcome="baseline not reproduced; extension is diagnostic only",
                selection_regime="test-oracle maximum over three sampled layers",
                source_path=source_path,
                source_sha256=source_hash,
                caveat="The Phi-4 baseline misses the paper mean substantially, so cross-model deltas cannot establish transfer.",
            )
        )
    return output


def build_iif(audit: Audit) -> list[dict[str, str]]:
    baseline_path = "data/results/internal-instruction-following/baseline_comparison.csv"
    extension_path = "data/results/internal-instruction-following/results_long.csv"
    baseline = audit.csv(
        baseline_path,
        {
            "model",
            "experiment",
            "layer",
            "token_position",
            "metric",
            "paper_mean",
            "replication_mean",
            "runs",
            "status",
        },
    )
    results = audit.csv(
        extension_path,
        {
            "scope",
            "model",
            "experiment",
            "layer",
            "token_position",
            "metric",
            "mean",
            "valid_runs",
            "expected_runs",
            "coverage",
        },
    )
    if len(baseline) != 48 or {row["metric"] for row in baseline} != {"auroc"}:
        raise ValueError(f"{baseline_path}: expected 48 AUROC comparison rows")
    if {row["status"] for row in baseline} != {"complete"}:
        raise ValueError(f"{baseline_path}: baseline contains incomplete rows")
    baseline_keys = {(row["model"], row["experiment"], row["layer"], row["token_position"]) for row in baseline}
    if len(baseline_keys) != 48:
        raise ValueError(f"{baseline_path}: duplicate comparison cells")

    by_scope = {scope: [row for row in results if row["scope"] == scope] for scope in {row["scope"] for row in results}}
    if set(by_scope) != {"paper", "baseline_replication", "extension"}:
        raise ValueError(f"{extension_path}: unexpected scopes {sorted(by_scope)}")
    for scope, field in [("paper", "paper_mean"), ("baseline_replication", "replication_mean")]:
        scoped = by_scope[scope]
        if len(scoped) != 48:
            raise ValueError(f"{extension_path}: expected 48 {scope} rows")
        indexed = {(row["model"], row["experiment"], row["layer"], row["token_position"]): row for row in scoped}
        if set(indexed) != baseline_keys:
            raise ValueError(f"{extension_path}: {scope} keys differ from baseline comparison")
        for row in baseline:
            key = (row["model"], row["experiment"], row["layer"], row["token_position"])
            observed = number(indexed[key]["mean"], field="mean", source=extension_path)
            expected = number(row[field], field=field, source=baseline_path)
            if abs(observed - expected) > 5e-8:
                raise ValueError(f"{extension_path}: {scope} value differs for {key}")

    extensions = by_scope["extension"]
    extension_models = sorted({row["model"] for row in extensions})
    if len(extensions) != 72 or len(extension_models) != 3:
        raise ValueError(f"{extension_path}: expected 72 rows across three extension models")
    source_path, source_hash = audit.source_fields(baseline_path, extension_path)
    output = []
    for experiment in ["task_generalization", "instruction_type_generalization"]:
        base = [row for row in baseline if row["experiment"] == experiment]
        if not base:
            raise ValueError(f"{baseline_path}: no rows for {experiment}")
        paper = mean(number(row["paper_mean"], field="paper_mean", source=baseline_path) for row in base)
        replication = mean(number(row["replication_mean"], field="replication_mean", source=baseline_path) for row in base)
        for model in extension_models:
            model_rows = [row for row in extensions if row["model"] == model and row["experiment"] == experiment]
            if len(model_rows) != 12:
                raise ValueError(f"{extension_path}: expected 12 {experiment} rows for {model}")
            layers = sorted({int(row["layer"]) for row in model_rows})
            if len(layers) != 4 or {row["token_position"] for row in model_rows} != {"first", "middle", "last"}:
                raise ValueError(f"{extension_path}: incomplete layer/token grid for {model}/{experiment}")
            for row in model_rows:
                coverage = number(row["coverage"], field="coverage", source=extension_path)
                valid = int(row["valid_runs"])
                expected = int(row["expected_runs"])
                if expected != 5 or abs(coverage - valid / expected) > 5e-12:
                    raise ValueError(f"{extension_path}: inconsistent run coverage for {model}/{experiment}")
            selected = model_rows if experiment == "task_generalization" else [row for row in model_rows if int(row["layer"]) == layers[0]]
            if experiment == "instruction_type_generalization" and len(selected) != 3:
                raise ValueError(f"{extension_path}: expected three early-layer type rows for {model}")
            extension_value = mean(number(row["mean"], field="mean", source=extension_path) for row in selected)
            task_label = "unseen tasks" if experiment == "task_generalization" else "unseen instruction types"
            output.append(
                make_row(
                    study="Internal Instruction Following",
                    claim_id=experiment,
                    phase="extension",
                    comparison_unit=(
                        "mean over four sampled layers and three token positions"
                        if experiment == "task_generalization"
                        else "mean over three token positions at the earliest sampled layer"
                    ),
                    model=model,
                    probe_or_method="bias-free linear sigmoid probe",
                    dataset_or_setting=task_label,
                    metric="AUROC",
                    paper=paper,
                    replication=replication,
                    extension=extension_value,
                    delta_valid=True,
                    retention=None,
                    retention_definition="",
                    retention_valid=False,
                    evidence_tier="C",
                    outcome=(
                        "unseen-task transfer remains detectable"
                        if experiment == "task_generalization"
                        else "unseen-type transfer is near chance and label-support dependent"
                    ),
                    selection_regime="four prespecified layers; paper-aligned earliest layer for unseen types",
                    source_path=source_path,
                    source_sha256=source_hash,
                    caveat=(
                        "Each model generates and labels its own answers. Some held-out instruction types have one class, "
                        "so extension means average only valid folds; the full 2x2 response/prompt controls are pending."
                    ),
                )
            )
    return output


def build_blp(audit: Audit) -> list[dict[str, str]]:
    comparison_path = "data/results/beyond-linear-probes/baseline_replication.csv"
    combined_path = "data/results/beyond-linear-probes/baseline_and_extensions.csv"
    comparison = audit.csv(
        comparison_path,
        {"dataset", "model", "layer", "method", "split", "metric", "paper_value_pct", "replication_value_pct"},
    )
    combined = audit.csv(
        combined_path,
        {"dataset", "scope", "model", "layer", "method", "split", "metric", "value_pct", "seeds"},
    )
    if len(combined) != 25 or {row["scope"] for row in combined} != {"paper", "replication", "extension"}:
        raise ValueError(f"{combined_path}: expected 25 paper/replication/extension rows")
    methods = {"linear", "bilinear", "mlp", "ee_mlp", "tpc_order5"}
    scoped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in combined:
        if row["dataset"] != "WildGuardMix" or row["split"] != "test" or row["metric"] != "F1":
            raise ValueError(f"{combined_path}: synthesis expects WildGuardMix test F1 only")
        key = (row["scope"], row["model"])
        if row["method"] in scoped[key]:
            raise ValueError(f"{combined_path}: duplicate method for {key}")
        scoped[key][row["method"]] = row
    for key, values in scoped.items():
        if set(values) != methods:
            raise ValueError(f"{combined_path}: {key} does not contain all five methods")

    test_comparison = [row for row in comparison if row["split"] == "test"]
    comparison_alias = {"early_exit_mlp": "ee_mlp"}
    comparison_methods = {comparison_alias.get(row["method"], row["method"]) for row in test_comparison}
    if len(test_comparison) != 5 or comparison_methods != methods:
        raise ValueError(f"{comparison_path}: expected five test-method rows")
    paper_rows = scoped[("paper", "Llama-3.2-3B")]
    replication_rows = scoped[("replication", "Llama-3.2-3B")]
    for row in test_comparison:
        method = comparison_alias.get(row["method"], row["method"])
        paper = number(row["paper_value_pct"], field="paper_value_pct", source=comparison_path)
        replication = number(row["replication_value_pct"], field="replication_value_pct", source=comparison_path)
        if abs(paper - number(paper_rows[method]["value_pct"], field="value_pct", source=combined_path)) > 5e-5:
            raise ValueError(f"{combined_path}: paper value mismatch for {method}")
        if abs(replication - number(replication_rows[method]["value_pct"], field="value_pct", source=combined_path)) > 5e-5:
            raise ValueError(f"{combined_path}: replication value mismatch for {method}")

    paper_gain = (
        number(paper_rows["tpc_order5"]["value_pct"], field="value_pct", source=combined_path)
        - number(paper_rows["linear"]["value_pct"], field="value_pct", source=combined_path)
    ) / 100
    replication_gain = (
        number(replication_rows["tpc_order5"]["value_pct"], field="value_pct", source=combined_path)
        - number(replication_rows["linear"]["value_pct"], field="value_pct", source=combined_path)
    ) / 100
    extension_keys = sorted(key for key in scoped if key[0] == "extension")
    if len(extension_keys) != 3:
        raise ValueError(f"{combined_path}: expected three extension models")
    source_path, source_hash = audit.source_fields(comparison_path, combined_path)
    output = []
    for _, model in extension_keys:
        values = scoped[("extension", model)]
        extension_gain = (
            number(values["tpc_order5"]["value_pct"], field="value_pct", source=combined_path)
            - number(values["linear"]["value_pct"], field="value_pct", source=combined_path)
        ) / 100
        best_method = max(methods, key=lambda method: number(values[method]["value_pct"], field="value_pct", source=combined_path))
        output.append(
            make_row(
                study="Beyond Linear Probes",
                claim_id="tpc_gain_over_linear",
                phase="extension",
                comparison_unit="TPC order-5 test F1 minus linear-probe test F1",
                model=model,
                probe_or_method="TPC order 5 versus linear",
                dataset_or_setting=f"WildGuardMix; layer {values['linear']['layer']}",
                metric="F1 gain",
                paper=paper_gain,
                replication=replication_gain,
                extension=extension_gain,
                delta_valid=True,
                retention=None,
                retention_definition="",
                retention_valid=False,
                evidence_tier="C",
                outcome="TPC beats linear, but is not the best architecture on every extension model",
                selection_regime="released validation-selected layer; five matched probe architectures",
                source_path=source_path,
                source_sha256=source_hash,
                caveat=(
                    f"The best extension method is {best_method}; only one dataset and one selected layer are covered. "
                    "A ratio of small F1 gains would be unstable and is not reported."
                ),
            )
        )
    return output


def build_sycophancy(audit: Audit) -> list[dict[str, str]]:
    summary_path = "data/results/sycophancy/mega_table.csv"
    long_path = "data/results/sycophancy/probe_accuracies_long.csv"
    summary = audit.csv(
        summary_path,
        {
            "model",
            "component",
            "metric",
            "paper_value",
            "replication_mean",
            "replication_sd",
            "paper_peak_layer",
            "replication_peak_layer",
            "seeds",
            "verdict",
        },
    )
    scores = audit.csv(long_path, {"model", "seed", "activation", "layer", "head", "accuracy"})
    if len(summary) != 2 or {row["verdict"] for row in summary} != {"not_reproduced"}:
        raise ValueError(f"{summary_path}: expected two failed Figure 10 component rows")
    if {row["metric"] for row in summary} != {"peak_validation_accuracy"}:
        raise ValueError(f"{summary_path}: unexpected metric")
    mapping = {"MLP": "mlp", "residual_stream": "residual"}
    source_path, source_hash = audit.source_fields(summary_path, long_path)
    output = []
    for row in summary:
        activation = mapping[row["component"]]
        selected = [item for item in scores if item["activation"] == activation]
        seeds = row["seeds"].split(";")
        if len(selected) != 84 or sorted({item["seed"] for item in selected}) != sorted(seeds):
            raise ValueError(f"{long_path}: expected 28 layers x three seeds for {activation}")
        by_layer: dict[int, list[float]] = defaultdict(list)
        for item in selected:
            by_layer[int(item["layer"])].append(number(item["accuracy"], field="accuracy", source=long_path))
        if len(by_layer) != 28 or any(len(values) != 3 for values in by_layer.values()):
            raise ValueError(f"{long_path}: incomplete layer/seed grid for {activation}")
        peak_layer, peak_values = max(by_layer.items(), key=lambda item: mean(item[1]))
        peak_mean = mean(peak_values)
        peak_sd = math.sqrt(sum((value - peak_mean) ** 2 for value in peak_values) / (len(peak_values) - 1))
        if peak_layer != int(row["replication_peak_layer"]):
            raise ValueError(f"{summary_path}: peak layer mismatch for {activation}")
        if abs(peak_mean - number(row["replication_mean"], field="replication_mean", source=summary_path)) > 5e-10:
            raise ValueError(f"{summary_path}: peak mean mismatch for {activation}")
        if abs(peak_sd - number(row["replication_sd"], field="replication_sd", source=summary_path)) > 5e-10:
            raise ValueError(f"{summary_path}: peak SD mismatch for {activation}")
        output.append(
            make_row(
                study="Sycophancy",
                claim_id="figure10_peak_validation_accuracy",
                phase="supporting_baseline_failure",
                comparison_unit="peak mean validation accuracy across 28 layers and three seeds",
                model=row["model"],
                probe_or_method=row["component"],
                dataset_or_setting=f"Figure 10; layer {peak_layer}",
                metric="accuracy",
                paper=number(row["paper_value"], field="paper_value", source=summary_path),
                replication=peak_mean,
                extension=None,
                delta_valid=False,
                retention=None,
                retention_definition="",
                retention_valid=False,
                evidence_tier="D",
                outcome="reported peak magnitude not reproduced; extension skipped",
                selection_regime="peak over all 28 layers in both paper and replication",
                source_path=source_path,
                source_sha256=source_hash,
                caveat="The middle-layer location recurs, but paper checkpoints, activations, and per-seed results were not released.",
            )
        )
    return output


def prism_gain(
    rows: list[dict[str, str]], model: str, method: str, value_field: str, source: str
) -> float:
    gains = []
    datasets = sorted({row["dataset"] for row in rows if row["model"] == model and row["method"] == method})
    if datasets != ["LogicStruct", "True-False"]:
        raise ValueError(f"{source}: {model}/{method} does not cover both datasets")
    for dataset in datasets:
        values = {
            row["representation"]: number(row[value_field], field=value_field, source=source)
            for row in rows
            if row["model"] == model and row["method"] == method and row["dataset"] == dataset
        }
        if set(values) != {"prompted", "raw"}:
            raise ValueError(f"{source}: {model}/{method}/{dataset} does not have raw and prompted rows")
        gains.append(values["prompted"] - values["raw"])
    return mean(gains)


def build_prism(audit: Audit) -> list[dict[str, str]]:
    path = "data/results/prism/cross_model_summary.csv"
    rows = audit.csv(
        path,
        {"dataset", "model", "method", "representation", "metric", "paper_value", "replication_value", "status"},
    )
    if {row["metric"] for row in rows} != {"accuracy"} or {row["status"] for row in rows} != {"completed"}:
        raise ValueError(f"{path}: expected completed accuracy rows only")
    baseline_models = ["Llama-2-7B-Chat", "Llama-2-13B-Chat"]
    extension_models = sorted({row["model"] for row in rows} - set(baseline_models))
    source_path, source_hash = audit.source_fields(path)
    output = []
    for method in ["MM", "SAPLMA"]:
        paper = mean(prism_gain(rows, model, method, "paper_value", path) for model in baseline_models)
        replication = mean(prism_gain(rows, model, method, "replication_value", path) for model in baseline_models)
        if replication <= 0:
            raise ValueError(f"{path}: non-positive baseline prompt gain for {method}")
        for model in extension_models:
            extension = prism_gain(rows, model, method, "replication_value", path)
            output.append(
                make_row(
                    study="PRISM",
                    claim_id="prompt_gain",
                    phase="extension",
                    comparison_unit="mean prompted-minus-raw accuracy over True-False and LogicStruct",
                    model=model,
                    probe_or_method=method,
                    dataset_or_setting="True-False and LogicStruct",
                    metric="accuracy gain",
                    paper=paper,
                    replication=replication,
                    extension=extension,
                    delta_valid=True,
                    retention=extension / replication,
                    retention_definition="extension prompt gain / replicated baseline prompt gain",
                    retention_valid=True,
                    evidence_tier="B",
                    outcome="prompt-guided improvement transfers to every tested extension model",
                    selection_regime="released aggregation; no extension-model maximum",
                    source_path=source_path,
                    source_sha256=source_hash,
                    caveat="Retention compares effect sizes, not absolute accuracies; uncertainty is not available in the summary artifact.",
                )
            )
    return output


def build_implicit(audit: Audit) -> list[dict[str, str]]:
    baseline_path = "data/results/implicit-personalization/probe_validation_baseline.csv"
    extension_path = "data/results/implicit-personalization/cross_model_extension.csv"
    baseline = audit.csv(
        baseline_path,
        {"model", "attribute", "layer", "metric", "paper_value_approx", "replication_value", "status"},
    )
    extension = audit.csv(
        extension_path,
        {"model", "attribute", "layer", "result", "fold", "accuracy_percent"},
    )
    if len(baseline) != 60:
        raise ValueError(f"{baseline_path}: expected 60 selected-layer baseline rows, found {len(baseline)}")
    if {row["metric"] for row in baseline} != {"accuracy_percent"}:
        raise ValueError(f"{baseline_path}: expected accuracy_percent only")
    attributes = sorted({row["attribute"] for row in baseline})
    if attributes != ["age", "gender", "race", "socio-economic status"]:
        raise ValueError(f"{baseline_path}: unexpected attributes {attributes}")

    baseline_by_attribute: dict[str, tuple[float, float]] = {}
    for attribute in attributes:
        selected = [row for row in baseline if row["attribute"] == attribute]
        if len(selected) != 15:
            raise ValueError(f"{baseline_path}: expected 15 baseline rows for {attribute}")
        baseline_by_attribute[attribute] = (
            mean(number(row["paper_value_approx"], field="paper_value_approx", source=baseline_path) for row in selected) / 100,
            mean(number(row["replication_value"], field="replication_value", source=baseline_path) for row in selected) / 100,
        )

    extension_models = sorted({row["model"] for row in extension if row["result"] == "Control"})
    if not extension_models:
        raise ValueError(f"{extension_path}: no extension models with controls")
    source_path, source_hash = audit.source_fields(baseline_path, extension_path)
    output = []
    for model in extension_models:
        for attribute in attributes:
            selected = [
                row
                for row in extension
                if row["model"] == model and row["attribute"] == attribute and row["result"] == "Replication"
            ]
            if len(selected) != 25 or {row["fold"] for row in selected} != {"0", "1", "2", "3", "4"}:
                raise ValueError(f"{extension_path}: expected 5 layers x 5 folds for {model}/{attribute}")
            extension_value = mean(
                number(row["accuracy_percent"], field="accuracy_percent", source=extension_path) for row in selected
            ) / 100
            paper, replication = baseline_by_attribute[attribute]
            output.append(
                make_row(
                    study="Implicit Personalization",
                    claim_id="explicit_identity_probe_validation",
                    phase="reduced_extension",
                    comparison_unit="mean true-label accuracy over five selected layers and five folds",
                    model=model,
                    probe_or_method="multiclass logistic probe",
                    dataset_or_setting=attribute,
                    metric="accuracy",
                    paper=paper,
                    replication=replication,
                    extension=extension_value,
                    delta_valid=True,
                    retention=None,
                    retention_definition="",
                    retention_valid=False,
                    evidence_tier="C",
                    outcome="explicit demographic identity remains linearly decodable",
                    selection_regime="five prespecified interior layers",
                    source_path=source_path,
                    source_sha256=source_hash,
                    caveat=(
                        "This is a reduced selected-layer validation with digitized paper curves. It does not test the "
                        "paper's stereotype-driven personalization or steering claims."
                    ),
                )
            )
    return output


def build_lookback(audit: Audit) -> list[dict[str, str]]:
    baseline_path = "data/results/lookback-lens/table2_replication.csv"
    extension_path = "data/results/lookback-lens/cross_model_transfer.csv"
    baseline = audit.csv(
        baseline_path,
        {"setting", "source", "target", "split", "metric", "paper", "replication"},
    )
    extension = audit.csv(
        extension_path,
        {"group", "model_key", "model", "value", "setting", "source", "target", "split", "metric"},
    )
    transfers = [row for row in baseline if row["split"] == "transfer"]
    if len(transfers) != 4:
        raise ValueError(f"{baseline_path}: expected four transfer cells")
    keys = {(row["setting"], row["source"], row["target"], row["metric"]) for row in transfers}
    paper = mean(number(row["paper"], field="paper", source=baseline_path) for row in transfers)
    replication = mean(number(row["replication"], field="replication", source=baseline_path) for row in transfers)

    for group, field in [("Paper", "paper"), ("Baseline", "replication")]:
        check = [row for row in extension if row["group"] == group]
        if len(check) != 4:
            raise ValueError(f"{extension_path}: expected four {group} rows")
        by_key = {(row["setting"], row["source"], row["target"], row["metric"]): row for row in check}
        if set(by_key) != keys:
            raise ValueError(f"{extension_path}: {group} transfer keys differ from Table 2")
        for row in transfers:
            key = (row["setting"], row["source"], row["target"], row["metric"])
            observed = number(by_key[key]["value"], field="value", source=extension_path)
            expected = number(row[field], field=field, source=baseline_path)
            if abs(observed - expected) > 5e-10:
                raise ValueError(f"{extension_path}: {group} value differs from Table 2 for {key}")

    source_path, source_hash = audit.source_fields(baseline_path, extension_path)
    output = []
    extension_models = sorted({row["model"] for row in extension if row["group"] == "Extension"})
    for model in extension_models:
        selected = [row for row in extension if row["group"] == "Extension" and row["model"] == model]
        if len(selected) != 4:
            raise ValueError(f"{extension_path}: expected four transfer rows for {model}")
        if {(row["setting"], row["source"], row["target"], row["metric"]) for row in selected} != keys:
            raise ValueError(f"{extension_path}: transfer keys differ for {model}")
        extension_value = mean(number(row["value"], field="value", source=extension_path) for row in selected)
        retention = (extension_value - 0.5) / (replication - 0.5)
        output.append(
            make_row(
                study="Lookback Lens",
                claim_id="cross_dataset_transfer",
                phase="extension",
                comparison_unit="mean of four matched NQ/CNN-DM transfer cells",
                model=model,
                probe_or_method="logistic regression on lookback-ratio features",
                dataset_or_setting="NQ and CNN/DailyMail; predefined and random splits",
                metric="AUROC",
                paper=paper,
                replication=replication,
                extension=extension_value,
                delta_valid=True,
                retention=retention,
                retention_definition="(extension AUROC - 0.5) / (replication AUROC - 0.5)",
                retention_valid=True,
                evidence_tier="B",
                outcome="cross-dataset transfer remains in the replicated baseline band",
                selection_regime="four fixed transfer cells; no extension-model maximum",
                source_path=source_path,
                source_sha256=source_hash,
                caveat="The statistic summarizes matched transfer cells; it does not pool train or test-split AUROCs.",
            )
        )
    return output


def build_political(audit: Audit) -> list[dict[str, str]]:
    baseline_path = "data/results/political-perspective/independent_baseline.csv"
    summary_path = "data/results/political-perspective/mega_summary.csv"
    heads_path = "data/results/political-perspective/extension_all_heads.csv"
    baseline = audit.csv(
        baseline_path,
        {"model", "metric", "paper", "replication", "paper_layer", "replication_layer", "paper_head", "replication_head"},
    )
    summary = audit.csv(
        summary_path,
        {"group", "model", "observed_peak", "peak_layer", "peak_head", "num_probes"},
    )
    heads = audit.csv(heads_path, {"model", "layer", "head", "cv_spearman"})
    if len(baseline) != 3 or {row["metric"] for row in baseline} != {"peak single-head CV Spearman"}:
        raise ValueError(f"{baseline_path}: expected three peak single-head rows")
    for row in baseline:
        if row["paper_layer"] != row["replication_layer"] or row["paper_head"] != row["replication_head"]:
            raise ValueError(f"{baseline_path}: peak location mismatch for {row['model']}")
    paper = mean(number(row["paper"], field="paper", source=baseline_path) for row in baseline)
    replication = mean(number(row["replication"], field="replication", source=baseline_path) for row in baseline)

    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in heads:
        by_model[row["model"]].append(row)
    extension_summary = [row for row in summary if row["group"] == "extension"]
    if len(extension_summary) != 5:
        raise ValueError(f"{summary_path}: expected five extension models")
    for row in extension_summary:
        model_rows = by_model.get(row["model"], [])
        if len(model_rows) != int(row["num_probes"]):
            raise ValueError(f"{heads_path}: probe count mismatch for {row['model']}")
        finite_rows = [item for item in model_rows if item["cv_spearman"].strip().lower() != "nan"]
        nonfinite = len(model_rows) - len(finite_rows)
        if not finite_rows:
            raise ValueError(f"{heads_path}: no finite head scores for {row['model']}")
        audit.notes.setdefault("political_nonfinite_cv_spearman", {})[row["model"]] = nonfinite
        best = max(finite_rows, key=lambda item: number(item["cv_spearman"], field="cv_spearman", source=heads_path))
        if abs(number(best["cv_spearman"], field="cv_spearman", source=heads_path) - number(row["observed_peak"], field="observed_peak", source=summary_path)) > 5e-12:
            raise ValueError(f"{summary_path}: peak value mismatch for {row['model']}")
        if best["layer"] != row["peak_layer"] or best["head"] != row["peak_head"]:
            raise ValueError(f"{summary_path}: peak location mismatch for {row['model']}")

    source_path, source_hash = audit.source_fields(baseline_path, summary_path, heads_path)
    output = []
    for row in sorted(extension_summary, key=lambda item: item["model"]):
        extension_value = number(row["observed_peak"], field="observed_peak", source=summary_path)
        output.append(
            make_row(
                study="Political Perspective",
                claim_id="peak_single_head_decodability",
                phase="exploratory_extension",
                comparison_unit="maximum single-head cross-validated Spearman",
                model=row["model"],
                probe_or_method="single-head ridge regression",
                dataset_or_setting=f"Congress speeches; layer {row['peak_layer']}, head {row['peak_head']}",
                metric="Spearman correlation",
                paper=paper,
                replication=replication,
                extension=extension_value,
                delta_valid=False,
                retention=None,
                retention_definition="",
                retention_valid=False,
                evidence_tier="C",
                outcome="baseline reproduced; extension peak decodability is uneven",
                selection_regime="test-oracle maximum over every exposed layer and head",
                source_path=source_path,
                source_sha256=source_hash,
                caveat="Architectures receive unequal head-layer search budgets; the extension peaks are not fair transfer estimates.",
            )
        )
    return output


def build_llms_know(audit: Audit) -> list[dict[str, str]]:
    baseline_path = "data/results/llms-know/table1_mistral_replication.csv"
    extension_path = "data/results/llms-know/table1_cross_model_extensions.csv"
    baseline = audit.csv(baseline_path, {"dataset", "source", "mean", "std"})
    extension = audit.csv(
        extension_path,
        {
            "model_key",
            "model_id",
            "dataset",
            "regime",
            "selected_layer",
            "n_train",
            "n_test",
            "mean",
            "std",
        },
    )
    datasets = {"triviaqa", "winobias", "math"}
    if len(baseline) != 6 or {row["dataset"] for row in baseline} != datasets:
        raise ValueError(f"{baseline_path}: expected paper and replication rows for three datasets")
    baseline_values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in baseline:
        if row["source"] not in {"Original paper", "Replication"}:
            raise ValueError(f"{baseline_path}: unexpected source {row['source']!r}")
        baseline_values[row["dataset"]][row["source"]] = number(
            row["mean"], field="mean", source=baseline_path
        )
    if any(set(values) != {"Original paper", "Replication"} for values in baseline_values.values()):
        raise ValueError(f"{baseline_path}: incomplete paper/replication pairs")

    regimes = {"on_policy_responses", "fixed_mistral_responses"}
    models = {row["model_key"] for row in extension}
    expected = {(model, dataset, regime) for model in models for dataset in datasets for regime in regimes}
    observed = {(row["model_key"], row["dataset"], row["regime"]) for row in extension}
    if len(models) != 3 or len(extension) != 18 or observed != expected:
        raise ValueError(f"{extension_path}: incomplete three-model extension grid")

    source_path, source_hash = audit.source_fields(baseline_path, extension_path)
    output = []
    for row in extension:
        dataset = row["dataset"]
        regime = row["regime"]
        paper = baseline_values[dataset]["Original paper"]
        replication = baseline_values[dataset]["Replication"]
        extension_value = number(row["mean"], field="mean", source=extension_path)
        fixed = regime == "fixed_mistral_responses"
        output.append(
            make_row(
                study="LLMs Know",
                claim_id="table1_exact_answer_decodability",
                phase="postfreeze_extension",
                comparison_unit="dataset-specific exact-answer test AUROC",
                model=row["model_id"],
                probe_or_method=(
                    "logistic regression; fixed Mistral responses"
                    if fixed
                    else "logistic regression; target-model responses"
                ),
                dataset_or_setting=f"{dataset}; validation-selected layer {row['selected_layer']}",
                metric="AUROC",
                paper=paper,
                replication=replication,
                extension=extension_value,
                delta_valid=fixed,
                retention=(extension_value - 0.5) / (replication - 0.5) if fixed else None,
                retention_definition=(
                    "(extension AUROC - 0.5) / (replication AUROC - 0.5)"
                    if fixed
                    else ""
                ),
                retention_valid=fixed,
                evidence_tier="B" if fixed else "C",
                outcome="exact-answer decodability transfers; magnitude is response-regime dependent",
                selection_regime="five prespecified layers; validation-only selection",
                source_path=source_path,
                source_sha256=source_hash,
                caveat=(
                    "Fixed Mistral responses preserve responses and labels, but target-model tokenization and the "
                    "validation-selected layer still differ."
                    if fixed
                    else "Target-model generations change the response and correctness-label distribution, so the "
                    "cross-model delta is descriptive rather than a pure representation-transfer effect."
                ),
            )
        )
    return output


def validate_output(rows: list[dict[str, str]]) -> None:
    observed_studies = {row["study"] for row in rows}
    expected_studies = CORE_STUDIES | SUPPORTING_STUDIES | POST_FREEZE_STUDIES
    if observed_studies != expected_studies:
        raise ValueError(f"synthesis has the wrong study corpus: {sorted(observed_studies)}")
    observed_core = {row["study"] for row in rows if row["manuscript_role"] == "core"}
    if observed_core != CORE_STUDIES:
        raise ValueError(f"synthesis does not cover exactly the six manuscript core studies: {sorted(observed_core)}")
    keys = set()
    for index, row in enumerate(rows, start=1):
        missing = [field for field in OUTPUT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"output row {index} is missing fields: {missing}")
        required_values = ["study", "claim_id", "phase", "comparison_unit", "model", "metric", "synthesis_grade", "source_path"]
        if any(not row[field] for field in required_values):
            raise ValueError(f"output row {index} has an empty required value")
        key = (row["study"], row["claim_id"], row["model"], row["probe_or_method"], row["dataset_or_setting"])
        if key in keys:
            raise ValueError(f"duplicate synthesis row: {key}")
        keys.add(key)
        if row["manuscript_role"] != STUDY_ROLES[row["study"]]:
            raise ValueError(f"incorrect manuscript role in output row {index}")
        replication = float(row["replication_baseline"])
        if row["paper_baseline"]:
            paper = float(row["paper_baseline"])
            baseline_delta = float(row["delta_replication_vs_paper"])
            if abs(baseline_delta - (replication - paper)) > 5e-9:
                raise ValueError(f"incorrect baseline delta in output row {index}")
        elif row["delta_replication_vs_paper"]:
            raise ValueError(f"baseline delta requires a paper value in output row {index}")
        if row["extension_result"]:
            extension = float(row["extension_result"])
            delta = float(row["delta_extension_vs_replication"])
            if abs(delta - (extension - replication)) > 5e-9:
                raise ValueError(f"incorrect extension delta in output row {index}")
        elif row["delta_extension_vs_replication"] or row["delta_valid"] == "true":
            raise ValueError(f"extension delta requires an extension result in output row {index}")
        if row["retention_valid"] == "true":
            if not row["normalized_retention"] or not row["retention_definition"]:
                raise ValueError(f"valid retention is incomplete in output row {index}")
        elif row["normalized_retention"] or row["retention_definition"]:
            raise ValueError(f"invalid retention must be blank in output row {index}")
        if row["metric"] in {"AUROC", "accuracy", "accuracy gain"}:
            for field in ["replication_baseline", "extension_result"]:
                if not row[field]:
                    continue
                value = float(row[field])
                if not -1e-9 <= value <= 1 + 1e-9:
                    raise ValueError(f"{field} outside [0,1] in output row {index}")


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, rows: list[dict[str, str]], audit: Audit) -> None:
    evidence_path = output_dir / "evidence_long.csv"
    write_csv(evidence_path, rows, OUTPUT_FIELDS)

    outcomes = [
        {
            "study": "Models Under Pressure",
            "manuscript_role": "core",
            "baseline_verdict": "headline reproduced with an activation-site specification gap",
            "extension_verdict": "signal found across sampled models; portable layer selection not established",
            "synthesis_grade": "C",
            "normalized_retention_valid": "false",
            "reason": "extension values are evaluation-selected sparse-layer maxima",
        },
        {
            "study": "Internal Instruction Following",
            "manuscript_role": "core",
            "baseline_verdict": "qualitative and aggregate baseline reproduced; exact table is partial",
            "extension_verdict": "unseen tasks transfer; unseen instruction types remain unstable and label-support dependent",
            "synthesis_grade": "C",
            "normalized_retention_valid": "false",
            "reason": "model-specific on-policy answers change label support and some folds are undefined",
        },
        {
            "study": "Political Perspective",
            "manuscript_role": "core",
            "baseline_verdict": "three original-model peaks reproduced at paper precision",
            "extension_verdict": "mixed oracle peaks; architecture-fair transfer unresolved",
            "synthesis_grade": "C",
            "normalized_retention_valid": "false",
            "reason": "all-head maxima give architectures unequal search budgets",
        },
        {
            "study": "PRISM",
            "manuscript_role": "core",
            "baseline_verdict": "baseline replicated",
            "extension_verdict": "prompt gain transfers to every tested extension model",
            "synthesis_grade": "B",
            "normalized_retention_valid": "true",
            "reason": "prompted-minus-raw accuracy is the same positive effect estimand across models",
        },
        {
            "study": "Lookback Lens",
            "manuscript_role": "core",
            "baseline_verdict": "Table 2 replicated closely",
            "extension_verdict": "cross-dataset transfer remains in the baseline band",
            "synthesis_grade": "B",
            "normalized_retention_valid": "true",
            "reason": "the same four binary-AUROC transfer cells permit chance-adjusted retention",
        },
        {
            "study": "Beyond Linear Probes",
            "manuscript_role": "core",
            "baseline_verdict": "selected Llama layer-16 condition reproduced",
            "extension_verdict": "TPC beats linear across three models but is not universally the best probe",
            "synthesis_grade": "C",
            "normalized_retention_valid": "false",
            "reason": "a ratio of small F1 gains is unstable and only one dataset/layer is covered",
        },
        {
            "study": "Among Us",
            "manuscript_role": "supporting_baseline_failure",
            "baseline_verdict": "Figure 7 mean not reproduced",
            "extension_verdict": "diagnostic only",
            "synthesis_grade": "D",
            "normalized_retention_valid": "false",
            "reason": "the baseline failed and extension layers were selected by evaluation performance",
        },
        {
            "study": "Sycophancy",
            "manuscript_role": "supporting_baseline_failure",
            "baseline_verdict": "Figure 10 peak magnitudes were about 19 percentage points below paper",
            "extension_verdict": "not run",
            "synthesis_grade": "D",
            "normalized_retention_valid": "false",
            "reason": "the baseline failed and no extension result exists",
        },
        {
            "study": "Implicit Personalization",
            "manuscript_role": "supporting_sanity_check",
            "baseline_verdict": "successful reduced selected-layer replication",
            "extension_verdict": "explicit-identity decodability transfers; main stereotype and steering claims untested",
            "synthesis_grade": "C",
            "normalized_retention_valid": "false",
            "reason": "multiclass chance floors are not stored and the experiment covers only a reduced claim",
        },
        {
            "study": "LLMs Know",
            "manuscript_role": "postfreeze_positive_control",
            "baseline_verdict": "principal exact-answer decoding claim recovered; Math magnitude was higher than reported",
            "extension_verdict": "strong transfer across three modern families; response regime changes individual results",
            "synthesis_grade": "B",
            "normalized_retention_valid": "true for fixed-response rows only",
            "reason": "fixed-response comparisons preserve Mistral responses and labels while target-model tokenization and validation-selected sites remain model-specific",
        },
    ]
    write_csv(
        output_dir / "study_outcomes.csv",
        outcomes,
        [
            "study",
            "manuscript_role",
            "baseline_verdict",
            "extension_verdict",
            "synthesis_grade",
            "normalized_retention_valid",
            "reason",
        ],
    )

    audit_payload = {
        "core_studies": sorted(CORE_STUDIES),
        "supporting_studies": sorted(SUPPORTING_STUDIES),
        "post_freeze_studies": sorted(POST_FREEZE_STUDIES),
        "evidence_rows": len(rows),
        "sources": dict(sorted(audit.sources.items())),
        "notes": audit.notes,
        "validation": {
            "exactly_six_core_studies": True,
            "supporting_studies_labelled": True,
            "post_freeze_studies_labelled": True,
            "required_columns_present": True,
            "source_hashes_recorded": True,
            "stored_aggregates_recomputed": True,
            "duplicate_output_rows": 0,
            "invalid_retention_values_emitted": 0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "schema_audit.json").write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in [evidence_path, output_dir / "study_outcomes.csv", output_dir / "schema_audit.json"]
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    requested_output = args.output_dir or Path("data/results/paper-synthesis")
    output_dir = requested_output if requested_output.is_absolute() else root / requested_output
    audit = Audit(root)
    rows = []
    for builder in [
        build_mup,
        build_iif,
        build_political,
        build_prism,
        build_lookback,
        build_blp,
        build_among_us,
        build_sycophancy,
        build_implicit,
        build_llms_know,
    ]:
        rows.extend(builder(audit))
    rows.sort(key=lambda row: (row["study"], row["claim_id"], row["model"], row["probe_or_method"], row["dataset_or_setting"]))
    validate_output(rows)
    if args.validate_only:
        print(
            f"validated {len(rows)} synthesis rows across {len(CORE_STUDIES)} core, "
            f"{len(SUPPORTING_STUDIES)} supporting, and {len(POST_FREEZE_STUDIES)} post-freeze studies"
        )
        return
    write_outputs(output_dir, rows, audit)
    print(f"wrote {len(rows)} synthesis rows to {output_dir}")


if __name__ == "__main__":
    main()
