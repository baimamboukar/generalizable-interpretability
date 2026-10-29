#!/usr/bin/env python3
"""Render paper-ready figures from the canonical artifact tables."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/generalizable-interpretability-matplotlib")

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/results/paper-artifacts"
OUT = ROOT / "figures/paper-artifacts"
FONT_DIR = ROOT / "assets/fonts/static"
FONT = FONT_DIR / "SpaceGrotesk-Regular.ttf"

INK = "#000103"
MUTED = "#515052"
GRID = "#e7ecef"
BLUE = "#296eb4"
LIGHT_BLUE = "#a3cef1"
GREEN = "#32a287"
LIGHT_GREEN = "#a9fdac"
AMBER = "#ffd07b"
RED = "#ef476f"
LIGHT_RED = "#f6b3c1"


def setup() -> None:
    if not FONT.is_file():
        raise FileNotFoundError(FONT)
    for path in FONT_DIR.glob("SpaceGrotesk-*.ttf"):
        fm.fontManager.addfont(path)
    family = fm.FontProperties(fname=FONT).get_name()
    mpl.rcParams.update({
        "font.family": family,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def read(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def title(fig: plt.Figure, heading: str, subtitle: str) -> None:
    fig.suptitle(heading, x=0.06, y=0.985, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.06, 0.935, subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_accounting() -> None:
    inventory = read("study_inventory.csv")
    recovered = sum(
        row["analysis_role"] not in {"artifact audit", "in-progress baseline", "out of scope"}
        and "not reproduced" not in row["baseline_status"]
        and "diagnostic" not in row["baseline_status"]
        for row in inventory
    )
    categories = [
        ("Baseline recovered or partially recovered", recovered, BLUE),
        ("Baseline not reproduced", sum("not reproduced" in row["baseline_status"] for row in inventory), RED),
        ("Artifact or protocol audit", sum(row["analysis_role"] == "artifact audit" for row in inventory), AMBER),
        ("Baseline in progress", sum(row["analysis_role"] == "in-progress baseline" for row in inventory), GREEN),
        ("Out of probing scope", sum(row["analysis_role"] == "out of scope" for row in inventory), MUTED),
    ]
    labels, values, colors = zip(*categories)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    fig.subplots_adjust(top=0.79, left=0.39, right=0.94, bottom=0.14)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.58)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) + 2.4)
    ax.set_xlabel("Number of studies")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.grid(axis="x", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(value + 0.25, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=11, fontweight="bold")
    ax.text(0, -0.95, f"{len(inventory)} studies audited", fontsize=12, fontweight="bold", color=INK)
    title(fig, "Study accounting", "Every audited study remains visible; blocked releases and out-of-scope methods are not counted as empirical failures.")
    fig.text(0.06, 0.035, "Counts describe the current repository state, not a literature prevalence estimate.", fontsize=9, color=MUTED)
    save(fig, "figure1_study_accounting")


def short_name(study_id: str) -> str:
    return {
        "models-under-pressure": "Models Under Pressure",
        "internal-instruction-following": "Instruction Following",
        "political-perspective": "Political Perspective",
        "prism": "PRISM",
        "lookback-lens": "Lookback Lens",
        "beyond-linear-probes": "Beyond Linear Probes",
        "among-us": "Among Us",
        "sycophancy": "Sycophancy",
        "implicit-personalization": "Implicit Personalization",
        "llms-know": "LLMs Know",
        "test-awareness-steering": "Test Awareness",
    }[study_id]


def plot_baseline_fidelity() -> None:
    rows = read("baseline_fidelity_summary.csv")
    failed_ids = {"among-us", "sycophancy"}
    failed = [row for row in rows if row["study_id"] in failed_ids]
    recovered = [row for row in rows if row["study_id"] not in failed_ids]
    recovered.sort(key=lambda row: float(row["mean_absolute_delta"]), reverse=True)
    failed.sort(key=lambda row: float(row["mean_absolute_delta"]), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.2), gridspec_kw={"width_ratios": [1.55, 0.9]})
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.22, right=0.96, wspace=0.30)
    panels = [
        (axes[0], recovered, "Recovered or partial baselines", BLUE, 3.2),
        (axes[1], failed, "Baselines not reproduced", RED, 21.5),
    ]
    for ax, members, heading, color, xmax in panels:
        labels = [short_name(row["study_id"]) for row in members]
        values = [100 * float(row["mean_absolute_delta"]) for row in members]
        y = np.arange(len(members))
        bars = ax.barh(y, values, color=color, height=0.58)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(heading, loc="left", pad=15)
        ax.set_xlabel("Mean absolute difference on the reported 0–100 scale")
        ax.grid(axis="x", color=GRID, linewidth=0.9)
        ax.set_axisbelow(True)
        for bar, row, score in zip(bars, members, values):
            label = f"{score:.2f} points  ·  n={row['comparison_cells']}"
            ax.text(score + xmax * 0.018, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8.7, color=INK)
    title(fig, "Baseline fidelity", "Within-study differences over directly comparable paper and replication cells; scales are separated so small deviations remain visible.")
    fig.text(0.06, 0.035, "Each source metric is first expressed on its reported 0–1 scale, then multiplied by 100. Metric families are not pooled for inference.", fontsize=9, color=MUTED)
    save(fig, "figure2_baseline_fidelity")


def clean_model(model: str) -> str:
    aliases = {
        "Qwen/Qwen3-14B": "Qwen3-14B",
        "google/gemma-4-12B-it": "Gemma-4-12B",
        "zai-org/GLM-4.7-Flash": "GLM-4.7-Flash",
        "GLM-4-9B-Chat": "GLM-4-9B",
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "Moonlight-16B-A3B-Instruct": "Moonlight-16B",
        "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    }
    return aliases.get(model, model)


def plot_portability() -> None:
    rows = [row for row in read("cross_model_extensions.csv") if row["retention_valid"] == "true"]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["study"], clean_model(row["model"]))].append(100 * float(row["normalized_retention"]))
    studies = ["PRISM", "Lookback Lens", "LLMs Know"]
    colors = {"PRISM": BLUE, "Lookback Lens": GREEN, "LLMs Know": "#8b8c89"}
    entries = []
    for study in studies:
        models = sorted((model, mean(values)) for (name, model), values in grouped.items() if name == study)
        for model, score in models:
            entries.append((study, model, score))
        if study != studies[-1]:
            entries.append(("", "", float("nan")))

    fig, ax = plt.subplots(figsize=(12.4, 8.2))
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.25, right=0.95)
    y = np.arange(len(entries))
    for index, (study, model, score) in enumerate(entries):
        if not np.isfinite(score):
            continue
        ax.barh(index, score, left=0, color=colors[study], height=0.58)
        ax.text(score + 1.2, index, f"{score:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y, [model for _, model, _ in entries])
    ax.invert_yaxis()
    ax.set_xlim(70, 160)
    ax.axvline(100, color=INK, linewidth=1.2, linestyle=(0, (3, 3)))
    ax.text(100.7, -0.75, "baseline parity", fontsize=8.5, color=MUTED)
    ax.set_xlabel("Normalized retention of the replicated baseline effect")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="x", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    offset = 0
    for study in studies:
        count = sum(1 for name, _, _ in entries if name == study)
        ax.text(71, offset - 0.72, study, fontsize=11, fontweight="bold", color=colors[study])
        offset += count + 1
    title(fig, "Comparable cross-model portability", "Only extensions with the same effect estimand are shown; oracle-selected and baseline-failed diagnostics are excluded.")
    fig.text(0.06, 0.035, "PRISM retains prompt-guided gain. Lookback Lens and LLMs Know use chance-adjusted AUROC retention.", fontsize=9, color=MUTED)
    save(fig, "figure3_cross_model_portability")


def contract_category(field: str, text: str) -> tuple[int, str]:
    lower = text.lower()
    if "not applicable" in lower or lower == "not run":
        return 3, "N/A"
    if "not retained" in lower:
        return 2, "Missing"
    if "oracle" in lower or "target-test" in lower or "diagnostic" in lower:
        return 2, "Test-selected"
    if field == "extension_estimand" and "response-policy control" in lower:
        return 1, "Controlled"
    if field == "extension_estimand" and "policy shift" in lower:
        return 1, "Policy-shifted"
    if "on-policy" in lower or "model-dependent" in lower or "target-selected" in lower:
        return 1, "Model-specific"
    if field == "activation_site_selection" and "validation-selected" in lower:
        return 1, "Dev-selected"
    if field == "extension_estimand" and "baseline" in lower:
        return 3, "Baseline only"
    if field == "probe_architecture" and "compared" in lower:
        return 1, "Compared"
    if field == "threshold_protocol" and "paper default" in lower:
        return 1, "Paper default"
    if "fixed-response control" in lower:
        return 1, "Controlled"
    if "fixed" in lower or "prespecified" in lower or "all layers" in lower or "method portability" in lower or "architecture portability" in lower:
        return 0, "Fixed"
    return 1, "Varies"


def plot_contracts() -> None:
    rows = read("evaluation_contracts.csv")
    fields = [
        "examples_across_models", "responses_across_models", "labels_across_models",
        "activation_site_selection", "probe_architecture", "threshold_protocol", "extension_estimand",
    ]
    columns = ["Examples", "Responses", "Labels", "Activation site", "Probe", "Threshold", "Estimand"]
    values = np.zeros((len(rows), len(fields)))
    labels: list[list[str]] = []
    for i, row in enumerate(rows):
        line = []
        for j, field in enumerate(fields):
            category, label = contract_category(field, row[field])
            values[i, j] = category
            line.append(label)
        labels.append(line)

    fig, ax = plt.subplots(figsize=(14.8, 8.1))
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.24, right=0.98)
    cmap = ListedColormap([LIGHT_GREEN, AMBER, LIGHT_RED, GRID])
    ax.imshow(values, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(np.arange(len(fields)), columns)
    ax.set_yticks(np.arange(len(rows)), [short_name(row["study_id"]) for row in rows])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    for i in range(len(rows)):
        for j in range(len(fields)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8.1, color=INK)
    ax.set_xticks(np.arange(-0.5, len(fields), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = [
        Patch(facecolor=LIGHT_GREEN, label="Fixed or prespecified"),
        Patch(facecolor=AMBER, label="Model- or development-dependent"),
        Patch(facecolor=LIGHT_RED, label="Test-selected or missing"),
        Patch(facecolor=GRID, label="Not applicable / not run"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.57, 0.075), fontsize=9)
    title(fig, "What changes when the model changes?", "The cross-model evidence varies not only in backbone, but also in responses, labels, activation-site selection, and operating points.")
    fig.text(0.06, 0.025, "This table describes evaluation contracts; it does not score scientific quality.", fontsize=9, color=MUTED)
    save(fig, "figure4_evaluation_contracts")


def main() -> None:
    setup()
    plot_accounting()
    plot_baseline_fidelity()
    plot_portability()
    plot_contracts()
    for path in sorted(OUT.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
