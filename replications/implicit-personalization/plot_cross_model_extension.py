from __future__ import annotations

import csv
import json
from pathlib import Path
import pickle

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "data/results/implicit-personalization/probe_validation_baseline.csv"
ARTIFACTS = ROOT / "artifacts/implicit-personalization/extensions"
TABLE = ROOT / "data/results/implicit-personalization/cross_model_extension.csv"
OUTPUT = ROOT / "figures/implicit-personalization"
MODELS = {
    "Gemma-2-9B-IT": "Gemma 2\n9B",
    "Llama-3.1-8B-Instruct": "Llama 3.1\n8B",
    "OLMo-2-7B-Instruct": "OLMo 2\n7B",
    "qwen3-8b": "Qwen 3\n8B",
    "glm4-9b": "GLM-4\n9B",
    "phi4-14b": "Phi-4\n14B",
}
ATTRIBUTES = {
    "race": "Race",
    "age": "Age",
    "gender": "Gender",
    "socio-economic status": "Socio-economic status",
}
COLORS = {
    "Paper": "#1D3557",
    "Replication": "#42B883",
    "Control": "#FF6905",
}
FONT = Path.home() / "Library/Fonts/SpaceGrotesk-Variable.ttf"
STATIC_FONT = Path("/tmp/SpaceGrotesk-Regular.ttf")


def setup() -> None:
    if FONT.exists():
        if not STATIC_FONT.exists():
            font = TTFont(FONT)
            instantiateVariableFont(font, {"wght": 400}, inplace=True)
            font.save(STATIC_FONT)
        font_manager.fontManager.addfont(STATIC_FONT)
    plt.rcParams.update(
        {
            "font.family": "Space Grotesk",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def baseline_rows() -> list[dict]:
    with BASELINE.open() as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for model in list(MODELS)[:3]:
        for attribute in ATTRIBUTES:
            selected = [
                row
                for row in source
                if row["model"] == model and row["attribute"] == attribute
            ]
            for result, field in (
                ("Paper", "paper_value_approx"),
                ("Replication", "replication_value"),
            ):
                for row in selected:
                    rows.append(
                        {
                            "model": model,
                            "attribute": attribute,
                            "layer": int(row["layer"]),
                            "result": result,
                            "fold": "",
                            "accuracy_percent": float(row[field]),
                        }
                    )
    return rows


def extension_rows() -> list[dict]:
    rows = []
    for model in list(MODELS)[3:]:
        path = ARTIFACTS / model
        layers = json.loads((path / "metadata.json").read_text())["layers"]
        true = pickle.loads((path / "true.pkl").read_bytes())
        control = pickle.loads((path / "control.pkl").read_bytes())
        for attribute in ATTRIBUTES:
            for result, values in (
                ("Replication", true[attribute]),
                ("Control", control[attribute]),
            ):
                for layer, folds in zip(layers, values, strict=True):
                    for fold, accuracy in enumerate(folds):
                        rows.append(
                            {
                                "model": model,
                                "attribute": attribute,
                                "layer": layer,
                                "result": result,
                                "fold": fold,
                                "accuracy_percent": 100 * float(accuracy),
                            }
                        )
    return rows


def write_table(rows: list[dict]) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], model: str, attribute: str, result: str):
    values = np.asarray(
        [
            row["accuracy_percent"]
            for row in rows
            if row["model"] == model
            and row["attribute"] == attribute
            and row["result"] == result
        ],
        dtype=float,
    )
    if not len(values):
        return None
    ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
    return values.mean(), ci


def main() -> None:
    setup()
    rows = baseline_rows() + extension_rows()
    write_table(rows)
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 8.1), facecolor="white")
    figure.subplots_adjust(
        left=0.075,
        right=0.98,
        top=0.79,
        bottom=0.18,
        hspace=0.45,
        wspace=0.18,
    )
    models = list(MODELS)
    positions = np.arange(len(models))
    offsets = {"Paper": -0.13, "Replication": 0, "Control": 0.13}
    markers = {"Paper": "D", "Replication": "o", "Control": "X"}

    for axis, (attribute, title) in zip(axes.flat, ATTRIBUTES.items(), strict=True):
        axis.axvspan(-0.5, 2.5, color="#F4F7FA", zorder=0)
        axis.axvspan(2.5, 5.5, color="#F5FBF8", zorder=0)
        axis.axvline(2.5, color="#CDD3D9", linewidth=1)
        for index, model in enumerate(models):
            for result in ("Paper", "Replication", "Control"):
                summary = summarize(rows, model, attribute, result)
                if summary is None:
                    continue
                mean, ci = summary
                axis.errorbar(
                    index + offsets[result],
                    mean,
                    yerr=ci,
                    color=COLORS[result],
                    marker=markers[result],
                    markersize=6.5,
                    markeredgecolor="white" if result != "Control" else COLORS[result],
                    markeredgewidth=0.8,
                    capsize=2.5,
                    linewidth=1.5,
                    zorder=3,
                )
        axis.set_title(title, fontsize=13, fontweight=600, pad=27)
        axis.text(
            0.25,
            1.015,
            "Original models",
            transform=axis.transAxes,
            ha="center",
            fontsize=8.3,
            color="#66717C",
        )
        axis.text(
            0.75,
            1.015,
            "Extensions",
            transform=axis.transAxes,
            ha="center",
            fontsize=8.3,
            color="#4D7968",
        )
        axis.set_xlim(-0.5, len(models) - 0.5)
        axis.set_ylim(0, 105)
        axis.set_xticks(positions, [MODELS[model] for model in models])
        axis.tick_params(axis="x", labelsize=8.7)
        axis.grid(axis="y", color="#E2E6EA", linewidth=0.8)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))

    axes[0, 0].set_ylabel("Mean probe accuracy")
    axes[1, 0].set_ylabel("Mean probe accuracy")
    figure.suptitle(
        "Demographic information remains linearly separable across model families",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=22,
        fontweight=600,
    )
    figure.text(
        0.075,
        0.91,
        "Reading Between the Prompts · paper baselines, reduced replication, and cross-model extensions",
        fontsize=10.4,
        color="#60666C",
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=markers[result],
                markersize=7,
                markerfacecolor=COLORS[result],
                markeredgecolor=COLORS[result],
                label=label,
            )
            for result, label in (
                ("Paper", "Paper (digitized)"),
                ("Replication", "Replication / extension"),
                ("Control", "Random-label control"),
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.075,
        0.025,
        "Points average five selected layers; extension points also average five folds. "
        "Error bars show 95% CIs. No paper reference exists for extension models.",
        fontsize=8.5,
        color="#737A81",
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(
            OUTPUT / f"cross_model_probe_extension.{extension}",
            dpi=240 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
