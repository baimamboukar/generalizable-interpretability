from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import pickle

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GI_ARTIFACT_DIR", ROOT / "artifacts"))
DATA = DATA / "implicit-personalization" / "probe-validation"
PAPER = DATA / "paper-baselines"
OUTPUT = ROOT / "figures" / "implicit-personalization"
MODELS = {
    "gemma": "Gemma",
    "llama": "Llama",
    "olmo": "OLMo",
}
ATTRIBUTES = {
    "race": "Figure 2 · Race",
    "age": "Figure 9 · Age",
    "gender": "Figure 10 · Gender",
    "socio-economic status": "Figure 11 · SES",
}
COLORS = {"True labels": "#3478C4", "Random labels": "#FF6905"}
FONT = Path.home() / "Library" / "Fonts" / "SpaceGrotesk-Variable.ttf"
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


def load(model: str) -> tuple[list[int], dict, dict]:
    folder = DATA / model
    metadata = json.loads((folder / "metadata.json").read_text())
    true = pickle.loads((folder / "true.pkl").read_bytes())
    control = pickle.loads((folder / "control.pkl").read_bytes())
    return metadata["layers"], true, control


def write_long_table(values: dict) -> None:
    records = []
    for model, (layers, true, control) in values.items():
        for attribute in ATTRIBUTES:
            for kind, scores in (("True labels", true), ("Random labels", control)):
                for layer, folds in zip(layers, scores[attribute], strict=True):
                    for fold, accuracy in enumerate(folds):
                        records.append(
                            {
                                "source": "Replication",
                                "model": MODELS[model],
                                "attribute": attribute,
                                "probe": kind,
                                "layer": layer,
                                "fold": fold,
                                "accuracy": float(accuracy),
                            }
                        )
    with (DATA / "probe_validation_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def plot_replication(axis, layers: list[int], true: dict, control: dict, attribute: str) -> None:
    for kind, scores in (("True labels", true), ("Random labels", control)):
        folds = np.asarray(scores[attribute], dtype=float) * 100
        mean = folds.mean(axis=1)
        ci = 1.96 * folds.std(axis=1, ddof=1) / np.sqrt(folds.shape[1])
        axis.plot(
            layers,
            mean,
            color=COLORS[kind],
            linewidth=2.2,
            marker="o",
            markersize=4,
        )
        axis.fill_between(
            layers,
            mean - ci,
            mean + ci,
            color=COLORS[kind],
            alpha=0.13,
            linewidth=0,
        )
    axis.set_ylim(0, 103)
    axis.set_xticks(layers)
    axis.tick_params(axis="x", labelrotation=35, labelsize=8)
    axis.grid(axis="y", color="#E6E8EB", linewidth=0.8)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    axis.set_xlabel("Layer", labelpad=2)


def comparison_figure(values: dict, attributes: tuple[str, str], stem: str) -> None:
    figure = plt.figure(figsize=(15.5, 8.2), facecolor="white")
    outer = figure.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.98,
        top=0.78,
        bottom=0.12,
        hspace=0.56,
        wspace=0.12,
        width_ratios=(1, 1.13),
    )

    for row, attribute in enumerate(attributes):
        paper_axis = figure.add_subplot(outer[row, 0])
        paper_axis.imshow(mpimg.imread(PAPER / f"{'ses' if attribute == 'socio-economic status' else attribute}.png"))
        paper_axis.axis("off")
        paper_axis.text(
            0,
            1.1,
            ATTRIBUTES[attribute],
            transform=paper_axis.transAxes,
            fontsize=15,
            fontweight=600,
            ha="left",
        )

        replication = outer[row, 1].subgridspec(1, 3, wspace=0.27)
        for column, model in enumerate(MODELS):
            axis = figure.add_subplot(replication[0, column])
            layers, true, control = values[model]
            plot_replication(axis, layers, true, control, attribute)
            axis.set_title(MODELS[model], fontsize=12, pad=7)
            if column == 0:
                axis.set_ylabel("Accuracy")
            else:
                axis.set_yticklabels([])

    figure.suptitle(
        "Probe validation: paper vs replication",
        x=0.055,
        y=0.955,
        ha="left",
        fontsize=24,
        fontweight=600,
    )
    figure.text(
        0.055,
        0.895,
        "Published all-layer baseline alongside our reduced five-layer replication",
        fontsize=11,
        color="#63686E",
    )
    figure.text(0.255, 0.82, "Paper", ha="center", fontsize=14, fontweight=600)
    figure.text(
        0.745,
        0.82,
        "Our replication",
        ha="center",
        fontsize=14,
        fontweight=600,
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color=COLORS["True labels"], linewidth=2.4, label="True labels"),
            Line2D([0], [0], color=COLORS["Random labels"], linewidth=2.4, label="Random-label control"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.98,
        0.035,
        "Paper panels: Neplenbroek et al. (2025) · Replication bands: 95% CI across 5 folds",
        ha="right",
        fontsize=8.5,
        color="#7A7F85",
    )
    for extension in ("png", "pdf"):
        figure.savefig(
            OUTPUT / f"{stem}.{extension}",
            dpi=240 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    setup()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    values = {model: load(model) for model in MODELS}
    write_long_table(values)
    comparison_figure(
        values,
        ("race", "age"),
        "probe_validation_paper_vs_replication_race_age",
    )
    comparison_figure(
        values,
        ("gender", "socio-economic status"),
        "probe_validation_paper_vs_replication_gender_ses",
    )


if __name__ == "__main__":
    main()
