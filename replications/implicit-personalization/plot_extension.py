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
SOURCE = ROOT / "artifacts/implicit-personalization/extensions/qwen3-8b"
OUTPUT = ROOT / "figures/implicit-personalization"
TABLE = ROOT / "data/results/implicit-personalization/qwen3_8b_extension.csv"
ATTRIBUTES = {
    "race": "Race",
    "age": "Age",
    "gender": "Gender",
    "socio-economic status": "Socio-economic status",
}
COLORS = {"True labels": "#42B883", "Random-label control": "#FF6905"}
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


def load() -> tuple[list[int], dict, dict]:
    metadata = json.loads((SOURCE / "metadata.json").read_text())
    true = pickle.loads((SOURCE / "true.pkl").read_bytes())
    control = pickle.loads((SOURCE / "control.pkl").read_bytes())
    return metadata["layers"], true, control


def write_table(layers: list[int], true: dict, control: dict) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for attribute in ATTRIBUTES:
        for probe, values in (
            ("True labels", true[attribute]),
            ("Random-label control", control[attribute]),
        ):
            for layer, folds in zip(layers, values, strict=True):
                for fold, accuracy in enumerate(folds):
                    rows.append(
                        {
                            "model": "Qwen3-8B",
                            "attribute": attribute,
                            "layer": layer,
                            "probe": probe,
                            "fold": fold,
                            "accuracy_percent": 100 * float(accuracy),
                        }
                    )
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    setup()
    layers, true, control = load()
    write_table(layers, true, control)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), facecolor="white")
    figure.subplots_adjust(
        left=0.08,
        right=0.97,
        top=0.78,
        bottom=0.15,
        hspace=0.42,
        wspace=0.22,
    )
    for axis, (attribute, title) in zip(axes.flat, ATTRIBUTES.items(), strict=True):
        for probe, values in (
            ("True labels", true[attribute]),
            ("Random-label control", control[attribute]),
        ):
            folds = 100 * np.asarray(values, dtype=float)
            mean = folds.mean(axis=1)
            ci = 1.96 * folds.std(axis=1, ddof=1) / np.sqrt(folds.shape[1])
            axis.plot(
                layers,
                mean,
                color=COLORS[probe],
                linewidth=2.4,
                marker="o",
                markersize=4.5,
            )
            axis.fill_between(
                layers,
                mean - ci,
                mean + ci,
                color=COLORS[probe],
                alpha=0.14,
                linewidth=0,
            )
        axis.set_title(title, fontsize=13, fontweight=600)
        axis.set_ylim(0, 104)
        axis.set_xticks(layers)
        axis.set_xlabel("Layer")
        axis.grid(axis="y", color="#E7E9EC", linewidth=0.8)
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.0f}%")
        )
    axes[0, 0].set_ylabel("Probe accuracy")
    axes[1, 0].set_ylabel("Probe accuracy")
    figure.suptitle(
        "Demographic probes generalize to Qwen3-8B",
        x=0.08,
        y=0.95,
        ha="left",
        fontsize=23,
        fontweight=600,
    )
    figure.text(
        0.08,
        0.885,
        "Extension of Reading Between the Prompts · five-fold accuracy across selected layers",
        fontsize=10.5,
        color="#646A70",
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=2.5, label=label)
            for label, color in COLORS.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(
            OUTPUT / f"qwen3_8b_probe_extension.{extension}",
            dpi=240 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
