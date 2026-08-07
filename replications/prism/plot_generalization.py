from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "results" / "prism" / "cross_model_summary.csv"
OUTPUT = ROOT / "figures" / "prism"
MODELS = [
    ("Llama-2-7B-Chat", "Llama-2\n7B"),
    ("Llama-2-13B-Chat", "Llama-2\n13B"),
    ("Llama-3.1-8B-Instruct", "Llama-3.1\n8B"),
    ("GLM-4-9B-Chat", "GLM-4\n9B"),
    ("Qwen2.5-14B-Instruct", "Qwen2.5\n14B"),
    ("Moonlight-16B-A3B-Instruct", "Moonlight\n16B-A3B"),
]
BASELINE_RAW = "#303746"
BASELINE_PRISM = "#2F7FD3"
EXTENSION_RAW = "#49B98A"
EXTENSION_PRISM = "#FF6905"
TEXT = "#17191C"
MUTED = "#73767C"
GRID = "#E6E8EB"
CHANCE = "#E6A19A"
FONT = Path.home() / "Library" / "Fonts" / "SpaceGrotesk-Variable.ttf"
STATIC_FONT = Path("/tmp/SpaceGrotesk-Regular.ttf")


def load() -> dict[tuple[str, str, str, str], float]:
    with DATA.open(newline="") as handle:
        return {
            (
                row["model"],
                row["dataset"],
                row["method"],
                row["representation"],
            ): float(row["replication_value"])
            for row in csv.DictReader(handle)
            if row["status"] == "completed"
        }


def mean_score(
    values: dict[tuple[str, str, str, str], float],
    model: str,
    dataset: str,
    representation: str,
) -> float:
    return np.mean(
        [
            values[(model, dataset, method, representation)]
            for method in ("MM", "SAPLMA")
        ]
    )


def main() -> None:
    if FONT.exists():
        if not STATIC_FONT.exists():
            font = TTFont(FONT)
            instantiateVariableFont(font, {"wght": 400}, inplace=True)
            font.save(STATIC_FONT)
        font_manager.fontManager.addfont(STATIC_FONT)
    plt.rcParams.update(
        {
            "font.family": "Space Grotesk",
            "font.size": 12,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": MUTED,
            "text.color": TEXT,
        }
    )
    values = load()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.19, top=0.63, wspace=0.10)

    for ax, dataset in zip(axes, ("True-False", "LogicStruct")):
        raw = [mean_score(values, model, dataset, "raw") for model, _ in MODELS]
        prompted = [
            mean_score(values, model, dataset, "prompted") for model, _ in MODELS
        ]
        x = np.arange(len(MODELS))
        width = 0.34
        raw_colors = [BASELINE_RAW] * 2 + [EXTENSION_RAW] * 4
        prism_colors = [BASELINE_PRISM] * 2 + [EXTENSION_PRISM] * 4
        raw_bars = ax.bar(x - width / 2, raw, width, color=raw_colors, zorder=3)
        prism_bars = ax.bar(
            x + width / 2,
            prompted,
            width,
            color=prism_colors,
            zorder=3,
        )

        ax.axhline(0.5, color=CHANCE, linewidth=1.5, linestyle=(0, (2, 3)))
        ax.axvline(1.5, color="#D0D2D5", linewidth=1.1)
        ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
        ax.set_ylim(0.45, 0.96)
        ax.set_xticks(x, [label for _, label in MODELS])
        ax.tick_params(axis="x", length=0, pad=12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.set_title(dataset, fontsize=16, pad=16)
        ax.text(
            0.5,
            0.94,
            "ORIGINAL BASELINE",
            ha="center",
            va="center",
            fontsize=9,
            color=BASELINE_PRISM,
            weight="bold",
        )
        ax.text(
            3.5,
            0.94,
            "CROSS-MODEL EXTENSION",
            ha="center",
            va="center",
            fontsize=9,
            color=EXTENSION_PRISM,
            weight="bold",
        )
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#D0D2D5")

        for bars in (raw_bars, prism_bars):
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.011,
                    f"{bar.get_height():.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    axes[0].set_ylabel("Cross-domain accuracy", labelpad=12)
    fig.text(
        0.04,
        0.92,
        "PRISM cross-model generalization",
        fontsize=28,
        ha="left",
    )
    fig.text(
        0.04,
        0.855,
        "Extension of original Tables 3 and 5 · MM (mass-mean probe) · SAPLMA (Statement Accuracy Prediction based on Language Model Activations)",
        fontsize=11.5,
        color=MUTED,
        ha="left",
    )
    fig.legend(
        handles=[
            Patch(color=BASELINE_RAW, label="Baseline · statement only"),
            Patch(color=BASELINE_PRISM, label="Baseline · truth prompt"),
            Patch(color=EXTENSION_RAW, label="Extension · statement only"),
            Patch(color=EXTENSION_PRISM, label="Extension · truth prompt"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.77),
        ncol=4,
        frameon=False,
        columnspacing=2.2,
    )
    fig.text(
        0.04,
        0.055,
        "Statement only = final-layer state for the original input · Truth prompt = state after asking whether the statement reflects truth · dotted line = chance",
        fontsize=11,
        color="#92959A",
        ha="left",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "cross_model_generalization.png", dpi=240, facecolor="white")
    fig.savefig(OUTPUT / "cross_model_generalization.pdf", facecolor="white")
    fig.savefig(OUTPUT / "cross_model_generalization.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
