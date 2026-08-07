from __future__ import annotations

import csv
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/results/political-perspective/independent_baseline.csv"
MEGA = ROOT / "data/results/political-perspective/mega_summary.csv"
REMAINING = ROOT / "data/results/political-perspective/remaining_baseline_claims.csv"
OUTPUT = ROOT / "figures/political-perspective"
FONT = Path.home() / "Library/Fonts/SpaceGrotesk-Variable.ttf"
STATIC_FONT = Path("/tmp/SpaceGrotesk-Regular.ttf")
PAPER = "#303746"
REPLICATION = "#42B883"
EXTENSION = "#FF6905"
DIVIDER = "#D74747"
TEXT = "#17191C"
MUTED = "#6D747B"
GRID = "#E4E7EA"


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
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
        }
    )


def load() -> list[dict]:
    with DATA.open(newline="") as handle:
        return list(csv.DictReader(handle))


def baseline_figure() -> None:
    rows = load()
    labels = [
        "Llama-2-7B-Chat",
        "Mistral-7B-Instruct-v0.1",
        "Vicuna-7B-v1.5",
    ]
    rows = sorted(rows, key=lambda row: labels.index(row["model"]))
    paper = np.array([float(row["paper"]) for row in rows])
    replication = np.array([float(row["replication"]) for row in rows])

    figure, axis = plt.subplots(figsize=(11.2, 6.6), facecolor="white")
    figure.subplots_adjust(left=0.23, right=0.96, top=0.67, bottom=0.20)
    y = np.arange(len(rows))
    height = 0.28
    paper_bars = axis.barh(
        y - height / 2,
        paper,
        height,
        color=PAPER,
        label="Original paper",
        zorder=3,
    )
    replication_bars = axis.barh(
        y + height / 2,
        replication,
        height,
        color=REPLICATION,
        label="Replication",
        zorder=3,
    )

    axis.set_xlim(0, 0.90)
    axis.set_xticks(np.arange(0, 0.91, 0.1))
    axis.set_yticks(y, [row["model"] for row in rows])
    axis.invert_yaxis()
    axis.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axis.tick_params(axis="y", length=0, pad=12)
    axis.tick_params(axis="x", length=0, pad=8)
    axis.set_xlabel("Peak single-head CV Spearman correlation", labelpad=12)
    axis.spines["bottom"].set_color("#C9CED3")

    for bars, values in ((paper_bars, paper), (replication_bars, replication)):
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                value + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                fontsize=10.5,
            )

    figure.text(
        0.07,
        0.93,
        "Political perspective probes: paper vs replication",
        fontsize=23,
        fontweight=600,
        ha="left",
    )
    figure.text(
        0.07,
        0.855,
        "Replication of Figure 2 in the original paper · peak attention-head ridge probe performance",
        fontsize=11,
        color=MUTED,
        ha="left",
    )
    figure.legend(
        handles=[
            Patch(color=PAPER, label="Original paper"),
            Patch(color=REPLICATION, label="Replication"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.065, 0.79),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=2.0,
    )
    figure.text(
        0.07,
        0.055,
        "Two-fold cross-validation, ridge alpha 1, no intercept. "
        "Peak layer and attention head matched exactly for all three models.",
        fontsize=9.5,
        color=MUTED,
        ha="left",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT / "paper_vs_replication.png",
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT / "paper_vs_replication.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def extension_figure() -> None:
    with MEGA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    order = [
        "Llama-2-7B-Chat",
        "Mistral-7B-Instruct-v0.1",
        "Vicuna-7B-v1.5",
        "Qwen/Qwen3-8B",
        "zai-org/GLM-4-9B-0414",
        "google/gemma-2-9b-it",
        "microsoft/phi-4",
        "google/gemma-2-27b-it",
    ]
    labels = {
        "Llama-2-7B-Chat": "Llama-2 7B Chat",
        "Mistral-7B-Instruct-v0.1": "Mistral 7B Instruct",
        "Vicuna-7B-v1.5": "Vicuna 7B",
        "Qwen/Qwen3-8B": "Qwen3 8B",
        "zai-org/GLM-4-9B-0414": "GLM-4 9B",
        "google/gemma-2-9b-it": "Gemma-2 9B",
        "microsoft/phi-4": "Phi-4 14B",
        "google/gemma-2-27b-it": "Gemma-2 27B",
    }
    rows = sorted(rows, key=lambda row: order.index(row["model"]))
    values = np.array([float(row["observed_peak"]) for row in rows])
    colors = [
        REPLICATION if row["group"] == "baseline replication" else EXTENSION
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(11.4, 8.0), facecolor="white")
    figure.subplots_adjust(left=0.24, right=0.95, top=0.74, bottom=0.16)
    y = np.arange(len(rows))
    bars = axis.barh(y, values, height=0.54, color=colors, zorder=3)
    axis.axhline(2.5, color=DIVIDER, linewidth=1.5, zorder=4)
    paper_mean = np.mean(
        [float(row["paper_peak"]) for row in rows if row["paper_peak"]]
    )
    axis.axvline(
        paper_mean,
        color="#8A9097",
        linewidth=1.2,
        linestyle=(0, (4, 4)),
        zorder=2,
    )
    axis.text(
        paper_mean + 0.006,
        -0.52,
        f"Original-paper mean  {paper_mean:.4f}",
        color=MUTED,
        fontsize=9.5,
        va="bottom",
    )

    axis.set_xlim(0, 0.94)
    axis.set_xticks(np.arange(0, 0.91, 0.1))
    axis.set_yticks(y, [labels[row["model"]] for row in rows])
    axis.invert_yaxis()
    axis.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axis.tick_params(axis="y", length=0, pad=12)
    axis.tick_params(axis="x", length=0, pad=8)
    axis.set_xlabel("Peak single-head CV Spearman correlation", labelpad=12)
    axis.spines["bottom"].set_color("#C9CED3")

    for bar, value in zip(bars, values, strict=True):
        axis.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=10.5,
        )

    axis.text(
        -0.22,
        1.0,
        "BASELINE\nREPLICATIONS",
        transform=axis.get_yaxis_transform(),
        color=REPLICATION,
        fontsize=9,
        fontweight=600,
        ha="right",
        va="center",
    )
    axis.text(
        -0.22,
        5.0,
        "CROSS-MODEL\nEXTENSIONS",
        transform=axis.get_yaxis_transform(),
        color=EXTENSION,
        fontsize=9,
        fontweight=600,
        ha="right",
        va="center",
    )

    figure.text(
        0.07,
        0.94,
        "Political perspective probes across model families",
        fontsize=23,
        fontweight=600,
        ha="left",
    )
    figure.text(
        0.07,
        0.865,
        "Extension of Figure 2 · strongest attention-head ridge probe from every layer and head",
        fontsize=11,
        color=MUTED,
        ha="left",
    )
    figure.legend(
        handles=[
            Patch(color=REPLICATION, label="Paper-model baseline replication"),
            Patch(color=EXTENSION, label="New model extension"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.065, 0.81),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=2.0,
    )
    figure.text(
        0.07,
        0.045,
        "Same 552 politicians, raw prompt, final-token pre-projection head activations, "
        "ridge (alpha 1, no intercept), and two-fold CV (seed 42).",
        fontsize=9.5,
        color=MUTED,
        ha="left",
    )

    figure.savefig(
        OUTPUT / "baseline_vs_extensions.png",
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT / "baseline_vs_extensions.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def remaining_claims_figure() -> None:
    with REMAINING.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ensemble = [row for row in rows if row["target"] == "Table A3 K=32"]
    news = [row for row in rows if row["target"] == "Figure 3 news"]
    labels = {
        "Llama-2-7B-Chat": "Llama-2 7B",
        "Mistral-7B-Instruct-v0.1": "Mistral 7B",
        "Vicuna-7B-v1.5": "Vicuna 7B",
    }

    figure, axes = plt.subplots(1, 2, figsize=(13.4, 6.8), facecolor="white")
    figure.subplots_adjust(left=0.08, right=0.97, top=0.70, bottom=0.21, wspace=0.30)

    x = np.arange(3)
    width = 0.34
    paper = np.array([float(row["paper"]) for row in ensemble])
    replication = np.array([float(row["replication"]) for row in ensemble])
    paper_bars = axes[0].bar(x - width / 2, paper, width, color=PAPER, zorder=3)
    replication_bars = axes[0].bar(
        x + width / 2, replication, width, color=REPLICATION, zorder=3
    )
    axes[0].set_title("Top-32 head ensemble", loc="left", fontsize=14, fontweight=600)
    axes[0].set_xticks(x, [labels[row["model"]] for row in ensemble])
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_ylim(0, 0.94)
    axes[0].grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axes[0].tick_params(axis="both", length=0)
    for bars, values in ((paper_bars, paper), (replication_bars, replication)):
        for bar, value in zip(bars, values, strict=True):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.4f}",
                ha="center",
                fontsize=9.5,
            )

    news_values = np.array([float(row["replication"]) for row in news])
    news_bars = axes[1].bar(x, news_values, 0.52, color=REPLICATION, zorder=3)
    paper_news = float(next(row["paper"] for row in news if row["paper"]))
    axes[1].axhline(
        paper_news, color=PAPER, linewidth=1.5, linestyle=(0, (4, 4)), zorder=2
    )
    axes[1].text(
        2.42,
        paper_news + 0.012,
        f"Paper Llama-2  {paper_news:.3f}",
        color=MUTED,
        ha="right",
        fontsize=9.5,
    )
    axes[1].set_title(
        "Politician → news transfer", loc="left", fontsize=14, fontweight=600
    )
    axes[1].set_xticks(x, [labels[row["model"]] for row in news])
    axes[1].set_ylim(0, 0.94)
    axes[1].grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axes[1].tick_params(axis="both", length=0)
    for bar, value in zip(news_bars, news_values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            fontsize=9.5,
        )

    figure.text(
        0.06,
        0.93,
        "Political perspective probes: remaining baseline claims",
        fontsize=23,
        fontweight=600,
        ha="left",
    )
    figure.text(
        0.06,
        0.855,
        "Replication of Table A3 and Figure 3 in the original paper",
        fontsize=11,
        color=MUTED,
        ha="left",
    )
    figure.legend(
        handles=[
            Patch(color=PAPER, label="Original paper"),
            Patch(color=REPLICATION, label="Replication"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.055, 0.80),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.06,
        0.055,
        "Top 32 heads selected from politician probe performance. "
        "The same two fold-trained ensembles are transferred to 400 Ad Fontes-labelled outlets; "
        "the paper reports the news result for Llama-2.",
        fontsize=9.3,
        color=MUTED,
        ha="left",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT / "remaining_baseline_claims.png",
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT / "remaining_baseline_claims.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    setup()
    baseline_figure()
    extension_figure()
    remaining_claims_figure()


if __name__ == "__main__":
    main()
