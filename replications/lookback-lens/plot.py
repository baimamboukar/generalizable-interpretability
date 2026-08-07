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
DATA = ROOT / "data/results/lookback-lens/table2_replication.csv"
OUTPUT = ROOT / "figures/lookback-lens"
FONT = Path.home() / "Library/Fonts/SpaceGrotesk-Variable.ttf"
STATIC_FONT = Path("/tmp/SpaceGrotesk-Regular.ttf")
PAPER = "#303746"
REPLICATION = "#42B883"
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
            "font.size": 10.5,
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
    with DATA.open(newline="") as stream:
        return list(csv.DictReader(stream))


def panel(axis, rows: list[dict], title: str) -> None:
    directions = ("nq-7b", "cnndm-7b")
    splits = ("train", "test", "transfer")
    rows = sorted(
        rows,
        key=lambda row: (directions.index(row["source"]), splits.index(row["split"])),
    )
    labels = [
        f"{'NQ → CNN/DM' if row['source'] == 'nq-7b' else 'CNN/DM → NQ'} · "
        f"{row['split'].title()}"
        for row in rows
    ]
    paper = np.array([float(row["paper"]) * 100 for row in rows])
    replication = np.array([float(row["replication"]) * 100 for row in rows])
    y = np.arange(len(rows))
    height = 0.29
    paper_bars = axis.barh(y - height / 2, paper, height, color=PAPER, zorder=3)
    replication_bars = axis.barh(
        y + height / 2, replication, height, color=REPLICATION, zorder=3
    )
    axis.set_xlim(60, 101)
    axis.set_xticks(np.arange(60, 101, 10))
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axis.tick_params(axis="x", length=0, pad=7)
    axis.tick_params(axis="y", length=0, pad=9)
    axis.spines["bottom"].set_color("#C9CED3")
    axis.set_title(title, fontsize=13, fontweight=600, loc="left", pad=16)
    for bars, values in ((paper_bars, paper), (replication_bars, replication)):
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                value + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                fontsize=8.8,
            )


def main() -> None:
    setup()
    rows = load()
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 7.4), facecolor="white")
    figure.subplots_adjust(left=0.15, right=0.97, top=0.66, bottom=0.17, wspace=0.34)
    panel(
        axes[0],
        [row for row in rows if row["setting"] == "predefined"],
        "Predefined spans",
    )
    panel(
        axes[1],
        [row for row in rows if row["setting"] == "sliding-window-8"],
        "Sliding window (8 tokens)",
    )
    axes[0].set_xlabel("AUROC (%)", labelpad=11)
    axes[1].set_xlabel("AUROC (%)", labelpad=11)
    figure.text(
        0.06,
        0.94,
        "Lookback Lens: paper vs replication",
        fontsize=23,
        fontweight=600,
        ha="left",
    )
    figure.text(
        0.06,
        0.87,
        "Replication of Table 2 · Llama-2-7B-Chat contextual-hallucination probes",
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
        bbox_to_anchor=(0.055, 0.79),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=2.0,
    )
    figure.text(
        0.06,
        0.055,
        "NQ (Natural Questions) · CNN/DM (CNN/DailyMail) · "
        "Train/Test are two-fold source validation; Transfer evaluates the other task.",
        fontsize=9.2,
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


if __name__ == "__main__":
    main()
