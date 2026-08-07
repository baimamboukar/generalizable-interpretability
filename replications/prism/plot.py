from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "results" / "prism" / "baseline_summary.csv"
OUTPUT = ROOT / "figures" / "prism"
PAPER = "#303746"
REPLICATION = "#2F7FD3"
TEXT = "#17191C"
MUTED = "#73767C"
GRID = "#E6E8EB"
CHANCE = "#E6A19A"


def load() -> dict[tuple[str, str, str, str], tuple[float, float]]:
    with DATA.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (row["model"], row["dataset"], row["method"], row["representation"]): (
                float(row["paper_value"]),
                float(row["replication_value"]),
            )
            for row in rows
            if row["status"] == "completed"
        }


def label(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Helvetica Neue",
            "font.size": 12,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
        }
    )
    values = load()
    conditions = [
        ("MM", "raw", "MM\nRaw"),
        ("MM", "prompted", "PRISM–MM\nPrompt-guided"),
        ("SAPLMA", "raw", "SAPLMA\nRaw"),
        ("SAPLMA", "prompted", "PRISM–SAPLMA\nPrompt-guided"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.12, top=0.70, hspace=0.42, wspace=0.12
    )

    panels = [
        (model, dataset)
        for model in ("Llama-2-7B-Chat", "Llama-2-13B-Chat")
        for dataset in ("True-False", "LogicStruct")
    ]
    for ax, (model, dataset) in zip(axes.flat, panels):
        paper = [
            values[(model, dataset, method, representation)][0]
            for method, representation, _ in conditions
        ]
        replication = [
            values[(model, dataset, method, representation)][1]
            for method, representation, _ in conditions
        ]
        x = np.arange(len(conditions))
        width = 0.34
        bars_paper = ax.bar(x - width / 2, paper, width, color=PAPER, zorder=3)
        bars_replication = ax.bar(x + width / 2, replication, width, color=REPLICATION, zorder=3)

        ax.axhline(0.5, color=CHANCE, linewidth=1.5, linestyle=(0, (2, 3)), zorder=1)
        ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
        ax.set_ylim(0.44, 0.83)
        ax.set_xticks(x, [item[2] for item in conditions])
        ax.tick_params(axis="x", length=0, pad=12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.set_title(
            f"{model.replace('-Chat', '')} · {dataset}",
            fontsize=15,
            pad=14,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D0D2D5"},
        )
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#D0D2D5")

        for bars in (bars_paper, bars_replication):
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.009,
                    label(bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

    for ax in axes[:, 0]:
        ax.set_ylabel("Cross-domain accuracy (%)", labelpad=12)
    fig.text(
        0.04,
        0.93,
        "PRISM cross-domain baseline: paper vs replication",
        fontsize=28,
        weight="regular",
        ha="left",
    )
    fig.text(
        0.04,
        0.855,
        "Original paper: Tables 3 and 5 · Llama-2-7B/13B-Chat · average accuracy across held-out domains",
        fontsize=15,
        color=MUTED,
        ha="left",
    )
    fig.legend(
        handles=[Patch(color=PAPER, label="Paper"), Patch(color=REPLICATION, label="Replication")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.80),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=2.0,
    )
    fig.text(
        0.04,
        0.055,
        "red dotted line = chance accuracy (50%)   ·   PRISM = prompt-guided internal states",
        fontsize=11.5,
        color="#92959A",
        ha="left",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "paper_vs_replication.png", dpi=240, facecolor="white")
    fig.savefig(OUTPUT / "paper_vs_replication.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
