from __future__ import annotations

import csv
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "data/results/lookback-lens/table2_replication.csv"
EXTENSIONS = ROOT / "artifacts/lookback-lens/extensions"
TABLE = ROOT / "data/results/lookback-lens/cross_model_transfer.csv"
OUTPUT = ROOT / "figures/lookback-lens"
FONT = Path.home() / "Library/Fonts/SpaceGrotesk-Variable.ttf"
STATIC_FONT = Path("/tmp/SpaceGrotesk-Regular.ttf")
GREEN = "#42B883"
ORANGE = "#FF6905"
TEXT = "#17191C"
MUTED = "#6D747B"
GRID = "#E4E7EA"
MODELS = {
    "llama-3.1-8b": "Llama 3.1 8B",
    "qwen3-8b": "Qwen3 8B",
    "gemma-2-9b": "Gemma 2 9B",
    "glm-4-9b": "GLM-4 9B",
    "phi-4": "Phi-4 14B",
}


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
    rows = []
    with BASELINE.open(newline="") as stream:
        baseline = list(csv.DictReader(stream))
    for row in baseline:
        if row["split"] != "transfer":
            continue
        shared = {
            "setting": row["setting"],
            "source": row["source"],
            "target": row["target"],
            "split": "transfer",
            "metric": "AUROC",
        }
        rows.extend(
            (
                {
                    "group": "Paper",
                    "model_key": "paper",
                    "model": "Original paper",
                    "value": row["paper"],
                    **shared,
                },
                {
                    "group": "Baseline",
                    "model_key": "llama-2-7b",
                    "model": "Llama 2 7B replication",
                    "value": row["replication"],
                    **shared,
                },
            )
        )
    for model_key, label in MODELS.items():
        path = EXTENSIONS / model_key / "results.csv"
        with path.open(newline="") as stream:
            model_rows = list(csv.DictReader(stream))
        for row in model_rows:
            if row["split"] != "transfer":
                continue
            rows.append(
                {
                    "group": "Extension",
                    "model_key": model_key,
                    "model": label,
                    "setting": row["setting"],
                    "source": row["source"],
                    "target": row["target"],
                    "split": row["split"],
                    "metric": row["metric"],
                    "value": row["value"],
                }
            )
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def panel(axis, rows: list[dict], title: str) -> None:
    labels = ["Original paper", "Llama 2 7B replication", *MODELS.values()]
    y = np.arange(len(labels))
    axis.axhspan(-0.5, 1.5, color="#F4F6F7", zorder=0)
    axis.axhline(1.5, color="#D44747", linewidth=1.25, zorder=1)
    for source, color, offset in (
        ("nq-7b", GREEN, -0.12),
        ("cnndm-7b", ORANGE, 0.12),
    ):
        values = [
            100
            * float(
                next(
                    row["value"]
                    for row in rows
                    if row["model"] == label and row["source"] == source
                )
            )
            for label in labels
        ]
        axis.scatter(values, y + offset, s=52, color=color, zorder=3)
        for value, position in zip(values, y + offset, strict=True):
            axis.text(
                value + 0.7,
                position,
                f"{value:.1f}",
                va="center",
                fontsize=8.4,
                color=TEXT,
            )
    axis.set_xlim(50, 101)
    axis.set_xticks(np.arange(50, 101, 10))
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axis.tick_params(axis="x", length=0, pad=7)
    axis.tick_params(axis="y", length=0, pad=10)
    axis.spines["bottom"].set_color("#C9CED3")
    axis.set_title(title, fontsize=13, fontweight=600, loc="left", pad=16)
    axis.set_xlabel("Cross-task transfer AUROC (%)", labelpad=11)


def main() -> None:
    setup()
    rows = load()
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 7.8), facecolor="white")
    figure.subplots_adjust(left=0.18, right=0.97, top=0.69, bottom=0.18, wspace=0.3)
    for axis, setting, title in (
        (axes[0], "predefined", "Predefined spans"),
        (axes[1], "sliding-window-8", "Sliding window (8 tokens)"),
    ):
        panel(axis, [row for row in rows if row["setting"] == setting], title)
    figure.text(
        0.06,
        0.94,
        "Lookback Lens across model families",
        fontsize=23,
        fontweight=600,
        ha="left",
    )
    figure.text(
        0.06,
        0.875,
        "Original Table 2, faithful Llama-2 replication, and off-policy cross-model extensions",
        fontsize=11,
        color=MUTED,
        ha="left",
    )
    figure.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", color=GREEN, label="NQ → CNN/DM"
            ),
            Line2D(
                [], [], marker="o", linestyle="", color=ORANGE, label="CNN/DM → NQ"
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.055, 0.815),
        ncol=2,
        frameon=False,
        handletextpad=0.55,
        columnspacing=2.1,
    )
    figure.text(
        0.06,
        0.055,
        "Extensions reuse the released Llama-2 generations and GPT-4o labels, "
        "teacher-forced through each target model; the red rule separates baselines from extensions.",
        fontsize=9.1,
        color=MUTED,
        ha="left",
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
