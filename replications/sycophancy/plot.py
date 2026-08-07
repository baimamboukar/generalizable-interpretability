from __future__ import annotations

import csv
from pathlib import Path
import pickle

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "results" / "sycophancy"
OUTPUT = ROOT / "figures" / "sycophancy"
SEEDS = (3407, 3408, 3409)
ACTIVATIONS = ("mha", "mlp", "residual")
GREEN = "#49B98A"
GREEN_DARK = "#167C5A"
ORANGE = "#FF6905"
TEXT = "#17191C"
MUTED = "#73767C"
GRID = "#E6E8EB"
CHANCE = "#E6A19A"
PAPER_GREEN = "#49B98A"
FONT = Path.home() / "Library" / "Fonts" / "SpaceGrotesk-Variable.ttf"
STATIC_FONT = Path("/tmp/SpaceGrotesk-Regular.ttf")
PAPER_FIGURE10 = {
    "mlp": np.array(
        [
            70.84, 74.67, 77.70, 80.87, 86.15, 91.16, 93.14,
            93.93, 95.38, 94.99, 96.17, 96.57, 97.76, 98.09,
            97.76, 96.17, 92.35, 91.95, 91.56, 89.31, 86.94,
            85.75, 87.34, 87.34, 89.71, 88.52, 88.92, 86.15,
        ]
    ),
    "residual": np.array(
        [
            72.96, 77.97, 88.13, 92.74, 95.12, 96.17, 95.51,
            95.51, 96.97, 96.57, 97.36, 97.76, 98.47, 98.15,
            97.95, 97.76, 97.36, 96.97, 96.57, 95.78, 95.51,
            95.51, 94.33, 94.72, 93.54, 94.33, 93.54, 90.50,
        ]
    ),
}


def source(seed: int, activation: str) -> Path:
    return (
        DATA
        / f"seed-{seed}"
        / "source"
        / "probe"
        / "trained_probe_sycophancy_challenged"
        / "llama-3.2"
        / f"linear_accuracies_dict_{activation}.pkl"
    )


def load() -> dict[tuple[int, str], dict[str, float]]:
    return {
        (seed, activation): pickle.loads(source(seed, activation).read_bytes())
        for seed in SEEDS
        for activation in ACTIVATIONS
    }


def rows(
    values: dict[tuple[int, str], dict[str, float]],
) -> list[dict[str, str | int | float]]:
    output = []
    for (seed, activation), scores in values.items():
        for component, accuracy in scores.items():
            layer, *head = component.split("_")
            output.append(
                {
                    "model": "meta-llama/Llama-3.2-3B-Instruct",
                    "seed": seed,
                    "activation": activation,
                    "layer": int(layer),
                    "head": int(head[0]) if head else "",
                    "accuracy": accuracy / 100,
                }
            )
    return output


def save_table(records: list[dict[str, str | int | float]]) -> None:
    path = DATA / "probe_accuracies_long.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def setup_font() -> None:
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
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
        }
    )


def main() -> None:
    setup_font()
    values = load()
    save_table(rows(values))

    mha = np.array(
        [
            [
                [values[(seed, "mha")][f"{layer}_{head}"] for head in range(24)]
                for layer in range(28)
            ]
            for seed in SEEDS
        ]
    )
    mha_mean = mha.mean(axis=0)
    mha_sorted = np.sort(mha_mean, axis=1)[:, ::-1]
    layerwise = {}
    for activation in ("mlp", "residual"):
        scores = np.array(
            [
                [values[(seed, activation)][str(layer)] for layer in range(28)]
                for seed in SEEDS
            ]
        )
        layerwise[activation] = (
            scores.mean(axis=0),
            scores.std(axis=0, ddof=1),
        )

    fig = plt.figure(figsize=(15, 11), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1, 0.035),
        height_ratios=(1.35, 1),
        left=0.08,
        right=0.94,
        bottom=0.10,
        top=0.80,
        hspace=0.34,
        wspace=0.03,
    )
    heat = fig.add_subplot(grid[0, 0])
    colorbar = fig.add_subplot(grid[0, 1])
    line = fig.add_subplot(grid[1, 0])

    cmap = LinearSegmentedColormap.from_list(
        "probe_green", ["#F5F7F4", "#C6E8D9", GREEN, GREEN_DARK]
    )
    image = heat.imshow(
        mha_sorted,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=45,
        vmax=100,
        interpolation="nearest",
    )
    bar = fig.colorbar(image, cax=colorbar)
    bar.set_label("Validation accuracy")
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    heat.set_title("Attention-head probes", loc="left", fontsize=16, pad=14)
    heat.set_xlabel("Heads ranked within each layer")
    heat.set_ylabel("Layer")
    heat.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    heat.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=14))
    for spine in heat.spines.values():
        spine.set_visible(False)

    x = np.arange(28)
    for activation, label, color in (
        ("mlp", "MLP", ORANGE),
        ("residual", "Residual stream", GREEN_DARK),
    ):
        mean, std = layerwise[activation]
        line.plot(x, mean, color=color, linewidth=2.4, label=label)
        line.fill_between(x, mean - std, mean + std, color=color, alpha=0.14)

    line.axhline(50, color=CHANCE, linewidth=1.4, linestyle=(0, (2, 3)))
    line.grid(axis="y", color=GRID, linewidth=0.9)
    line.set_title("Layerwise probes", loc="left", fontsize=16, pad=14)
    line.set_xlabel("Layer")
    line.set_ylabel("Validation accuracy")
    line.set_xlim(0, 27)
    line.set_ylim(45, 100)
    line.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    line.legend(frameon=False, ncol=2, loc="upper left")
    for spine in ("top", "right", "left"):
        line.spines[spine].set_visible(False)
    line.spines["bottom"].set_color("#D0D2D5")

    fig.text(
        0.06,
        0.93,
        "Where does sycophancy become linearly accessible?",
        fontsize=28,
        ha="left",
    )
    fig.text(
        0.06,
        0.875,
        "Replication of original Figures 9(b) and 10 · Llama-3.2-3B-Instruct · mean ± SD across three seeds",
        fontsize=13,
        color=MUTED,
        ha="left",
    )
    fig.text(
        0.06,
        0.04,
        "Attention heads are sorted independently within each layer, following the original paper · dotted line = chance",
        fontsize=10.5,
        color="#92959A",
        ha="left",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            OUTPUT / f"llama32_probe_localization.{suffix}",
            dpi=240 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)

    figure10, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6.8),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    for axis, activation, title, color in (
        (axes[0], "mlp", "MLP probes", ORANGE),
        (axes[1], "residual", "Residual-stream probes", GREEN_DARK),
    ):
        mean, std = layerwise[activation]
        axis.plot(x, mean, color=color, linewidth=2.8)
        axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        axis.axhline(50, color=CHANCE, linewidth=1.4, linestyle=(0, (2, 3)))
        axis.grid(axis="y", color=GRID, linewidth=0.9)
        axis.set_title(title, loc="left", fontsize=16, pad=14)
        axis.set_xlabel("Layer")
        axis.set_xlim(0, 27)
        axis.set_ylim(45, 100)
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.0f}%")
        )
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#D0D2D5")
    axes[0].set_ylabel("Validation accuracy")

    figure10.subplots_adjust(
        left=0.08,
        right=0.96,
        bottom=0.16,
        top=0.72,
        wspace=0.14,
    )
    figure10.text(
        0.06,
        0.91,
        "Layerwise sycophancy probes",
        fontsize=27,
        ha="left",
    )
    figure10.text(
        0.06,
        0.83,
        "Replication of the Llama-3.2 panel in original Figure 10 · mean ± SD across three seeds",
        fontsize=12.5,
        color=MUTED,
        ha="left",
    )
    figure10.text(
        0.06,
        0.055,
        "Shaded bands show standard deviation · dotted line = chance",
        fontsize=10.5,
        color="#92959A",
        ha="left",
    )
    for suffix in ("png", "pdf", "svg"):
        figure10.savefig(
            OUTPUT / f"llama32_figure10_layerwise.{suffix}",
            dpi=240 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(figure10)

    comparison, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    panels = (
        ("mlp", "MLP probes"),
        ("residual", "Residual-stream probes"),
    )
    for column, (activation, title) in enumerate(panels):
        paper_axis = axes[0, column]
        paper = PAPER_FIGURE10[activation]
        paper_axis.plot(x, paper, color=PAPER_GREEN, linewidth=2.8)
        best = int(np.argmax(paper))
        paper_axis.scatter(
            best,
            paper[best],
            color=GREEN_DARK,
            s=42,
            zorder=3,
        )
        paper_axis.set_title(title, loc="left", fontsize=16, pad=12)

        replication_axis = axes[1, column]
        mean, std = layerwise[activation]
        replication_axis.plot(x, mean, color=ORANGE, linewidth=2.8)
        replication_axis.fill_between(
            x,
            mean - std,
            mean + std,
            color=ORANGE,
            alpha=0.15,
        )

    for axis in axes.flat:
        axis.axhline(50, color=CHANCE, linewidth=1.3, linestyle=(0, (2, 3)))
        axis.grid(axis="y", color=GRID, linewidth=0.9)
        axis.set_xlim(0, 27)
        axis.set_ylim(45, 102)
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.0f}%")
        )
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#D0D2D5")
    axes[0, 0].set_ylabel("Paper\nvalidation accuracy", labelpad=18)
    axes[1, 0].set_ylabel("Replication\nvalidation accuracy", labelpad=18)
    axes[1, 0].set_xlabel("Layer")
    axes[1, 1].set_xlabel("Layer")

    comparison.subplots_adjust(
        left=0.11,
        right=0.96,
        bottom=0.12,
        top=0.76,
        hspace=0.34,
        wspace=0.14,
    )
    comparison.text(
        0.06,
        0.93,
        "Published result vs. replication",
        fontsize=27,
        ha="left",
    )
    comparison.text(
        0.06,
        0.865,
        "Llama-3.2 panel from original Figure 10 · replication reports mean ± SD across three seeds",
        fontsize=12.5,
        color=MUTED,
        ha="left",
    )
    comparison.text(
        0.06,
        0.035,
        "Published curves digitized from Figure 10 · shaded bands show replication standard deviation · dotted line = chance",
        fontsize=10.5,
        color="#92959A",
        ha="left",
    )
    for suffix in ("png", "pdf", "svg"):
        comparison.savefig(
            OUTPUT / f"llama32_figure10_paper_vs_replication.{suffix}",
            dpi=240 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(comparison)


if __name__ == "__main__":
    main()
