from __future__ import annotations

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
BASELINE = ROOT / "artifacts/implicit-personalization/probe-validation"
EXTENSIONS = ROOT / "artifacts/implicit-personalization/extensions"
OUTPUT = ROOT / "figures/implicit-personalization"
MODELS = [
    (BASELINE / "gemma", "Gemma 2\n9B"),
    (BASELINE / "llama", "Llama 3.1\n8B"),
    (BASELINE / "olmo", "OLMo 2\n7B"),
    (EXTENSIONS / "qwen3-8b", "Qwen 3\n8B"),
    (EXTENSIONS / "glm4-9b", "GLM-4\n9B"),
    (EXTENSIONS / "phi4-14b", "Phi-4\n14B"),
]
ATTRIBUTES = {
    "race": "Race",
    "age": "Age",
    "gender": "Gender",
    "socio-economic status": "Socio-economic status",
}
COLORS = {"True labels": "#3478C4", "Random-label control": "#FF6905"}
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


def load(path: Path) -> tuple[list[int], dict, dict]:
    layers = json.loads((path / "metadata.json").read_text())["layers"]
    true = pickle.loads((path / "true.pkl").read_bytes())
    control = pickle.loads((path / "control.pkl").read_bytes())
    return layers, true, control


def draw(axis, layers: list[int], values: dict, attribute: str) -> None:
    for label, scores in (
        ("True labels", values[0][attribute]),
        ("Random-label control", values[1][attribute]),
    ):
        folds = 100 * np.asarray(scores, dtype=float)
        mean = folds.mean(axis=1)
        ci = 1.96 * folds.std(axis=1, ddof=1) / np.sqrt(folds.shape[1])
        axis.plot(
            layers,
            mean,
            color=COLORS[label],
            linewidth=2.2,
            marker="o",
            markersize=4,
        )
        axis.fill_between(
            layers,
            mean - ci,
            mean + ci,
            color=COLORS[label],
            alpha=0.15,
            linewidth=0,
        )
    axis.set_ylim(0, 103)
    axis.set_xticks(layers)
    axis.set_yticks([0, 50, 100])
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    axis.grid(axis="y", color="#E4E7EA", linewidth=0.8)
    axis.set_xlabel("Layer", labelpad=3)


def plot(attribute: str, title: str) -> None:
    figure, axes = plt.subplots(1, 6, figsize=(15, 5.1), facecolor="white")
    figure.subplots_adjust(left=0.06, right=0.985, top=0.61, bottom=0.24, wspace=0.24)
    for index, (axis, (path, model)) in enumerate(zip(axes, MODELS, strict=True)):
        layers, true, control = load(path)
        draw(axis, layers, (true, control), attribute)
        axis.set_title(model, fontsize=11.5, fontweight=600, pad=8)
        if index == 0:
            axis.set_ylabel("Probe accuracy")
        else:
            axis.set_yticklabels([])
        if index == 3:
            axis.spines["left"].set_color("#AAB1B8")
            axis.spines["left"].set_linewidth(1.5)

    figure.suptitle(
        f"{title} probe accuracy across models",
        x=0.06,
        y=0.96,
        ha="left",
        fontsize=22,
        fontweight=600,
    )
    figure.text(
        0.06,
        0.875,
        "Reading Between the Prompts · reduced baseline replication versus cross-model extensions",
        fontsize=10.5,
        color="#636A70",
    )
    figure.text(
        0.29,
        0.715,
        "Baseline replication",
        ha="center",
        fontsize=12,
        fontweight=600,
        color="#4D5965",
    )
    figure.text(
        0.755,
        0.715,
        "Extensions",
        ha="center",
        fontsize=12,
        fontweight=600,
        color="#4D5965",
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=2.4, label=label)
            for label, color in COLORS.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.06,
        0.025,
        "Lines show mean five-fold accuracy; bands show 95% CIs. "
        "Layer axes differ because each model uses five architecture-appropriate interior layers.",
        fontsize=8.5,
        color="#737A81",
    )
    stem = "ses" if attribute == "socio-economic status" else attribute
    for extension in ("png", "pdf"):
        figure.savefig(
            OUTPUT / f"probe_replication_vs_extensions_{stem}.{extension}",
            dpi=240 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def plot_combined() -> None:
    figure, axes = plt.subplots(4, 6, figsize=(15, 11.5), facecolor="white")
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.81,
        bottom=0.11,
        hspace=0.42,
        wspace=0.22,
    )
    loaded = [(model, load(path)) for path, model in MODELS]
    for row, (attribute, title) in enumerate(ATTRIBUTES.items()):
        for column, (model, (layers, true, control)) in enumerate(loaded):
            axis = axes[row, column]
            draw(axis, layers, (true, control), attribute)
            if row == 0:
                axis.set_title(model, fontsize=11.5, fontweight=600, pad=8)
            if column == 0:
                axis.set_ylabel(f"{title}\naccuracy", fontsize=10.5)
            else:
                axis.set_yticklabels([])
            if row != len(ATTRIBUTES) - 1:
                axis.set_xlabel("")
            if column == 3:
                axis.spines["left"].set_visible(False)

    separator = (
        axes[0, 2].get_position().x1 + axes[0, 3].get_position().x0
    ) / 2
    figure.add_artist(
        Line2D(
            [separator, separator],
            [0.105, 0.835],
            transform=figure.transFigure,
            color="#D94B4B",
            linewidth=2,
            alpha=0.9,
        )
    )
    figure.suptitle(
        "Probe accuracy across baseline and extension models",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=23,
        fontweight=600,
    )
    figure.text(
        0.075,
        0.92,
        "Reading Between the Prompts · reduced replication of three paper models and three cross-model extensions",
        fontsize=10.5,
        color="#636A70",
    )
    figure.text(
        0.29,
        0.855,
        "Baseline replication",
        ha="center",
        fontsize=13,
        fontweight=600,
        color="#4D5965",
    )
    figure.text(
        0.755,
        0.855,
        "Extensions",
        ha="center",
        fontsize=13,
        fontweight=600,
        color="#4D5965",
    )
    figure.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=2.4, label=label)
            for label, color in COLORS.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.075,
        0.015,
        "Lines show mean five-fold accuracy; bands show 95% CIs. "
        "The red divider separates replicated paper models from new model extensions.",
        fontsize=8.5,
        color="#737A81",
    )
    for extension in ("png", "pdf"):
        figure.savefig(
            OUTPUT / f"probe_replication_vs_extensions_combined.{extension}",
            dpi=240 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    setup()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for attribute, title in ATTRIBUTES.items():
        plot(attribute, title)
    plot_combined()


if __name__ == "__main__":
    main()
