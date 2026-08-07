from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ROOT / "data/results/beyond-linear-probes/baseline_targets.csv"
DEFAULT_RESULTS = ROOT / "artifacts/beyond-linear-probes/results"
OUTPUT = ROOT / "figures/beyond-linear-probes"
PAPER = "#303746"
REPLICATION = "#42B883"
TEXT = "#17191C"
MUTED = "#73767C"
GRID = "#E5E8EB"

MODEL_LABELS = {
    "gemma": "Gemma-3 27B-it",
    "qwen3": "Qwen3 30B-A3B",
    "gpt-oss": "GPT-OSS 20B",
    "llama3": "Llama-3.2 3B",
}
METHODS = ("linear", "bilinear", "mlp", "ee_mlp", "tpc_order5")
METHOD_LABELS = {
    "linear": "Linear",
    "bilinear": "Bilinear",
    "mlp": "MLP",
    "ee_mlp": "Early-exit\nMLP",
    "tpc_order5": "TPC\n(order 5)",
}
FILES = {
    "linear": "linear-vals.pkl",
    "bilinear": "bi-vals.pkl",
    "mlp": "mlp-vals.pkl",
    "ee_mlp": "eemlp-matched-vals.pkl",
    "tpc_order5": "poly-vals.pkl",
}


def setup_font() -> None:
    for path in (
        Path.home() / "Library/Fonts/SpaceGrotesk-Variable.ttf",
        Path("/tmp/SpaceGrotesk-Regular.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(path)
            break
    plt.rcParams.update(
        {
            "font.family": "Space Grotesk",
            "font.size": 11,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
        }
    )


def load_targets() -> dict[tuple[str, int, str], float]:
    with TARGETS.open(newline="") as handle:
        return {
            (row["model"], int(row["layer"]), row["method"]): float(row["test_f1_pct"])
            for row in csv.DictReader(handle)
        }


def final_value(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return float(array[..., -1].mean())


def replication_value(path: Path, method: str) -> float:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    values = payload["y_vals_test"]
    if method == "mlp":
        values = values[-1]
    return 100 * final_value(values)


def result_path(results: Path, model: str, layer: int, method: str) -> Path:
    model_name = {
        "gemma": "gemma-3-27b-it",
        "qwen3": "Qwen3-30B-A3B-Base",
        "gpt-oss": "gpt-oss-20b",
        "llama3": "Llama-3.2-3B",
    }[model]
    directory = results / "numbers" / "wildguard" / (
        f"{model_name}-L{layer}-mean-progressive-maxorder-5_rank-64"
    )
    return directory / FILES[method]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    setup_font()
    paper = load_targets()
    models = list(MODEL_LABELS)
    layers = {model: min(layer for m, layer, _ in paper if m == model) for model in models}

    values: dict[tuple[str, str], float] = {}
    missing: list[Path] = []
    for model in models:
        for method in METHODS:
            path = result_path(args.results_dir, model, layers[model], method)
            if not path.exists():
                missing.append(path)
            else:
                values[(model, method)] = replication_value(path, method)
    if missing:
        raise SystemExit("Missing baseline result files; run the baseline first:\n" + "\n".join(map(str, missing)))

    figure, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), sharey=True)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.15, hspace=0.42, wspace=0.12)
    x = np.arange(len(METHODS))
    width = 0.34
    for axis, model in zip(axes.flat, models, strict=True):
        layer = layers[model]
        paper_values = [paper[(model, layer, method)] for method in METHODS]
        replication_values = [values[(model, method)] for method in METHODS]
        paper_bars = axis.bar(x - width / 2, paper_values, width, color=PAPER, zorder=3)
        replication_bars = axis.bar(x + width / 2, replication_values, width, color=REPLICATION, zorder=3)
        axis.set_title(f"{MODEL_LABELS[model]} · layer {layer}", loc="left", fontsize=15, pad=12)
        axis.set_xticks(x, [METHOD_LABELS[method] for method in METHODS])
        axis.set_ylim(75, 100)
        axis.set_yticks([75, 80, 85, 90, 95, 100])
        axis.set_ylabel("Test F1 (%)")
        axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        axis.tick_params(axis="x", length=0, pad=9)
        axis.tick_params(axis="y", length=0, pad=6)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#C9CED3")
        for bars in (paper_bars, replication_bars):
            for bar in bars:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.45,
                    f"{bar.get_height():.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                )

    figure.text(0.06, 0.94, "Beyond Linear Probes: paper versus replication", fontsize=25, fontweight=600, ha="left")
    figure.text(
        0.06,
        0.875,
        "WildGuardMix · Table 1 test F1 · selected paper layers · five-seed upstream training",
        fontsize=11.5,
        color=MUTED,
        ha="left",
    )
    figure.legend(
        handles=[Patch(color=PAPER, label="Original paper"), Patch(color=REPLICATION, label="Faithful replication")],
        loc="upper left",
        bbox_to_anchor=(0.06, 0.82),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
    )
    figure.text(
        0.06,
        0.055,
        "Paper values are frozen from Table 1. Replication values are averaged across the five seeds in the released training script.",
        fontsize=9.5,
        color=MUTED,
        ha="left",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / "paper_vs_replication.png", dpi=240, facecolor="white", bbox_inches="tight")
    figure.savefig(args.output_dir / "paper_vs_replication.pdf", facecolor="white", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
