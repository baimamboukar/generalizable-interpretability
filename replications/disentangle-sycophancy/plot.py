from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "figures" / "disentangle-sycophancy"
COLORS = {
    "syc": "#2D6FA3",
    "ga": "#E07A2D",
    "pr": "#3C9B6A",
    "rand": "#C74747",
}
LABELS = {"syc": "SyA", "ga": "GA", "pr": "SyPr", "rand": "Random"}

# Digitized from the published Figure 1 image; values are AUROC by layer.
PAPER = {
    "syc": [
        .54, .82, .825, .82, .824, .821, .823, .824, .819, .82, .825, .839,
        .824, .834, .856, .862, .858, .893, .88, .947, .976, .98, .987, .989,
        .99, .99, .99, .988, .987, .986, .985, .984, .984, .975, .974, .972,
        .972, .968, .964, .96, .951, .936, .926, .919, .914, .906, .901, .901,
    ],
    "ga": [
        .51, .71, .703, .717, .699, .714, .708, .705, .72, .719, .722, .71,
        .759, .787, .775, .776, .817, .899, .928, .936, .957, .968, .98, .984,
        .984, .983, .984, .985, .985, .983, .983, .982, .981, .978, .977, .975,
        .976, .974, .973, .973, .971, .969, .969, .968, .966, .964, .964, .965,
    ],
    "pr": [
        .819, .817, .805, .811, .814, .828, .89, .895, .916, .915, .919, .906,
        .922, .922, .914, .912, .892, .876, .907, .907, .909, .897, .897, .879,
        .885, .918, .924, .944, .952, .955, .95, .949, .958, .962, .96, .962,
        .96, .962, .958, .956, .956, .958, .954, .957, .955, .954, .949, .949,
    ],
    "rand": [
        .514, .526, .531, .536, .536, .538, .538, .537, .539, .539, .541, .538,
        .537, .536, .536, .536, .534, .537, .537, .537, .538, .539, .541, .542,
        .543, .54, .539, .533, .529, .528, .52, .52, .516, .517, .52, .51,
        .511, .509, .516, .514, .52, .519, .515, .516, .514, .517, .517, .517,
    ],
}


def setup_font() -> None:
    candidates = [
        Path.home() / "Library/Fonts/SpaceGrotesk-Variable.ttf",
        Path("/tmp/SpaceGrotesk-Regular.ttf"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(path)
            break
    plt.rcParams.update(
        {
            "font.family": "Space Grotesk",
            "font.size": 10.5,
            "axes.labelcolor": "#17191C",
            "text.color": "#17191C",
            "xtick.color": "#73767C",
            "ytick.color": "#73767C",
        }
    )


def load_replication(output_dir: Path) -> dict[str, dict[int, tuple[float, float, float]]]:
    values = {label: {} for label in LABELS}
    for path in sorted(output_dir.glob("diffmean_*_L*.json")):
        name = path.stem.removeprefix("diffmean_")
        label, layer_text = name.rsplit("_L", 1)
        if label not in values:
            continue
        payload = json.loads(path.read_text())
        auc = float(payload["test_auc_raw"])
        ci = payload.get("test_auc_raw_ci") or [auc, auc]
        values[label][int(layer_text)] = (auc, float(ci[0]), float(ci[1]))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    setup_font()
    replication = load_replication(args.output_dir)
    if not any(replication.values()):
        raise SystemExit(f"No DiffMean JSON files found in {args.output_dir}")

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), sharex=True, sharey=True)
    figure.subplots_adjust(left=.085, right=.98, top=.79, bottom=.13, hspace=.34, wspace=.15)
    layers = np.arange(1, 49)
    for axis, label in zip(axes.flat, LABELS, strict=True):
        color = COLORS[label]
        axis.plot(layers, PAPER[label], color=color, linewidth=2.2, label="Paper (digitized)")
        points = replication[label]
        if points:
            x = np.array(sorted(points))
            y = np.array([points[layer][0] for layer in x])
            low = np.array([points[layer][1] for layer in x])
            high = np.array([points[layer][2] for layer in x])
            axis.plot(x, y, color=color, linewidth=2.0, linestyle=(0, (5, 2)), marker="o", markersize=2.8, label="Replication")
            axis.fill_between(x, low, high, color=color, alpha=.13, linewidth=0)
        axis.set_title(LABELS[label], loc="left", fontsize=14, pad=8)
        axis.set_xlim(1, 48)
        axis.set_ylim(.48, 1.01)
        axis.set_xticks([1, 10, 20, 30, 40, 48])
        axis.set_yticks([.5, .6, .7, .8, .9, 1.0])
        axis.set_yticklabels(["50%", "60%", "70%", "80%", "90%", "100%"])
        axis.grid(axis="y", color="#E5E8EB", linewidth=.8)
        axis.tick_params(length=0, pad=6)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#C9CED3")

    axes[0, 0].set_ylabel("AUROC")
    axes[1, 0].set_ylabel("AUROC")
    axes[1, 0].set_xlabel("Transformer layer")
    axes[1, 1].set_xlabel("Transformer layer")
    figure.text(.06, .935, "Sycophancy is not one thing: Figure 1", fontsize=25, fontweight=600, ha="left")
    figure.text(.06, .875, "Qwen3-30B-Instruct · math factorial · DiffMean residual-stream AUROC · paper versus faithful replication", fontsize=11.5, color="#73767C", ha="left")
    figure.legend(
        handles=[
            Line2D([0], [0], color="#303746", linewidth=2.2, label="Paper (digitized from Figure 1)"),
            Line2D([0], [0], color="#303746", linewidth=2.0, linestyle=(0, (5, 2)), marker="o", markersize=3, label="Replication"),
        ],
        loc="upper left", bbox_to_anchor=(.06, .835), frameon=False, ncol=2, columnspacing=2.0,
    )
    figure.text(.06, .045, "Paper curves are digitized from the published Figure 1; replication bands show the released script's 95% bootstrap CIs.", fontsize=9.5, color="#92959A", ha="left")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(OUTPUT / f"figure1_paper_vs_replication.{suffix}", dpi=240 if suffix == "png" else None, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
