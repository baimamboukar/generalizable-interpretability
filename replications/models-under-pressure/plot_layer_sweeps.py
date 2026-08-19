from __future__ import annotations

import csv
import json
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "models-under-pressure"
OUTPUT = ROOT / "figures" / "model-under-pressure"
TABLE_OUTPUT = ROOT / "data" / "results" / "model-under-pressure"
FONT = Path.home() / "Library" / "Fonts" / "SpaceGrotesk-Variable.ttf"
STATIC_FONT = Path("/tmp/SpaceGrotesk-Regular.ttf")

TEXT = "#17191C"
MUTED = "#73767C"
GRID = "#E7E9EC"
ACCENT = "#A3263B"
THRESHOLD = 0.85
DATASETS = (
    ("anthropic_hh_balanced", "Anthropic HH-RLHF"),
    ("toolace_balanced", "ToolACE"),
    ("mt_balanced", "MT Samples"),
    ("mts_balanced", "MTS Dialog"),
    ("mental_health_balanced", "Mental Health"),
    ("aya_redteaming_balanced", "Aya Redteaming"),
)
GAP_DATASETS = (
    ("anthropic_hh_balanced", "Anthropic-HH"),
    ("aya_redteaming_balanced", "Aya-RT"),
    ("mt_balanced", "MT"),
    ("mts_balanced", "MTS"),
    ("mental_health_balanced", "Mental Health"),
    ("toolace_balanced", "ToolACE"),
)
GAP_COLORMAP = LinearSegmentedColormap.from_list(
    "baseline_gap",
    ("#8B001B", "#D55A4A", "#F3F1EE", "#75B1D1", "#124E84"),
)
GAP_NORM = TwoSlopeNorm(vmin=-0.40, vcenter=0.0, vmax=0.40)
LLAMA_BASELINE = {
    "attention": {
        "anthropic_hh_balanced": 0.9501,
        "toolace_balanced": 0.8744,
        "mt_balanced": 0.9838,
        "mts_balanced": 0.9931,
        "mental_health_balanced": 0.8074,
        "aya_redteaming_balanced": 0.9014,
    },
    "softmax": {
        "anthropic_hh_balanced": 0.9651,
        "toolace_balanced": 0.8352,
        "mt_balanced": 0.9906,
        "mts_balanced": 0.9808,
        "mental_health_balanced": 0.8515,
        "aya_redteaming_balanced": 0.9234,
    },
}
RUNS = (
    {
        "artifact": "qwen25-32b-eval-a100-r11",
        "prefix": "qwen25_32b_r11",
        "title": "Qwen2.5-32B-Instruct",
        "filename": "qwen25_32b_layer_sweep",
        "attention_color": "#4F46E5",
        "softmax_color": "#98A2B3",
    },
    {
        "artifact": "gemma4-31b-eval-a100-r11",
        "prefix": "gemma4_31b_r11",
        "title": "Gemma 4 31B IT",
        "filename": "gemma4_31b_layer_sweep",
        "attention_color": "#2563EB",
        "softmax_color": "#D92D3A",
    },
)
QWEN3_EXTRA_RUN = {
    "artifact": "qwen3-32b-extra5-h200-r8",
    "prefix": "qwen3_32b_extra5_r8",
    "title": "Qwen3-32B",
    "filename": "qwen3_32b_followup_layer_screen",
    "attention_color": "#0B6E99",
    "softmax_color": "#262A30",
}
QWEN3_PRIOR = {
    "attention": {
        32: {
            "anthropic_hh_balanced": 0.8894,
            "toolace_balanced": 0.8905,
            "mt_balanced": 0.9703,
            "mts_balanced": 0.9557,
            "mental_health_balanced": 0.5810,
            "aya_redteaming_balanced": 0.8635,
        },
        47: {
            "anthropic_hh_balanced": 0.9173,
            "toolace_balanced": 0.8964,
            "mt_balanced": 0.9652,
            "mts_balanced": 0.9746,
            "mental_health_balanced": 0.6386,
            "aya_redteaming_balanced": 0.8949,
        },
    },
    "softmax": {
        32: {
            "anthropic_hh_balanced": 0.8525,
            "toolace_balanced": 0.6500,
            "mt_balanced": 0.8661,
            "mts_balanced": 0.8588,
            "mental_health_balanced": 0.8791,
            "aya_redteaming_balanced": 0.8170,
        },
        47: {
            "anthropic_hh_balanced": 0.8757,
            "toolace_balanced": 0.7131,
            "mt_balanced": 0.8807,
            "mts_balanced": 0.8039,
            "mental_health_balanced": 0.5000,
            "aya_redteaming_balanced": 0.8269,
        },
    },
}
HISTORICAL_MEANS = {
    "attention": (
        ("Llama-3.3-70B", ((31, 0.9184),), "#AEB7C3"),
        ("DeepSeek-R1-32B", ((28, 0.9069), (36, 0.9002), (44, 0.8843), (54, 0.8326)), "#27A78A"),
        ("GLM-4-32B", ((30, 0.8882), (35, 0.8971), (40, 0.9166)), "#FF6905"),
    ),
    "softmax": (
        ("Llama-3.3-70B", ((31, 0.9244),), "#AEB7C3"),
        ("DeepSeek-R1-32B", ((28, 0.8864), (44, 0.9169), (54, 0.8595)), "#27A78A"),
        ("GLM-4-32B", ((35, 0.8880),), "#FF6905"),
    ),
}


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
            "font.size": 10,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": MUTED,
            "text.color": TEXT,
        }
    )


def load_run(run: dict[str, str]) -> tuple[list[int], dict[str, dict[str, dict[int, float]]]]:
    result_dir = ARTIFACTS / run["artifact"] / "results_evaluate_probes"
    values: dict[str, dict[str, dict[int, float]]] = {
        probe: {dataset: {} for dataset, _ in DATASETS}
        for probe in ("attention", "softmax")
    }
    layers: set[int] = set()
    for path in sorted(result_dir.glob(f"results_{run['prefix']}_*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        config = records[0]["config"]
        raw_probe = config["probe_spec"]["name"]
        probe = "softmax" if raw_probe == "linear_then_softmax" else raw_probe
        layer = int(config["layer"])
        layers.add(layer)
        latest = {record["dataset_name"]: record for record in records}
        for dataset, _ in DATASETS:
            values[probe][dataset][layer] = latest[dataset]["metrics"]["metrics"]["auroc"]
    ordered_layers = sorted(layers)
    expected = set(ordered_layers)
    for probe in values:
        for dataset in values[probe]:
            if set(values[probe][dataset]) != expected:
                raise ValueError(f"Incomplete {run['title']} results for {probe}, {dataset}")
    return ordered_layers, values


def write_run_table() -> None:
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = TABLE_OUTPUT / "model_layer_sweeps.csv"
    fields = ("model", "probe", "layer", "dataset", "auroc")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in RUNS:
            layers, values = load_run(run)
            for probe in ("attention", "softmax"):
                for layer in layers:
                    rows = [
                        (dataset, values[probe][dataset][layer])
                        for dataset, _ in DATASETS
                    ]
                    rows.append(
                        ("mean", np.mean([value for _, value in rows]))
                    )
                    for dataset, value in rows:
                        writer.writerow(
                            {
                                "model": run["title"],
                                "probe": probe,
                                "layer": layer,
                                "dataset": dataset,
                                "auroc": f"{value:.8f}",
                            }
                        )


def load_qwen3_combined() -> tuple[list[int], dict[str, dict[str, dict[int, float]]]]:
    layers, values = load_run(QWEN3_EXTRA_RUN)
    for probe, layer_values in QWEN3_PRIOR.items():
        for layer, dataset_values in layer_values.items():
            layers.append(layer)
            for dataset, value in dataset_values.items():
                values[probe][dataset][layer] = value
    layers = sorted(set(layers))
    expected = set(layers)
    for probe in values:
        for dataset in values[probe]:
            if set(values[probe][dataset]) != expected:
                raise ValueError(f"Incomplete Qwen3 results for {probe}, {dataset}")
    return layers, values


def qwen3_means(
    layers: list[int], values: dict[str, dict[str, dict[int, float]]]
) -> dict[str, dict[int, float]]:
    return {
        probe: {
            layer: np.mean([values[probe][dataset][layer] for dataset, _ in DATASETS])
            for layer in layers
        }
        for probe in ("attention", "softmax")
    }


def write_qwen3_table(
    layers: list[int], values: dict[str, dict[str, dict[int, float]]]
) -> None:
    means = qwen3_means(layers, values)
    TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = TABLE_OUTPUT / "qwen3_32b_layer_screen.csv"
    fields = (
        "model",
        "probe",
        "layer",
        "relative_depth",
        "sampling_phase",
        "dataset",
        "auroc",
        "passes_085",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for probe in ("attention", "softmax"):
            for layer in layers:
                rows = [(dataset, values[probe][dataset][layer]) for dataset, _ in DATASETS]
                rows.append(("mean", means[probe][layer]))
                for dataset, value in rows:
                    writer.writerow(
                        {
                            "model": "Qwen/Qwen3-32B",
                            "probe": probe,
                            "layer": layer,
                            "relative_depth": f"{layer / 64:.6f}",
                            "sampling_phase": "initial" if layer in (32, 47) else "follow-up",
                            "dataset": dataset,
                            "auroc": f"{value:.8f}",
                            "passes_085": value >= THRESHOLD,
                        }
                    )


def plot_run(run: dict[str, str]) -> None:
    layers, values = load_run(run)
    attention_color = run["attention_color"]
    softmax_color = run["softmax_color"]
    figure, axes = plt.subplots(2, 3, figsize=(15.8, 9.0), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.68, hspace=0.42, wspace=0.15)
    positions = np.arange(len(layers))
    width = 0.36

    for axis, (dataset, label) in zip(axes.flat, DATASETS):
        attention = [values["attention"][dataset][layer] for layer in layers]
        softmax = [values["softmax"][dataset][layer] for layer in layers]
        attention_bars = axis.bar(
            positions - width / 2,
            attention,
            width,
            color=attention_color,
            linewidth=0,
            zorder=3,
        )
        softmax_bars = axis.bar(
            positions + width / 2,
            softmax,
            width,
            color=softmax_color,
            linewidth=0,
            zorder=3,
        )
        axis.axhline(LLAMA_BASELINE["attention"][dataset], color=attention_color, alpha=0.35, linestyle=(0, (3, 3)), linewidth=0.9)
        axis.axhline(LLAMA_BASELINE["softmax"][dataset], color=softmax_color, alpha=0.35, linestyle=(0, (3, 3)), linewidth=0.9)
        for bars in (attention_bars, softmax_bars):
            axis.bar_label(
                bars,
                labels=[f"{bar.get_height() * 100:.2f}%" for bar in bars],
                padding=2.5,
                fontsize=6.2,
                color=TEXT,
                fontweight=500,
            )
        axis.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
        axis.set_title(label, fontsize=13, fontweight=600, loc="left", pad=11)
        axis.set_xticks(positions, [f"L{layer}" for layer in layers])
        axis.set_ylim(0.2, 1.02)
        axis.set_yticks((0.2, 0.4, 0.6, 0.8, 1.0))
        axis.set_yticklabels(("20%", "40%", "60%", "80%", "100%"))
        axis.tick_params(axis="x", length=0, pad=7)
        axis.tick_params(axis="y", length=0, pad=6)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#D0D3D8")

    axes[0, 0].set_ylabel("AUROC", labelpad=11)
    axes[1, 0].set_ylabel("AUROC", labelpad=11)
    for axis in axes[1]:
        axis.set_xlabel("Probed transformer layer", labelpad=10)

    figure.text(0.07, 0.955, "MODELS UNDER PRESSURE · EXTENSION", fontsize=11, color=ACCENT, fontweight=600, ha="left")
    figure.text(0.07, 0.895, f"Layer sweep | {run['title']}", fontsize=25, fontweight=600, ha="left")
    figure.text(0.07, 0.848, "Five selected layers · six balanced deployment datasets · AUROC (%)", fontsize=11.5, color=MUTED, ha="left")
    figure.legend(
        handles=[
            Line2D([], [], color=attention_color, linewidth=8, label="Attention probe"),
            Line2D([], [], color=softmax_color, linewidth=8, label="Softmax probe"),
            Line2D([], [], color="#73767C", linestyle=(0, (3, 3)), linewidth=0.9, label="Dashed reference · Llama-3.3-70B@L31 replication"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.812),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.0,
    )
    figure.text(0.07, 0.055, "Bar labels report AUROC (%). Each dashed line marks the corresponding Llama-3.3-70B@L31 replication value.", fontsize=9.5, color="#92959A", ha="left")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / f"{run['filename']}.png", facecolor="white", dpi=300)
    plt.close(figure)


def plot_baseline_gaps(run: dict[str, str]) -> None:
    layers, values = load_run(run)
    figure, axes = plt.subplots(2, 1, figsize=(18, 12.5), sharex=True)
    figure.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.68, hspace=0.62)

    for axis, probe in zip(axes, ("attention", "softmax")):
        gaps = np.array(
            [
                [values[probe][dataset][layer] - LLAMA_BASELINE[probe][dataset] for dataset, _ in GAP_DATASETS]
                for layer in layers
            ]
        )
        image = axis.imshow(gaps, cmap=GAP_COLORMAP, norm=GAP_NORM, aspect="auto")
        axis.set_title(f"{probe.title()} probe", fontsize=13, fontweight=600, loc="left", pad=13)
        axis.set_yticks(np.arange(len(layers)), [f"L{layer}" for layer in layers])
        axis.set_xticks(np.arange(len(GAP_DATASETS)), [label for _, label in GAP_DATASETS])
        axis.tick_params(axis="x", length=0, pad=10, labelsize=10.5)
        axis.tick_params(axis="y", length=0, pad=8, labelsize=10.5)

        for row, layer in enumerate(layers):
            for column, (dataset, _) in enumerate(GAP_DATASETS):
                gap = gaps[row, column]
                value = values[probe][dataset][layer]
                color = "white" if abs(gap) >= 0.24 else TEXT
                axis.text(column, row - 0.18, f"{gap * 100:+.2f} pp", ha="center", va="center", fontsize=10.5, fontweight=600, color=color)
                axis.text(column, row + 0.23, f"{value * 100:.2f}%", ha="center", va="center", fontsize=8.7, color=color, alpha=0.84)

        for index in np.argsort(gaps, axis=None)[:4]:
            row, column = np.unravel_index(index, gaps.shape)
            axis.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=TEXT, linewidth=2.1))

        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.text(0.07, 0.955, "FAILURE LOCALIZATION", fontsize=11, color=ACCENT, fontweight=600, ha="left")
    figure.text(0.07, 0.895, f"How far does {run['title']} fall from the Llama baseline?", fontsize=25, fontweight=600, ha="left")
    figure.text(0.07, 0.845, "Cell top: AUROC gap versus Llama-3.3-70B@L31 replication · small number: absolute AUROC", fontsize=11.5, color=MUTED, ha="left")
    figure.text(0.18, 0.725, "Black boxes mark the four largest negative gaps in each probe panel.", fontsize=9.5, color=MUTED, ha="left")

    colorbar = figure.colorbar(image, ax=axes, orientation="horizontal", fraction=0.06, pad=0.16, aspect=45)
    colorbar.set_ticks((-0.40, -0.20, 0.0, 0.20, 0.40))
    colorbar.set_ticklabels(("−40 pp", "−20 pp", "0", "+20 pp", "+40 pp"))
    colorbar.set_label("AUROC gap versus Llama baseline", color=MUTED, labelpad=10)
    colorbar.ax.tick_params(length=0, labelsize=9, colors=MUTED)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / f"{run['filename']}_baseline_gaps.png", facecolor="white", dpi=300)
    plt.close(figure)


def style_layer_axis(axis: plt.Axes) -> None:
    axis.set_xlim(23, 62)
    axis.set_ylim(0.72, 0.96)
    axis.set_xticks((25, 32, 37, 42, 47, 54, 60), ("L25", "L32", "L37", "L42", "L47", "L54", "L60"))
    axis.set_yticks((0.75, 0.80, 0.85, 0.90, 0.95), ("75%", "80%", "85%", "90%", "95%"))
    axis.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    axis.tick_params(axis="both", length=0, pad=7)
    axis.set_xlabel("Probed transformer layer", labelpad=10)
    axis.set_ylabel("Mean AUROC", labelpad=10)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#D0D3D8")


def plot_qwen3_layer_screen() -> None:
    layers, values = load_qwen3_combined()
    means = qwen3_means(layers, values)
    colors = {
        "attention": QWEN3_EXTRA_RUN["attention_color"],
        "softmax": QWEN3_EXTRA_RUN["softmax_color"],
    }
    positions = np.arange(len(layers))
    figure = plt.figure(figsize=(18, 12.2))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.975,
        bottom=0.09,
        top=0.78,
        hspace=0.54,
        wspace=0.18,
        height_ratios=(0.95, 1.12),
    )
    bar_axes = [figure.add_subplot(grid[0, column]) for column in range(2)]
    heat_axes = [figure.add_subplot(grid[1, column]) for column in range(2)]

    for axis, probe in zip(bar_axes, ("attention", "softmax")):
        color = colors[probe]
        layer_means = means[probe]
        bar_colors = ["white" if layer in (32, 47) else color for layer in layers]
        bars = axis.bar(
            positions,
            [layer_means[layer] for layer in layers],
            width=0.62,
            color=bar_colors,
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )
        axis.axhline(
            THRESHOLD,
            color="#8A8F98",
            linestyle=(0, (3, 4)),
            linewidth=1.1,
            alpha=0.78,
            zorder=2,
        )
        best_layer = max(layers, key=layer_means.get)
        best = layer_means[best_layer]
        axis.bar_label(
            bars,
            labels=[f"{layer_means[layer] * 100:.2f}%" for layer in layers],
            padding=4,
            fontsize=9.2,
            color=TEXT,
            fontweight=500,
        )
        axis.text(
            len(layers) - 0.15,
            THRESHOLD + 0.003,
            "85% screen",
            fontsize=8.5,
            color=MUTED,
            ha="right",
            va="bottom",
        )
        axis.text(
            1.0,
            1.04,
            f"Best sampled layer: L{best_layer}  ·  {best * 100:.2f}%",
            transform=axis.transAxes,
            fontsize=10.2,
            color=color,
            fontweight=600,
            ha="right",
            va="center",
        )
        axis.set_title(f"{probe.title()} probe", fontsize=15, fontweight=600, loc="left", pad=13)
        axis.set_xlim(-0.65, len(layers) - 0.35)
        axis.set_ylim(0.70, 0.96)
        axis.set_yticks((0.70, 0.75, 0.80, 0.85, 0.90, 0.95), ("70%", "75%", "80%", "85%", "90%", "95%"))
        axis.set_xticks(positions, [f"L{layer}" for layer in layers])
        axis.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
        axis.tick_params(axis="both", length=0, pad=7)
        axis.set_ylabel("Mean AUROC", labelpad=10)
        axis.set_xlabel("Probed transformer layer", labelpad=10)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#D0D3D8")

    cmaps = {
        "attention": LinearSegmentedColormap.from_list(
            "qwen_attention", ("#F5F8FA", "#B9DCE9", colors["attention"])
        ),
        "softmax": LinearSegmentedColormap.from_list(
            "qwen_softmax", ("#F7F7F8", "#BCC1C8", colors["softmax"])
        ),
    }
    norm = Normalize(vmin=0.25, vmax=1.0)
    row_labels = [label for _, label in DATASETS] + ["Six-dataset mean"]
    for axis, probe in zip(heat_axes, ("attention", "softmax")):
        matrix = np.array(
            [
                [values[probe][dataset][layer] for layer in layers]
                for dataset, _ in DATASETS
            ]
            + [[means[probe][layer] for layer in layers]]
        )
        axis.imshow(matrix, cmap=cmaps[probe], norm=norm, aspect="auto")
        axis.set_xticks(np.arange(len(layers)), [f"L{layer}" for layer in layers])
        axis.set_yticks(np.arange(len(row_labels)), row_labels)
        axis.tick_params(axis="both", length=0, pad=8, labelsize=9.4)
        axis.axhline(len(DATASETS) - 0.5, color="white", linewidth=3.0)
        best_layer = max(layers, key=means[probe].get)
        best_column = layers.index(best_layer)
        axis.add_patch(
            Rectangle(
                (best_column - 0.5, -0.5),
                1,
                len(row_labels),
                fill=False,
                edgecolor=colors[probe],
                linewidth=2.0,
            )
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value * 100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="white" if value >= 0.80 else TEXT,
                    fontweight=600 if row == len(DATASETS) else 400,
                )
        for tick, layer in zip(axis.get_xticklabels(), layers):
            tick.set_color(colors[probe] if layer not in (32, 47) else MUTED)
            tick.set_fontweight(600 if layer not in (32, 47) else 400)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(f"{probe.title()} · dataset breakdown", fontsize=13, fontweight=600, loc="left", pad=12)

    figure.text(0.075, 0.925, "Qwen3-32B layer screen", fontsize=25, fontweight=600, ha="left")
    figure.text(
        0.075,
        0.875,
        "Six-dataset mean AUROC and dataset-level results across seven sampled layers",
        fontsize=11.5,
        color=MUTED,
        ha="left",
    )
    figure.text(
        0.075,
        0.025,
        "Filled bars are the five added layers; outlined bars are the original L32 and L47 samples. Heatmap outlines mark the best mean layer. Single seed per layer.",
        fontsize=9,
        color="#92959A",
        ha="left",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "qwen3_32b_followup_layer_screen.png", facecolor="white", dpi=300)
    plt.close(figure)


def model_layer_means() -> dict[str, list[tuple[str, tuple[tuple[int, float], ...], str]]]:
    means = {probe: list(rows) for probe, rows in HISTORICAL_MEANS.items()}
    qwen_layers, qwen_values = load_qwen3_combined()
    qwen_means = qwen3_means(qwen_layers, qwen_values)
    for probe in means:
        means[probe].insert(
            1,
            (
                "Qwen3-32B",
                tuple((layer, qwen_means[probe][layer]) for layer in qwen_layers),
                "#0B6E99",
            ),
        )
    for run in RUNS:
        layers, values = load_run(run)
        for probe in means:
            rows = tuple(
                (
                    layer,
                    sum(values[probe][dataset][layer] for dataset, _ in DATASETS) / len(DATASETS),
                )
                for layer in layers
            )
            means[probe].append(
                (
                    run["title"].replace("-Instruct", "").replace(" IT", ""),
                    rows,
                    run[f"{probe}_color"],
                )
            )
    return means


def plot_model_layer_means() -> None:
    means = model_layer_means()
    figure, axes = plt.subplots(2, 1, figsize=(20, 12.5), sharex=False)
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.16, top=0.68, hspace=0.54)

    for axis, probe in zip(axes, ("attention", "softmax")):
        rows = means[probe]
        entries = [(model, layer, value, color) for model, layers, color in rows for layer, value in layers]
        positions = np.arange(len(entries))
        axis.bar(positions, [value for _, _, value, _ in entries], color=[color for _, _, _, color in entries], width=0.72, zorder=3)
        axis.axhline(THRESHOLD, color=ACCENT, alpha=0.60, linestyle=(0, (3, 3)), linewidth=1.2, zorder=4)
        axis.text(len(entries) - 0.25, THRESHOLD + 0.005, "all-layer sweep threshold · 85%", color=ACCENT, fontsize=9.5, ha="right", va="bottom")

        start = 0
        for index, (model, layers, _) in enumerate(rows):
            end = start + len(layers)
            if index % 2 == 1:
                axis.axvspan(start - 0.5, end - 0.5, color="#F6F7F8", zorder=0)
            best = max(value for _, value in layers)
            if model != "Llama-3.3-70B" and best < THRESHOLD:
                axis.text((start + end - 1) / 2, 0.706, "all-layer candidate", color=ACCENT, fontsize=9, fontweight=600, ha="center", va="bottom")
            axis.text((start + end - 1) / 2, -0.20, model, transform=axis.get_xaxis_transform(), color=TEXT, fontsize=10, fontweight=600, ha="center", va="top")
            if end < len(entries):
                axis.axvline(end - 0.5, color="#D8DBDF", linewidth=0.8, zorder=1)
            start = end

        for position, (_, _, value, _) in zip(positions, entries):
            axis.text(position, value + 0.006, f"{value * 100:.2f}%", ha="center", va="bottom", fontsize=7.6, color=TEXT, rotation=90)

        axis.set_title(f"{probe.title()} probe", fontsize=14, fontweight=600, loc="left", pad=14)
        axis.set_xlim(-0.7, len(entries) - 0.3)
        axis.set_ylim(0.70, 1.0)
        axis.set_yticks((0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0), ("70%", "75%", "80%", "85%", "90%", "95%", "100%"))
        axis.set_xticks(positions, [f"L{layer}" for _, layer, _, _ in entries])
        axis.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
        axis.tick_params(axis="x", length=0, pad=8, labelsize=9)
        axis.tick_params(axis="y", length=0, pad=6)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#D0D3D8")
        axis.set_ylabel("Mean AUROC", labelpad=11)

    figure.text(0.06, 0.955, "ALL-LAYER SWEEP SCREEN", fontsize=11, color=ACCENT, fontweight=600, ha="left")
    figure.text(0.06, 0.895, "Which model-probe pairs still need an all-layer sweep?", fontsize=25, fontweight=600, ha="left")
    figure.text(0.06, 0.848, "Bars are the six-dataset mean AUROC at each selected layer. Trigger a full sweep only when the best sampled layer stays below 85%.", fontsize=11.5, color=MUTED, ha="left")
    figure.text(0.06, 0.055, "Llama-3.3-70B@L31 is the replication reference. The 85% rule is a screening decision, not evidence that a probe cannot work at another layer.", fontsize=9.5, color="#92959A", ha="left")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "model_layer_mean_screen.png", facecolor="white", dpi=300)
    plt.close(figure)


def main() -> None:
    setup_font()
    write_run_table()
    for run in RUNS:
        plot_run(run)
        plot_baseline_gaps(run)
    qwen_layers, qwen_values = load_qwen3_combined()
    write_qwen3_table(qwen_layers, qwen_values)
    plot_qwen3_layer_screen()
    plot_model_layer_means()


if __name__ == "__main__":
    main()
