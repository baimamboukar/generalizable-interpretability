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
TARGETS = ROOT / "data/results/internal-instruction-following/paper_targets.csv"
COMPARISON = (
    ROOT / "data/results/internal-instruction-following/baseline_comparison.csv"
)
RESULTS_LONG = ROOT / "data/results/internal-instruction-following/results_long.csv"
RUN_METRICS = ROOT / "data/results/internal-instruction-following/run_metrics.csv"
METRICS_LONG = ROOT / "data/results/internal-instruction-following/metrics_long.csv"
ARTIFACTS = ROOT / "artifacts/internal-instruction-following"
OUTPUT = ROOT / "figures/internal-instruction-following"
FONT = Path.home() / "Library/Fonts/SpaceGrotesk-Variable.ttf"
STATIC_FONT = Path("/tmp/SpaceGrotesk-Regular.ttf")

MODELS = {
    "llama2-7b": ("Llama-2-7B-chat", "Llama-2-7B-Chat"),
    "llama2-13b": ("Llama-2-13B-chat", "Llama-2-13B-Chat"),
    "mistral-7b": (
        "Mistral-7B-Instruct-v0.3",
        "Mistral-7B-Instruct-v0.3",
    ),
    "phi3-mini": (
        "Phi-3-mini-128k-instruct",
        "Phi-3-Mini-128K-Instruct",
    ),
}
MODEL_COLORS = {
    "llama2-7b": "#0866FF",
    "llama2-13b": "#0866FF",
    "mistral-7b": "#FF6B00",
    "phi3-mini": "#00A4EF",
}
EXTENSIONS = {
    "qwen3-14b": ("Qwen3-14B", "Qwen3-14B", "#008C95"),
    "glm4.7-flash": ("GLM-4.7-Flash", "GLM-4.7-Flash", "#E45756"),
    "gemma4-12b": ("Gemma-4-12B-it", "Gemma-4-12B-IT", "#2F6BFF"),
}
EXPERIMENTS = ("task_generalization", "instruction_type_generalization")
LAYER_ORDER = {"early": 0, "middle": 1, "last": 2}
TOKEN_ORDER = {"first": 0, "middle": 1, "last": 2}
PAPER = "#303746"
TEXT = "#17191C"
MUTED = "#737B84"
GRID = "#E5E8EB"
CHANCE = "#C35B62"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def available_models() -> list[tuple[str, str, str]]:
    return [
        (key, paper_name, display_name)
        for key, (paper_name, display_name) in MODELS.items()
        if (ARTIFACTS / key / "probe_summary.csv").exists()
    ]


def available_extensions() -> list[tuple[str, str, str, str]]:
    return [
        (key, model_name, display_name, color)
        for key, (model_name, display_name, color) in EXTENSIONS.items()
        if (ARTIFACTS / key / "probe_summary.csv").exists()
    ]


def write_long_results(
    rows: list[dict[str, str]],
    models: list[tuple[str, str, str]],
    extensions: list[tuple[str, str, str, str]],
) -> Path:
    keys = {paper_name: key for key, paper_name, _ in models}
    output: list[dict[str, object]] = []
    for row in rows:
        common = {
            "model": row["model"],
            "model_key": keys[row["model"]],
            "experiment": row["experiment"],
            "layer": row["layer"],
            "token_position": row["token_position"],
            "metric": row["metric"],
            "expected_runs": 5,
        }
        output.append(
            common
            | {
                "scope": "paper",
                "mean": row["paper_mean"],
                "std": row["paper_std"],
                "valid_runs": "",
                "coverage": "",
                "paper_mean": row["paper_mean"],
                "delta_to_paper": 0.0,
            }
        )
        valid = int(row["runs"])
        output.append(
            common
            | {
                "scope": "baseline_replication",
                "mean": row["replication_mean"],
                "std": row["replication_std"],
                "valid_runs": valid,
                "coverage": valid / 5,
                "paper_mean": row["paper_mean"],
                "delta_to_paper": row["delta"],
            }
        )
    for key, model_name, _, _ in extensions:
        for row in extension_rows(key):
            valid = int(row["runs"])
            output.append(
                {
                    "scope": "extension",
                    "model": model_name,
                    "model_key": key,
                    "experiment": row["experiment"],
                    "layer": row["layer"],
                    "token_position": row["token_position"],
                    "metric": row["metric"],
                    "mean": row["replication_mean"],
                    "std": row["replication_std"],
                    "valid_runs": valid,
                    "expected_runs": 5,
                    "coverage": valid / 5,
                    "paper_mean": "",
                    "delta_to_paper": "",
                }
            )
    fields = [
        "scope",
        "model",
        "model_key",
        "experiment",
        "layer",
        "token_position",
        "metric",
        "mean",
        "std",
        "valid_runs",
        "expected_runs",
        "coverage",
        "paper_mean",
        "delta_to_paper",
    ]
    RESULTS_LONG.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_LONG.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return RESULTS_LONG


def write_diagnostic_results(
    models: list[tuple[str, str, str]],
    extensions: list[tuple[str, str, str, str]],
) -> tuple[Path, Path]:
    available = [
        ("baseline_replication", key, paper_name)
        for key, paper_name, _ in models
    ] + [("extension", key, model_name) for key, model_name, _, _ in extensions]
    run_fields = [
        "scope",
        "model_key",
        "model",
        "experiment",
        "layer",
        "token_position",
        "split",
        "heldout_instruction",
        "n_train",
        "n_test",
        "auroc",
        "threshold",
        "tn",
        "fp",
        "fn",
        "tp",
        "support_negative",
        "support_positive",
        "predicted_negative",
        "predicted_positive",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_tpr",
        "specificity_tnr",
        "fpr",
        "fnr",
        "f1",
        "status",
        "diagnostic_available",
    ]
    run_rows = []
    metric_rows = []
    for scope, key, model_name in available:
        run_path = ARTIFACTS / key / "probe_runs.csv"
        if run_path.exists():
            for row in read_csv(run_path):
                diagnostic = all(name in row for name in ("tn", "fp", "fn", "tp"))
                run_rows.append(
                    {
                        "scope": scope,
                        "model_key": key,
                        "model": model_name,
                        **{field: row.get(field, "") for field in run_fields[3:-1]},
                        "diagnostic_available": diagnostic,
                    }
                )
        metric_path = ARTIFACTS / key / "probe_metrics_summary.csv"
        if metric_path.exists():
            for row in read_csv(metric_path):
                metric_rows.append(
                    {
                        "scope": scope,
                        "model_key": key,
                        "model": model_name,
                        **row,
                    }
                )
    with RUN_METRICS.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(run_rows)
    metric_fields = [
        "scope",
        "model_key",
        "model",
        "experiment",
        "layer",
        "token_position",
        "metric",
        "paper_mean",
        "paper_std",
        "replication_mean",
        "replication_std",
        "delta",
        "runs",
        "status",
    ]
    with METRICS_LONG.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    return RUN_METRICS, METRICS_LONG


def build_comparison(models: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    targets = {
        (
            row["model"],
            row["experiment"],
            int(row["layer"]),
            row["token_position"],
        ): row
        for row in read_csv(TARGETS)
        if row["experiment"] in EXPERIMENTS
    }
    rows: list[dict[str, str]] = []
    for key, paper_name, _ in models:
        for result in read_csv(ARTIFACTS / key / "probe_summary.csv"):
            experiment = result["experiment"]
            if experiment not in EXPERIMENTS:
                continue
            lookup = (
                paper_name,
                experiment,
                int(result["layer"]),
                result["token_position"],
            )
            target = targets.get(lookup)
            if target is None:
                continue
            paper_mean = float(target["paper_mean"])
            replication_mean = float(result["replication_mean"])
            rows.append(
                {
                    "model": paper_name,
                    "experiment": experiment,
                    "table": target["table"],
                    "layer_group": target["layer_group"],
                    "layer": result["layer"],
                    "token_position": result["token_position"],
                    "metric": "auroc",
                    "paper_mean": f"{paper_mean:.8f}",
                    "paper_std": target["paper_std"],
                    "replication_mean": f"{replication_mean:.8f}",
                    "replication_std": result["replication_std"],
                    "delta": f"{replication_mean - paper_mean:.8f}",
                    "runs": result["runs"],
                    "status": result["status"],
                }
            )
    rows.sort(
        key=lambda row: (
            [model[1] for model in models].index(row["model"]),
            EXPERIMENTS.index(row["experiment"]),
            LAYER_ORDER[row["layer_group"]],
            TOKEN_ORDER[row["token_position"]],
        )
    )
    fields = [
        "model",
        "experiment",
        "table",
        "layer_group",
        "layer",
        "token_position",
        "metric",
        "paper_mean",
        "paper_std",
        "replication_mean",
        "replication_std",
        "delta",
        "runs",
        "status",
    ]
    COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def experiment_rows(
    rows: list[dict[str, str]], model: str, experiment: str
) -> list[dict[str, str]]:
    selected = [
        row for row in rows if row["model"] == model and row["experiment"] == experiment
    ]
    return sorted(
        selected,
        key=lambda row: (
            LAYER_ORDER[row["layer_group"]],
            TOKEN_ORDER[row["token_position"]],
        ),
    )


def panel(
    axis,
    rows: list[dict[str, str]],
    *,
    replication_color: str,
    show_group_dividers: bool,
    label_size: float,
) -> None:
    paper = np.array([float(row["paper_mean"]) * 100 for row in rows])
    replication = np.array([float(row["replication_mean"]) * 100 for row in rows])
    paper_std = np.array([float(row["paper_std"]) * 100 for row in rows])
    replication_std = np.array([float(row["replication_std"]) * 100 for row in rows])
    y = np.arange(len(rows), dtype=float) * 1.45
    if show_group_dividers:
        y += np.repeat([0.0, 0.80, 1.60], 3)
    labels = [
        (
            f"L{row['layer']} · {row['token_position'].title()}"
            if show_group_dividers
            else row["token_position"].title()
        )
        for row in rows
    ]
    height = 0.64
    error = {
        "ecolor": "#59616A",
        "elinewidth": 0.75,
        "capsize": 1.8,
        "capthick": 0.75,
    }
    paper_bars = axis.barh(
        y - height / 2,
        paper,
        height,
        xerr=paper_std,
        color=PAPER,
        error_kw=error,
        zorder=3,
    )
    replication_bars = axis.barh(
        y + height / 2,
        replication,
        height,
        xerr=replication_std,
        color=replication_color,
        error_kw=error,
        zorder=3,
    )
    axis.set_xlim(0, 104)
    axis.set_xticks(np.arange(0, 101, 25))
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.axvline(50, color=CHANCE, linestyle=(0, (3, 4)), linewidth=0.9, zorder=1)
    axis.grid(axis="x", color=GRID, linewidth=0.75, zorder=0)
    axis.tick_params(axis="x", length=0, pad=6, labelsize=8.5)
    axis.tick_params(axis="y", length=0, pad=7, labelsize=label_size)
    axis.spines["bottom"].set_color("#C9CED3")
    if show_group_dividers:
        axis.axhline((y[2] + y[3]) / 2, color=GRID, linewidth=0.7)
        axis.axhline((y[5] + y[6]) / 2, color=GRID, linewidth=0.7)
    for bars, values in ((paper_bars, paper), (replication_bars, replication)):
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                value / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%",
                ha="center",
                va="center",
                fontsize=label_size - 0.7,
                fontweight=600,
                color="white",
            )


def decorate_model(
    figure,
    *,
    title: str,
    subtitle: str,
    replication_color: str,
    footnote: str,
) -> None:
    figure.text(0.055, 0.965, title, fontsize=22, fontweight=600, ha="left")
    figure.text(0.055, 0.925, subtitle, fontsize=10.5, color=MUTED, ha="left")
    figure.legend(
        handles=[
            Patch(color=PAPER, label="Original paper"),
            Patch(color=replication_color, label="Replication"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.052, 0.895),
        ncol=2,
        frameon=False,
        handlelength=1.45,
        columnspacing=1.9,
    )
    figure.text(
        0.055,
        0.022,
        footnote,
        fontsize=8.5,
        color=MUTED,
        ha="left",
    )


def overview(rows: list[dict[str, str]], models: list[tuple[str, str, str]]) -> Path:
    figure, axes = plt.subplots(1, len(models), figsize=(15.8, 7.2), facecolor="white")
    axes = np.atleast_1d(axes)
    figure.subplots_adjust(
        left=0.07,
        right=0.975,
        top=0.72,
        bottom=0.17,
        wspace=0.25,
    )
    categories = ("Unseen tasks", "Unseen instruction types")
    x = np.arange(2)
    width = 0.44
    for axis, (key, paper_name, display_name) in zip(axes, models, strict=True):
        paper_values = []
        replication_values = []
        for experiment in EXPERIMENTS:
            selected = experiment_rows(rows, paper_name, experiment)
            paper_values.append(
                np.mean([float(row["paper_mean"]) for row in selected]) * 100
            )
            replication_values.append(
                np.mean([float(row["replication_mean"]) for row in selected]) * 100
            )
        paper_bars = axis.bar(x - width / 2, paper_values, width, color=PAPER, zorder=3)
        replication_bars = axis.bar(
            x + width / 2,
            replication_values,
            width,
            color=MODEL_COLORS[key],
            zorder=3,
        )
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 25))
        axis.set_xticks(x, categories)
        axis.set_ylabel("AUROC", labelpad=8)
        axis.set_title(display_name, fontsize=13, fontweight=600, pad=15)
        axis.axhline(50, color=CHANCE, linestyle=(0, (3, 4)), linewidth=0.9, zorder=1)
        axis.grid(axis="y", color=GRID, linewidth=0.75, zorder=0)
        axis.tick_params(axis="both", length=0, pad=7)
        axis.spines["bottom"].set_color("#C9CED3")
        for bars, values in (
            (paper_bars, paper_values),
            (replication_bars, replication_values),
        ):
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 2.0,
                    f"{value:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9.2,
                    fontweight=600,
                )
    figure.text(
        0.055,
        0.955,
        "Instruction-following probes: paper vs replication",
        fontsize=22,
        fontweight=600,
        ha="left",
    )
    figure.text(
        0.055,
        0.905,
        "Tables 1–2 · mean probe performance for each completed baseline model",
        fontsize=10.5,
        color=MUTED,
        ha="left",
    )
    families = [("Llama", "llama2-7b"), ("Mistral", "mistral-7b")]
    if any(key == "phi3-mini" for key, _, _ in models):
        families.append(("Phi", "phi3-mini"))
    figure.legend(
        handles=[Patch(color=PAPER, label="Original paper")]
        + [
            Patch(color=MODEL_COLORS[key], label=f"{family} replication")
            for family, key in families
        ],
        loc="upper left",
        bbox_to_anchor=(0.052, 0.845),
        ncol=len(families) + 1,
        frameon=False,
        handlelength=1.45,
        columnspacing=1.8,
    )
    figure.text(
        0.055,
        0.045,
        "Unseen tasks averages 3 layers × 3 token positions. Unseen instruction types averages "
        "first, middle, and last tokens at the early layer. Single-class folds are excluded. Dashed line = chance (50%).",
        fontsize=8.8,
        color=MUTED,
        ha="left",
    )
    path = OUTPUT / "paper_vs_replication_completed.png"
    figure.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def model_figure(rows: list[dict[str, str]], model: tuple[str, str, str]) -> Path:
    key, paper_name, display_name = model
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14.2, 11.8),
        facecolor="white",
        gridspec_kw={"height_ratios": [2.6, 1.15]},
    )
    figure.subplots_adjust(
        left=0.14,
        right=0.975,
        top=0.77,
        bottom=0.11,
        hspace=0.38,
    )
    panel(
        axes[0],
        experiment_rows(rows, paper_name, "task_generalization"),
        replication_color=MODEL_COLORS[key],
        show_group_dividers=True,
        label_size=9.5,
    )
    panel(
        axes[1],
        experiment_rows(rows, paper_name, "instruction_type_generalization"),
        replication_color=MODEL_COLORS[key],
        show_group_dividers=False,
        label_size=9.8,
    )
    axes[0].set_title(
        "Generalization to unseen tasks · Table 2",
        fontsize=13,
        fontweight=600,
        loc="left",
        pad=16,
    )
    axes[1].set_title(
        "Generalization to unseen instruction types · Table 1",
        fontsize=13,
        fontweight=600,
        loc="left",
        pad=16,
    )
    axes[0].set_xlabel("AUROC", labelpad=8)
    axes[1].set_xlabel("AUROC", labelpad=8)
    layer_rows = experiment_rows(rows, paper_name, "task_generalization")
    layers = [
        next(row["layer"] for row in layer_rows if row["layer_group"] == group)
        for group in ("early", "middle", "last")
    ]
    decorate_model(
        figure,
        title=f"{display_name}: paper vs replication",
        subtitle="Replication of Tables 1–2 · five planned splits or folds; valid AUROC runs shown",
        replication_color=MODEL_COLORS[key],
        footnote=(
            "Bars show mean AUROC; whiskers show one standard deviation. "
            f"L{layers[0]}, L{layers[1]}, and L{layers[2]} are the early, middle, and final tested layers; "
            "First, Middle, and Last are response-token positions. Dashed line = chance (50%)."
        ),
    )
    path = OUTPUT / f"paper_vs_replication_{key.replace('-', '_')}.png"
    figure.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def summaries(rows: list[dict[str, str]], models: list[tuple[str, str, str]]) -> None:
    for _, paper_name, display_name in models:
        parts = []
        for experiment in EXPERIMENTS:
            selected = experiment_rows(rows, paper_name, experiment)
            gaps = np.array([abs(float(row["delta"])) * 100 for row in selected])
            parts.append(f"{experiment}: MAE={gaps.mean():.2f} pp")
        print(f"{display_name} | " + " | ".join(parts))


def extension_rows(key: str) -> list[dict[str, str]]:
    return [
        row
        for row in read_csv(ARTIFACTS / key / "probe_summary.csv")
        if row["experiment"] in EXPERIMENTS
    ]


def aggregate_extension(
    key: str, experiment: str, *, earliest_layer_only: bool
) -> float:
    rows = [row for row in extension_rows(key) if row["experiment"] == experiment]
    if earliest_layer_only:
        earliest = min(int(row["layer"]) for row in rows)
        rows = [row for row in rows if int(row["layer"]) == earliest]
    return float(np.mean([float(row["replication_mean"]) for row in rows]))


def baseline_extension_overview(
    rows: list[dict[str, str]],
    models: list[tuple[str, str, str]],
    extensions: list[tuple[str, str, str, str]],
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(17.8, 8.2), facecolor="white")
    figure.subplots_adjust(left=0.07, right=0.985, top=0.70, bottom=0.21, wspace=0.15)
    labels = [model[2] for model in models] + [model[2] for model in extensions]
    x = np.arange(len(labels), dtype=float)
    width = 0.34

    for axis, experiment, title in zip(
        axes,
        EXPERIMENTS,
        (
            "Generalization to unseen tasks",
            "Generalization to unseen instruction types",
        ),
        strict=True,
    ):
        for index, (key, paper_name, _) in enumerate(models):
            selected = experiment_rows(rows, paper_name, experiment)
            paper_value = np.mean([float(row["paper_mean"]) for row in selected]) * 100
            replication_value = (
                np.mean([float(row["replication_mean"]) for row in selected]) * 100
            )
            paper_bar = axis.bar(
                index - width / 2,
                paper_value,
                width,
                color=PAPER,
                zorder=3,
            )[0]
            result_bar = axis.bar(
                index + width / 2,
                replication_value,
                width,
                color=MODEL_COLORS[key],
                zorder=3,
            )[0]
            for bar, value in (
                (paper_bar, paper_value),
                (result_bar, replication_value),
            ):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value - 1.5,
                    f"{value:.2f}%",
                    ha="center",
                    va="top",
                    fontsize=8.2,
                    fontweight=600,
                    color="white",
                )

        offset = len(models)
        for extension_index, (key, _, _, color) in enumerate(extensions):
            index = offset + extension_index
            value = (
                aggregate_extension(
                    key,
                    experiment,
                    earliest_layer_only=experiment == "instruction_type_generalization",
                )
                * 100
            )
            bar = axis.bar(index, value, width * 1.45, color=color, zorder=3)[0]
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8.4,
                fontweight=600,
            )

        if models and extensions:
            divider = len(models) - 0.5
            axis.axvline(divider, color="#CDD2D7", linewidth=1.0, zorder=1)
            axis.text(
                (len(models) - 1) / 2,
                102.5,
                "BASELINE REPLICATIONS",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color=MUTED,
                fontweight=600,
            )
            axis.text(
                len(models) + (len(extensions) - 1) / 2,
                102.5,
                "CROSS-MODEL EXTENSIONS",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color=MUTED,
                fontweight=600,
            )

        axis.set_title(title, fontsize=14, fontweight=600, loc="left", pad=28)
        axis.set_ylim(35, 107)
        axis.set_yticks(
            (40, 50, 60, 70, 80, 90, 100),
            [f"{value}%" for value in (40, 50, 60, 70, 80, 90, 100)],
        )
        axis.set_xticks(x, labels, rotation=22, ha="right")
        axis.axhline(50, color=CHANCE, linestyle=(0, (3, 4)), linewidth=0.9, zorder=1)
        axis.grid(axis="y", color=GRID, linewidth=0.75, zorder=0)
        axis.tick_params(axis="both", length=0, pad=7, labelsize=9)
        axis.spines["bottom"].set_color("#C9CED3")
        axis.set_ylabel("Mean AUROC", labelpad=8)

    figure.text(
        0.055,
        0.955,
        "Instruction-following probes: baseline replication and extensions",
        fontsize=22,
        fontweight=600,
        ha="left",
    )
    figure.text(
        0.055,
        0.905,
        "Paper-aligned baseline results followed by the same probe protocol on newer model families",
        fontsize=10.5,
        color=MUTED,
        ha="left",
    )
    legend_handles = [Patch(color=PAPER, label="Original paper")]
    baseline_families = [("Llama", "llama2-7b"), ("Mistral", "mistral-7b")]
    if any(key == "phi3-mini" for key, _, _ in models):
        baseline_families.append(("Phi", "phi3-mini"))
    legend_handles.extend(
        Patch(color=MODEL_COLORS[key], label=f"{family} replication")
        for family, key in baseline_families
    )
    legend_handles.extend(
        Patch(color=color, label=f"{display_name} extension")
        for _, _, display_name, color in extensions
    )
    figure.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.052, 0.845),
        ncol=3,
        frameon=False,
        handlelength=1.45,
        columnspacing=1.8,
    )
    figure.text(
        0.055,
        0.04,
        "Unseen tasks averages all sampled layers and token positions. Unseen instruction types uses the earliest sampled layer, matching Table 1. "
        "Single-class held-out folds have undefined AUROC and are excluded; coverage is retained in results_long.csv. Dashed line = chance (50%).",
        fontsize=8.7,
        color=MUTED,
        ha="left",
    )
    path = OUTPUT / "baseline_vs_extensions.png"
    figure.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def extension_layer_profiles(
    extensions: list[tuple[str, str, str, str]],
) -> Path:
    figure, axes = plt.subplots(
        2,
        len(extensions),
        figsize=(18.4, 9.8),
        facecolor="white",
        sharey="row",
    )
    axes = np.asarray(axes).reshape(2, len(extensions))
    figure.subplots_adjust(
        left=0.07, right=0.985, top=0.76, bottom=0.14, hspace=0.45, wspace=0.14
    )

    for column, (key, _, display_name, color) in enumerate(extensions):
        rows = extension_rows(key)
        layers = sorted({int(row["layer"]) for row in rows})
        for row_index, (experiment, row_title) in enumerate(
            zip(EXPERIMENTS, ("Unseen tasks", "Unseen instruction types"), strict=True)
        ):
            axis = axes[row_index, column]
            values = []
            for layer in layers:
                selected = [
                    float(row["replication_mean"])
                    for row in rows
                    if row["experiment"] == experiment and int(row["layer"]) == layer
                ]
                values.append(float(np.mean(selected)) * 100)
            positions = np.arange(len(layers))
            bars = axis.bar(positions, values, width=0.62, color=color, zorder=3)
            axis.bar_label(
                bars,
                labels=[f"{value:.2f}%" for value in values],
                padding=4,
                fontsize=8.7,
                fontweight=600,
                color=TEXT,
            )
            axis.set_ylim(35, 103)
            axis.set_yticks(
                (40, 50, 60, 70, 80, 90, 100),
                [f"{value}%" for value in (40, 50, 60, 70, 80, 90, 100)],
            )
            axis.set_xticks(positions, [f"L{layer}" for layer in layers])
            axis.axhline(
                50, color=CHANCE, linestyle=(0, (3, 4)), linewidth=0.9, zorder=1
            )
            axis.grid(axis="y", color=GRID, linewidth=0.75, zorder=0)
            axis.tick_params(axis="both", length=0, pad=7, labelsize=9)
            axis.spines["bottom"].set_color("#C9CED3")
            axis.set_title(
                f"{display_name}\n{row_title}",
                fontsize=12.5,
                fontweight=600,
                loc="left",
                pad=12,
            )
            if column == 0:
                axis.set_ylabel("Mean AUROC", labelpad=8)
            axis.set_xlabel("Probed transformer layer", labelpad=8)

    figure.text(
        0.055,
        0.955,
        "Cross-model extension: layer profiles",
        fontsize=22,
        fontweight=600,
        ha="left",
    )
    figure.text(
        0.055,
        0.905,
        "Mean probe performance across first, middle, and last response-token positions",
        fontsize=10.5,
        color=MUTED,
        ha="left",
    )
    figure.text(
        0.055,
        0.04,
        "Each bar averages the three token-position probes at one layer. AUROC summaries use valid folds only; coverage is retained in results_long.csv. "
        "Dashed line = chance (50%).",
        fontsize=8.7,
        color=MUTED,
        ha="left",
    )
    path = OUTPUT / "extension_layer_profiles.png"
    figure.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> None:
    setup()
    models = available_models()
    if not models:
        raise RuntimeError("No completed baseline summaries found")
    rows = build_comparison(models)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [overview(rows, models)]
    paths.extend(model_figure(rows, model) for model in models)
    extensions = available_extensions()
    long_results = write_long_results(rows, models, extensions)
    run_metrics, metrics_long = write_diagnostic_results(models, extensions)
    if extensions:
        paths.append(baseline_extension_overview(rows, models, extensions))
        paths.append(extension_layer_profiles(extensions))
    summaries(rows, models)
    print(COMPARISON)
    print(long_results)
    print(run_metrics)
    print(metrics_long)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
