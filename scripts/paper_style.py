"""Shared visual style for the final paper figures."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Literal, Sequence

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


PALETTES = {
    "primary": ("#26547c", "#ef476f", "#ffd166", "#06d6a0", "#fffcf9"),
    "neutral": ("#fffffa", "#515052", "#000103", "#333138", "#ff312e"),
    "warm_blue": ("#b1740f", "#ffd07b", "#fdb833", "#296eb4", "#1789fc"),
    "green": ("#f1ffe7", "#a9fdac", "#44cf6c", "#32a287", "#6c464e"),
}
ALLOWED_COLORS = frozenset(color for palette in PALETTES.values() for color in palette)

BACKGROUND = "#fffcf9"
TEXT = "#000103"
MUTED = "#515052"
GRID = "#515052"
NEGATIVE = "#ef476f"
POSITIVE = "#32a287"
REFERENCE = "#fdb833"

CHARCOAL = "#333138"
BLUE = "#296eb4"
TEAL = "#32a287"
GREEN = "#06d6a0"
PINK = "#ef476f"
ORANGE = "#fdb833"
OCHRE = "#b1740f"

MODEL_FAMILY_COLORS = {
    "Llama": "#296eb4",
    "Mistral": "#fdb833",
    "Qwen": "#26547c",
    "Gemma": "#ef476f",
    "GLM": "#06d6a0",
    "Phi": "#6c464e",
    "Vicuna": "#1789fc",
    "DeepSeek": "#b1740f",
    "GPT-OSS": "#333138",
    "Moonlight": "#ffd166",
    "MiniMax": "#ff312e",
    "Other": "#515052",
}

STAGE_COLORS = {
    "paper": CHARCOAL,
    "replication": BLUE,
    "extension": TEAL,
}


@dataclass(frozen=True)
class StageStyle:
    label: str
    marker: str
    linestyle: str | tuple[int, tuple[int, ...]]
    hatch: str | None
    filled: bool


STAGE_STYLES = {
    "paper": StageStyle("Original paper", "o", (0, (4, 2)), None, False),
    "replication": StageStyle("Replication", "o", "-", None, True),
    "extension": StageStyle("Cross-model extension", "D", "-.", "///", False),
}

MODEL_ALIASES = (
    ("gpt-oss", "GPT-OSS"),
    ("deepseek", "DeepSeek"),
    ("moonlight", "Moonlight"),
    ("minimax", "MiniMax"),
    ("vicuna", "Vicuna"),
    ("mistral", "Mistral"),
    ("mixtral", "Mistral"),
    ("llama", "Llama"),
    ("qwen", "Qwen"),
    ("gemma", "Gemma"),
    ("glm", "GLM"),
    ("phi", "Phi"),
)

SINGLE_COLUMN_WIDTH = 3.25
DOUBLE_COLUMN_WIDTH = 6.75


def validate_colors(colors: Iterable[str]) -> None:
    invalid = sorted({color.lower() for color in colors} - ALLOWED_COLORS)
    if invalid:
        raise ValueError(f"colors outside the approved palettes: {invalid}")


def model_family(model_name: str) -> str:
    normalized = model_name.casefold()
    for alias, family in MODEL_ALIASES:
        if alias in normalized:
            return family
    return "Other"


def model_color(model_name: str) -> str:
    return MODEL_FAMILY_COLORS[model_family(model_name)]


def _font_candidates(font_path: str | Path | None = None) -> tuple[Path, ...]:
    explicit = font_path or os.environ.get("SPACE_GROTESK_PATH")
    paths = []
    if explicit:
        paths.append(Path(explicit).expanduser())
    paths.extend(
        (
            Path("/tmp/SpaceGrotesk-Regular.ttf"),
            Path.home() / "Library/Fonts/SpaceGrotesk-Variable.ttf",
            Path.home() / ".local/share/fonts/SpaceGrotesk-Variable.ttf",
        )
    )
    return tuple(paths)


def register_space_grotesk(font_path: str | Path | None = None) -> Path:
    for path in _font_candidates(font_path):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            if family != "Space Grotesk":
                raise ValueError(f"expected Space Grotesk, found {family!r} in {path}")
            return path
    raise FileNotFoundError(
        "Space Grotesk was not found; set SPACE_GROTESK_PATH to the font file"
    )


def configure_style(font_path: str | Path | None = None, scale: float = 1.0) -> Path:
    path = register_space_grotesk(font_path)
    mpl.rcParams.update(
        {
            "font.family": "Space Grotesk",
            "font.size": 8.0 * scale,
            "axes.titlesize": 9.0 * scale,
            "axes.titleweight": 600,
            "axes.labelsize": 8.2 * scale,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": CHARCOAL,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.3 * scale,
            "ytick.labelsize": 7.3 * scale,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.fontsize": 7.3 * scale,
            "legend.frameon": False,
            "text.color": TEXT,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return path


def figure_size(columns: Literal[1, 2] = 2, aspect: float = 0.62) -> tuple[float, float]:
    width = SINGLE_COLUMN_WIDTH if columns == 1 else DOUBLE_COLUMN_WIDTH
    return width, width * aspect


def style_axis(
    axis: Axes,
    *,
    grid_axis: Literal["x", "y", "both", "none"] = "y",
    percent_axis: Literal["x", "y", "none"] = "none",
    percent_xmax: float = 1.0,
    percent_decimals: int = 0,
) -> None:
    axis.set_axisbelow(True)
    if grid_axis != "none":
        axis.grid(axis=grid_axis, color=GRID, alpha=0.11, linewidth=0.58)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(CHARCOAL)
    axis.spines["bottom"].set_alpha(0.32)
    axis.tick_params(length=0, pad=3)
    if percent_axis == "x":
        axis.xaxis.set_major_formatter(PercentFormatter(percent_xmax, percent_decimals))
    elif percent_axis == "y":
        axis.yaxis.set_major_formatter(PercentFormatter(percent_xmax, percent_decimals))


def stage_bar_kwargs(stage: str, color: str) -> dict[str, object]:
    validate_colors((color,))
    style = STAGE_STYLES[stage]
    return {
        "facecolor": color if style.filled else BACKGROUND,
        "edgecolor": color,
        "hatch": style.hatch,
        "linewidth": 1.15,
    }


def stage_line_kwargs(stage: str, color: str) -> dict[str, object]:
    validate_colors((color,))
    style = STAGE_STYLES[stage]
    return {
        "color": color,
        "linestyle": style.linestyle,
        "marker": style.marker,
        "markerfacecolor": color if style.filled else BACKGROUND,
        "markeredgecolor": color,
        "linewidth": 1.7,
        "markersize": 4.2,
    }


def label_panels(axes: Sequence[Axes], labels: str = "abcdefghijklmnopqrstuvwxyz") -> None:
    for axis, label in zip(axes, labels, strict=False):
        axis.text(
            -0.08,
            1.04,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            fontweight=700,
            color=TEXT,
        )


def add_bar_labels(
    axis: Axes,
    bars: Iterable[object],
    *,
    percent: bool = True,
    decimals: int = 2,
    fraction_values: bool = True,
    padding: float = 2.0,
) -> None:
    for bar in bars:
        value = float(bar.get_height())
        shown = value * 100 if percent and fraction_values else value
        suffix = "%" if percent else ""
        axis.annotate(
            f"{shown:.{decimals}f}{suffix}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=TEXT,
            clip_on=False,
        )


def save_figure(
    figure: Figure,
    output_stem: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=dpi if suffix == "png" else None)
        outputs.append(path)
    return tuple(outputs)


validate_colors(
    (
        BACKGROUND,
        TEXT,
        MUTED,
        GRID,
        NEGATIVE,
        POSITIVE,
        REFERENCE,
        CHARCOAL,
        BLUE,
        TEAL,
        GREEN,
        PINK,
        ORANGE,
        OCHRE,
        *MODEL_FAMILY_COLORS.values(),
        *STAGE_COLORS.values(),
    )
)
