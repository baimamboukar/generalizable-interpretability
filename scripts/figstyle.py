"""Figure design system for the reproduction-first probing manuscript.

One import, one call:

    from figstyle import use_style, PALETTE, ...
    use_style(font="DMSans")

Design contract enforced by this module
---------------------------------------
* One canvas tone, one ink tone, one accent per figure. Colour is never
  decorative: it either encodes model family or it encodes evidence stage,
  never both in the same figure.
* No chartjunk: no top/right spines, no vertical gridlines on categorical
  axes, no drop shadows, no 3-D, no redundant legends.
* Text has exactly four roles (panel title, axis label, tick, annotation)
  with a fixed size ladder. Nothing else may set a font size.
* Every glyph is >= 7 pt at final column width.
* Titles and long protocol prose live in the LaTeX caption, not the image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator

# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "static"

AVAILABLE_FONTS = (
    "DMSans",
    "SpaceGrotesk",
    "Inter",
    "IBMPlexSans",
    "Manrope",
    "Outfit",
    "SourceSans3",
    "Avenir Next",  # system font, macOS only
)


def register_fonts(font_dir: Path | None = None) -> tuple[str, ...]:
    """Register every vendored static font. Returns the family names found."""
    directory = Path(font_dir) if font_dir else FONT_DIR
    found: set[str] = set()
    if directory.is_dir():
        for path in sorted(directory.glob("*.ttf")):
            fm.fontManager.addfont(str(path))
            found.add(fm.ttfFontProperty(fm.get_font(str(path))).name)
    return tuple(sorted(found))


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """The paper's colour system.

    ``canvas``/``ink`` set the page. ``accent`` is the single colour a reader's
    eye should land on. ``support`` carries secondary series. ``muted`` is for
    reference material that must recede.
    """

    canvas: str = "#FFFFFF"          # white page
    canvas_alt: str = "#F5F6F6"      # light neutral panel wash
    ink: str = "#1F2421"             # near-black with a green cast
    ink_soft: str = "#4A5450"        # secondary text
    ink_faint: str = "#8B948F"       # ticks, rules
    rule: str = "#DDD5C7"            # hairlines, gridlines

    accent: str = "#4A7C59"          # deep sage - the primary voice
    accent_deep: str = "#2F5943"     # darkest sage, for emphasis
    accent_soft: str = "#8FC0A9"     # light sage fill
    accent_pale: str = "#C8D5B9"     # palest sage wash

    teal: str = "#68B0AB"            # secondary series
    teal_deep: str = "#3E7C78"
    steel: str = "#568EA3"           # tertiary series
    steel_deep: str = "#3A6577"
    clay: str = "#826251"            # warm counterpoint
    clay_soft: str = "#B08968"
    sand: str = "#FFE8D1"            # warm wash
    ember: str = "#C1654F"           # the only "alarm" colour

    positive: str = "#4A7C59"
    negative: str = "#C1654F"
    reference: str = "#8B948F"


PALETTE = Palette()

#: Ordered series ramp. Perceptually separated and greyscale-safe.
SERIES = (
    PALETTE.accent,
    PALETTE.steel,
    PALETTE.clay_soft,
    PALETTE.teal,
    PALETTE.clay,
    PALETTE.accent_soft,
)

#: Sequential ramp for ordered / continuous data (light -> dark sage).
SEQUENTIAL = ("#EDF2E6", "#C8D5B9", "#8FC0A9", "#68B0AB", "#4A7C59", "#2F5943")

#: Stable colour per model family. Used only in cross-model figures.
MODEL_FAMILY_COLORS = {
    "Llama": "#568EA3",
    "Qwen": "#2F5943",
    "Gemma": "#C1654F",
    "GLM": "#68B0AB",
    "Phi": "#826251",
    "Mistral": "#B08968",
    "Moonlight": "#8FC0A9",
    "Vicuna": "#8AA8B8",
    "DeepSeek": "#6B7F6E",
    "Other": "#8B948F",
}

#: Evidence-stage colours. Used only in figures where colour is not family.
STAGE_COLORS = {
    "paper": "#8B948F",
    "replication": "#4A7C59",
    "extension": "#568EA3",
    "failed": "#C1654F",
}

_MODEL_ALIASES = (
    ("moonlight", "Moonlight"),
    ("deepseek", "DeepSeek"),
    ("vicuna", "Vicuna"),
    ("mistral", "Mistral"),
    ("mixtral", "Mistral"),
    ("llama", "Llama"),
    ("qwen", "Qwen"),
    ("gemma", "Gemma"),
    ("glm", "GLM"),
    ("phi", "Phi"),
)


def model_family(name: str) -> str:
    low = name.casefold()
    for alias, family in _MODEL_ALIASES:
        if alias in low:
            return family
    return "Other"


def model_color(name: str) -> str:
    return MODEL_FAMILY_COLORS[model_family(name)]


# --------------------------------------------------------------------------
# Type ladder and geometry
# --------------------------------------------------------------------------

#: The only sanctioned font sizes, in points, at scale 1.0.
TYPE = {
    "panel_title": 9.0,
    "panel_note": 7.6,
    "axis_label": 8.0,
    "tick": 7.4,
    "annot": 7.2,
    "legend": 7.6,
    "panel_letter": 9.5,
}

# Page geometry, read off iclr2026_conference.sty: \textwidth 5.5in,
# \textheight 9.0in, single column. A figure built WIDER than TEXT_W does not
# overflow - \includegraphics[width=\linewidth] silently scales it down, and
# every font scales with it, so a 6pt annotation lands at 4.8pt on the page.
# Build at the true text width so the point sizes declared in TYPE are the point
# sizes the reader gets.
TEXT_W = 5.50   # \textwidth, inches
TEXT_H = 9.00   # \textheight, inches
COL1 = TEXT_W / 2 - 0.10   # half-width figure, allowing a gutter
COL2 = TEXT_W              # full text width
MAX_FIG_H = 8.60           # a figure taller than this cannot share a page


def figure_size(width: Literal[1, 2] | float = 2, aspect: float = 0.62) -> tuple[float, float]:
    w = COL1 if width == 1 else COL2 if width == 2 else float(width)
    return w, w * aspect


#: The adopted pairing. Space Grotesk carries display type - figure titles and
#: block tags - and DM Sans carries everything a reader compares numerically,
#: because its numerals sit in more even columns at annotation sizes. Figures
#: must not hardcode a family; they ask for a role via :func:`font_for`.
DISPLAY_FONT = "SpaceGrotesk"
DATA_FONT = "DMSans"

#: Which family serves each type role.
ROLE_FONT = {
    "figure_title": DISPLAY_FONT,
    "block_tag": DISPLAY_FONT,
    "panel_title": DISPLAY_FONT,
    "deck": DATA_FONT,
    "panel_note": DATA_FONT,
    "axis_label": DATA_FONT,
    "tick": DATA_FONT,
    "annot": DATA_FONT,
    "legend": DATA_FONT,
    "value": DATA_FONT,
}

_ACTIVE_FONTS: dict[str, str] = {}


def font_for(role: str) -> str:
    """Return the family a type role should use, after fallback resolution.

    Activates the style if it has not been. Figure scripts compute layout
    budgets at IMPORT time (a wrapped footnote's line count, a measured column
    width) but call ``use_style`` inside ``main``, so without this the budget
    would be measured in the fallback font and the drawing done in the real one
    - a silent disagreement that shows up as overflow only on the page.
    """
    if not _ACTIVE_FONTS:
        use_style()
    want = ROLE_FONT.get(role, DATA_FONT)
    return _ACTIVE_FONTS.get(want, _ACTIVE_FONTS.get(DATA_FONT, "DejaVu Sans"))


def _resolve(want: str, families: set[str]) -> str:
    if want in families:
        return want
    for candidate in AVAILABLE_FONTS:
        if candidate in families:
            return candidate
    return "DejaVu Sans"


def use_style(font: str | None = None, scale: float = 1.0) -> str:
    """Install the paper style. Returns the data font family actually used.

    ``font`` overrides the data family - used by the specimen sheet to render
    the same figure in each candidate. Production figures leave it unset so
    the adopted pairing applies.
    """
    families = set(register_fonts()) | {f.name for f in fm.fontManager.ttflist}
    _ACTIVE_FONTS.clear()
    _ACTIVE_FONTS[DATA_FONT] = _resolve(font or DATA_FONT, families)
    _ACTIVE_FONTS[DISPLAY_FONT] = (
        _ACTIVE_FONTS[DATA_FONT] if font else _resolve(DISPLAY_FONT, families))
    chosen = _ACTIVE_FONTS[DATA_FONT]
    mpl.rcParams.update(
        {
            "font.family": chosen,
            "font.size": TYPE["tick"] * scale,
            "font.weight": 400,
            "axes.titlesize": TYPE["panel_title"] * scale,
            "axes.titleweight": 600,
            "axes.labelsize": TYPE["axis_label"] * scale,
            "axes.labelweight": 500,
            "axes.labelcolor": PALETTE.ink_soft,
            "axes.edgecolor": PALETTE.ink_faint,
            "axes.linewidth": 0.6,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": TYPE["tick"] * scale,
            "ytick.labelsize": TYPE["tick"] * scale,
            "xtick.color": PALETTE.ink_soft,
            "ytick.color": PALETTE.ink_soft,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.pad": 2.6,
            "ytick.major.pad": 2.4,
            "legend.fontsize": TYPE["legend"] * scale,
            "legend.frameon": False,
            "legend.handlelength": 1.05,
            "legend.handleheight": 0.72,
            "legend.handletextpad": 0.48,
            "legend.columnspacing": 1.25,
            "legend.labelspacing": 0.42,
            "legend.borderpad": 0.0,
            "text.color": PALETTE.ink,
            "figure.facecolor": PALETTE.canvas,
            "axes.facecolor": PALETTE.canvas,
            "savefig.facecolor": PALETTE.canvas,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.02,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.0,
            "hatch.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
        }
    )
    return chosen


# --------------------------------------------------------------------------
# Axis furniture
# --------------------------------------------------------------------------


def clean_axes(
    ax: Axes,
    *,
    grid: Literal["x", "y", "both", "none"] = "y",
    spine: Literal["x", "y", "both", "none"] = "x",
    pct: Literal["x", "y", "none"] = "none",
    pct_decimals: int = 0,
    nbins: int | None = 5,
) -> Axes:
    """Strip an axis to data + one baseline + faint gridlines."""
    ax.set_facecolor("none")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if spine in ("x", "both"):
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(PALETTE.rule)
        ax.spines["bottom"].set_linewidth(0.7)
    if spine in ("y", "both"):
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(PALETTE.rule)
        ax.spines["left"].set_linewidth(0.7)

    if grid != "none":
        ax.grid(
            axis=grid,
            color=PALETTE.rule,
            linewidth=0.6,
            alpha=0.9,
            zorder=0,
        )
    ax.set_axisbelow(True)
    ax.tick_params(which="both", length=0, pad=2.6)

    if nbins:
        target = ax.yaxis if grid in ("y", "both") else ax.xaxis
        target.set_major_locator(MaxNLocator(nbins=nbins, prune=None))

    if pct != "none":
        fmt = FuncFormatter(lambda v, _: f"{v:.{pct_decimals}f}%")
        (ax.xaxis if pct == "x" else ax.yaxis).set_major_formatter(fmt)
    return ax


def panel_title(ax: Axes, title: str, note: str | None = None, *, pad: float = 3.0) -> None:
    """Left-aligned panel title with an optional single-line note beneath.

    Offsets are in points, not axes fractions, so the title block sits at a
    constant optical distance from the axes regardless of panel height.
    The note carries the one fact needed to read the panel (metric, selection
    regime). Anything longer belongs in the LaTeX caption.
    """
    note_h = TYPE["panel_note"] + 2.0
    title_dy = pad + (note_h if note else 0.0)
    ax.annotate(
        title, xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(0.0, title_dy), textcoords="offset points",
        ha="left", va="bottom",
        fontsize=TYPE["panel_title"], fontweight=600, color=PALETTE.ink,
        annotation_clip=False,
    )
    if note:
        ax.annotate(
            note, xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(0.0, pad), textcoords="offset points",
            ha="left", va="bottom",
            fontsize=TYPE["panel_note"], fontweight=400, color=PALETTE.ink_faint,
            annotation_clip=False,
        )


def title_block_height(has_note: bool = True) -> float:
    """Height of a panel title block in points."""
    h = 3.0 + TYPE["panel_title"] + 2.0
    if has_note:
        h += TYPE["panel_note"] + 2.0
    return h


def panel_letter(ax: Axes, letter: str, *, dx: float = -26.0, dy: float = 3.0) -> None:
    """Panel letter, offset in points from the axes' top-left corner."""
    ax.annotate(
        letter, xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(dx, dy + title_block_height()), textcoords="offset points",
        ha="left", va="bottom",
        fontsize=TYPE["panel_letter"], fontweight=700, color=PALETTE.accent_deep,
        annotation_clip=False,
    )


def figure_caption(fig: Figure, text: str, *, y: float = 0.006) -> None:
    """A single-line provenance note at the foot of the figure."""
    fig.text(
        0.5, y, text,
        ha="center", va="bottom",
        fontsize=TYPE["annot"], color=PALETTE.ink_faint,
    )


def legend(
    ax: Axes,
    handles=None,
    labels=None,
    *,
    loc: str = "upper left",
    ncol: int = 1,
    bbox=None,
    title: str | None = None,
) -> None:
    kwargs = dict(loc=loc, ncol=ncol, frameon=False, borderaxespad=0.0)
    if bbox is not None:
        kwargs["bbox_to_anchor"] = bbox
    lg = ax.legend(handles, labels, **kwargs) if handles is not None else ax.legend(**kwargs)
    if title:
        lg.set_title(title)
        lg.get_title().set_fontsize(TYPE["legend"])
        lg.get_title().set_fontweight(600)
        lg.get_title().set_color(PALETTE.ink_soft)
    for text in lg.get_texts():
        text.set_color(PALETTE.ink_soft)


# --------------------------------------------------------------------------
# Marks
# --------------------------------------------------------------------------


def rounded_bars(
    ax: Axes,
    x: Sequence[float],
    heights: Sequence[float],
    *,
    width: float = 0.62,
    color: str | Sequence[str] = PALETTE.accent,
    radius: float = 0.035,
    baseline: float = 0.0,
    horizontal: bool = False,
    alpha: float = 1.0,
    edge: str | None = None,
    zorder: float = 3,
) -> list[FancyBboxPatch]:
    """Bars with a softly rounded cap on the free end only.

    Rounding is applied in axis-fraction space so radius is visually equal
    across panels regardless of the data range.
    """
    colors = [color] * len(x) if isinstance(color, str) else list(color)
    patches: list[FancyBboxPatch] = []
    span = None
    for xi, h, c in zip(x, heights, colors, strict=False):
        if not np.isfinite(h):
            continue
        if horizontal:
            lo, hi = (baseline, h) if h >= baseline else (h, baseline)
            rect = FancyBboxPatch(
                (lo, xi - width / 2), max(hi - lo, 1e-12), width,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                mutation_aspect=1.0,
                facecolor=c, edgecolor=edge or "none",
                linewidth=0.0 if edge is None else 0.7,
                alpha=alpha, zorder=zorder, clip_on=False,
            )
        else:
            lo, hi = (baseline, h) if h >= baseline else (h, baseline)
            span = span or 1.0
            rect = FancyBboxPatch(
                (xi - width / 2, lo), width, max(hi - lo, 1e-12),
                boxstyle=f"round,pad=0,rounding_size={radius}",
                mutation_aspect=1.0,
                facecolor=c, edgecolor=edge or "none",
                linewidth=0.0 if edge is None else 0.7,
                alpha=alpha, zorder=zorder, clip_on=False,
            )
        ax.add_patch(rect)
        patches.append(rect)
    return patches


def value_labels(
    ax: Axes,
    x: Sequence[float],
    heights: Sequence[float],
    *,
    fmt: str = "{:.2f}",
    horizontal: bool = False,
    color: str | Sequence[str] | None = None,
    offset: float = 3.0,
    weight: int = 600,
    inside: bool = False,
) -> None:
    """Direct value labels. Direct labelling replaces a value axis, so use
    one or the other, never both."""
    colors = (
        [color] * len(x) if isinstance(color, str) or color is None else list(color)
    )
    for xi, h, c in zip(x, heights, colors, strict=False):
        if not np.isfinite(h):
            continue
        sign = 1 if h >= 0 else -1
        if horizontal:
            xy = (h, xi)
            off = (sign * (-offset if inside else offset), 0)
            ha = ("right" if sign > 0 else "left") if inside else ("left" if sign > 0 else "right")
            va = "center"
        else:
            xy = (xi, h)
            off = (0, sign * (-offset if inside else offset))
            ha = "center"
            va = ("top" if sign > 0 else "bottom") if inside else ("bottom" if sign > 0 else "top")
        ax.annotate(
            fmt.format(h), xy, xytext=off, textcoords="offset points",
            ha=ha, va=va, fontsize=TYPE["annot"], fontweight=weight,
            color=c or PALETTE.ink, clip_on=False, zorder=6,
        )


def reference_line(
    ax: Axes,
    value: float,
    *,
    label: str | None = None,
    horizontal: bool = True,
    color: str | None = None,
    label_x: float = 0.995,
) -> None:
    """A dashed reference level with the label riding on the line itself."""
    c = color or PALETTE.ink_faint
    if horizontal:
        ax.axhline(value, color=c, linewidth=0.75, linestyle=(0, (3.5, 2.5)), zorder=2)
        if label:
            ax.text(
                label_x, value, f" {label} ",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=TYPE["annot"], color=c, zorder=7,
                bbox=dict(facecolor=PALETTE.canvas, edgecolor="none", pad=0.9),
            )
    else:
        ax.axvline(value, color=c, linewidth=0.75, linestyle=(0, (3.5, 2.5)), zorder=2)
        if label:
            ax.text(
                value, label_x, f" {label} ",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=TYPE["annot"], color=c, zorder=7,
                bbox=dict(facecolor=PALETTE.canvas, edgecolor="none", pad=0.9),
            )


def section_band(
    fig: Figure,
    axes: Sequence[Axes],
    label: str,
    *,
    color: str | None = None,
    pad_pt: float = 6.0,
    above_title: bool = True,
) -> None:
    """Tag a group of panels with a small label and a hairline rule.

    The band clears the panel title block, so it never lands on a title.
    """
    boxes = [a.get_position() for a in axes]
    x0 = min(b.x0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    fig_h_pt = fig.get_size_inches()[1] * 72.0
    offset = (pad_pt + (title_block_height() if above_title else 0.0)) / fig_h_pt
    y = y1 + offset
    c = color or PALETTE.ink_faint
    tag = fig.text(
        x0, y, label,
        ha="left", va="bottom", fontsize=TYPE["annot"],
        fontweight=700, color=c,
    )
    bb = tag.get_window_extent(renderer=_renderer(fig)).transformed(
        fig.transFigure.inverted()
    )
    y_rule = (bb.y0 + bb.y1) / 2.0
    fig.add_artist(
        mpl.lines.Line2D(
            [bb.x1 + 0.009, x1], [y_rule, y_rule],
            transform=fig.transFigure, color=PALETTE.rule,
            linewidth=0.7, zorder=0,
        )
    )


def headroom(ax: Axes, *, top: float = 0.16, bottom: float = 0.0, symmetric: bool = False) -> None:
    """Reserve space above bars for value labels so nothing ever collides."""
    lo, hi = ax.get_ylim()
    rng = hi - lo
    ax.set_ylim(lo - rng * bottom, hi + rng * top)
    if symmetric:
        m = max(abs(lo), abs(hi))
        ax.set_ylim(-m * (1 + top), m * (1 + top))


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


SAVE_DPI = 400   #: audit() measures at this too - see its docstring


def save(
    fig: Figure,
    stem: str | Path,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = SAVE_DPI,
) -> tuple[Path, ...]:
    path = Path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix in formats:
        p = path.with_suffix(f".{suffix}")
        fig.savefig(p, dpi=dpi if suffix == "png" else None)
        out.append(p)
    return tuple(out)


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------


def _renderer(fig: Figure):
    """Attach an Agg canvas if needed and return a usable renderer."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    return fig.canvas.get_renderer()


# Raw checkpoint identifiers ("Qwen/Qwen3-32B", "Mistral-Small-24B-Instruct-2501")
# are repository paths, not names a reader parses on an axis. Every figure that
# labels a model routes through display_model() so the same checkpoint is never
# spelled two ways across the figure set.
MODEL_DISPLAY = {
    "Original Llama baseline": "Original Llama",
    "Original-model mean": "Original-model mean",
    "Original Llama mean": "Original Llama mean",
    "Llama-2 baseline": "Original Llama-2",
    "Llama-3 baseline": "Original Llama-3",
    "Gemma 4 31B IT": "Gemma-4 31B",
    "Qwen/Qwen3-32B": "Qwen3 32B",
    "Qwen2.5-32B-Instruct": "Qwen2.5 32B",
    "GLM-4.7-Flash": "GLM-4.7 Flash",
    "Gemma-4-12B-it": "Gemma-4 12B",
    "Qwen3-14B": "Qwen3 14B",
    "Qwen/Qwen3-8B": "Qwen3 8B",
    "google/gemma-2-27b-it": "Gemma-2 27B",
    "google/gemma-2-9b-it": "Gemma-2 9B",
    "microsoft/phi-4": "Phi-4",
    "zai-org/GLM-4-9B-0414": "GLM-4 9B",
    "GLM-4-9B-Chat": "GLM-4 9B Chat",
    "Llama-3.1-8B-Instruct": "Llama-3.1 8B",
    "Moonlight-16B-A3B-Instruct": "Moonlight 16B",
    "Qwen2.5-14B-Instruct": "Qwen2.5 14B",
    "GLM-4 9B": "GLM-4 9B",
    "Gemma 2 9B": "Gemma-2 9B",
    "Llama 3.1 8B": "Llama-3.1 8B",
    "Phi-4 14B": "Phi-4 14B",
    "Qwen3 8B": "Qwen3 8B",
    "GLM-4-9B-0414": "GLM-4 9B",
    "Mistral-Small-24B-Instruct-2501": "Mistral-Small 24B",
    "Phi-4-14B": "Phi-4 14B",
}


# The generated CSVs spell one study two ways ("Instruction Following" in the
# fidelity table, "Internal Instruction Following" elsewhere). The paper prose
# uses the short form throughout, so that is canonical: a reader must not meet
# two names for one study across the figure set.
STUDY_DISPLAY = {
    "Internal Instruction Following": "Instruction Following",
}


def display_study(name: str) -> str:
    """Reader-facing name for a study, matching the paper's prose."""
    return STUDY_DISPLAY.get(name, name)


def display_model(name: str) -> str:
    """Reader-facing name for a model checkpoint identifier."""
    return MODEL_DISPLAY.get(name, name)


def text_width(text: str, fontsize: float, *, role: str = "annot",
               weight: int = 400) -> float:
    """Width of ``text`` in INCHES if rendered at ``fontsize``.

    Layout decisions that depend on how wide a string actually is - wrapping a
    footnote, placing a note clear of a label - must measure rather than guess,
    because the answer changes with the typeface. Uses a throwaway figure so it
    is safe to call before the real figure exists.
    """
    probe = Figure(figsize=(10.0, 1.0))
    t = probe.text(0.0, 0.0, text, fontsize=fontsize,
                   family=font_for(role), fontweight=weight)
    bb = t.get_window_extent(renderer=_renderer(probe))
    return float(bb.width) / probe.dpi


def tick_bins(axis_w_in: float, sample_label: str, fontsize: float,
              *, role: str = "tick", max_bins: int = 5,
              gap_in: float = 0.10) -> int:
    """How many x-axis ticks fit in ``axis_w_in`` inches without touching.

    A fixed tick count is a bug waiting for a narrower column: the same four
    ticks that read cleanly across a full-width axis collide once the axis is
    half as wide. Ask for the count the width supports instead, measured from
    a representative label.
    """
    w = text_width(sample_label, fontsize, role=role) + gap_in
    if w <= 0:
        return max_bins
    return max(2, min(max_bins, int(axis_w_in // w)))


def text_height(lines, fontsize: float, *, role: str = "annot",
                weight: int = 400, linespacing: float = 1.55) -> float:
    """Height of a wrapped text block in INCHES, as matplotlib will draw it.

    The twin of :func:`text_width`, and needed for the same reason: a block's
    rendered height is NOT ``len(lines) * linespacing * fontsize``, because the
    line height comes from the font's own metrics. Budgeting a footer with that
    arithmetic under-reserves space and the block grows up into the plot.
    Callers pass the SAME ``linespacing`` they will draw with.
    """
    text = lines if isinstance(lines, str) else "\n".join(lines)
    probe = Figure(figsize=(10.0, 10.0))
    t = probe.text(0.0, 0.0, text, fontsize=fontsize, family=font_for(role),
                   fontweight=weight, linespacing=linespacing)
    bb = t.get_window_extent(renderer=_renderer(probe))
    return float(bb.height) / probe.dpi


def check_sources(script_dir: Path | None = None) -> list[str]:
    """Flag figure scripts that hardcode a font family or a raw hex colour.

    The design contract is that families come from :func:`font_for` and colours
    from :data:`PALETTE` / the series maps. A literal in a plotting script is
    how a figure set drifts out of alignment, so it is a build error here
    rather than something a reader has to notice.
    """
    directory = Path(script_dir) if script_dir else Path(__file__).resolve().parent
    problems: list[str] = []
    families = set(AVAILABLE_FONTS)
    hexpat = re.compile(r"#[0-9A-Fa-f]{6}\b")
    for path in sorted(directory.glob("plot_fig_*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            for fam in families:
                if f'"{fam}"' in code or f"'{fam}'" in code:
                    problems.append(f"{path.name}:{n} hardcoded font {fam!r}")
            for m in hexpat.finditer(line):
                if "color" in code or "facecolor" in code:
                    problems.append(f"{path.name}:{n} raw hex {m.group(0)}")
    return problems


def audit(fig: Figure, *, min_pt: float = 7.0, dpi: int = SAVE_DPI) -> list[str]:
    """Return a list of design-contract violations for this figure.

    Checks: (1) no text below the legibility floor; (2) no text-on-text
    collisions; (3) no artist extending outside the canvas.

    Measured at ``dpi`` - the dpi the figure is SAVED at, not the interactive
    default. Glyph advances are hinted per-dpi, so a string that measures just
    inside the canvas at 100 dpi can run off the edge in a 400 dpi render: the
    audit has to look at the figure the reader gets.
    """
    import warnings as _warnings

    was_dpi = fig.dpi
    fig.set_dpi(dpi)
    try:
        return _audit_at_dpi(fig, min_pt)
    finally:
        fig.set_dpi(was_dpi)


def _audit_at_dpi(fig: Figure, min_pt: float) -> list[str]:
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        renderer = _renderer(fig)
    problems: list[str] = []
    seen_glyph = set()
    for w in caught:
        msg = str(w.message)
        if "missing from font" in msg and msg not in seen_glyph:
            seen_glyph.add(msg)
            problems.append(f"missing glyph: {msg}")

    texts = [
        t for t in fig.findobj(mpl.text.Text)
        if t.get_visible() and t.get_text().strip()
    ]
    for t in texts:
        size = t.get_fontsize()
        if size < min_pt - 1e-6:
            problems.append(f"text below {min_pt}pt ({size:.1f}): {t.get_text()[:42]!r}")

    boxes = []
    for t in texts:
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append((t, bb))

    for i in range(len(boxes)):
        ti, bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            tj, bj = boxes[j]
            ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ox > 1.2 and oy > 1.2:
                area = ox * oy
                smaller = min(bi.width * bi.height, bj.width * bj.height)
                if smaller > 0 and area / smaller > 0.16:
                    problems.append(
                        f"text overlap: {ti.get_text()[:26]!r} vs {tj.get_text()[:26]!r}"
                    )

    w, h = fig.canvas.get_width_height()
    for t, bb in boxes:
        if bb.x0 < -1.5 or bb.y0 < -1.5 or bb.x1 > w + 1.5 or bb.y1 > h + 1.5:
            problems.append(f"text outside canvas: {t.get_text()[:34]!r}")

    return problems

def wrap(text, avail_in: float, fontsize: float, *, role: str = "annot",
         weight: int = 400, sep: str = "  ") -> list[str]:
    """Pack ``text`` into lines no wider than ``avail_in`` INCHES.

    ``text`` may be one string or a sequence of them; a sequence is packed
    greedily, so two short notes share a line and a long one takes its own.
    Words are split only when a single item exceeds the width.

    Prose in a figure - the deck under the title, the footnote under the axes -
    is the one element whose length is authored rather than derived from data,
    so it is the element that breaks when the canvas narrows. Wrapping here,
    against measured width, means a figure ported to a different text width
    reflows instead of running off the page. Callers must derive their height
    budget from ``len()`` of this result rather than a hardcoded line count.
    """
    items = [text] if isinstance(text, str) else list(text)
    lines: list[str] = []
    cur = ""

    def fits(t: str) -> bool:
        return text_width(t, fontsize, role=role, weight=weight) <= avail_in

    for item in items:
        trial = item if not cur else cur + sep + item
        if fits(trial):
            cur = trial
            continue
        if cur:
            lines.append(cur)
            cur = ""
        if fits(item):
            cur = item
            continue
        # single item too wide: break it on words
        for word in item.split():
            trial = word if not cur else cur + " " + word
            if fits(trial):
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = word
    if cur:
        lines.append(cur)
    return lines


def finish(fig, out, *, extra=None):
    """Audit, save, and report one figure - the single exit path for every script.

    Each figure script used to print its own verdict, in its own shape; one of
    them printed no verdict at all, so a silent run looked like a pass without
    being one. Routing every script through here means a figure cannot be
    written without its audit result appearing next to it, and a non-zero exit
    makes a dirty figure fail a build rather than scroll past.

    Returns the number of problems found (0 when clean).
    """
    problems = audit(fig)
    w, h = fig.get_size_inches()
    # A figure wider than the text block is scaled DOWN by \linewidth, shrinking
    # every declared point size with it; one taller than the text block cannot
    # share a page with prose. Both are page-layout faults, invisible in the PNG.
    if w > TEXT_W + 0.01:
        problems.append("width %.2f in exceeds \\textwidth %.2f in - will be "
                        "scaled down, shrinking all type" % (w, TEXT_W))
    if h > MAX_FIG_H:
        problems.append("height %.2f in exceeds %.2f in - will own a full page"
                        % (h, MAX_FIG_H))
    for p in problems:
        print("AUDIT:", p)
    for p in check_sources():
        print("SOURCE:", p)
    paths = list(save(fig, out))
    if extra:
        paths += list(extra)
    print("%-34s %s  %.2f x %.2f in" % (
        Path(out).stem,
        "CLEAN" if not problems else "%d PROBLEM(S)" % len(problems),
        *fig.get_size_inches()))
    for q in paths:
        print("   ", Path(q).name)
    return len(problems)
