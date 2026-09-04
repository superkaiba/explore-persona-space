"""Visual system for the context-to-answer-map paper (``c2a-v2``).

This module owns every style choice a paper figure makes: fonts (text and
math), the semantic palette, axis treatment, panel furniture, the fixed
authoring scale, and export.  Paper plotting scripts import these constants
and helpers instead of copying rcParams, colors, or axis styling, so a later
global change propagates across the paper from one place.

The fixed scale is the load rule of the system: every figure is authored on a
canvas whose width is ``include_fraction * TEXT_WIDTH_IN / C2A_SCALE`` inches
and included in the manuscript at ``include_fraction * \\textwidth``, so the
same script point sizes realize the same printed sizes in every figure.  With
the pinned rcParams this is 7.6 pt body, 7.1 pt ticks, 8.4 pt axis labels, and
9.2 pt panel titles.  ``save_c2a_figure`` refuses a canvas that is off-scale.

Spec: ``docs/paper_context_answer_map/figure_standard.md`` (sections 2 and 5)
and ``docs/paper_context_answer_map/plotting_style.md``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from PIL import Image

STYLE_VERSION = "c2a-v2"
FONT_CANDIDATES = ("Inter", "Noto Sans", "DejaVu Sans")

INK = "#22272B"
MUTED = "#687078"
GRID = "#C8C6BF"
PAPER = "#FFFFFF"
SEAM = "#A9A69E"

# ---------------------------------------------------------------------------
# Fixed authoring scale
# ---------------------------------------------------------------------------

TEXT_WIDTH_IN = 5.5
"""ICLR ``\\textwidth`` in inches."""

C2A_SCALE = 0.42
"""Realized printed size divided by script point size, identical for every figure."""

INCLUDE_WIDTHS: dict[str, float] = {"full": 1.0, "wide": 0.75, "half": 0.5}
"""The only ``\\includegraphics`` width fractions a paper figure may use."""

BASE_FONT_PT: dict[str, float] = {"body": 18, "tick": 17, "label": 20, "title": 22}
"""Script-side point sizes pinned by :func:`set_c2a_style`."""

_SCALE_TOLERANCE = 0.02


def realized_font_pt() -> dict[str, float]:
    """Printed point sizes every figure realizes under the fixed scale."""

    return {name: round(size * C2A_SCALE, 2) for name, size in BASE_FONT_PT.items()}


def canvas_width_in(include_fraction: float) -> float:
    """Authoring canvas width (inches) for a given include fraction."""

    if include_fraction <= 0:
        raise ValueError(f"include_fraction must be positive, got {include_fraction}")
    return include_fraction * TEXT_WIDTH_IN / C2A_SCALE


def c2a_figure(
    width: str = "full",
    aspect: float = 0.43,
    *,
    constrained_layout: bool = False,
) -> tuple[plt.Figure, float]:
    """Create a canvas that realizes :data:`C2A_SCALE` at the given include width.

    ``width`` is one of :data:`INCLUDE_WIDTHS`; ``aspect`` is height over width.
    Returns the figure and the include fraction to pass to
    :func:`save_c2a_figure` and to write into the LaTeX ``\\includegraphics``.
    """

    if width not in INCLUDE_WIDTHS:
        raise ValueError(f"width must be one of {sorted(INCLUDE_WIDTHS)}, got {width!r}")
    if aspect <= 0:
        raise ValueError(f"aspect must be positive, got {aspect}")
    frac = INCLUDE_WIDTHS[width]
    w_in = canvas_width_in(frac)
    fig = plt.figure(figsize=(w_in, w_in * aspect), constrained_layout=constrained_layout)
    return fig, frac


def include_width_for(fig: plt.Figure) -> float:
    """Infer the include fraction a canvas was authored for, or raise if off-scale."""

    frac = fig.get_figwidth() * C2A_SCALE / TEXT_WIDTH_IN
    for allowed in INCLUDE_WIDTHS.values():
        if abs(frac - allowed) <= _SCALE_TOLERANCE:
            return allowed
    allowed_widths = {k: round(canvas_width_in(v), 2) for k, v in INCLUDE_WIDTHS.items()}
    raise ValueError(
        f"canvas width {fig.get_figwidth():.2f} in realizes include fraction {frac:.3f}, "
        f"which is not one of {INCLUDE_WIDTHS}; author at one of {allowed_widths} inches "
        "(use c2a_figure) so the printed type size matches every other paper figure"
    )


def latex_include_line(
    stem_name: str, include_fraction: float, subdir: str = "figures/paper"
) -> str:
    """The exact ``\\includegraphics`` line the manuscript must carry for this figure."""

    width = (
        "\\textwidth" if abs(include_fraction - 1.0) < 1e-9 else f"{include_fraction:g}\\textwidth"
    )
    return f"\\includegraphics[width={width}]{{{subdir}/{stem_name}.pdf}}"


# ---------------------------------------------------------------------------
# Semantic palette: one color, one meaning, paper-wide
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesStyle:
    """Redundant color-and-marker encoding for one conceptual series."""

    label: str
    color: str
    marker: str


ROLES: dict[str, SeriesStyle] = {
    "linear": SeriesStyle("Linear", "#176B87", "o"),
    "nonlinear": SeriesStyle("Nonlinear", "#C4553D", "D"),
    "control": SeriesStyle("Control", MUTED, "x"),
    "base_model": SeriesStyle("Base", "#C98A1B", "s"),
    "post_trained": SeriesStyle("Post-trained", "#176B87", "o"),
    "other_source": SeriesStyle("Other model", "#C98A1B", "^"),
    "needs_reasoning": SeriesStyle("Needs-reasoning corpora", "#7B3294", "s"),
    "no_reasoning": SeriesStyle("No-reasoning corpora", "#5AAE61", "o"),
}
"""Role -> style.  ``control`` covers shuffled, identity+bias, random-direction,
and every other null.  ``post_trained`` shares the ``linear`` hue on purpose: it
is the map on the model the paper studies.  ``other_source`` (answers written
by another model or persona) is amber so purple stays reserved for reasoning
demand."""

PREDICTOR_STYLES = {
    "ridge": ROLES["linear"],
    "mlp_w8192": ROLES["nonlinear"],
}
"""Backward-compatible alias used by the Figure 3 script."""

METRIC_LABELS: dict[str, str] = {"r2": "Held-out $R^2$", "top1": "Top-1 retrieval"}
"""The only two metric strings a paper figure may render (decision D2, 2026-09-03)."""


def metric_style(metric: str) -> dict[str, Any]:
    """Fill-based encoding of the metric: R^2 solid/filled, top-1 dashed/open/hatched."""

    if metric == "r2":
        return {"linestyle": "-", "fillstyle": "full", "hatch": None}
    if metric == "top1":
        return {"linestyle": "--", "fillstyle": "none", "hatch": "////"}
    raise ValueError(f"metric must be 'r2' or 'top1', got {metric!r}")


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------


def resolved_font() -> str:
    """Return the first installed font in the paper's pinned fallback chain."""

    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in FONT_CANDIDATES:
        if candidate in installed:
            return candidate
    # Matplotlib ships DejaVu Sans, so this is only reachable in a broken install.
    raise RuntimeError(f"none of the paper fonts are installed: {FONT_CANDIDATES}")


def set_c2a_style() -> str:
    """Apply the paper rcParams (text and math in one font) and return the font name."""

    font = resolved_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font],
            "font.size": BASE_FONT_PT["body"],
            "axes.labelsize": BASE_FONT_PT["label"],
            "axes.titlesize": BASE_FONT_PT["title"],
            "xtick.labelsize": BASE_FONT_PT["tick"],
            "ytick.labelsize": BASE_FONT_PT["tick"],
            "legend.fontsize": BASE_FONT_PT["tick"],
            "mathtext.fontset": "custom",
            "mathtext.rm": font,
            "mathtext.it": f"{font}:italic",
            "mathtext.bf": f"{font}:bold",
            "mathtext.default": "it",
            "axes.linewidth": 1.2,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    return font


# ---------------------------------------------------------------------------
# Axis treatment and panel furniture
# ---------------------------------------------------------------------------


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    """Shared minimal axis treatment: two seams, no ticks, one grid direction."""

    if grid_axis not in {"x", "y", "both", "none"}:
        raise ValueError(f"grid_axis must be x, y, both, or none, got {grid_axis!r}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SEAM)
    ax.spines["bottom"].set_color(SEAM)
    ax.tick_params(length=0, pad=8)
    ax.grid(False)
    if grid_axis != "none":
        ax.grid(axis=grid_axis, color=GRID, lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)


def style_score_axis(
    ax: plt.Axes,
    *,
    y_min: float = 0.5,
    y_max: float = 1.01,
    y_step: float = 0.1,
) -> None:
    """Score axis (R^2 / top-1) with a fixed range and trimmed tick labels."""

    if not y_min < y_max:
        raise ValueError(f"expected y_min < y_max, got {y_min} >= {y_max}")
    if y_step <= 0:
        raise ValueError(f"expected y_step > 0, got {y_step}")
    style_axis(ax, grid_axis="y")
    ax.set_ylim(y_min, y_max)

    ticks = []
    value = y_min
    while value <= y_max + 1e-9:
        ticks.append(round(value, 10))
        value += y_step
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:.2f}".rstrip("0").rstrip(".")))


_KICKER_GID = "c2a-kicker"
_TITLE_GID = "c2a-title"


def panel_header(
    ax: plt.Axes,
    letter: str,
    kicker: str,
    title: str | None = None,
    *,
    kicker_y: float = 1.16,
    title_y: float = 1.055,
) -> None:
    """Panel letter plus uppercase kicker, and an optional DESCRIPTIVE title.

    The kicker reads ``A  ·  LAYER 19``.  ``title`` states what is plotted
    ("Predictability across layers"); it never carries a claim (decision D1,
    2026-09-03).  ``letter`` is a single uppercase letter, or ``""`` for an
    unlettered facet.
    """

    if letter and not (len(letter) == 1 and letter.isupper()):
        raise ValueError(f"panel letter must be one uppercase letter or empty, got {letter!r}")
    text = f"{letter}  ·  {kicker}" if letter else kicker
    kicker_artist = ax.text(
        0.0,
        kicker_y,
        text.upper(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight=700,
        color=MUTED,
    )
    kicker_artist.set_gid(_KICKER_GID)
    if title:
        ax.set_title(title, loc="left", y=title_y, pad=0, fontweight=650)
        ax._left_title.set_gid(_TITLE_GID)


def legend_kicker(fig: plt.Figure, x: float, y: float, heading: str) -> None:
    """Uppercase heading placed above a frameless figure-level legend."""

    fig.text(
        x, y, heading.upper(), color=MUTED, fontsize=11.5, fontweight=750, ha="left", va="center"
    )


def better_label(label: str, *, higher_is_better: bool = True) -> str:
    """Axis label with the direction arrow the paper uses when larger is better."""

    return f"{label} \u2191" if higher_is_better else f"{label} \u2193"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


_EXPORT_OVERFLOW_TOLERANCE = 0.05


def _spilling_artists(fig: plt.Figure, renderer) -> list[str]:
    """Names of text/legend artists whose drawn extent crosses the canvas edge (inches)."""

    width_in = fig.get_figwidth()
    dpi = fig.dpi
    out: list[str] = []
    for artist in list(fig.findobj(Text)) + list(fig.findobj(Legend)):
        try:
            bb = artist.get_window_extent(renderer)
        except Exception:
            continue
        if bb.width == 0 and bb.height == 0:
            continue
        x0, x1 = bb.x0 / dpi, bb.x1 / dpi
        if x0 < -0.03 or x1 > width_in + 0.03:
            label = artist.get_text() if isinstance(artist, Text) else "legend"
            out.append(f"{label[:40]!r}@[{x0:.2f},{x1:.2f}]in")
    return out


def _export_bbox(fig: plt.Figure):
    """Tight export box padded horizontally to the authored canvas width.

    ``bbox_inches="tight"`` crops the margins, which silently rescales the
    printed type: a crop 5% narrower than the canvas renders 5% larger at the
    fixed include width.  This keeps the tight vertical crop but pads the box
    out to the authored width (centered), so exported width == authored width
    and the realized point sizes are exactly ``BASE_FONT_PT x C2A_SCALE``.
    Artists that spill past the canvas by more than 5% raise instead of
    shrinking the figure.
    """

    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    width_in = fig.get_figwidth()
    spill = _spilling_artists(fig, renderer)
    if tight.width > width_in * (1 + _EXPORT_OVERFLOW_TOLERANCE):
        raise ValueError(
            f"figure content is {tight.width:.2f} in wide but the canvas is {width_in:.2f} in; "
            "move legends/labels inside the canvas so the printed scale stays fixed. "
            f"Artists past the canvas edge: {spill}"
        )
    if spill:
        print(
            f"[c2a] WARNING: content {tight.width:.2f} in on a {width_in:.2f} in canvas "
            f"(tolerated); artists past the edge: {spill}",
            file=sys.stderr,
        )
    pad = max(0.0, (width_in - tight.width) / 2.0)
    return Bbox([[tight.x0 - pad, tight.y0 - 0.02], [tight.x1 + pad, tight.y1 + 0.02]])


def rendered_text(fig: plt.Figure) -> list[str]:
    """Every non-empty string drawn on the figure, for the sidecar and the slug scan."""

    seen: list[str] = []
    for artist in fig.findobj(Text):
        s = artist.get_text().strip()
        if s and s not in seen:
            seen.append(s)
    return seen


def save_c2a_figure(
    fig: plt.Figure,
    stem: Path,
    *,
    title: str,
    subject: str,
    creator: str,
    png_dpi: int = 240,
    include_width: float | None = None,
    enforce_scale: bool = True,
) -> dict[str, Any]:
    """Save vector, color-raster, and grayscale-audit versions of a figure.

    Returns the three output paths under ``pdf`` / ``png`` / ``grayscale`` plus a
    ``record`` dict (style version, font, canvas size, include fraction, the
    LaTeX include line, realized point sizes, every rendered string) for the
    caller to embed in its provenance sidecar.  With ``enforce_scale`` the canvas
    must realize :data:`C2A_SCALE` at ``include_width`` (inferred when omitted).
    """

    if png_dpi < 1:
        raise ValueError(f"png_dpi must be positive, got {png_dpi}")
    if enforce_scale:
        inferred = include_width_for(fig)
        if include_width is not None and abs(include_width - inferred) > _SCALE_TOLERANCE:
            raise ValueError(
                f"include_width={include_width} does not match the canvas, which realizes "
                f"{inferred} at C2A_SCALE={C2A_SCALE}"
            )
        include_width = inferred
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    grayscale = stem.with_name(f"{stem.name}_grayscale.png")

    export_bbox = _export_bbox(fig)
    fig.savefig(
        pdf,
        facecolor=PAPER,
        bbox_inches=export_bbox,
        metadata={
            "Title": title,
            "Subject": subject,
            "Creator": creator,
            # Matplotlib otherwise inserts the current time, making identical
            # inputs produce different PDF hashes on every render.
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(png, dpi=png_dpi, facecolor=PAPER, bbox_inches=export_bbox)
    with Image.open(png) as image:
        image.convert("L").save(grayscale)

    record: dict[str, Any] = {
        "style_version": STYLE_VERSION,
        "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
        "resolved_font": mpl.rcParams["font.sans-serif"][0],
        "authoring_size_inches": [round(fig.get_figwidth(), 3), round(fig.get_figheight(), 3)],
        "exported_size_inches": [round(export_bbox.width, 3), round(export_bbox.height, 3)],
        "scale": C2A_SCALE,
        "include_width_frac": include_width,
        "latex_include_line": (
            latex_include_line(stem.name, include_width) if include_width is not None else None
        ),
        "realized_font_pt": realized_font_pt(),
        "png_dpi": png_dpi,
        "background": PAPER,
        "text": rendered_text(fig),
    }
    return {"pdf": pdf, "png": png, "grayscale": grayscale, "record": record}
