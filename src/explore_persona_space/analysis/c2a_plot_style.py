"""Visual system for the context-to-answer-map paper.

This module owns the style choices established by Figure 2.  Paper plotting
scripts should import these constants and helpers instead of copying rcParams,
colors, or axis styling.  The oversized authoring canvas is intentional: the
figures are included at 5.5 inches in the ICLR manuscript, which realizes the
font sizes documented in ``docs/paper_context_answer_map/plotting_style.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from PIL import Image

STYLE_VERSION = "c2a-v1"
FONT_CANDIDATES = ("Inter", "Noto Sans", "DejaVu Sans")

INK = "#22272B"
MUTED = "#687078"
GRID = "#C8C6BF"
PAPER = "#FFFFFF"
SEAM = "#A9A69E"


@dataclass(frozen=True)
class SeriesStyle:
    """Redundant color-and-marker encoding for one conceptual series."""

    label: str
    color: str
    marker: str


PREDICTOR_STYLES = {
    "ridge": SeriesStyle("Linear", "#176B87", "o"),
    "mlp_w8192": SeriesStyle("Nonlinear", "#C4553D", "D"),
}


def resolved_font() -> str:
    """Return the first installed font in the paper's pinned fallback chain."""

    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in FONT_CANDIDATES:
        if candidate in installed:
            return candidate
    # Matplotlib ships DejaVu Sans, so this is only reachable in a broken install.
    raise RuntimeError(f"none of the paper fonts are installed: {FONT_CANDIDATES}")


def set_c2a_style() -> str:
    """Apply Figure-2-compatible rcParams and return the resolved font name."""

    font = resolved_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font],
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "axes.linewidth": 1.2,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
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


def style_score_axis(
    ax: plt.Axes,
    *,
    y_min: float = 0.5,
    y_max: float = 1.01,
    y_step: float = 0.1,
) -> None:
    """Apply the shared minimal axis treatment used by Figure 2."""

    if not y_min < y_max:
        raise ValueError(f"expected y_min < y_max, got {y_min} >= {y_max}")
    if y_step <= 0:
        raise ValueError(f"expected y_step > 0, got {y_step}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SEAM)
    ax.spines["bottom"].set_color(SEAM)
    ax.tick_params(length=0, pad=8)
    ax.set_ylim(y_min, y_max)

    ticks = []
    value = y_min
    while value <= y_max + 1e-9:
        ticks.append(round(value, 10))
        value += y_step
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:.2f}".rstrip("0").rstrip(".")))
    ax.grid(axis="y", color=GRID, lw=1.0, alpha=0.55)
    ax.set_axisbelow(True)


def save_c2a_figure(
    fig: plt.Figure,
    stem: Path,
    *,
    title: str,
    subject: str,
    creator: str,
    png_dpi: int = 240,
) -> dict[str, Path]:
    """Save vector, color-raster, and grayscale-audit versions of a figure."""

    if png_dpi < 1:
        raise ValueError(f"png_dpi must be positive, got {png_dpi}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    grayscale = stem.with_name(f"{stem.name}_grayscale.png")

    fig.savefig(
        pdf,
        facecolor=PAPER,
        bbox_inches="tight",
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
    fig.savefig(png, dpi=png_dpi, facecolor=PAPER, bbox_inches="tight")
    with Image.open(png) as image:
        image.convert("L").save(grayscale)
    return {"pdf": pdf, "png": png, "grayscale": grayscale}
