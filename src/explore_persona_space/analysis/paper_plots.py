"""Paper-quality plotting utilities.

This module centralises the rcParams, palette, save-format, and reproducibility
metadata conventions for every figure that ships to a clean-result issue, the
research log, or the paper.

Design rationale
----------------
* **Paper-ready rcParams** — NeurIPS single-column-friendly sizing, DejaVu Sans
  (cross-platform, no LaTeX required), Type-42 fonts so figures remain editable
  in Illustrator / Inkscape for camera-ready. Grid at low alpha, top/right
  spines removed (despine), colorblind-safe prop cycle.
* **Colorblind palette** — the 8-colour Wong 2011 / IBM scheme, widely cited as
  safe for the two most common colour-vision deficiencies (deuteranopia,
  protanopia). Limit yourself to ≤ 3-5 colours per chart (see
  `.claude/skills/clean-results/SPEC.md` §14).
* **Commit-pinned metadata** — every saved figure carries the git commit hash
  embedded in PDF metadata / PNG pnginfo plus a sidecar `<stem>.meta.json` so a
  reader can always trace a figure back to the code that produced it.
* **Per-point data in the sidecar** — `savefig_paper` reads the plotted data
  back off the matplotlib artists (scatter offsets + nearest text labels, line
  vertices, bar heights, error-bar magnitudes) and embeds it under a `points`
  key in the `.meta.json`, in the shape the EPS dashboard data viewer
  (`dashboard/lib/task-data.ts`) reads directly. Every new figure thus
  auto-populates the per-figure sortable/filterable table with no per-call-site
  change. Pass `embed_data=False` to opt out.

Public API
----------
    set_paper_style        — set rcParams for paper-quality figures
    savefig_paper          — save to <dir>/<stem>.png AND <dir>/<stem>.pdf plus .meta.json
    add_direction_arrow    — append ↑ better / ↓ better to an axis label
    paper_palette          — return N colorblind-safe hex colours (Wong 2011; N > 8
                             extends via perceptually-uniform colormap sampling)
    paper_palette_blog     — return N colours from the soft "blog" palette (N > 8
                             extends via perceptually-uniform colormap sampling)
    paper_palette_role     — look up a colour by semantic role (primary / baseline / ...)
    set_title_subtitle     — left-aligned title + subtitle (Anthropic-blog register)
    proportion_ci          — 95% Wald CI for a proportion

Exemplar usage
--------------
The default target is ``"blog"`` (Anthropic / Apollo / LessWrong-blog
register). Use it for clean-result issue figures and mentor slides:

>>> from explore_persona_space.analysis.paper_plots import (
...     set_paper_style, set_title_subtitle, paper_palette_role, savefig_paper,
... )
>>> set_paper_style()  # equivalent to set_paper_style("blog")
>>> fig, ax = plt.subplots()
>>> colors = [paper_palette_role("primary"), paper_palette_role("baseline")]
>>> ax.bar([0, 1], [0.7, 0.4], color=colors)
>>> set_title_subtitle(
...     ax,
...     "Marker uptake holds across personas",
...     subtitle="Source-rate per persona, n=200 each",
... )
>>> savefig_paper(fig, "issue_281/marker_uptake", dir="figures/")

For paper figures (NeurIPS / ICML / ICLR — narrow column, dense, camera-ready):

>>> set_paper_style("neurips")
>>> # ... build your figure ...
>>> savefig_paper(fig, "em_defense/pre_post_alignment", dir="figures/")
"""

from __future__ import annotations

import json
import subprocess
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import colors as mcolors
from matplotlib import font_manager

# Preferred font fallback chain for the "blog" style. matplotlib walks the
# list and uses the first installed font; we filter to installed entries
# at `set_paper_style("blog")` time to avoid per-text findfont warnings.
_BLOG_FONT_CANDIDATES: tuple[str, ...] = (
    "Inter",
    "Source Sans 3",
    "Source Sans Pro",
    "Helvetica Neue",
    "Arial",
    "DejaVu Sans",
)


def _resolve_blog_fonts() -> list[str]:
    """Return the installed subset of the blog font candidates.

    Always keeps ``DejaVu Sans`` (ships with matplotlib) at the tail so the
    list is never empty and matplotlib never has to fall back through a
    missing-font search at draw time.
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    out = [name for name in _BLOG_FONT_CANDIDATES if name in installed]
    if "DejaVu Sans" not in out:
        out.append("DejaVu Sans")
    return out


# Wong 2011 / IBM colorblind-safe palette. Order chosen so the first three
# (blue / orange / green) give the widest contrast and remain distinguishable
# under deuteranopia and protanopia.
_PALETTE: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

# Soft-warm "blog" palette — Anthropic / Apollo / LessWrong-blog register.
# Hand-tuned to stay colorblind-safe (verified against deuteranopia +
# protanopia simulators) while reading warmer and more polished than the
# Wong palette. Use with `set_paper_style("blog")`.
_BLOG_PALETTE: list[str] = [
    "#1F4E9F",  # primary    — deep blue (slightly warmer than Wong's #0072B2)
    "#E08220",  # baseline   — warm orange
    "#3FA577",  # control    — forest green
    "#C0413B",  # accent     — warm red
    "#8064A2",  # purple
    "#5A6975",  # slate (neutral)
    "#E0B834",  # yellow (fill only — never as a line colour)
    "#000000",  # black (reference / ground truth)
]

# Fallback colormap for n > 8: perceptually uniform + colorblind-robust
# (van der Walt & Smith, SciPy 2015). Sampled on interior points per the
# seaborn ``mpl_palette`` convention so the near-black start and pale-yellow
# end are skipped.
_EXTENSION_COLORMAP = "viridis"


def _extended_palette(base: list[str], n: int, fn_name: str) -> list[str]:
    """Return ``n`` hex colours: the full curated ``base`` then colormap-sampled extras.

    Colours beyond ``len(base)`` are sampled deterministically from the
    interior of ``_EXTENSION_COLORMAP`` (``(i + 1) / (k + 1)`` for
    ``i in range(k)``, ``k = n - len(base)``) and are less distinguishable
    than the curated set — a one-time ``UserWarning`` says so. The tail is
    a pure function of ``n`` but NOT stable across n: growing a figure from
    9 to 10 series recolours its entire >8 tail (only the curated first 8
    are stable). At large n the 256-entry colormap quantises and hex
    duplicates can appear (measured first at n=144; non-monotone in n).
    """
    k = n - len(base)
    cmap = mpl.colormaps[_EXTENSION_COLORMAP]
    extra = [mcolors.to_hex(cmap((i + 1) / (k + 1))) for i in range(k)]
    warnings.warn(
        f"{fn_name}({n}) exceeds the {len(base)}-colour curated palette; colours "
        f"{len(base) + 1}..{n} are sampled from {_EXTENSION_COLORMAP!r} and are less "
        "distinguishable — prefer <= 8 series per chart",
        UserWarning,
        stacklevel=3,
    )
    return list(base) + extra


# Named-role aliases. Plot code reads `paper_palette_role("primary")` instead
# of indexing into the palette by integer slot, so semantic intent travels
# with the chart code. Roles map to per-style palettes — `"primary"` stays
# semantically primary whether the active style is `"blog"` or `"neurips"`.
_ROLE_ALIASES: dict[str, dict[str, str]] = {
    "neurips": {
        "primary": _PALETTE[0],  # Wong blue
        "baseline": _PALETTE[1],  # Wong orange
        "control": _PALETTE[2],  # Wong bluish green
        "accent": _PALETTE[5],  # Wong vermillion
        "neutral": "#5A5A5A",  # neutral grey for reference lines
    },
    "generic": {
        "primary": _PALETTE[0],
        "baseline": _PALETTE[1],
        "control": _PALETTE[2],
        "accent": _PALETTE[5],
        "neutral": "#5A5A5A",
    },
    "blog": {
        "primary": _BLOG_PALETTE[0],
        "baseline": _BLOG_PALETTE[1],
        "control": _BLOG_PALETTE[2],
        "accent": _BLOG_PALETTE[3],
        "neutral": _BLOG_PALETTE[5],
    },
}

# Active style — set by `set_paper_style`. Used by `paper_palette_role` to
# return colours from the right palette without forcing the caller to pass
# the style name on every lookup.
_ACTIVE_STYLE: str = "neurips"


def paper_palette(n: int) -> list[str]:
    """Return ``n`` colorblind-safe hex colours.

    ``n <= 8`` returns the first ``n`` colours of the curated Wong 2011
    palette (unchanged, byte-identical to the historical contract).
    ``n > 8`` degrades gracefully: the full 8-colour curated palette
    followed by ``n - 8`` colours sampled from a perceptually-uniform
    colormap (see :func:`_extended_palette`), with a ``UserWarning``.

    Raises
    ------
    ValueError
        If ``n`` is not a positive int.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")
    if n > len(_PALETTE):
        return _extended_palette(_PALETTE, n, "paper_palette")
    return list(_PALETTE[:n])


def paper_palette_blog(n: int) -> list[str]:
    """Return ``n`` colours from the soft "blog" palette.

    Companion to :func:`paper_palette` for the Anthropic-blog-register style.
    ``n <= 8`` returns the first ``n`` curated blog colours (unchanged);
    ``n > 8`` degrades gracefully via the same perceptually-uniform colormap
    extension (see :func:`_extended_palette`), with a ``UserWarning``.

    Raises
    ------
    ValueError
        If ``n`` is not a positive int.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")
    if n > len(_BLOG_PALETTE):
        return _extended_palette(_BLOG_PALETTE, n, "paper_palette_blog")
    return list(_BLOG_PALETTE[:n])


def paper_palette_role(role: str) -> str:
    """Return the colour assigned to ``role`` for the active style.

    Roles: ``"primary"``, ``"baseline"``, ``"control"``, ``"accent"``,
    ``"neutral"``. Reads the style most recently set via :func:`set_paper_style`
    so the same call produces the right palette for `"neurips"` vs `"blog"`.

    Raises
    ------
    ValueError
        If ``role`` is not a known role.
    """
    aliases = _ROLE_ALIASES[_ACTIVE_STYLE]
    if role not in aliases:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(aliases)}")
    return aliases[role]


def set_paper_style(
    target: Literal["neurips", "generic", "blog"] = "blog",
    font_scale: float = 1.0,
) -> None:
    """Configure ``matplotlib`` rcParams for paper-quality figures.

    Idempotent: calling twice produces the same state as calling once.

    Parameters
    ----------
    target
        ``"neurips"`` — ~single-column NeurIPS sizing (5.5 x 3.4 in), Wong
        palette, dense paper-camera-ready aesthetic.
        ``"generic"`` — slightly larger default (6.0 x 4.0 in), otherwise
        identical to ``"neurips"``.
        ``"blog"`` — Anthropic / Apollo / LessWrong-blog register: soft
        sans-serif (Inter with fallbacks), wider canvas (6.5 x 4.0 in),
        off-white outer with white plotting area, light y-grid only,
        frameless legend, semibold left-aligned titles. Use for clean-result
        issues and mentor slides.
    font_scale
        Multiplier applied to every font size. ``1.0`` leaves the defaults.
        Use ``1.2`` for talk-slide variants of the same figure.
    """
    global _ACTIVE_STYLE

    if target not in ("neurips", "generic", "blog"):
        raise ValueError(f"target must be 'neurips', 'generic', or 'blog', got {target!r}")

    if target == "blog":
        _set_blog_style(font_scale)
    else:
        _set_neurips_style(target, font_scale)

    _ACTIVE_STYLE = target


def _set_neurips_style(target: Literal["neurips", "generic"], font_scale: float) -> None:
    """Apply the original paper-camera-ready rcParams (NeurIPS / generic)."""
    figsize = (5.5, 3.4) if target == "neurips" else (6.0, 4.0)

    base_font = 10.0 * font_scale
    label_font = 11.0 * font_scale
    title_font = 11.0 * font_scale
    tick_font = 9.0 * font_scale
    legend_font = 9.0 * font_scale

    mpl.rcParams.update(
        {
            # Fonts
            "font.family": "DejaVu Sans",
            "font.size": base_font,
            "axes.labelsize": label_font,
            "axes.titlesize": title_font,
            "xtick.labelsize": tick_font,
            "ytick.labelsize": tick_font,
            "legend.fontsize": legend_font,
            # Figure / save DPI
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.figsize": figsize,
            # Despine (hide top + right spines)
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            # Lines / markers
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # Error bars
            "errorbar.capsize": 3,
            # Legend
            "legend.frameon": True,
            "legend.edgecolor": "lightgrey",
            "legend.facecolor": "white",
            # Type-42 fonts so PDF/PS remain editable in Illustrator / Inkscape
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # Colorblind-safe default prop cycle
            "axes.prop_cycle": cycler(color=_PALETTE),
        }
    )


def _set_blog_style(font_scale: float) -> None:
    """Apply the Anthropic-blog-register rcParams.

    Differences from `_set_neurips_style` are commented inline; everything
    else is intentionally aligned.
    """
    base_font = 11.0 * font_scale
    label_font = 12.0 * font_scale
    title_font = 13.0 * font_scale
    tick_font = 10.0 * font_scale
    legend_font = 10.0 * font_scale

    mpl.rcParams.update(
        {
            # Fonts — Inter primary with graceful fallback. We filter the
            # candidate list to installed fonts only (DejaVu Sans always
            # included as last resort) so matplotlib never logs a per-text
            # findfont warning. Install Inter on the dev VM / pods to get
            # the intended look; otherwise figures degrade gracefully.
            "font.family": _resolve_blog_fonts(),
            "font.size": base_font,
            "axes.labelsize": label_font,
            "axes.titlesize": title_font,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "xtick.labelsize": tick_font,
            "ytick.labelsize": tick_font,
            "legend.fontsize": legend_font,
            # Figure / save DPI — wider canvas, fits ~900px GitHub-inline well.
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.figsize": (6.5, 4.0),
            # Background — off-white outer with a clean white plotting area.
            "figure.facecolor": "#FAFAFA",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FAFAFA",
            # Spines — keep top/right off; soften left/bottom to a light grey.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#B0B0B0",
            "axes.linewidth": 0.5,
            # Grid — y-axis only, very light. Anchors the reader to values
            # without competing with the data. `axisbelow=True` forces the
            # grid behind all artists (matplotlib's default `"line"` lets it
            # show through patches like bars).
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": "#EEEEEE",
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            # Ticks — outward, short, soft grey, no top/right.
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.color": "#999999",
            "ytick.color": "#999999",
            "xtick.labelcolor": "#1A1A1A",
            "ytick.labelcolor": "#1A1A1A",
            "xtick.top": False,
            "ytick.right": False,
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
            # Lines / markers — round caps for a softer feel; clear marker
            # edges so points don't get a halo at small sizes.
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "lines.markeredgewidth": 0,
            "lines.solid_capstyle": "round",
            # Error bars — thinner caps for refinement.
            "errorbar.capsize": 2,
            # Bar patches — no edge, slightly translucent for layered fills.
            "patch.edgecolor": "none",
            "patch.linewidth": 0.0,
            # Legend — frameless (Anthropic-blog convention).
            "legend.frameon": False,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "none",
            "legend.borderpad": 0.4,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.6,
            # Layout — constrained_layout reserves space for titles, subtitles,
            # legends, and annotations automatically. Replaces the historical
            # tight_layout dance.
            "figure.constrained_layout.use": True,
            # Typography polish — true unicode minus, anti-aliased text.
            "axes.unicode_minus": True,
            "text.antialiased": True,
            "path.simplify": True,
            # Type-42 fonts so PDF/PS remain editable in Illustrator / Inkscape
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # Soft-warm prop cycle — see `_BLOG_PALETTE` for hex sources.
            "axes.prop_cycle": cycler(color=_BLOG_PALETTE),
        }
    )


def set_title_subtitle(
    ax: plt.Axes,
    title: str,
    subtitle: str | None = None,
    *,
    source: str | None = None,
    title_color: str = "#1A1A1A",
    subtitle_color: str = "#5A5A5A",
    source_color: str = "#7A7A7A",
) -> None:
    """Set a left-aligned title + subtitle above ``ax`` (Anthropic-blog style).

    Replaces any existing axes title. Title is bold/semibold (using the
    active ``axes.titleweight``); subtitle is regular weight, one point
    smaller than the body font, in a secondary grey.

    Parameters
    ----------
    ax
        The axes to title.
    title
        The bold lede line. Should state the finding, not just describe
        the chart axes (Sanders / Alley assertion-evidence).
    subtitle
        Optional regular-weight line under the title — usually carries the
        descriptive context (sample size, condition, comparison anchor).
    source
        Optional italic source line at the bottom of the figure (e.g.
        ``"Source: eval_results/issue_281, n=200/persona, commit abc123"``).
    title_color
        Title text colour. Default is near-black (``#1A1A1A``).
    subtitle_color
        Subtitle text colour. Default is mid-grey (``#5A5A5A``).
    source_color
        Source-line text colour. Default is light grey (``#7A7A7A``).
    """
    title_size = mpl.rcParams.get("axes.titlesize", 13)
    title_weight = mpl.rcParams.get("axes.titleweight", "semibold")
    body_size = mpl.rcParams.get("font.size", 11)
    subtitle_size = max(body_size - 1, 8)

    pad = 24 if subtitle else 12
    ax.set_title(
        title,
        loc="left",
        color=title_color,
        fontweight=title_weight,
        fontsize=title_size,
        pad=pad,
    )

    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0.0, 1.0),
            xytext=(0, 6),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="left",
            va="bottom",
            color=subtitle_color,
            fontsize=subtitle_size,
            fontweight="normal",
        )

    if source:
        # `supxlabel` registers with constrained_layout so the source line
        # never overlaps the x-tick labels. Left-aligned by setting x=0.02.
        ax.figure.supxlabel(
            source,
            x=0.02,
            ha="left",
            color=source_color,
            fontsize=max(body_size - 2, 7),
            fontstyle="italic",
        )


def add_direction_arrow(
    ax: plt.Axes,
    axis: Literal["x", "y"] = "y",
    direction: Literal["up", "down"] = "up",
    label: str | None = None,
) -> None:
    """Append ``↑ better`` / ``↓ better`` to an existing axis label.

    Parameters
    ----------
    ax
        The ``Axes`` whose label should be annotated.
    axis
        ``"x"`` or ``"y"``.
    direction
        ``"up"`` for ``↑ better`` (higher is better), ``"down"`` for
        ``↓ better`` (lower is better).
    label
        If given, replace the axis label with this string verbatim and do not
        append an arrow. Useful when the caller wants a fully-custom label that
        already includes a direction indicator.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

    if label is not None:
        if axis == "x":
            ax.set_xlabel(label)
        else:
            ax.set_ylabel(label)
        return

    arrow = "↑" if direction == "up" else "↓"
    suffix = f" {arrow} better"
    current = ax.get_xlabel() if axis == "x" else ax.get_ylabel()
    if not current:
        raise ValueError(
            f"Cannot add direction arrow to an empty {axis}-axis label. "
            f"Set the label first via ax.set_{axis}label(...)."
        )
    new_label = current + suffix
    if axis == "x":
        ax.set_xlabel(new_label)
    else:
        ax.set_ylabel(new_label)


def proportion_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wald ``(lo, hi)`` confidence interval for a proportion.

    Uses ``p ± z * sqrt(p * (1 - p) / n)``. Clamps the result to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``n <= 0`` or ``p`` is outside ``[0, 1]``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    half = z * ((p * (1.0 - p)) / n) ** 0.5
    lo = max(0.0, p - half)
    hi = min(1.0, p + half)
    return (lo, hi)


# Maximum number of plotted points to embed in a figure's `.meta.json` sidecar.
# The dashboard viewer (`dashboard/lib/task-data.ts`) caps client-side at 5000
# rows; we keep the in-sidecar embed well under that so committed sidecars stay
# small and a pathological scatter cannot bloat the repo. Past the cap the
# sidecar still records the full row count + that it was truncated. This 2000
# in-sidecar cap is the BINDING one (it always trips first, below the viewer's
# 5000), so the viewer's own `truncated` flag never fires on a sidecar we wrote.
_MAX_SIDECAR_ROWS = 2000


def _finite_or_none(x: object) -> float | None:
    """Coerce ``x`` to a finite float, or ``None`` for NaN/±Inf/unparseable.

    Sidecar rows are emitted via ``json.dumps`` (default ``allow_nan=True``),
    which serializes ``NaN`` / ``Infinity`` as bare JS-invalid literals. The
    dashboard viewer's ``JSON.parse`` (``dashboard/lib/task-data.ts``) THROWS on
    those, silently degrading the whole figure to provenance-only — and the loss
    hits exactly the figures most likely to need the viewer (masked / missing
    cells, common in plots that use ``np.nan`` / ``fillna``). Map every
    non-finite numeric cell to ``None`` (→ JSON ``null``, which the viewer's
    column-typer already handles) so the row survives with only the bad cell
    dropped.
    """
    import math

    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _nearest_label(
    px: float,
    py: float,
    text_artifacts: list[tuple[float, float, str]],
    x_span: float,
    y_span: float,
) -> str | None:
    """Return the text label whose anchor sits closest to point ``(px, py)``.

    Matplotlib stores point labels as free-floating ``Text`` artifacts (often
    offset from the point with a leader line — see #657's persona-name scatter),
    so the label→point association has to be recovered geometrically. A label is
    attached to a point only when it is meaningfully the closest one AND within a
    generous radius (25% of each axis span), so an axis-corner annotation or a
    title placed via ``ax.text`` is not mis-attached to a data point.

    Returns the matched label string, or ``None`` when no text anchor is close
    enough. Distances are normalized by the axis spans so x and y units are
    comparable.

    Mapping is point→nearest-text, NOT a 1:1 assignment: in a dense scatter with
    sparse labels one text anchor can attach to several points. The one-label-
    per-point case (#657's persona scatter) is unambiguous; a dense scatter with
    a few callout labels may duplicate a label across nearby points.
    """
    if not text_artifacts:
        return None
    x_span = x_span if x_span > 0 else 1.0
    y_span = y_span if y_span > 0 else 1.0
    best: tuple[float, str] | None = None
    for tx, ty, label in text_artifacts:
        dx = (tx - px) / x_span
        dy = (ty - py) / y_span
        d2 = dx * dx + dy * dy
        if best is None or d2 < best[0]:
            best = (d2, label)
    if best is None:
        return None
    # 0.25 of an axis span (squared, summed over two axes) is a generous but
    # bounded radius: a leader-line-offset label still matches, a corner
    # annotation does not.
    return best[1] if best[0] <= (0.25**2) * 2 else None


@dataclass
class _AxesCtx:
    """Per-Axes context shared by the per-artist-kind extractors."""

    xlabel: str
    ylabel: str
    x_span: float
    y_span: float
    text_artifacts: list[tuple[float, float, str]]
    err_by_x: dict[float, float]


def _axes_ctx(ax: plt.Axes) -> _AxesCtx:
    """Build the shared per-Axes context: labels, spans, point-labels, errors."""
    import numpy as np
    from matplotlib.container import ErrorbarContainer

    xlabel = (ax.get_xlabel() or "").strip() or "x"
    ylabel = (ax.get_ylabel() or "").strip() or "y"
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # Free-floating text artifacts that may label data points (empty skipped).
    text_artifacts: list[tuple[float, float, str]] = []
    for t in ax.texts:
        s = (t.get_text() or "").strip()
        if not s:
            continue
        try:
            tx, ty = t.get_position()
            text_artifacts.append((float(tx), float(ty), s))
        except (TypeError, ValueError):
            continue

    # Symmetric y-error magnitude per x, recovered from errorbar bar-line
    # segments: half the vertical span of each (same-x) two-point segment.
    err_by_x: dict[float, float] = {}
    for cont in ax.containers:
        if not isinstance(cont, ErrorbarContainer):
            continue
        _data_line, _caps, barlinecols = cont
        for blc in barlinecols or []:
            try:
                for seg in blc.get_segments():
                    seg = np.asarray(seg, dtype=float)
                    if seg.shape == (2, 2) and abs(seg[0, 0] - seg[1, 0]) < 1e-12:
                        cx = round(float(seg[0, 0]), 6)
                        err_by_x[cx] = abs(seg[1, 1] - seg[0, 1]) / 2.0
            except (TypeError, ValueError):
                continue

    return _AxesCtx(
        xlabel=xlabel,
        ylabel=ylabel,
        x_span=abs(x1 - x0),
        y_span=abs(y1 - y0),
        text_artifacts=text_artifacts,
        err_by_x=err_by_x,
    )


def _extract_bars(ax: plt.Axes, ctx: _AxesCtx) -> list[dict[str, object]]:
    """Extract bar-chart rows: category/x + height (+ error) per bar.

    Category labels come from x-tick labels keyed by the patch CENTER. NOTE:
    grouped / dodged bar charts (multiple series at offset positions like
    ``x±0.2``) place patches off the tick locations, so ``tick_map`` misses and
    those rows fall back to the numeric x position. Single (non-grouped) bars
    keep their category labels. Non-finite heights / errors are emitted as JSON
    ``null`` (see ``_finite_or_none``).
    """
    from matplotlib.container import BarContainer

    tick_labels = [t.get_text().strip() for t in ax.get_xticklabels()]
    tick_locs = [round(float(v), 6) for v in ax.get_xticks()]
    tick_map = {loc: lab for loc, lab in zip(tick_locs, tick_labels, strict=False) if lab}

    out: list[dict[str, object]] = []
    for cont in ax.containers:
        if not isinstance(cont, BarContainer):
            continue
        try:
            heights = [float(v) for v in cont.datavalues]
        except (TypeError, ValueError, AttributeError):
            continue
        rows: list[dict[str, object]] = []
        for patch, height in zip(cont.patches, heights, strict=False):
            try:
                cx = round(float(patch.get_x() + patch.get_width() / 2.0), 6)
            except (TypeError, ValueError):
                continue
            cat = tick_map.get(cx)
            row: dict[str, object] = {}
            if cat:
                row[ctx.xlabel if ctx.xlabel != "x" else "category"] = cat
            else:
                row[ctx.xlabel] = _finite_or_none(cx)
            row[ctx.ylabel] = _finite_or_none(height)
            if cx in ctx.err_by_x:
                row["error"] = _finite_or_none(ctx.err_by_x[cx])
            rows.append(row)
        if rows:
            out.append({"kind": "bar", "label": (cont.get_label() or "").strip(), "rows": rows})
    return out


def _xy_rows(
    xy: object, ctx: _AxesCtx, series: str, *, with_error: bool, with_label: bool
) -> list[dict[str, object]]:
    """Turn an (N,2) xy array into flat rows, optionally attaching error/label."""
    import numpy as np

    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        return []

    rows: list[dict[str, object]] = []
    for px, py in arr:
        fx = _finite_or_none(px)
        fy = _finite_or_none(py)
        # Non-finite coordinates (NaN/Inf — e.g. masked cells) → JSON null so
        # the row survives and the viewer's JSON.parse never throws.
        row: dict[str, object] = {ctx.xlabel: fx, ctx.ylabel: fy}
        if with_error and fx is not None:
            cx = round(fx, 6)
            if cx in ctx.err_by_x:
                row["error"] = _finite_or_none(ctx.err_by_x[cx])
        if with_label and fx is not None and fy is not None:
            lbl = _nearest_label(fx, fy, ctx.text_artifacts, ctx.x_span, ctx.y_span)
            if lbl is not None:
                row["label"] = lbl
        if series and not series.startswith("_"):
            row["series"] = series
        rows.append(row)
    return rows


def _extract_errorbars(ax: plt.Axes, ctx: _AxesCtx) -> list[dict[str, object]]:
    """Extract non-bar errorbar data lines: x/y (+ error) per vertex."""
    from matplotlib.container import ErrorbarContainer

    out: list[dict[str, object]] = []
    for cont in ax.containers:
        if not isinstance(cont, ErrorbarContainer):
            continue
        data_line = cont[0]
        if data_line is None:
            continue
        series = (cont.get_label() or "").strip()
        try:
            rows = _xy_rows(data_line.get_xydata(), ctx, series, with_error=True, with_label=False)
        except (TypeError, ValueError):
            continue
        if rows:
            out.append({"kind": "errorbar", "label": series, "rows": rows})
    return out


def _extract_scatters(ax: plt.Axes, ctx: _AxesCtx) -> list[dict[str, object]]:
    """Extract scatter point offsets: x/y (+ nearest text label) per point."""
    import numpy as np

    out: list[dict[str, object]] = []
    for coll in ax.collections:
        # Only PathCollections carry point offsets; LineCollections (error bars)
        # are handled by _extract_errorbars.
        try:
            offsets = np.asarray(coll.get_offsets(), dtype=float)
        except (TypeError, ValueError, AttributeError):
            continue
        if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.shape[0] == 0:
            continue
        # A lone (0,0) offset is matplotlib's default for empty collections.
        if offsets.shape[0] == 1 and not np.any(offsets):
            continue
        series = (coll.get_label() or "").strip()
        rows = _xy_rows(offsets, ctx, series, with_error=False, with_label=True)
        if rows:
            out.append({"kind": "scatter", "label": series, "rows": rows})
    return out


def _extract_lines(ax: plt.Axes, ctx: _AxesCtx) -> list[dict[str, object]]:
    """Extract plain line-plot vertices: x/y per vertex, series-labeled.

    Unlabeled lines (matplotlib's default ``_childN`` label) are KEPT — a single
    ``ax.plot(x, y)`` sweep with no explicit legend label is the common case — but
    their ``series`` column is omitted (``_xy_rows`` drops ``_``-prefixed labels).
    Degenerate unlabeled 2-vertex lines (grey leader lines connecting a scatter
    point to its offset text label) are dropped as non-data.
    """
    import numpy as np

    out: list[dict[str, object]] = []
    for line in ax.get_lines():
        series = (line.get_label() or "").strip()
        try:
            xy = np.asarray(line.get_xydata(), dtype=float)
        except (TypeError, ValueError):
            continue
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
            continue
        is_internal = series.startswith("_")
        # A degenerate ≤2-vertex line with no explicit label is a leader /
        # reference segment, not data — skip it.
        if is_internal and xy.shape[0] <= 2:
            continue
        rows = _xy_rows(xy, ctx, series, with_error=False, with_label=False)
        if rows:
            out.append({"kind": "line", "label": "" if is_internal else series, "rows": rows})
    return out


def _extract_axes_data(fig: plt.Figure) -> list[dict[str, object]]:
    """Extract the plotted per-point data from a matplotlib ``Figure``.

    Reads the data back off the rendered artists so EVERY figure auto-carries
    its underlying data with no per-call-site changes:

    * **scatter** (``PathCollection.get_offsets()``) — x/y per point, plus the
      nearest free-floating text label as an identifier column when present.
    * **line** (``Line2D.get_xydata()``) — x/y per vertex, the series' legend
      label as a ``series`` column.
    * **bar** (``BarContainer.datavalues`` + patch centers) — category x-tick
      label (or numeric x) and height, with the matching error-bar magnitude
      when a paired ``ErrorbarContainer`` is present.
    * **errorbar** (``ErrorbarContainer``) — the data line's x/y plus the
      symmetric error magnitude recovered from the bar-line segments.

    Column names come from the axis labels (``set_xlabel`` / ``set_ylabel``)
    where set, falling back to ``x`` / ``y``. The return is a list of artifact
    descriptors, one per extractable artist group, each with ``kind``,
    ``label``, and a flat ``rows`` list of scalar-valued dicts — exactly the
    shape ``dashboard/lib/task-data.ts`` normalizes (``points`` / ``rows`` /
    array-of-objects).

    Never raises: any artist whose data cannot be read is skipped, and a fully
    unreadable figure returns ``[]`` (the sidecar then carries provenance only,
    and the viewer falls back to the figure link-out).
    """
    artifacts: list[dict[str, object]] = []
    for ax in fig.get_axes():
        try:
            ctx = _axes_ctx(ax)
            artifacts.extend(_extract_bars(ax, ctx))
            artifacts.extend(_extract_errorbars(ax, ctx))
            artifacts.extend(_extract_scatters(ax, ctx))
            artifacts.extend(_extract_lines(ax, ctx))
        except Exception:
            # A single bad Axes never sinks the whole extraction.
            continue
    return artifacts


def _build_sidecar_data(
    artifacts: list[dict[str, object]],
) -> dict[str, object] | None:
    """Collapse extracted artifacts into the sidecar's ``points`` payload.

    Returns a dict with ``points`` (the flat row list the viewer reads), plus
    ``n_series`` / ``truncated`` / ``total_points`` provenance, or ``None`` when
    nothing extractable was found. Each row gets a ``_kind`` tag and — when more
    than one artist group is present — a ``_group`` index so the viewer's
    filter/sort can separate co-plotted series (e.g. raw scatter + binned line).
    """
    if not artifacts:
        return None

    multi = len(artifacts) > 1
    points: list[dict[str, object]] = []
    for gi, art in enumerate(artifacts):
        rows = art.get("rows") or []
        kind = art.get("kind", "")
        for row in rows:  # type: ignore[union-attr]
            tagged: dict[str, object] = dict(row)
            tagged["_kind"] = kind
            if multi:
                tagged["_group"] = gi
            points.append(tagged)

    if not points:
        return None

    total = len(points)
    truncated = total > _MAX_SIDECAR_ROWS
    if truncated:
        points = points[:_MAX_SIDECAR_ROWS]

    return {
        "points": points,
        "n_series": len(artifacts),
        "total_points": total,
        "truncated": truncated,
    }


_MAX_TEXT_ITEMS = 100  # per-axes cap on annotations / tick labels (+ the series list)


def _artist_text(t) -> str:
    """The stripped text of a matplotlib ``Text``-like artist, or ``""`` when
    the artist is None / unreadable (best-effort — never raises)."""
    try:
        return t.get_text().strip() if t is not None else ""
    except Exception:
        return ""


def _axes_text_and_series(ax) -> tuple[dict[str, object], list[str]]:
    """Rendered text of ONE ``Axes`` for ``_extract_fig_text``: the per-axes
    text dict (titles at all three locs — the house style renders at
    ``loc="left"`` — axis labels, legend labels + title, annotations, tick
    labels) plus the axes' legend-eligible artist labels (series names).
    May raise on a pathological Axes; the caller's per-axes try/except skips
    it (one bad Axes never sinks the capture)."""
    ax_d: dict[str, object] = {}
    for loc in ("center", "left", "right"):
        t = ax.get_title(loc=loc)
        if t.strip():
            ax_d["title" if loc == "center" else f"title_{loc}"] = t.strip()
    for key, val in (("xlabel", ax.get_xlabel()), ("ylabel", ax.get_ylabel())):
        if val.strip():
            ax_d[key] = val.strip()
    leg = ax.get_legend()
    if leg is not None:
        labs = [s for s in (_artist_text(t) for t in leg.get_texts()) if s]
        if labs:
            ax_d["legend_labels"] = labs
        lt = _artist_text(leg.get_title())
        if lt:
            ax_d["legend_title"] = lt
    # set_title_subtitle's subtitle is an ax.annotate → lands in ax.texts,
    # alongside free-floating point labels — both rendered text.
    ann = [s for s in (_artist_text(t) for t in ax.texts) if s]
    if ann:
        ax_d["annotations"] = ann[:_MAX_TEXT_ITEMS]
    for key, ticks in (
        ("xticklabels", ax.get_xticklabels()),
        ("yticklabels", ax.get_yticklabels()),
    ):
        tl = [s for s in (_artist_text(t) for t in ticks) if s]
        if tl:
            ax_d[key] = tl[:_MAX_TEXT_ITEMS]
    # Series names: legend-eligible artist labels ('_'-prefixed =
    # matplotlib's no-legend convention, excluded).
    labels: list[str] = []
    for art in (*ax.get_lines(), *ax.containers, *ax.collections):
        lab = str(art.get_label() or "")
        if lab and not lab.startswith("_"):
            labels.append(lab)
    return ax_d, labels


def _extract_fig_text(fig: plt.Figure) -> dict[str, object] | None:
    """Best-effort capture of the figure's RENDERED text for the sidecar.

    Returns ``{"suptitle": str|None, "fig_texts": [str], "series": [str],
    "axes": [{...}]}`` or ``None`` when nothing was captured. Values are ONLY
    ``str`` / ``None`` / ``list`` — never nested dicts-of-dicts — so the
    dashboard viewer's ``normalizeToRows`` fallback can never mistake the
    block for data rows (``dashboard/lib/task-data.ts``: ``isPlainObject``
    excludes arrays/null, and the object-of-objects fallback descends at most
    two levels, so a ``text`` dict whose values are str/None/list never
    matches). Never raises: any unreadable artist/axes is skipped (the
    ``_extract_axes_data`` contract). The block is what lets
    ``scripts/verify_task_body.py`` checks 24/28/34 mechanically scan the
    figure's titles / legends / series names (incident #1092: bare cell slugs
    as panel titles + a beat claiming an unrendered series structure passed
    every mechanical figure check because the sidecar carried no rendered
    text).
    """
    suptitle_obj = getattr(fig, "_suptitle", None)
    supx_obj = getattr(fig, "_supxlabel", None)
    supy_obj = getattr(fig, "_supylabel", None)
    # On matplotlib 3.10 the suptitle AND supx/supy labels all appear in
    # `fig.texts`; exclude them by identity so they are not duplicated into
    # `fig_texts` (suptitle gets its own key; supx/supy are re-added exactly
    # once below, AFTER the cap, so a text-dense figure cannot truncate them
    # away).
    special = {id(suptitle_obj), id(supx_obj), id(supy_obj)}
    out: dict[str, object] = {"suptitle": _artist_text(suptitle_obj) or None}
    fig_texts = [s for s in (_artist_text(t) for t in fig.texts if id(t) not in special) if s]
    fig_texts = fig_texts[:_MAX_TEXT_ITEMS]
    fig_texts += [s for s in (_artist_text(supx_obj), _artist_text(supy_obj)) if s]
    out["fig_texts"] = fig_texts

    # `series` is fig-GLOBAL, deduplicated across axes.
    series: list[str] = []
    axes_out: list[dict[str, object]] = []
    for ax in fig.get_axes():
        try:
            ax_d, labels = _axes_text_and_series(ax)
        except Exception:
            continue  # one bad Axes never sinks the capture
        series.extend(labels)
        if ax_d:
            axes_out.append(ax_d)
    if axes_out:
        out["axes"] = axes_out
    dedup = list(dict.fromkeys(series))[:_MAX_TEXT_ITEMS]
    if dedup:
        out["series"] = dedup
    has_content = out["suptitle"] or out["fig_texts"] or axes_out or dedup
    return out if has_content else None


def _git_commit_hash() -> str:
    """Return the current git commit short hash, or ``"uncommitted"`` on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "uncommitted"
    if out.returncode != 0:
        return "uncommitted"
    sha = out.stdout.strip()
    return sha if sha else "uncommitted"


def savefig_paper(
    fig: plt.Figure,
    stem: str,
    dir: str | Path = "figures/",
    formats: tuple[str, ...] = ("png", "pdf"),
    embed_data: bool = True,
    embed_text: bool = True,
) -> dict[str, Path]:
    """Save ``fig`` to ``<dir>/<stem>.<fmt>`` for every ``fmt`` in ``formats``.

    Embeds the current git commit hash in PDF metadata (``Commit``) and in PNG
    ``pnginfo``. Also writes a sidecar ``<dir>/<stem>.meta.json`` containing
    commit hash, ISO-8601 UTC timestamp, figure size (inches), AND — when
    ``embed_data`` is true (the default) — the figure's per-point data under a
    ``points`` key, plus — when ``embed_text`` is true (the default) — the
    figure's rendered text under a ``text`` key.

    **Per-point data (dashboard data viewer).** The plotted data is read back
    off the rendered matplotlib artists (``_extract_axes_data``): scatter point
    offsets (with the nearest text label as an identifier column — e.g. persona
    names), line vertices, bar heights, and error-bar magnitudes, with the axis
    labels as column names. The result is emitted under ``points`` in the shape
    the EPS dashboard's ``dashboard/lib/task-data.ts`` resolver reads directly
    (an array of flat scalar-cell objects → a sortable/filterable table), so
    every NEW figure auto-populates the per-figure data viewer with NO per-call
    or dashboard change. Extraction never raises: an unreadable figure simply
    yields a provenance-only sidecar and the viewer falls back to the figure
    link-out. Large sets are capped at ``_MAX_SIDECAR_ROWS`` (with
    ``data_truncated`` + ``total_points`` recorded). See the SPEC "Dashboard
    data-artifact interface (Phase 2 contract)".

    Caller responsibility: every label baked into ``fig`` (axis labels, tick
    labels, legend entries, in-figure annotations) must be plain English, not
    a Hydra slug (``sw_eng_C1``, ``c1_evil_wrong_em``, ``cond_4``) or
    short-letter label (``M1``, ``Method A``, ``Bin C``, ``BS_E0``). See
    ``.claude/skills/paper-plots/SKILL.md`` § 3.5 for the relabel pattern.
    The clean-result-critic Lens 3 and interpretation-critic Lens 6 enforce
    this on review; doing it here avoids a regenerate-the-figure bounce. Because
    the axis labels also become the sidecar's column names, plain-English labels
    make the data viewer's columns readable too. The rendered labels are ALSO
    serialized into the sidecar under ``text`` (``_extract_fig_text``; opt out
    with ``embed_text=False``) and mechanically scanned by
    ``scripts/verify_task_body.py`` checks 24/28/34, so an opaque slug in a
    title / legend / series name now WARNs at body-verification time instead
    of waiting for the multimodal critics.

    Parameters
    ----------
    fig
        The ``Figure`` to save.
    stem
        Filename stem (no extension). May contain subdirectories; the full
        parent directory will be created.
    dir
        Parent directory for the outputs. Created if missing.
    formats
        Tuple of extensions to save. Supported: ``"png"``, ``"pdf"``.
    embed_data
        When true (default), extract and embed the figure's per-point data in
        the sidecar under ``points``. Set false to write a provenance-only
        sidecar (e.g. for a figure whose data is huge or already exposed via a
        committed ``data_path`` the caller adds afterward).
    embed_text
        When true (default), capture the figure's rendered text (suptitle,
        per-axes titles incl. the house ``loc="left"`` style, axis labels,
        legend labels + title, series names, annotations, tick labels) into
        the sidecar under ``text`` (``_extract_fig_text``). Independent of
        ``embed_data`` — a huge-data ``embed_data=False`` figure still gets
        its cheap text captured. Best-effort: a capture failure omits the
        key, never fails the save.

    Returns
    -------
    dict
        Mapping from format to the ``Path`` that was written. Includes the key
        ``"meta"`` for the sidecar ``.meta.json``.
    """
    out_dir = Path(dir)
    target = out_dir / stem
    target.parent.mkdir(parents=True, exist_ok=True)

    commit = _git_commit_hash()
    written: dict[str, Path] = {}

    for fmt in formats:
        if fmt == "png":
            from PIL import PngImagePlugin  # local import to keep module light

            png_path = target.with_suffix(".png")
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("Commit", commit)
            fig.savefig(png_path, format="png", metadata={"Software": f"commit={commit}"})
            # Re-tag with pnginfo chunk so the commit is greppable from the file.
            from PIL import Image as _Image

            with _Image.open(png_path) as img:
                img.save(png_path, format="png", pnginfo=pnginfo)
            written["png"] = png_path
        elif fmt == "pdf":
            pdf_path = target.with_suffix(".pdf")
            fig.savefig(pdf_path, format="pdf", metadata={"Keywords": f"commit={commit}"})
            written["pdf"] = pdf_path
        else:
            raise ValueError(f"Unsupported format {fmt!r}; supported: png, pdf")

    meta_path = target.with_suffix(".meta.json")
    fig_size = fig.get_size_inches().tolist()
    meta: dict[str, object] = {
        "commit": commit,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "figsize": [float(fig_size[0]), float(fig_size[1])],
    }

    if embed_data:
        # Extraction is best-effort: a figure whose artists cannot be read back
        # (custom collections, image plots, ...) just gets a provenance-only
        # sidecar — never a save failure.
        try:
            payload = _build_sidecar_data(_extract_axes_data(fig))
        except Exception:
            payload = None
        if payload is not None:
            meta["points"] = payload["points"]
            meta["n_series"] = payload["n_series"]
            meta["total_points"] = payload["total_points"]
            if payload["truncated"]:
                meta["data_truncated"] = True

    # Rendered-text capture (verifier + review window into figure text).
    # Best-effort, same contract as the data embed above: a text-extraction
    # failure NEVER fails the save — the key is simply omitted.
    if embed_text:
        try:
            fig_text = _extract_fig_text(fig)
        except Exception:
            fig_text = None
        if fig_text is not None:
            meta["text"] = fig_text

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    written["meta"] = meta_path
    return written
