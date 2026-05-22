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

Public API
----------
    set_paper_style        — set rcParams for paper-quality figures
    savefig_paper          — save to <dir>/<stem>.png AND <dir>/<stem>.pdf plus .meta.json
    add_direction_arrow    — append ↑ better / ↓ better to an axis label
    paper_palette          — return N colorblind-safe hex colours (Wong 2011)
    paper_palette_blog     — return N colours from the soft "blog" palette
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
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
    """Return the first ``n`` colours of the curated colorblind-safe palette.

    Raises
    ------
    ValueError
        If ``n`` is not in ``[1, 8]``.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")
    if n > len(_PALETTE):
        raise ValueError(f"paper_palette supports at most {len(_PALETTE)} colors; requested {n}")
    return list(_PALETTE[:n])


def paper_palette_blog(n: int) -> list[str]:
    """Return the first ``n`` colours of the soft "blog" palette.

    Companion to :func:`paper_palette` for the Anthropic-blog-register style.

    Raises
    ------
    ValueError
        If ``n`` is not in ``[1, 8]``.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")
    if n > len(_BLOG_PALETTE):
        raise ValueError(
            f"paper_palette_blog supports at most {len(_BLOG_PALETTE)} colors; requested {n}"
        )
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
) -> dict[str, Path]:
    """Save ``fig`` to ``<dir>/<stem>.<fmt>`` for every ``fmt`` in ``formats``.

    Embeds the current git commit hash in PDF metadata (``Commit``) and in PNG
    ``pnginfo``. Also writes a sidecar ``<dir>/<stem>.meta.json`` containing
    commit hash, ISO-8601 UTC timestamp, and figure size (inches).

    Caller responsibility: every label baked into ``fig`` (axis labels, tick
    labels, legend entries, in-figure annotations) must be plain English, not
    a Hydra slug (``sw_eng_C1``, ``c1_evil_wrong_em``, ``cond_4``) or
    short-letter label (``M1``, ``Method A``, ``Bin C``, ``BS_E0``). See
    ``.claude/skills/paper-plots/SKILL.md`` § 3.5 for the relabel pattern.
    The clean-result-critic Lens 3 and interpretation-critic Lens 6 enforce
    this on review; doing it here avoids a regenerate-the-figure bounce.

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
    meta = {
        "commit": commit,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "figsize": [float(fig_size[0]), float(fig_size[1])],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    written["meta"] = meta_path
    return written
