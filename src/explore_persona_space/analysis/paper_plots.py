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
  `.claude/skills/clean-results/principles.md`).
* **Commit-pinned metadata** — every saved figure carries the git commit hash
  embedded in PDF metadata / PNG pnginfo plus a sidecar `<stem>.meta.json` so a
  reader can always trace a figure back to the code that produced it.

Public API
----------
    set_paper_style        — set rcParams for paper-quality figures
    savefig_paper          — save to <dir>/<stem>.png AND <dir>/<stem>.pdf plus .meta.json
    add_direction_arrow    — append ↑ better / ↓ better to an axis label
    paper_palette          — return N colorblind-safe hex colours
    proportion_ci          — 95% Wald CI for a proportion

Exemplar usage
--------------
>>> from explore_persona_space.analysis.paper_plots import (
...     set_paper_style, savefig_paper, add_direction_arrow, paper_palette,
...     proportion_ci,
... )
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


def set_paper_style(
    target: Literal["neurips", "generic"] = "neurips",
    font_scale: float = 1.0,
) -> None:
    """Configure ``matplotlib`` rcParams for paper-quality figures.

    Idempotent: calling twice produces the same state as calling once.

    Parameters
    ----------
    target
        ``"neurips"`` produces ~single-column NeurIPS sizing (5.5 x 3.4 in).
        ``"generic"`` is a slightly larger default (6.0 x 4.0 in).
    font_scale
        Multiplier applied to every font size. ``1.0`` leaves the defaults.
    """
    if target not in ("neurips", "generic"):
        raise ValueError(f"target must be 'neurips' or 'generic', got {target!r}")

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
