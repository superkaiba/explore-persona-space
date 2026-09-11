#!/usr/bin/env python3
"""Render the Section 4.2 per-element answer-shift figure as three metric columns.

The shipped ``c3_element_shifts`` carries two columns (direction, shift size).
This script renders the same rows with the two-way discrimination rate added as
a third column, in two variants that differ only in which rows are drawn:

* ``c3_element_shifts_three_panel`` -- the seven rows left after the two
  refusal-holds rows are dropped.
* ``c3_element_shifts_three_panel_withholds`` -- all nine rows, with the two
  holds rows in their own subordinate block.

Both read the banked rows in
``eval_results/issue_2564/section42_element_shifts.json`` (rebuilt by
``scripts/issue2564_element_shift_rows.py``; it needs staging mounts this
script does not) and write vector PDF, color PNG, grayscale-audit PNG, and a
provenance JSON for each variant.  Plot-only: no statistic is recomputed here.

The two-way column is drawn on a cut axis.  Every rate sits far above the 0.5
chance reference, so one linear axis either buries the rows against the right
edge or drops the reference off the left.  The axis is therefore two linear
segments -- a narrow one holding the chance reference, then the data segment --
whose plotted widths are proportional to their data spans, so both segments
share ONE scale and a distance means the same thing in either.  Diagonal marks
sit on the cut, and the renderer refuses to draw any value or interval endpoint
that falls in the omitted range rather than clipping it to a segment edge.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE the matplotlib/numpy imports below.  On the
# shared VM load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, and
# the BLAS pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
    MUTED,
    PAPER,
    ROLES,
    STYLE_VERSION,
    better_label,
    c2a_figure,
    canvas_width_in,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.analysis import c2a_row_labels as L  # noqa: E402

# Reused from the sibling Section 4.2 renderer so both figures carry byte-identical
# provenance fields; the row source is the same banked file it reads.
from scripts.make_paper_section42_figures import (  # noqa: E402
    ELEMENT_SHIFT_SOURCE,
    _display_path,
    _git_state,
    _sha256,
    _tint,
)

SCRIPT = "scripts/issue2564_element_shifts_three_panel.py"
DEFAULT_OUT = ROOT / "figures/paper"

LINEAR = ROLES["linear"].color
CONTROL = ROLES["control"].color
# The subordinate block keeps the map's hue mixed toward paper, the same tone the
# sibling renderer gives a second row group, and takes an open marker so the
# distinction survives the grayscale audit.
SUBORDINATE = _tint(LINEAR, 0.45)

# ---------------------------------------------------------------------------
# Row groups.  Identity-side elements, then topic-side, then the refusal rows;
# the holds rows form their own block in the second variant.
# ---------------------------------------------------------------------------

_IDENTITY = ("Identity", (L.TONE, L.PERSONA, L.OUTPUT_FORMAT), False)
_TOPIC = ("Topic", (L.QUESTION_TOPIC, L.ONE_WORD_TOPIC), False)
_REFUSAL = (
    L.REFUSAL_REVERSES_GROUP,
    (L.REFUSAL_REVERSES_INTENT, L.REFUSAL_REVERSES_FRAMING),
    False,
)
_REFUSAL_HOLDS = (
    L.REFUSAL_HOLDS_GROUP,
    (L.REFUSAL_HOLDS_INTENT, L.REFUSAL_HOLDS_FRAMING),
    True,
)

# A group is separated from the one above it by this many row heights.  The gap
# carries the group's kicker, so the break states what it separates.
_GROUP_GAP = 0.95
# Constant row pitch in inches: the two variants draw different row counts at the
# same size, so the canvas height follows the rows rather than squashing them.
# Shared with the sibling Section 4.2 renderer so a wrapped refusal label clears
# its neighbours in every figure that draws one row per element.
_ROW_PITCH_IN = L.WRAPPED_ROW_PITCH_IN
_HEADER_IN = 0.78
_FOOTER_IN = 1.10
# Kicker and title offsets above the plot box, in inches, so both variants place
# their header at the same distance whatever the plot height.
_KICKER_OFF_IN = 0.42
_TITLE_OFF_IN = 0.15
# One x-label offset below the plot box, in inches, shared by all three columns
# (the cut column places its label by hand, so the others must match it).
_XLABEL_OFF_IN = 0.34

_KICKER = "controlled minimal pairs · qwen2.5-7b-instruct · layer 19"
_TITLE = "What the map keeps when one context element changes"

# ---------------------------------------------------------------------------
# Columns.  Axis wording follows the banked ``metrics`` definitions: the cosine
# between predicted and observed answer shift, their size ratio, and the rate at
# which the predicted answer is nearer its own target than the other member of
# the pair.  The first two ranges match the shipped two-column figure so the
# three-column figure can be read against it.
# ---------------------------------------------------------------------------

_DIRECTION = {
    "key": "direction",
    "xlabel": better_label("Predicted shift\ndirection (cosine)"),
    "xlim": (0.15, 1.0),
    "ticks": (0.2, 0.4, 0.6, 0.8, 1.0),
    "reference": None,
}
_MAGNITUDE = {
    "key": "magnitude",
    "xlabel": "Predicted / observed\nshift size",
    "xlim": (0.55, 1.28),
    "ticks": (0.6, 0.8, 1.0, 1.2),
    "reference": 1.0,
}
# The chance segment is identical in both variants; the data segment starts just
# below the lowest interval endpoint the variant draws.
_CHANCE_SEGMENT = (0.46, 0.54)
_TWOWAY = {
    "key": "twoway",
    "xlabel": better_label("Two-way\ndiscrimination rate"),
    "reference": 0.5,
}

VARIANTS: dict[str, dict] = {
    "c3_element_shifts_three_panel": {
        "groups": (_IDENTITY, _TOPIC, _REFUSAL),
        "twoway_segments": (_CHANCE_SEGMENT, (0.78, 1.015)),
        "twoway_ticks": ((0.5,), (0.8, 0.9, 1.0)),
        "subject": (
            "Per-element answer shift under one controlled context change: predicted "
            "shift direction, predicted over observed shift size, and two-way "
            "discrimination rate, for the seven rows left after the two "
            "refusal-holds rows are dropped"
        ),
    },
    "c3_element_shifts_three_panel_withholds": {
        "groups": (_IDENTITY, _TOPIC, _REFUSAL, _REFUSAL_HOLDS),
        "twoway_segments": (_CHANCE_SEGMENT, (0.58, 1.015)),
        "twoway_ticks": ((0.5,), (0.7, 0.8, 0.9, 1.0)),
        "subject": (
            "Per-element answer shift under one controlled context change: predicted "
            "shift direction, predicted over observed shift size, and two-way "
            "discrimination rate, for all nine rows, the two refusal-holds rows "
            "drawn as a subordinate block"
        ),
    },
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_rows() -> tuple[dict[str, dict], dict]:
    banked = json.loads(ELEMENT_SHIFT_SOURCE.read_text())
    return {row["row"]: row for row in banked["panel_rows"]}, banked


def _variant_rows(banked_rows: dict[str, dict], groups: tuple) -> list[dict]:
    """Rows in drawing order, tagged with their group and subordinate flag."""
    ordered: list[dict] = []
    for label, names, subordinate in groups:
        for name in names:
            if name not in banked_rows:
                raise KeyError(f"banked panel_rows has no row {name!r}")
            ordered.append({**banked_rows[name], "group": label, "subordinate": subordinate})
    return ordered


def _row_layout(groups: tuple) -> tuple[dict[str, float], list[tuple[str, float]], float, float]:
    """Row y positions, group kicker positions, and the y limits that hold them."""
    positions: dict[str, float] = {}
    kickers: list[tuple[str, float]] = []
    depth = 0.0
    for label, names, _subordinate in groups:
        kickers.append((label, -(depth + _GROUP_GAP / 2.0)))
        depth += _GROUP_GAP
        for name in names:
            positions[name] = -depth
            depth += 1.0
    return positions, kickers, -(depth - 1.0) - 0.6, 0.05


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _assert_drawable(rows: list[dict], key: str, segments: tuple[tuple[float, float], ...]) -> None:
    """Every point and interval endpoint must land inside one drawn segment."""
    for row in rows:
        for value in (row[key], *row[f"{key}_ci95"]):
            if not any(lo <= value <= hi for lo, hi in segments):
                raise ValueError(
                    f"{key}={value} for row {row['row']!r} falls outside the drawn "
                    f"segments {segments}; widen the axis rather than clipping the value"
                )


def _plot_points(ax: plt.Axes, rows: list[dict], positions: dict[str, float], key: str) -> None:
    """Point estimate plus its 95% interval, one row per element."""
    for subordinate in (False, True):
        block = [row for row in rows if row["subordinate"] is subordinate]
        if not block:
            continue
        y = np.asarray([positions[row["row"]] for row in block])
        values = np.asarray([row[key] for row in block], dtype=float)
        ci = np.asarray([row[f"{key}_ci95"] for row in block], dtype=float)
        xerr = np.vstack([values - ci[:, 0], ci[:, 1] - values]).clip(min=0)
        color = SUBORDINATE if subordinate else LINEAR
        ax.errorbar(
            values,
            y,
            xerr=xerr,
            fmt="o",
            color=color,
            markerfacecolor=PAPER if subordinate else LINEAR,
            markeredgecolor=color,
            markeredgewidth=1.6,
            markersize=7,
            capsize=3,
            capthick=1.5 if subordinate else 1.6,
            elinewidth=1.5 if subordinate else 1.8,
            lw=0,
            zorder=3,
        )


def _shade_subordinate(ax: plt.Axes, groups: tuple, positions: dict[str, float]) -> None:
    """A stripe behind a subordinate group, so the block reads without color."""
    for _label, names, subordinate in groups:
        if not subordinate:
            continue
        ys = [positions[name] for name in names]
        ax.axhspan(
            min(ys) - 0.5,
            max(ys) + 0.5 + _GROUP_GAP,
            color=GRID,
            alpha=0.20,
            zorder=0,
            lw=0,
        )


def _finish_axis(
    ax: plt.Axes,
    *,
    xlim: tuple[float, float],
    ticks: tuple[float, ...],
    reference: float | None,
    ylim: tuple[float, float],
) -> None:
    if reference is not None and xlim[0] <= reference <= xlim[1]:
        ax.axvline(reference, color=CONTROL, lw=1.6, linestyle=(0, (5, 4)), zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    style_axis(ax, grid_axis="x")
    ax.set_xticks(list(ticks))


def _draw_cut_marks(left: plt.Axes, right: plt.Axes) -> None:
    """Diagonal marks on both sides of the cut, as the paper's other cut axis uses."""
    mark = {
        "marker": [(-1.0, -0.6), (1.0, 0.6)],
        "markersize": 11,
        "linestyle": "none",
        "color": MUTED,
        "markeredgecolor": MUTED,
        "markeredgewidth": 1.5,
        "clip_on": False,
        "zorder": 6,
    }
    # The x-axis is the cut axis, so the marks sit on the bottom seam; the top
    # edge carries no spine to break.
    left.plot([1], [0], transform=left.transAxes, **mark)
    right.plot([0], [0], transform=right.transAxes, **mark)


def _place_xlabel(ax: plt.Axes, label: str, *, x_axes: float, plot_h_in: float) -> None:
    ax.set_xlabel(label)
    ax.xaxis.set_label_coords(x_axes, -_XLABEL_OFF_IN / plot_h_in)


def make_figure(variant: str, banked_rows: dict[str, dict]) -> tuple[plt.Figure, float, dict]:
    """One row per context element; direction, shift size, and two-way rate."""
    spec = VARIANTS[variant]
    groups = spec["groups"]
    rows = _variant_rows(banked_rows, groups)
    positions, kickers, y_lo, y_hi = _row_layout(groups)

    # Row pitch is constant, so the canvas grows with the row count instead of
    # squashing the rows: the two variants draw their rows at the same size.
    width_in = canvas_width_in(1.0)
    height_in = _HEADER_IN + (y_hi - y_lo) * _ROW_PITCH_IN + _FOOTER_IN
    fig, include_frac = c2a_figure("full", aspect=height_in / width_in)
    height_in = fig.get_figheight()
    plot_h_in = height_in - _HEADER_IN - _FOOTER_IN

    grid = fig.add_gridspec(
        1,
        3,
        left=0.21,
        right=0.975,
        top=1.0 - _HEADER_IN / height_in,
        bottom=_FOOTER_IN / height_in,
        wspace=0.16,
    )
    columns = (_DIRECTION, _MAGNITUDE, _TWOWAY)
    drawn: list[dict] = []

    for index, column in enumerate(columns):
        key = column["key"]
        if key == "twoway":
            segments = spec["twoway_segments"]
            seg_ticks = spec["twoway_ticks"]
            _assert_drawable(rows, key, segments)
            spans = [hi - lo for lo, hi in segments]
            inner = grid[0, index].subgridspec(1, len(segments), width_ratios=spans, wspace=0.10)
            axes = [
                fig.add_subplot(inner[0, seg], label=f"{variant}-twoway-seg{seg}")
                for seg in range(len(segments))
            ]
            for ax, seg_range, ticks in zip(axes, segments, seg_ticks, strict=True):
                _shade_subordinate(ax, groups, positions)
                _plot_points(ax, rows, positions, key)
                _finish_axis(
                    ax,
                    xlim=seg_range,
                    ticks=ticks,
                    reference=column["reference"],
                    ylim=(y_lo, y_hi),
                )
                ax.set_yticks([positions[row["row"]] for row in rows], [""] * len(rows))
            for lower, upper in pairwise(axes):
                lower.spines["right"].set_visible(False)
                upper.spines["left"].set_visible(False)
                upper.tick_params(axis="y", length=0, labelleft=False)
                _draw_cut_marks(lower, upper)
            boxes = [ax.get_position() for ax in axes]
            center = (boxes[0].x0 + boxes[-1].x1) / 2.0
            _place_xlabel(
                axes[0],
                column["xlabel"],
                x_axes=(center - boxes[0].x0) / boxes[0].width,
                plot_h_in=plot_h_in,
            )
            drawn.append(
                {
                    "metric": key,
                    "segments": [list(seg) for seg in segments],
                    "reference": column["reference"],
                }
            )
            continue

        _assert_drawable(rows, key, (column["xlim"],))
        ax = fig.add_subplot(grid[0, index])
        _shade_subordinate(ax, groups, positions)
        _plot_points(ax, rows, positions, key)
        _finish_axis(
            ax,
            xlim=column["xlim"],
            ticks=column["ticks"],
            reference=column["reference"],
            ylim=(y_lo, y_hi),
        )
        _place_xlabel(ax, column["xlabel"], x_axes=0.5, plot_h_in=plot_h_in)
        drawn.append(
            {
                "metric": key,
                "segments": [list(column["xlim"])],
                "reference": column["reference"],
            }
        )

        if index == 0:
            ax.set_yticks(
                [positions[row["row"]] for row in rows],
                [L.tick_label(row["row"]) for row in rows],
            )
            for label in ax.get_yticklabels():
                label.set_linespacing(L.WRAPPED_TICK_LINESPACING)
            for label, row in zip(ax.get_yticklabels(), rows, strict=True):
                if row["subordinate"]:
                    label.set_color(SUBORDINATE)
            blended = blended_transform_factory(ax.transAxes, ax.transData)
            for group_label, y in kickers:
                ax.text(
                    0.0,
                    y,
                    group_label.upper(),
                    transform=blended,
                    ha="left",
                    va="center",
                    color=MUTED,
                    fontsize=11.5,
                    fontweight=750,
                )
            panel_header(
                ax,
                "",
                _KICKER,
                _TITLE,
                kicker_y=1.0 + _KICKER_OFF_IN / plot_h_in,
                title_y=1.0 + _TITLE_OFF_IN / plot_h_in,
            )
        else:
            ax.set_yticks([positions[row["row"]] for row in rows], [""] * len(rows))

    displayed = {
        "row_order": [row["row"] for row in rows],
        "groups": [
            {"group": label, "rows": list(names), "subordinate": subordinate}
            for label, names, subordinate in groups
        ],
        "columns": drawn,
        "rows": [
            {
                "row": row["row"],
                "group": row["group"],
                "subordinate": row["subordinate"],
                "n_pairs": row["n_pairs"],
                "source": row["source"],
                **{
                    field: row[field]
                    for field in ("direction", "direction_ci95", "magnitude", "magnitude_ci95")
                },
                "twoway": row["twoway"],
                "twoway_ci95": row["twoway_ci95"],
            }
            for row in rows
        ],
    }
    return fig, include_frac, displayed


# ---------------------------------------------------------------------------
# Provenance + export
# ---------------------------------------------------------------------------


def _write_metadata(
    *,
    stem: Path,
    outputs: dict,
    title: str,
    subject: str,
    sources: list[Path],
    displayed_data: dict,
) -> Path:
    """Sidecar in the Section 4.2 shape, attributed to this script."""
    metadata = stem.with_suffix(".meta.json")
    metadata.write_text(
        json.dumps(
            {
                "status": "Results Section 4.2 manuscript figure",
                "style_version": STYLE_VERSION,
                "plotting_script": SCRIPT,
                "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
                "reproduction_command": f"uv run python {SCRIPT}",
                "title": title,
                "subject": subject,
                "git": _git_state(),
                "sources": [
                    {"path": _display_path(path), "sha256": _sha256(path)} for path in sources
                ],
                "render": outputs["record"],
                "displayed_data": displayed_data,
                "output_sha256": {
                    kind: _sha256(path) for kind, path in outputs.items() if isinstance(path, Path)
                },
            },
            indent=2,
        )
        + "\n"
    )
    return metadata


def render(variant: str, out_dir: Path, banked_rows: dict[str, dict], banked: dict) -> list[Path]:
    fig, include_frac, displayed = make_figure(variant, banked_rows)
    displayed = {
        **displayed,
        "map": banked["map"],
        "bootstrap": banked["bootstrap"],
        "metrics": banked["metrics"],
        "caveat": banked["caveat"],
    }
    title = "Per-element answer shift: direction, size, and two-way discrimination"
    stem = out_dir / variant
    outputs = save_c2a_figure(
        fig,
        stem,
        title=title,
        subject=VARIANTS[variant]["subject"],
        creator=SCRIPT,
        include_width=include_frac,
    )
    metadata = _write_metadata(
        stem=stem,
        outputs=outputs,
        title=title,
        subject=VARIANTS[variant]["subject"],
        sources=[ELEMENT_SHIFT_SOURCE],
        displayed_data=displayed,
    )
    plt.close(fig)
    return [outputs["pdf"], outputs["png"], outputs["grayscale"], metadata]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(VARIANTS),
        help="render one variant; repeat to render a named subset",
    )
    args = parser.parse_args()

    set_c2a_style()
    banked_rows, banked = _load_rows()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.only or sorted(VARIANTS):
        for path in render(variant, args.out_dir, banked_rows, banked):
            print(_display_path(path))


if __name__ == "__main__":
    main()
