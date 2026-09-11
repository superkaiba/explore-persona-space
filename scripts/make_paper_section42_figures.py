#!/usr/bin/env python3
"""Render the publication figures used in Results Section 4.2.

Ten figures: the SAE feature-property panels, the minimal-pair
predicted-over-observed shift-size figure, the four combined panel figures, the
per-element answer-shift figure and its appendix slot companion, and the two
appendix companions (refusal swaps by class, and held-out R-squared by
answer-variance rank).  (The qualitative retrieval-failure
figure c3_qualitative_discrimination is produced by
scripts/issue1901_qualitative_retrieval_failures.py.) The
script is plot-only: it reads checked-in summaries and per-pair records,
performs a deterministic bootstrap only for the one-word pilot intervals that
were not banked in its summary, and writes vector PDF, color PNG, grayscale PNG,
and provenance JSON for each figure.

Two inputs are banked by their own rebuild scripts rather than recomputed here:
eval_results/issue_1901/section42_panels.json by
scripts/issue1901_section42_panel_data.py, and
eval_results/issue_2564/section42_element_shifts.json by
scripts/issue2564_element_shift_rows.py.  Rebuild those when their upstream
captures change.  Both need staging mounts this script does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE the matplotlib/numpy imports below. On the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, and
# the BLAS pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    PAPER,
    ROLES,
    STYLE_VERSION,
    better_label,
    c2a_figure,
    canvas_width_in,
    legend_kicker,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.analysis import c2a_row_labels as L  # noqa: E402


DEFAULT_OUT = ROOT / "figures/paper"
# Decoder-direction DV (the paper's primary SAE analysis from 2026-09-08); the
# activation-target twin is plot4_redesign.json in the same directory.
SAE_SOURCE = ROOT / "eval_results/issue_1482/plot4_redesign/plot4_decoder_direction.json"
MINPAIR_SOURCE = ROOT / "eval_results/issue_2564/minpair_delta.json"
PERSONA_SOURCE = ROOT / "eval_results/issue_2564/floor-failed-reelicitation/minpair_delta_ffr.json"
ONEWORD_SOURCE = ROOT / "eval_results/issue_2564/lang_oneword_pilot/summary.json"
ONEWORD_PAIRS = ROOT / "eval_results/issue_2564/lang_oneword_pilot/perpair.jsonl"
SPECTRUM_SOURCE = ROOT / "eval_results/issue_779/plot3_redesign/plot3_redesign.json"
# Panels C and D of the Section 4.2 main figure: retrieval-failure shift sizes
# against the candidate-pool background, and variance explained per controlled
# change.  Built by the consolidation step recorded in the file's own provenance.
SECTION42_PANELS = ROOT / "eval_results/issue_1901/section42_panels.json"
# Per-element answer-shift rows of the Section 4.2 element figure and its
# appendix slot companion, banked by scripts/issue2564_element_shift_rows.py.
ELEMENT_SHIFT_SOURCE = ROOT / "eval_results/issue_2564/section42_element_shifts.json"
# Nested-dictionary tier concordance, banked by scripts/issue1482_tier_concordance.py.
# A different population from the five feature properties, scored on the same
# statistic, so it joins the property panel as its own group.
TIER_CONCORDANCE_SOURCE = ROOT / "eval_results/issue_1482/tier_concordance.json"
SVMP_DIR = Path(os.environ.get("C2A_SVMP_DIR", ROOT / "eval_results/issue_2617/svmp_verbharm"))

LINEAR = ROLES["linear"].color
# Controls / null references take the paper-wide control role (muted gray).
CONTROL = ROLES["control"].color


def _tint(color: str, amount: float) -> str:
    """Mix ``color`` toward the paper background; 0 keeps it, 1 returns white."""
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be between 0 and 1, got {amount}")
    base = np.asarray(to_rgb(color))
    return to_hex(base + (np.asarray(to_rgb(PAPER)) - base) * amount)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _write_metadata(
    *,
    stem: Path,
    outputs: dict[str, Path | dict],
    title: str,
    subject: str,
    sources: list[Path],
    displayed_data: dict,
) -> Path:
    metadata = stem.with_suffix(".meta.json")
    metadata.write_text(
        json.dumps(
            {
                "status": "Results Section 4.2 manuscript figure",
                "style_version": STYLE_VERSION,
                "plotting_script": "scripts/make_paper_section42_figures.py",
                "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
                "reproduction_command": "uv run python scripts/make_paper_section42_figures.py",
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


def _save(
    fig: plt.Figure,
    out_dir: Path,
    stem_name: str,
    *,
    title: str,
    subject: str,
    include_frac: float,
    sources: list[Path],
    displayed_data: dict,
) -> dict[str, Path | dict]:
    stem = out_dir / stem_name
    outputs = save_c2a_figure(
        fig,
        stem,
        title=title,
        subject=subject,
        creator="scripts/make_paper_section42_figures.py",
        include_width=include_frac,
    )
    metadata = _write_metadata(
        stem=stem,
        outputs=outputs,
        title=title,
        subject=subject,
        sources=sources,
        displayed_data=displayed_data,
    )
    return {**outputs, "metadata": metadata}


def _sae_data() -> dict:
    source = json.loads(SAE_SOURCE.read_text())
    all_rows = source["left_panel"]["rows"]
    rendered = source["left_panel"]["rendered_labels"]
    rows = all_rows[: len(rendered)]
    assert [row["label"] for row in rows] == rendered, "rendered prefix drifted"
    assert rows[0]["banked_name"] == "Variance explained in answer space", rows[0]["banked_name"]
    assert rows[-1]["banked_name"] == "Content type: topic", rows[-1]["banked_name"]

    rows = [{**row, "kind": "forward-selected association"} for row in rows]
    tiers = []
    for key, label in (("0", "Coarsest"), ("1", "Middle"), ("2", "Finest")):
        cell = source["right_panel"]["per_tier"][key]
        tiers.append(
            {
                "label": label,
                "n": int(cell["n"]),
                "median": float(cell["median_adjusted"]),
                "q25": float(cell["q25_adjusted"]),
                "q75": float(cell["q75_adjusted"]),
            }
        )
    return {
        "properties": rows,
        "dv": source["left_panel"]["dv"],
        "tiers": tiers,
        "spearman_raw": float(source["right_panel"]["spearman_tier_r2_raw"]),
        "spearman_adjusted": float(source["right_panel"]["spearman_tier_r2_activity_centered"]),
        "centering": source["right_panel"]["centering"],
    }


# A second group of rows sits below the properties, separated by this many row
# heights.  The gap carries the group's own kicker, so the break states its
# reason instead of leaving the reader to infer one.
_GROUP_GAP_ROWS = 1.0
# Fill for that group: the property-bar hue mixed toward paper, so the two
# groups stay apart once the color is thrown away.
_GROUP_FILL = _tint(LINEAR, 0.55)


def _draw_property_rows(
    ax: plt.Axes,
    properties: list[dict],
    *,
    xlabel: str,
    extra_group: dict | None = None,
    xticks: list[float] | None = None,
) -> None:
    """Feature-property concordance as one horizontal bar per property.

    Everything inside the axes: the bars, the zero line, the property names down
    the left edge, the axis range and the axis treatment.  A negative
    association is drawn hollow and hatched so its sign survives the grayscale
    audit.  Panel furniture (kicker and title) stays with the caller, so the
    same drawer serves the SAE figure's left panel and panel A of
    ``c3_features_and_shifts``.

    ``xticks`` pins the tick locations; the default five fit only a wide panel.

    ``extra_group`` carries rows measured on a DIFFERENT population under the
    same statistic.  They are pushed below a one-row gap, backed by the shaded
    stripe the lower panels already use for row groups, filled in a lighter tone
    of the bar hue, and headed by their own kicker in the gap.  Three of those
    four cues survive the grayscale audit.  Left out, the drawer behaves exactly
    as it did before the argument existed, so the SAE figure is untouched.
    """
    y = np.arange(len(properties))[::-1]
    labels = [row["label"] for row in properties]
    if extra_group is not None:
        y = y + (len(extra_group["rows"]) + _GROUP_GAP_ROWS)
    values = np.asarray([row["value"] for row in properties])
    bars = ax.barh(y, values, height=0.58, color=LINEAR, edgecolor=LINEAR, linewidth=1.2)
    for bar, value in zip(bars, values, strict=True):
        if value < 0:
            bar.set_facecolor(PAPER)
            bar.set_hatch("////")
    if extra_group is not None:
        extra_rows = extra_group["rows"]
        extra_y = np.arange(len(extra_rows))[::-1].astype(float)
        gap_center = extra_y[0] + 0.5 + _GROUP_GAP_ROWS / 2
        ax.axhspan(
            extra_y[-1] - 0.5,
            extra_y[0] + 0.5 + _GROUP_GAP_ROWS,
            color=GRID,
            alpha=0.20,
            zorder=0,
            lw=0,
        )
        extra_values = np.asarray([row["value"] for row in extra_rows])
        extra_bars = ax.barh(
            extra_y,
            extra_values,
            height=0.58,
            color=_GROUP_FILL,
            edgecolor=LINEAR,
            linewidth=1.2,
        )
        for bar, value in zip(extra_bars, extra_values, strict=True):
            if value < 0:
                bar.set_facecolor(PAPER)
                bar.set_hatch("////")
        # x sits just inside the left limit set below, so the kicker starts at
        # the plot box edge rather than hanging into the label gutter.
        ax.text(
            -0.163,
            gap_center,
            extra_group["kicker"].upper(),
            ha="left",
            va="center",
            color=MUTED,
            fontsize=11.5,
            fontweight=750,
        )
        y = np.concatenate([y, extra_y])
        labels = labels + [row["label"] for row in extra_rows]
        ax.set_ylim(extra_y[-1] - 0.62, float(y.max()) + 0.62)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.17, 0.37)
    # ``xticks`` pins the tick locations for a narrow panel, where the default
    # five would overlap; omitted, the drawer behaves as it did before.
    ax.set_xticks(np.arange(-0.1, 0.31, 0.1) if xticks is None else xticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:+.1f}" if x else "0"))
    ax.set_xlabel(xlabel)
    style_axis(ax, grid_axis="x")


def make_sae_figure(data: dict) -> tuple[plt.Figure, float]:
    fig, include_frac = c2a_figure("full", aspect=0.45)
    grid = fig.add_gridspec(1, 2, left=0.315, right=0.985, top=0.75, bottom=0.16, wspace=0.30)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    _draw_property_rows(
        ax_left, data["properties"], xlabel="Concordance with feature $R^2$, above chance"
    )
    panel_header(
        ax_left,
        "A",
        "Forward-selected associations",
        "Feature-property concordance",
        kicker_y=1.21,
        title_y=1.08,
    )

    tiers = data["tiers"]
    x = np.arange(3)
    med = np.asarray([row["median"] for row in tiers])
    lo = med - np.asarray([row["q25"] for row in tiers])
    hi = np.asarray([row["q75"] for row in tiers]) - med
    ax_right.errorbar(
        x,
        med,
        yerr=np.vstack([lo, hi]),
        fmt="o",
        color=LINEAR,
        markerfacecolor=LINEAR,
        markeredgecolor=LINEAR,
        markersize=9,
        capsize=7,
        capthick=2,
        elinewidth=2.4,
        lw=0,
        zorder=3,
    )
    ax_right.axhline(0, color=INK, lw=1.2)
    ax_right.set_xticks(x, [row["label"] for row in tiers])
    ax_right.set_xlim(-0.45, 2.45)
    ax_right.set_ylim(-0.12, 0.22)
    ax_right.set_yticks(np.arange(-0.1, 0.21, 0.1))
    ax_right.set_ylabel("Activity-adjusted feature $R^2$")
    ax_right.set_xlabel("Nested SAE tier")
    style_axis(ax_right, grid_axis="y")
    panel_header(
        ax_right,
        "",
        "B",
        "Median feature $R^2$ by tier",
        kicker_y=1.21,
        title_y=1.08,
    )
    return fig, include_frac


def _read_jsonl(path: Path) -> list[dict]:
    # str.split("\n"), not splitlines(): JSON strings may contain U+2028/U+2029, which
    # splitlines() treats as row breaks and would corrupt the record.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def _bootstrap_slope(
    all_rows: list[dict], axis: str, *, n_boot: int = 10_000, seed: int = 21620
) -> dict:
    """Through-origin norm slope + 95% pair-bootstrap CI for one pilot axis."""
    rows = [row for row in all_rows if row["axis"] == axis]
    obs = np.asarray([row["norm_obs_tail_L19"] for row in rows], dtype=float)
    pred = np.asarray([row["norm_pred_arm_779ce"] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    slopes = np.sum(obs[draws] * pred[draws], axis=1) / np.sum(obs[draws] ** 2, axis=1)
    return {
        "n": len(rows),
        "slope": float(np.sum(obs * pred) / np.sum(obs**2)),
        "slope_ci95": np.quantile(slopes, [0.025, 0.975]).tolist(),
        "bootstrap": {"unit": "pair", "draws": n_boot, "seed": seed},
    }


def _pair_shift_data() -> list[dict]:
    parent = json.loads(MINPAIR_SOURCE.read_text())
    persona = json.loads(PERSONA_SOURCE.read_text())
    oneword = json.loads(ONEWORD_SOURCE.read_text())
    pilot_rows = _read_jsonl(ONEWORD_PAIRS)

    one = _bootstrap_slope(pilot_rows, "query_content_oneword")
    lang = _bootstrap_slope(pilot_rows, "answer_language")
    assert one["n"] == 24 and lang["n"] == 72
    for axis, computed in (("query_content_oneword", one), ("answer_language", lang)):
        assert np.isclose(computed["slope"], oneword["calibration_slope"]["arm_779ce"][axis])

    # The banked ``n_primary_pairs`` counts every primary-class pair of the axis,
    # but ``axis_slope`` is fitted on the headline subset (primary class AND the
    # changed instruction followed in at least 70% of the rollouts of both
    # contexts).  Only format differs between the two, 36 of 120, so reporting
    # n_primary_pairs alongside the slope mislabels that one row.  Recount from
    # the per-pair records and assert the slope reproduces on that subset.
    def headline_rows(records: list[dict], axis: str, primary_class: str) -> list[dict]:
        return [
            row
            for row in records
            if row["axis"] == axis and row["pair_class"] == primary_class and row["in_headline_70"]
        ]

    parent_rows = _read_jsonl(ROOT / "eval_results/issue_2564/perpair.jsonl")
    persona_rows = _read_jsonl(
        ROOT / "eval_results/issue_2564/floor-failed-reelicitation/perpair_ffr.jsonl"
    )

    def banked(source: dict, axis: str, label: str, records: list[dict]) -> dict:
        cell = source["axes"][axis]
        cal = cell["calibration"]["arm_779ce"]
        rows = headline_rows(records, axis, cell["primary_class"])
        obs = np.asarray([row["norm_obs_tail_L19"] for row in rows], dtype=float)
        pred = np.asarray([row["norm_pred"]["arm_779ce"] for row in rows], dtype=float)
        slope = float(np.sum(pred * obs) / np.sum(obs**2))
        assert np.isclose(slope, cal["axis_slope"]), (axis, slope, cal["axis_slope"])
        return {
            "label": label,
            "axis": axis,
            "n": len(rows),
            "n_primary_pairs": int(cell["n_primary_pairs"]),
            "slope": float(cal["axis_slope"]),
            "slope_ci95": [float(v) for v in cal["axis_slope_ci95"]],
        }

    rows = [
        banked(parent, "format", "Output\nformat", parent_rows),
        banked(persona, "persona", "Persona", persona_rows),
        banked(parent, "register", "Tone", parent_rows),
        {"label": "Answer\nlanguage", "axis": "answer_language", **lang},
        banked(parent, "query_content", "Question\ntopic", parent_rows),
        {"label": "One-word\ntopic", "axis": "query_content_oneword", **one},
    ]
    rows.sort(key=lambda row: -row["slope"])
    return rows


def _draw_pair_shift_axes(
    ax: plt.Axes,
    rows: list[dict],
    *,
    ylabel: str = "Predicted / observed shift size",
    stagger_xticklabels: bool = False,
) -> None:
    """Errorbar panel content shared by the standalone and combined figures.

    Everything inside the axes: the dashed reference line at one, the per-element
    point estimates with 95% bootstrap CIs, the value labels, the two-line x tick
    labels, and the axis treatment.  Panel furniture (kicker and title) stays
    with the caller.  ``stagger_xticklabels`` drops every second tick label a
    full label height so the six labels stay legible on a narrow panel (adjacent
    labels need a wider slot than a half-width panel provides).
    """
    x = np.arange(len(rows))
    values = np.asarray([row["slope"] for row in rows])
    ci = np.asarray([row["slope_ci95"] for row in rows])
    yerr = np.vstack([values - ci[:, 0], ci[:, 1] - values])
    ax.axhline(1.0, color=CONTROL, lw=1.7, linestyle=(0, (5, 4)), zorder=1)
    ax.errorbar(
        x,
        values,
        yerr=yerr,
        fmt="o",
        color=LINEAR,
        markerfacecolor=LINEAR,
        markeredgecolor=LINEAR,
        markersize=9,
        capsize=6,
        capthick=2,
        elinewidth=2.3,
        lw=0,
        zorder=3,
    )
    for xi, value, hi in zip(x, values, ci[:, 1], strict=True):
        ax.text(
            xi, hi + 0.035, f"{value:.2f}", fontsize=14, fontweight=650, ha="center", va="bottom"
        )
    ax.set_xticks(x, [row["label"] for row in rows])
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(0.45, 1.42)
    ax.set_yticks([0.5, 0.75, 1.0, 1.25])
    ax.set_ylabel(ylabel)
    style_axis(ax, grid_axis="y")
    if stagger_xticklabels:
        # Two-line labels at 17 pt are ~42 pt tall; +44 pt of pad fully separates rows.
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(tick.get_pad() + 44)


def make_pair_shift_figure(rows: list[dict]) -> tuple[plt.Figure, float]:
    fig, include_frac = c2a_figure("wide", aspect=0.53)
    # No model, read layer or error-bar definition on the canvas: the caption
    # carries them, so the row the second kicker line used sits in the axes
    # instead (top 0.78 -> 0.82) and the furniture keeps its inch offsets.
    grid = fig.add_gridspec(1, 1, left=0.115, right=0.98, top=0.82, bottom=0.155)
    ax = fig.add_subplot(grid[0, 0])
    _draw_pair_shift_axes(ax, rows)
    panel_header(
        ax,
        "",
        "Controlled minimal pairs",
        "Predicted over observed answer-shift size by element",
        kicker_y=1.188,
        title_y=1.070,
    )
    return fig, include_frac


# Panel-A label offsets for the combined figure (panel geometry differs from the
# standalone wide canvas, so the shared PAPER_OFFSETS do not transfer).
# (dx, dy) in points from the data point, then ha/va.
_COMBINED_SPECTRUM_OFFSETS = {
    "evil": (69, 14, "left", "bottom"),
    "sycophancy": (56, 6, "right", "bottom"),
    "hallucination": (-45, 1, "center", "top"),
    "refusal": (14, 9, "left", "bottom"),
    "assistant axis": (3, -47, "right", "top"),
    "casualness": (6, -46, "left", "top"),
    "impoliteness": (-35, -2, "right", "bottom"),
    "harmful compliance": (-45, -33, "center", "bottom"),
    "correctness (math)": (79, 11, "center", "center"),
    "correctness (MMLU-Pro)": (81, -3, "left", "center"),
    "correctness (code)": (102, -14, "left", "center"),
}


def make_directions_and_pairs_figure(spectrum: dict, rows: list[dict]) -> tuple[plt.Figure, float]:
    """Combined manuscript figure: A = useful-directions spectrum, B = pair shifts.

    Panel A reuses the ax-level spectrum drawer from
    ``scripts/issue779_plot3_redesign.py`` (same plotted values as the standalone
    ``c3_persona_direction_spectrum``); panel B reuses the errorbar panel of
    ``c3_pair_shifts``.  Full include width, one figure-level eyebrow, per-panel
    kickers and descriptive titles.
    """
    # Sibling script import (lazy: pulls torch and the #779 analysis modules).
    from issue779_plot3_redesign import draw_spectrum_panel

    fig, include_frac = c2a_figure("full", aspect=0.40)
    # No figure-level provenance eyebrow (model and read layer live in the
    # caption), so the eyebrow row and the second kicker line go to the axes
    # and the panel furniture keeps its inch offsets. 0.80 is the last safe
    # step: above it panel A picks up a wider y tick set, which pushes the
    # rotated y label past the left canvas edge and off-scales the export.
    grid = fig.add_gridspec(
        1, 2, width_ratios=[0.6, 0.4], left=0.065, right=0.985, top=0.80, bottom=0.26, wspace=0.18
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    draw_spectrum_panel(ax_a, spectrum, offsets=_COMBINED_SPECTRUM_OFFSETS, legend_frame=True)
    panel_header(
        ax_a,
        "",
        "A",
        title="Per-direction held-out $R^2$ vs variance rank",
        kicker_y=1.148,
        title_y=1.051,
    )
    _draw_pair_shift_axes(ax_b, rows, ylabel="Predicted / observed", stagger_xticklabels=True)
    panel_header(
        ax_b,
        "B",
        "Controlled minimal pairs",
        title="Answer-shift size by element",
        kicker_y=1.148,
        title_y=1.051,
    )
    return fig, include_frac


# ---------------------------------------------------------------------------
# Section 4.2 main figure (four panels, one per claim of the section)
# ---------------------------------------------------------------------------

# Panel-A label offsets: the spectrum sits in the top-left cell of a two-row
# canvas, so its aspect is close to the c3_directions_and_pairs panel and the
# combined offsets transfer with only the crowded upper-right cluster retuned.
_INFO_SPECTRUM_OFFSETS = dict(_COMBINED_SPECTRUM_OFFSETS)
# The upper-left cluster fans further out than in c3_directions_and_pairs, whose
# offsets leave "sycophancy refusal evil" and "assistant axis casualness" reading
# as single phrases.
_INFO_SPECTRUM_OFFSETS.update(
    {
        "evil": (104, 30, "left", "bottom"),
        "refusal": (8, 30, "right", "bottom"),
        "casualness": (40, -46, "left", "top"),
        "assistant axis": (-6, -47, "right", "top"),
    }
)

# Short row labels for panel D.  The full element names live in the prose; a
# half-width panel cannot carry them without stealing the neighbouring axes.
# Panel-B tick labels.  A quarter-width panel cannot carry the property names,
# which are spelled out in the prose and in the appendix table.
_PROPERTY_TICK_LABELS = {
    "variance along decoder direction": "direction\nvariance",
    "speaker identity / disposition": "speaker\nidentity",
    "promotes specific output tokens": "promotes\ntokens",
    "suppresses specific output tokens": "suppresses\ntokens",
    "topic content": "topic\ncontent",
}

_VE_ROW_LABELS = {
    "Output format": "Output format",
    "Persona": "Persona",
    "Tone": "Tone",
    "Answer language": "Language",
    "Question topic": "Topic",
    "One-word topic": "One word",
    "Refusal flip": "Refusal flip",
    "Refusal non-flip": "Refusal same",
}


def _spectrum_displayed(spectrum: dict) -> dict:
    """The plotted slice of the answer-variance spectrum, for a provenance sidecar."""
    return {
        "layer": spectrum.get("layer", 19),
        "ranks_evaluated": spectrum["ranks_evaluated"],
        "r2_by_rank": spectrum["r2_by_rank"],
        "random_directions": spectrum["random_directions"],
        "directions": {
            name: {
                "plotted_rank_1based": entry["plotted_rank_1based"],
                "heldout_r2": entry["heldout_r2"],
            }
            for name, entry in spectrum["directions"].items()
        },
    }


def _information_data() -> dict:
    """Assemble the four panels from checked-in sources."""
    panels = json.loads(SECTION42_PANELS.read_text())
    sae = _sae_data()
    spectrum = json.loads(SPECTRUM_SOURCE.read_text())

    rows = []
    for element in panels["panel_d"]["elements"]:
        rows.append(
            {
                "label": _VE_ROW_LABELS[element["label"]],
                "element": element["label"],
                "n": int(element["n"]),
                "ve": float(element["ve"]),
                "ve_size_corrected": float(element["ve_size_corrected"]),
                "ratio": float(element["ratio"]),
                "kind": element["kind"],
            }
        )
    rows.sort(key=lambda row: row["ve"])
    return {
        "panel_a": _spectrum_displayed(spectrum),
        "panel_b": {"properties": sae["properties"], "dv": sae["dv"]},
        "panel_c": panels["panel_c"],
        "panel_d": {
            "elements": rows,
            "natural_reference_ve": float(panels["panel_d"]["natural_reference_ve"]),
            "natural_reference_note": panels["panel_d"]["natural_reference_note"],
            "order": "ascending variance explained",
        },
        "_spectrum": spectrum,
    }


def _draw_property_panel(ax: plt.Axes, properties: list[dict]) -> None:
    """Panel B: forward-selected SAE feature properties, strongest first."""
    labels = [_PROPERTY_TICK_LABELS[row["label"]] for row in properties]
    values = np.asarray([row["value"] for row in properties])
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.62, color=LINEAR, edgecolor=LINEAR, linewidth=1.2)
    for bar, value in zip(bars, values, strict=True):
        if value < 0:
            bar.set_facecolor(PAPER)
            bar.set_hatch("////")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_ylim(-0.17, 0.37)
    ax.set_yticks(np.arange(-0.1, 0.31, 0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:+.1f}" if v else "0"))
    ax.set_ylabel("Concordance with feature $R^2$")
    style_axis(ax, grid_axis="y")
    # Two-line labels need a full label height of separation on a narrow panel.
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(tick.get_pad() + 40)


def _draw_failure_panel(ax: plt.Axes, panel: dict) -> None:
    """Panel C: where the retrieval failures sit on the pool's shift-size plane."""
    bg_ctx = np.asarray(panel["background"]["ctx"], dtype=float)
    bg_ans = np.asarray(panel["background"]["ans"], dtype=float)
    # Thin the cloud for the vector export; the density is already saturated.
    step = max(1, len(bg_ctx) // 9000)
    ax.scatter(
        bg_ctx[::step],
        bg_ans[::step],
        s=2.0,
        color=MUTED,
        alpha=0.13,
        linewidths=0,
        zorder=1,
        label="pool context pairs",
    )
    slope = float(panel["background_slope"])
    xs = np.array([0.0, float(bg_ctx.max()) * 1.02])
    ax.plot(xs, slope * xs, color=MUTED, lw=1.6, linestyle=(0, (5, 4)), zorder=2)
    floor = float(panel["rollout_floor"])
    ax.axhline(floor, color=INK, lw=1.1, linestyle=(0, (1, 2.2)), zorder=2)
    ax.text(xs[1] * 0.80, floor, "noise floor ", ha="right", va="bottom", color=INK)

    fails = panel["failures"]
    ctx = np.asarray([row["ctx"] for row in fails], dtype=float)
    ans = np.asarray([row["ans"] for row in fails], dtype=float)
    ax.scatter(
        ctx,
        ans,
        s=13,
        color=LINEAR,
        linewidths=0,
        zorder=5,
        label="retrieval failures",
    )
    ax.set_xlim(0, xs[1])
    ax.set_ylim(0, float(max(bg_ans.max(), ans.max())) * 1.04)
    ax.set_xlabel("Context-vector shift")
    ax.set_ylabel("Answer-vector shift")
    style_axis(ax)
    ax.legend(loc="upper left", markerscale=2.2, handletextpad=0.5)


def _draw_variance_explained_panel(ax: plt.Axes, panel: dict) -> None:
    """Panel D: variance explained per controlled change, before and after the
    predicted size is rescaled to its optimum."""
    rows = panel["elements"]
    y = np.arange(len(rows))
    ve = np.asarray([row["ve"] for row in rows])
    corrected = np.asarray([row["ve_size_corrected"] for row in rows])
    ax.barh(
        y,
        corrected,
        height=0.68,
        facecolor=PAPER,
        edgecolor=LINEAR,
        linewidth=1.6,
        zorder=2,
        label="size corrected",
    )
    ax.barh(
        y,
        ve,
        height=0.40,
        color=LINEAR,
        edgecolor=LINEAR,
        linewidth=0,
        zorder=3,
        label="as predicted",
    )
    reference = float(panel["natural_reference_ve"])
    ax.axvline(reference, color=MUTED, lw=1.6, linestyle=(0, (5, 4)), zorder=1)
    ax.text(reference, -0.72, "natural pairs ", color=MUTED, ha="right", va="center")
    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.set_yticks(y, [row["label"] for row in rows])
    ax.set_ylim(-1.05, len(rows) - 0.35)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel(better_label("Variance explained"))
    style_axis(ax, grid_axis="x")
    ax.legend(loc="upper left", handlelength=1.4, handletextpad=0.5, labelspacing=0.25)


def make_directions_and_features_figure(data: dict) -> tuple[plt.Figure, float]:
    """Which parts of an answer the map predicts: variance rank, then SAE properties."""
    from issue779_plot3_redesign import draw_spectrum_panel

    fig, include_frac = c2a_figure("full", aspect=0.42)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.60, 0.40],
        left=0.062,
        right=0.988,
        # No figure-level provenance eyebrow (model and read layer live in
        # the caption), so its row goes to the axes: 0.76 -> 0.83, with the
        # panel furniture keeping its inch offsets. 0.83 is the last safe
        # step: above it panel A picks up a wider y tick set, which pushes
        # the rotated y label past the left canvas edge.
        top=0.83,
        bottom=0.245,
        wspace=0.22,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    draw_spectrum_panel(ax_a, data["_spectrum"], offsets=_INFO_SPECTRUM_OFFSETS, legend_frame=True)
    # Headroom for the top label row: the drawer's 1.05 ceiling leaves "refusal"
    # touching either its point or the panel title.
    ax_a.set_ylim(-0.32, 1.20)
    panel_header(
        ax_a,
        "",
        "A",
        title="Held-out $R^2$ by variance rank",
        kicker_y=1.141,
        title_y=1.048,
    )
    _draw_property_panel(ax_b, data["panel_b"]["properties"])
    panel_header(
        ax_b,
        "B",
        "120,716 SAE features",
        title="Concordance by property",
        kicker_y=1.141,
        title_y=1.048,
    )
    return fig, include_frac


def make_failures_and_shifts_figure(data: dict) -> tuple[plt.Figure, float]:
    """Which contexts the map fails on, and how well it predicts a controlled change."""
    fig, include_frac = c2a_figure("full", aspect=0.44)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.52, 0.48],
        left=0.070,
        right=0.988,
        # No figure-level provenance eyebrow (model and read layer live in
        # the caption), so its row goes to the axes: 0.755 -> 0.85, with the
        # panel furniture keeping its inch offsets.
        top=0.85,
        bottom=0.155,
        wspace=0.30,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    _draw_failure_panel(ax_a, data["panel_c"])
    panel_header(
        ax_a,
        "A",
        "10,000-context candidate pool",
        title="Retrieval failures",
        kicker_y=1.138,
        title_y=1.047,
    )
    _draw_variance_explained_panel(ax_b, data["panel_d"])
    panel_header(
        ax_b,
        "B",
        "Controlled context pairs",
        title="Variance explained per element",
        kicker_y=1.138,
        title_y=1.047,
    )
    return fig, include_frac


# ---------------------------------------------------------------------------
# Section 4.2 per-element answer shifts (main panel + appendix slot companion)
# ---------------------------------------------------------------------------

# The main element-shift panel reads as five bands of related rows.  Every second
# band carries a shaded stripe so the eye groups them without a second legend.
_ELEMENT_SHIFT_GROUPS = (
    ("identity", (L.TONE, L.PERSONA)),
    ("format", (L.OUTPUT_FORMAT,)),
    ("content", (L.QUESTION_TOPIC, L.ONE_WORD_TOPIC)),
    ("refusal_word", (L.REFUSAL_REVERSES_INTENT, L.REFUSAL_HOLDS_INTENT)),
    ("refusal_framing", (L.REFUSAL_REVERSES_FRAMING, L.REFUSAL_HOLDS_FRAMING)),
)

# Header and footer of the element-shift panel, pinned in inches: the canvas
# grows with the row count instead of squeezing the rows, so the wrapped
# refusal labels get their pitch without narrowing the panels or moving the
# printed type size.
_ELEMENT_HEADER_IN = 1.15
_ELEMENT_FOOTER_IN = 0.89
_ELEMENT_KICKER_OFF_IN = 0.41
_ELEMENT_TITLE_OFF_IN = 0.15

# Appendix companion: the main-panel row the slots decompose, then the slots.
_SLOT_ROW_ORDER = (L.ONE_WORD_TOPIC, L.SLOT_SUBJECT, L.SLOT_VERB, L.SLOT_OBJECT)
_SLOT_ROW_LABELS = {
    L.ONE_WORD_TOPIC: L.ONE_WORD_TOPIC,
    L.SLOT_SUBJECT: "…subject swapped",
    L.SLOT_VERB: "…verb swapped",
    L.SLOT_OBJECT: "…object swapped",
}


def _element_shift_data() -> dict:
    """The banked element rows, ordered as each of the two figures draws them."""
    banked = json.loads(ELEMENT_SHIFT_SOURCE.read_text())
    panel = {row["row"]: row for row in banked["panel_rows"]}
    appendix = {row["row"]: row for row in banked["appendix_rows"]}
    order = [label for _group, labels in _ELEMENT_SHIFT_GROUPS for label in labels]
    assert set(order) == set(panel), (sorted(order), sorted(panel))
    return {
        "elements": [panel[label] for label in order],
        "bands": [len(labels) for _group, labels in _ELEMENT_SHIFT_GROUPS],
        "slots": [(panel | appendix)[name] for name in _SLOT_ROW_ORDER],
        "map": banked["map"],
        "bootstrap": banked["bootstrap"],
        "metrics": banked["metrics"],
        "caveat": banked["caveat"],
    }


def _draw_row_metric_panel(
    ax: plt.Axes,
    rows: list[dict],
    *,
    key: str,
    xlabel: str,
    xlim: tuple[float, float],
    reference: float | None = None,
    bands: list[int] | None = None,
    ytick_labels: list[str] | None = None,
    xticks: list[float] | None = None,
) -> None:
    """One column of a row-per-element figure: point estimate plus its 95% interval.

    ``bands`` are consecutive row-group sizes.  Every second group gets a shaded
    stripe.  ``xticks`` pins the tick locations, which keeps the automatic
    locator from placing a label outside ``xlim`` and past the canvas edge.
    Panel furniture (kicker and title) stays with the caller.
    """
    y = np.arange(len(rows))[::-1]
    if bands is not None:
        start = 0
        for index, size in enumerate(bands):
            if index % 2 == 1:
                ax.axhspan(
                    y[start + size - 1] - 0.5,
                    y[start] + 0.5,
                    color=GRID,
                    alpha=0.20,
                    zorder=0,
                    lw=0,
                )
            start += size
    if reference is not None:
        ax.axvline(reference, color=CONTROL, lw=1.6, linestyle=(0, (5, 4)), zorder=1)
    values = np.asarray([row[key] for row in rows])
    ci = np.asarray([row[f"{key}_ci95"] for row in rows])
    xerr = np.vstack([values - ci[:, 0], ci[:, 1] - values]).clip(min=0)
    ax.errorbar(
        values,
        y,
        xerr=xerr,
        fmt="o",
        color=LINEAR,
        markerfacecolor=LINEAR,
        markeredgecolor=LINEAR,
        markersize=7,
        capsize=3,
        capthick=1.6,
        elinewidth=1.8,
        lw=0,
        zorder=3,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    style_axis(ax, grid_axis="x")
    ax.set_xlabel(xlabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_yticks(y, ytick_labels if ytick_labels is not None else [""] * len(rows))
    for label in ax.get_yticklabels():
        label.set_linespacing(L.WRAPPED_TICK_LINESPACING)


def make_element_shifts_figure(data: dict) -> tuple[plt.Figure, float]:
    """What the map keeps when one context element changes: direction, then size."""
    rows = data["elements"]
    # ylim below spans len(rows) + 0.2 units, so the canvas is sized from that.
    plot_h_in = L.WRAPPED_ROW_PITCH_IN * (len(rows) + 0.2)
    height_in = _ELEMENT_HEADER_IN + plot_h_in + _ELEMENT_FOOTER_IN
    fig, include_frac = c2a_figure("full", aspect=height_in / canvas_width_in(1.0))
    height_in = fig.get_figheight()
    plot_h_in = height_in - _ELEMENT_HEADER_IN - _ELEMENT_FOOTER_IN
    grid = fig.add_gridspec(
        1,
        2,
        left=0.285,
        right=0.985,
        top=1.0 - _ELEMENT_HEADER_IN / height_in,
        bottom=_ELEMENT_FOOTER_IN / height_in,
        wspace=0.16,
    )
    columns = (
        ("direction", better_label("Predicted shift direction (cosine)"), (0.15, 1.0), None),
        ("magnitude", "Predicted / observed shift size", (0.55, 1.28), 1.0),
    )
    for index, (key, xlabel, xlim, reference) in enumerate(columns):
        ax = fig.add_subplot(grid[0, index])
        _draw_row_metric_panel(
            ax,
            rows,
            key=key,
            xlabel=xlabel,
            xlim=xlim,
            reference=reference,
            bands=data["bands"],
            # The pair count rides the row label so nothing is placed by hand.
            ytick_labels=(
                [L.tick_label(row["row"], row["n_pairs"]) for row in rows] if index == 0 else None
            ),
        )
        if index == 0:
            panel_header(
                ax,
                "C",
                "controlled minimal pairs",
                "What the map keeps when one context element changes",
                kicker_y=1.0 + _ELEMENT_KICKER_OFF_IN / plot_h_in,
                title_y=1.0 + _ELEMENT_TITLE_OFF_IN / plot_h_in,
            )
    return fig, include_frac


# ---------------------------------------------------------------------------
# Section 4.2 results figure: feature properties beside the per-element shifts,
# with the answer-variance-rank spectrum moved to its own appendix figure.
# ---------------------------------------------------------------------------

# The figure is ONE horizontal row of four panels, so its width budget is the
# binding constraint and is therefore written in inches rather than fractions.
# On the 13.10 in full-width canvas, measured at the pinned tick size:
#
#   panel A keeps its own label column (feature-property names, a different
#   population from the element rows, so the two sets cannot share a gutter);
#   B, C and D share one element-row gutter, with the labels drawn on B.
#
#   Both gutters are cut rather than the panels.  The property names wrap to
#   two lines (widest line "suppresses specific" at 2.23 in, against 3.84 in
#   set on one line), which costs nothing because a single row of four panels
#   gives panel A's six bars the 7.2-row pitch of the element column, about
#   0.48 in each, and a two-line label needs about 0.44 in.  The per-row pair
#   count leaves the element labels (widest line 2.67 in with it, 1.94 in
#   without) and is carried by the caption and the sidecar instead.
#
# Together those two cuts return 2.34 in to the four plot boxes.
_FS_LEFT_MARGIN_IN = 0.06
_FS_A_LABEL_IN = 2.30
_FS_BCD_LABEL_IN = 2.10
_FS_A_WIDTH_IN = 2.45
_FS_COL_GAP_IN = 0.40
_FS_RIGHT_MARGIN_IN = 0.20
# Panel A is the widest of the four: its axis carries five tick labels on a
# signed axis and the separate-dictionary group kicker inside the plot box,
# neither of which the three metric columns have.
# No figure-level provenance eyebrow (model and read layer live in the
# caption), so the header keeps only the panel kicker: 0.20 in of offset plus
# one 13 pt line, and about 0.10 in of top margin.
_FS_HEADER_IN = 0.52
_FS_FOOTER_IN = 1.24
_FS_KICKER_OFF_IN = 0.20
_FS_XLABEL_OFF_IN = 0.34

# Rows drawn by this figure: the two refusal-holds rows are not among them, so
# it keeps its own group tuple rather than sharing the nine-row one above.
_FEATURES_AND_SHIFTS_GROUPS = (
    ("identity", (L.TONE, L.PERSONA)),
    ("format", (L.OUTPUT_FORMAT,)),
    ("content", (L.QUESTION_TOPIC, L.ONE_WORD_TOPIC)),
    ("refusal_reverses", (L.REFUSAL_REVERSES_INTENT, L.REFUSAL_REVERSES_FRAMING)),
)

# Panel A label wraps.  Keyed by the exact banked string, so a renamed label in
# the source JSON raises here instead of silently shipping an unwrapped name
# that widens the gutter and squeezes the three metric columns.
_FS_PROPERTY_LABEL_WRAP = {
    "variance along decoder direction": "variance along\ndecoder direction",
    "speaker identity / disposition": "speaker identity /\ndisposition",
    "promotes specific output tokens": "promotes specific\noutput tokens",
    "suppresses specific output tokens": "suppresses specific\noutput tokens",
    "topic content": "topic content",
    "coarsest nested-dictionary tier": "coarsest nested-\ndictionary tier",
}
# The group kicker sits inside panel A's plot box, so it wraps to the box width
# too.  "Nested" is dropped from it because the row it heads already says
# "nested-dictionary tier"; the feature count and layer are what it adds.
_FS_TIER_KICKER = "SEPARATE DICTIONARY\n16,384 FEATURES"

# Panel D, the two-way discrimination rate, is drawn on a cut axis, following
# scripts/issue2564_element_shifts_three_panel.py, which solved this axis for
# the same seven rows.  Every rate here sits between 0.875 and 1.0 with its
# interval reaching 0.8125, so one linear 0-to-1 axis flattens the rows into a
# stripe and a truncated axis drops the 0.5 chance reference off the left.  The
# axis is therefore two linear segments whose plotted widths are proportional
# to their data spans, so both share ONE scale and a distance means the same
# thing in either; diagonal marks sit on the cut, and a value or interval
# endpoint landing in the omitted range raises rather than being clipped.
_FS_TWOWAY_SEGMENTS = ((0.46, 0.54), (0.78, 1.015))
_FS_TWOWAY_SEGMENT_TICKS = ((0.5,), (0.8, 0.9, 1.0))
_FS_TWOWAY_REFERENCE = 0.5


def _fs_panel_boxes() -> list[tuple[float, float]]:
    """Left and right figure fractions of the four panel boxes, from the inch budget.

    Panel A takes a pinned width; B, C and D split what is left equally, so a
    later change to either gutter moves width into or out of the three metric
    columns rather than silently overrunning the canvas.
    """
    width_in = canvas_width_in(1.0)
    a_left = _FS_LEFT_MARGIN_IN + _FS_A_LABEL_IN
    b_left = a_left + _FS_A_WIDTH_IN + _FS_BCD_LABEL_IN
    remaining = width_in - b_left - _FS_RIGHT_MARGIN_IN - 2 * _FS_COL_GAP_IN
    column_in = remaining / 3.0
    if column_in <= 0:
        raise ValueError(
            f"no width left for the three metric columns on a {width_in:.2f} in canvas; "
            f"gutters and margins already take {b_left + _FS_RIGHT_MARGIN_IN:.2f} in"
        )
    boxes = [(a_left, a_left + _FS_A_WIDTH_IN)]
    left = b_left
    for _ in range(3):
        boxes.append((left, left + column_in))
        left += column_in + _FS_COL_GAP_IN
    return [(lo / width_in, hi / width_in) for lo, hi in boxes]


def _fs_grouped_rows(elements: dict, groups: tuple) -> tuple[list[dict], list[int]]:
    """Banked rows in drawing order plus the group sizes the stripes read."""
    banked = {row["row"]: row for row in elements["elements"]}
    rows: list[dict] = []
    for _group, names in groups:
        for name in names:
            if name not in banked:
                raise KeyError(f"banked panel_rows has no row {name!r}")
            rows.append(banked[name])
    return rows, [len(names) for _group, names in groups]


def _fs_wrap_labels(properties: list[dict]) -> list[dict]:
    """Panel A rows with their labels wrapped to the narrower four-across gutter."""
    wrapped = []
    for row in properties:
        label = row["label"]
        if label not in _FS_PROPERTY_LABEL_WRAP:
            raise KeyError(
                f"no wrap for property label {label!r}; add one to _FS_PROPERTY_LABEL_WRAP "
                "so the label column stays inside its 2.30 in gutter"
            )
        wrapped.append({**row, "label": _FS_PROPERTY_LABEL_WRAP[label]})
    return wrapped


def _fs_place_xlabel(ax: plt.Axes, *, x_axes: float, plot_h_in: float) -> None:
    """One x-label offset below every plot box, so all four labels share a top edge."""
    ax.xaxis.set_label_coords(x_axes, -_FS_XLABEL_OFF_IN / plot_h_in)


def _fs_assert_in_segments(
    rows: list[dict], key: str, segments: tuple[tuple[float, float], ...]
) -> None:
    """Every point and interval endpoint must land inside one drawn segment."""
    for row in rows:
        for value in (row[key], *row[f"{key}_ci95"]):
            if not any(lo <= value <= hi for lo, hi in segments):
                raise ValueError(
                    f"{key}={value} for row {row['row']!r} falls outside the drawn segments "
                    f"{segments}; widen the axis rather than clipping the value"
                )


def _fs_draw_cut_marks(left: plt.Axes, right: plt.Axes) -> None:
    """Diagonal marks on both sides of the cut, as the paper's other cut axes use."""
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
    # The cut is on the x-axis, so the marks sit on the bottom seam; the top
    # edge carries no spine to break.
    left.plot([1], [0], transform=left.transAxes, **mark)
    right.plot([0], [0], transform=right.transAxes, **mark)


def _tier_concordance_group() -> dict:
    """The nested-dictionary tier row that joins the feature-property panel.

    Same statistic and same scale as the five property rows, measured on a
    different dictionary at a different layer under a different matching scheme,
    so the figure gives it its own group rather than letting it pass as a sixth
    property.  The label stays inside the shared 3.84 in gutter (it measures
    3.43 in), so the export crop, and with it the two lower panels, do not move.
    """
    banked = json.loads(TIER_CONCORDANCE_SOURCE.read_text())
    head = banked["headline"]
    assert head["coding"] == "coarsest tier versus the rest", head["coding"]
    universe = banked["universe"]
    return {
        "kicker": (
            f"separate {universe['n_features_scored']:,}-feature nested dictionary"
            f" · layer {universe['layer']}"
        ),
        "rows": [
            {
                "banked_name": head["coding"],
                "label": "coarsest nested-dictionary tier",
                "value": float(head["activity_quintile_matched"]),
                "ci95": [float(bound) for bound in head["activity_quintile_matched_ci95"]],
                "n": int(head["n"]),
                "n_positive": int(head["n_positive"]),
                "kind": "nested-dictionary tier, activity-quintile matched",
            }
        ],
        "universe": universe,
        "matching": banked["matching"],
        "bootstrap": banked["bootstrap"],
        "property_rows_universe": banked["property_panel_universe"],
        "caption_note": banked["caption_note"],
    }


def make_features_and_shifts_figure(
    sae: dict, elements: dict, tier_group: dict
) -> tuple[plt.Figure, float]:
    """Feature-property concordance beside what the map keeps per changed element.

    One horizontal row of four panels: the SAE feature properties, then the
    direction, the size and the two-way discrimination rate of the answer shift
    under one controlled context change.  Seven element rows; the two
    refusal-holds rows are not drawn here.

    Four panels across leave each metric column about 1.8 in wide on the
    13.10 in canvas, with about 2.1 in of pitch from one column's left edge to
    the next.  Descriptive panel titles do not fit that pitch: the three the
    stacked layout carried measure 3.40 in to 4.46 in, so they would overlap
    their neighbours.  Each panel therefore carries its letter and the
    estimator as a kicker, and the x-axis label states the metric in full, in
    the same words the other Section 4.2 figures use.
    """
    rows, bands = _fs_grouped_rows(elements, _FEATURES_AND_SHIFTS_GROUPS)
    # Row pitch is pinned, so the canvas follows the row count instead of
    # squeezing the rows: the element labels wrap to two lines and clear their
    # neighbours, and panel A's six bars inherit the same pitch.
    plot_h_in = L.WRAPPED_ROW_PITCH_IN * (len(rows) + 0.2)
    height_in = _FS_HEADER_IN + plot_h_in + _FS_FOOTER_IN
    fig, include_frac = c2a_figure("full", aspect=height_in / canvas_width_in(1.0))
    height_in = fig.get_figheight()
    plot_h_in = height_in - _FS_HEADER_IN - _FS_FOOTER_IN
    top = 1.0 - _FS_HEADER_IN / height_in
    bottom = _FS_FOOTER_IN / height_in
    kicker_y = 1.0 + _FS_KICKER_OFF_IN / plot_h_in
    boxes = _fs_panel_boxes()
    # Panel letters come from one iterator consumed in axes-creation order, so a
    # reordered or added panel cannot ship a stale hand-typed letter.
    letters = iter("ABCD")

    a_left, a_right = boxes[0]
    grid_a = fig.add_gridspec(1, 1, left=a_left, right=a_right, top=top, bottom=bottom)
    ax_a = fig.add_subplot(grid_a[0, 0])
    wrapped_tier = {
        **tier_group,
        "kicker": _FS_TIER_KICKER,
        "rows": _fs_wrap_labels(tier_group["rows"]),
    }
    _draw_property_rows(
        ax_a,
        _fs_wrap_labels(sae["properties"]),
        xlabel="Concordance with feature $R^2$,\nabove chance",
        extra_group=wrapped_tier,
        # Three ticks, not five: at 2.45 in the five-tick pitch is 0.45 in and
        # the widest label ("+0.3") is 0.52 in, so they would collide.  The zero
        # line is drawn as a rule, so it needs no tick of its own.
        xticks=[-0.1, 0.1, 0.3],
    )
    for label in ax_a.get_yticklabels():
        label.set_linespacing(L.WRAPPED_TICK_LINESPACING)
    _fs_place_xlabel(ax_a, x_axes=0.5, plot_h_in=plot_h_in)
    panel_header(ax_a, next(letters), "120,716 SAE features", kicker_y=kicker_y)

    columns = (
        (
            "direction",
            better_label("Predicted shift\ndirection\n(cosine)"),
            (0.15, 1.0),
            [0.2, 0.6, 1.0],
            None,
            "mean cosine",
        ),
        (
            "magnitude",
            "Predicted /\nobserved\nshift size",
            (0.55, 1.28),
            [0.6, 0.8, 1.0, 1.2],
            1.0,
            "median ratio",
        ),
    )
    for index, (key, xlabel, xlim, xticks, reference, kicker) in enumerate(columns):
        left, right = boxes[1 + index]
        grid = fig.add_gridspec(1, 1, left=left, right=right, top=top, bottom=bottom)
        ax = fig.add_subplot(grid[0, 0])
        _draw_row_metric_panel(
            ax,
            rows,
            key=key,
            xlabel=xlabel,
            xlim=xlim,
            reference=reference,
            bands=bands,
            xticks=xticks,
            # The three metric columns share one label column, drawn on the
            # first of them.
            ytick_labels=[L.tick_label(row["row"]) for row in rows] if index == 0 else None,
        )
        _fs_place_xlabel(ax, x_axes=0.5, plot_h_in=plot_h_in)
        panel_header(ax, next(letters), kicker, kicker_y=kicker_y)

    d_left, d_right = boxes[3]
    _fs_assert_in_segments(rows, "twoway", _FS_TWOWAY_SEGMENTS)
    spans = [hi - lo for lo, hi in _FS_TWOWAY_SEGMENTS]
    grid_d = fig.add_gridspec(
        1,
        len(spans),
        left=d_left,
        right=d_right,
        top=top,
        bottom=bottom,
        width_ratios=spans,
        # Wider than a within-panel gap: the break has to read as a cut, and it
        # is what separates the chance segment's tick label from the data
        # segment's first one on a column this narrow.
        wspace=0.28,
    )
    segments = []
    for index, (span, ticks) in enumerate(
        zip(_FS_TWOWAY_SEGMENTS, _FS_TWOWAY_SEGMENT_TICKS, strict=True)
    ):
        ax = fig.add_subplot(grid_d[0, index], label=f"features-and-shifts-twoway-{index}")
        low, high = span
        _draw_row_metric_panel(
            ax,
            rows,
            key="twoway",
            # The two segments are one axis, so the label is placed once, below
            # their shared center, rather than once per segment.
            xlabel="",
            xlim=span,
            reference=(_FS_TWOWAY_REFERENCE if low <= _FS_TWOWAY_REFERENCE <= high else None),
            bands=bands,
            xticks=list(ticks),
        )
        segments.append(ax)
    chance_ax, data_ax = segments
    chance_ax.spines["right"].set_visible(False)
    data_ax.spines["left"].set_visible(False)
    data_ax.tick_params(axis="y", length=0, labelleft=False)
    _fs_draw_cut_marks(chance_ax, data_ax)
    positions = [ax.get_position() for ax in segments]
    center = (positions[0].x0 + positions[-1].x1) / 2.0
    chance_ax.set_xlabel(better_label("Two-way\ndiscrimination\nrate"))
    _fs_place_xlabel(
        chance_ax,
        x_axes=(center - positions[0].x0) / positions[0].width,
        plot_h_in=plot_h_in,
    )
    panel_header(chance_ax, next(letters), "rate", kicker_y=kicker_y)

    return fig, include_frac


def make_direction_spectrum_figure(spectrum: dict) -> tuple[plt.Figure, float]:
    """Appendix figure: held-out $R^2$ of a direction against its variance rank."""
    from issue779_plot3_redesign import draw_spectrum_panel

    # The whole canvas goes to one panel, so the plot box is 11.9 x 4.7 in, both
    # wider and taller than the 0.75-width standalone the drawer's own label
    # offsets were tuned on.  The extra room spreads the labeled cluster further
    # apart rather than crowding it.  A flatter canvas (aspect 0.40) squeezed the
    # cluster's vertical room and collided the leader lines.
    fig, include_frac = c2a_figure("full", aspect=0.52)
    grid = fig.add_gridspec(1, 1, left=0.078, right=0.985, top=0.830, bottom=0.135)
    ax = fig.add_subplot(grid[0, 0])
    draw_spectrum_panel(ax, spectrum, legend_frame=True)
    panel_header(
        ax,
        "",
        "",
        title="Held-out $R^2$ by answer-variance rank",
        kicker_y=1.100,
        title_y=1.030,
    )
    return fig, include_frac


def make_element_shifts_by_slot_figure(data: dict) -> tuple[plt.Figure, float]:
    """Appendix companion: the one-word topic swap, by the grammatical slot that moved."""
    rows = data["slots"]
    fig, include_frac = c2a_figure("full", aspect=0.30)
    grid = fig.add_gridspec(1, 4, left=0.305, right=0.975, top=0.66, bottom=0.30, wspace=0.42)
    columns = (
        ("separation", "Answer separation", (0.94, 1.002), None),
        ("twoway", better_label("Two-way"), (0.5, 1.03), 0.5),
        ("direction", better_label("Direction"), (0.15, 0.75), None),
        ("magnitude", "Shift size", (0.4, 1.15), 1.0),
    )
    for index, (key, xlabel, xlim, reference) in enumerate(columns):
        ax = fig.add_subplot(grid[0, index])
        _draw_row_metric_panel(
            ax,
            rows,
            key=key,
            xlabel=xlabel,
            xlim=xlim,
            reference=reference,
            ytick_labels=(
                [f"{_SLOT_ROW_LABELS[row['row']]} (n={row['n_pairs']})" for row in rows]
                if index == 0
                else None
            ),
        )
        if index == 0:
            panel_header(
                ax,
                "",
                "one-word query swaps",
                "One-word topic change, by grammatical slot",
                kicker_y=1.26,
                title_y=1.07,
            )
    return fig, include_frac


REFUSAL_CLASS_LABELS = (
    ("obj_flip", "Object swap\n(flips refusal)"),
    ("verb_flip", "Verb swap\n(flips refusal)"),
    ("xstest", "XSTest\n(unsafe / safe)"),
    ("verb_harm", "Verb swap\n(both harmful)"),
    ("subj_ctl", "Subject swap\n(harmful)"),
    ("benign", "Benign\nswaps"),
)
BENIGN_SWAP_CLASSES = ("obj_benign", "verb_benign", "subj_benign")


def _boot_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    """95% pair-bootstrap CI of the mean (resampled pair indices)."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    mean = np.mean(values[idx], axis=1)
    return [float(np.percentile(mean, 2.5)), float(np.percentile(mean, 97.5))]


def _boot_slope(
    obs: np.ndarray, pred: np.ndarray, rng: np.random.Generator, n_boot: int
) -> list[float]:
    """95% pair-bootstrap CI of the through-origin norm slope."""
    idx = rng.integers(0, len(obs), size=(n_boot, len(obs)))
    slopes = np.sum(obs[idx] * pred[idx], axis=1) / np.sum(obs[idx] ** 2, axis=1)
    return [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]


def _refusal_group_stats(
    rs: list[dict], label: str, null_p95: float | None, rng: np.random.Generator, n_boot: int
) -> dict:
    """Mean-statistic reads for one group of #2617 safety-swap pairs.

    Uses the same three quantities as the #2564 minimal pairs: through-origin
    slope of predicted on observed shift norm (tail pooling), mean
    shift-direction cosine, and mean loading of the predicted shift on the
    refusal direction. Intervals are pair bootstraps.
    """
    obs = np.asarray([r["norm_obs_tail"] for r in rs], dtype=float)
    pred = np.asarray([r["norm_pred_arm_779ce"] for r in rs], dtype=float)
    base_pred = np.asarray([r["norm_pred_arm_iddelta"] for r in rs], dtype=float)
    cos = np.asarray([r["cos_arm_779ce"] for r in rs], dtype=float)
    base_cos = np.asarray([r["cos_arm_iddelta"] for r in rs], dtype=float)
    load = np.asarray([r["axis_cos_pred_arm_779ce"] for r in rs], dtype=float)
    base_load = np.asarray([r["axis_cos_pred_arm_iddelta"] for r in rs], dtype=float)
    return {
        "label": label,
        "n": len(rs),
        "slope": float(np.sum(obs * pred) / np.sum(obs**2)),
        "slope_ci95": _boot_slope(obs, pred, rng, n_boot),
        "base_slope": float(np.sum(obs * base_pred) / np.sum(obs**2)),
        "base_slope_ci95": _boot_slope(obs, base_pred, rng, n_boot),
        "mean_cos": float(np.mean(cos)),
        "mean_cos_ci95": _boot_mean(cos, rng, n_boot),
        "base_cos": float(np.mean(base_cos)),
        "base_cos_ci95": _boot_mean(base_cos, rng, n_boot),
        "null_cos_q": None if null_p95 is None else [0.0, float(null_p95)],
        "load": float(np.mean(load)),
        "load_ci95": _boot_mean(load, rng, n_boot),
        "base_load": float(np.mean(base_load)),
        "base_load_ci95": _boot_mean(base_load, rng, n_boot),
        "median_cos": float(np.median(cos)),
        "median_load": float(np.median(load)),
        "r10_mean": float(np.mean([r["r10"] for r in rs])),
    }


def _refusal_by_class_data(*, n_boot: int = 10_000, seed: int = 26170) -> dict:
    """Per-class reads (mean statistics) for the appendix refusal-swaps companion."""
    rows = _read_jsonl(SVMP_DIR / "perpair.jsonl")
    summary = json.loads((SVMP_DIR / "summary.json").read_text())
    null_p95 = summary["per_arm"]["arm_779ce"]["by_class_p95"]
    rng = np.random.default_rng(seed)
    present = {r["pair_class"] for r in rows}
    classes = []
    for key, label in REFUSAL_CLASS_LABELS:
        if key == "benign":
            rs = [r for r in rows if r["pair_class"] in BENIGN_SWAP_CLASSES]
            null = max(null_p95[c] for c in BENIGN_SWAP_CLASSES if c in null_p95)
        else:
            if key not in present:
                continue
            rs = [r for r in rows if r["pair_class"] == key]
            null = null_p95.get(key)
        classes.append({"key": key, **_refusal_group_stats(rs, label, null, rng, n_boot)})
    return {"classes": classes}


def _class_point_panel(
    ax: plt.Axes,
    rows: list[dict],
    *,
    value_key: str,
    ci_key: str,
    reference: float,
    ylabel: str,
    letter: str,
    kicker: str,
    title: str,
    ylim: tuple[float, float],
    yticks: list[float],
    base_key: str,
    base_ci_key: str,
    null_key: str | None = None,
    label_side: str = "above",
    kicker_y: float,
    title_y: float,
) -> None:
    """One per-class panel: linear-map points, raw context-shift squares, optional null band."""
    x = np.arange(len(rows))
    values = np.asarray([row[value_key] for row in rows])
    ci = np.asarray([row[ci_key] for row in rows])
    yerr = np.vstack([np.maximum(0.0, values - ci[:, 0]), np.maximum(0.0, ci[:, 1] - values)])
    ax.axhline(reference, color=INK, lw=1.4, linestyle=(0, (5, 4)), zorder=1)
    if null_key is not None:
        for xi, row in zip(x, rows, strict=True):
            band = row.get(null_key)
            if band is None:
                continue
            ax.fill_between(
                [xi - 0.36, xi + 0.36], band[0], band[1], color=MUTED, alpha=0.22, lw=0, zorder=1
            )
    base = np.asarray([row[base_key] for row in rows])
    base_ci = np.asarray([row[base_ci_key] for row in rows])
    base_err = np.vstack(
        [np.maximum(0.0, base - base_ci[:, 0]), np.maximum(0.0, base_ci[:, 1] - base)]
    )
    base_x = x + 0.22
    ax.errorbar(
        base_x,
        base,
        yerr=base_err,
        fmt="s",
        color=CONTROL,
        markerfacecolor=PAPER,
        markeredgecolor=CONTROL,
        markeredgewidth=2.0,
        markersize=8,
        capsize=5,
        capthick=1.8,
        elinewidth=1.8,
        lw=0,
        zorder=2,
    )
    ax.errorbar(
        x,
        values,
        yerr=yerr,
        fmt="o",
        color=LINEAR,
        markerfacecolor=LINEAR,
        markeredgecolor=LINEAR,
        markersize=9,
        capsize=6,
        capthick=2,
        elinewidth=2.3,
        lw=0,
        zorder=3,
    )
    for xi, value, lo, hi in zip(x, values, ci[:, 0], ci[:, 1], strict=True):
        if label_side == "below":
            anchor, dy, va = min(value, lo), -5, "top"
        else:
            anchor, dy, va = max(value, hi), 5, "bottom"
        ax.annotate(
            f"{value:.2f}",
            (xi, anchor),
            xytext=(0, dy),
            textcoords="offset points",
            fontsize=14,
            fontweight=650,
            ha="center",
            va=va,
        )
    ax.set_xticks(x, [row["label"] for row in rows])
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlim(-0.5, len(rows) - 0.25)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    style_axis(ax, grid_axis="y")
    panel_header(ax, letter, kicker, title, kicker_y=kicker_y, title_y=title_y)


def make_refusal_by_class_figure(data: dict) -> tuple[plt.Figure, float]:
    classes = data["classes"]
    fig, include_frac = c2a_figure("wide", aspect=1.10)
    grid = fig.add_gridspec(3, 1, left=0.115, right=0.985, top=0.85, bottom=0.07, hspace=0.8)
    ax_cos = fig.add_subplot(grid[0, 0])
    ax_load = fig.add_subplot(grid[1, 0])
    ax_slope = fig.add_subplot(grid[2, 0])
    _class_point_panel(
        ax_cos,
        classes,
        value_key="mean_cos",
        ci_key="mean_cos_ci95",
        reference=0.0,
        ylabel="Mean cosine",
        letter="A",
        kicker="Shift direction, by pair class",
        title="Cosine of predicted and observed shift",
        ylim=(-0.06, 1.14),
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        base_key="base_cos",
        base_ci_key="base_cos_ci95",
        null_key="null_cos_q",
        kicker_y=1.32,
        title_y=1.13,
    )
    _class_point_panel(
        ax_load,
        classes,
        value_key="load",
        ci_key="load_ci95",
        reference=0.0,
        ylabel="Mean projection",
        letter="B",
        kicker="Refusal-direction projection, by pair class",
        title="Projection on the refusal direction",
        ylim=(-0.32, 0.98),
        yticks=[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
        base_key="base_load",
        base_ci_key="base_load_ci95",
        kicker_y=1.32,
        title_y=1.13,
    )
    top = max(
        max(row["base_slope_ci95"][1] for row in classes),
        max(row["slope_ci95"][1] for row in classes),
    )
    ymax = float(np.ceil((top + 0.08) * 4) / 4)
    _class_point_panel(
        ax_slope,
        classes,
        value_key="slope",
        ci_key="slope_ci95",
        reference=1.0,
        ylabel="Size ratio",
        letter="C",
        kicker="Shift magnitude, by pair class",
        title="Predicted over observed shift size",
        ylim=(0.4, ymax),
        yticks=[float(t) for t in np.arange(0.5, ymax + 1e-9, 0.5)],
        base_key="base_slope",
        base_ci_key="base_slope_ci95",
        label_side="below",
        kicker_y=1.32,
        title_y=1.13,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=LINEAR,
            markerfacecolor=LINEAR,
            markeredgecolor=LINEAR,
            markersize=9,
            lw=0,
            label="Linear map",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color=CONTROL,
            markerfacecolor=PAPER,
            markeredgecolor=CONTROL,
            markeredgewidth=2.0,
            markersize=8,
            lw=0,
            label="Raw context shift",
        ),
        Patch(facecolor=MUTED, alpha=0.22, label="Shuffled-pair null"),
    ]
    row_y = 0.985
    legend_kicker(fig, 0.115, row_y, "Prediction")
    fig.legend(
        handles=handles[:2],
        loc="upper left",
        bbox_to_anchor=(0.114, row_y - 0.012),
        ncol=2,
        frameon=False,
        columnspacing=1.45,
        handlelength=1.4,
        handletextpad=0.65,
        borderaxespad=0,
    )
    legend_kicker(fig, 0.62, row_y, "Null")
    fig.legend(
        handles=handles[2:],
        loc="upper left",
        bbox_to_anchor=(0.619, row_y - 0.012),
        ncol=1,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.65,
        borderaxespad=0,
    )
    return fig, include_frac


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only",
        action="append",
        choices=(
            "sae",
            "pair_shifts",
            "refusal_by_class",
            "directions_and_pairs",
            "directions_and_features",
            "direction_r2_spectrum",
            "failures_and_shifts",
            "features_and_shifts",
            "element_shifts",
            "element_shifts_by_slot",
        ),
        default=None,
        help="render one figure, repeat the flag to select several (default: all ten)",
    )
    args = parser.parse_args()
    set_c2a_style()

    selected = set(args.only) if args.only else None

    def wanted(*names: str) -> bool:
        return selected is None or any(name in selected for name in names)

    report: list[tuple[str, dict]] = []

    if wanted("sae"):
        sae = _sae_data()
        sae_fig, sae_frac = make_sae_figure(sae)
        sae_outputs = _save(
            sae_fig,
            args.out_dir,
            "c3_sae_tier_gradient",
            title="SAE feature properties and context-to-answer predictability",
            subject=(
                "Conditional feature-property associations and activity-adjusted "
                "nested-tier gradient, decoder-direction target"
            ),
            include_frac=sae_frac,
            sources=[SAE_SOURCE],
            displayed_data=sae,
        )
        plt.close(sae_fig)
        report.append(("sae", sae_outputs))

    pair_shifts: list[dict] | None = None
    if wanted("pair_shifts", "directions_and_pairs"):
        pair_shifts = _pair_shift_data()

    if wanted("pair_shifts"):
        pair_fig, pair_frac = make_pair_shift_figure(pair_shifts)
        pair_outputs = _save(
            pair_fig,
            args.out_dir,
            "c3_pair_shifts",
            title="Predicted over observed answer-shift size for controlled minimal pairs",
            subject=(
                "Through-origin calibration slope of predicted over observed answer-shift size "
                "per changed element, with 95% confidence intervals"
            ),
            include_frac=pair_frac,
            sources=[MINPAIR_SOURCE, PERSONA_SOURCE, ONEWORD_SOURCE, ONEWORD_PAIRS],
            displayed_data={
                "elements": pair_shifts,
                "reference_line": 1.0,
                "order": "descending slope",
            },
        )
        plt.close(pair_fig)
        report.append(("pair_shifts", pair_outputs))

    if wanted("refusal_by_class"):
        refusal = _refusal_by_class_data()
        refusal_fig, refusal_frac = make_refusal_by_class_figure(refusal)
        refusal_outputs = _save(
            refusal_fig,
            args.out_dir,
            "c3_refusal_swaps_by_class",
            title="One-word safety swaps by pair class",
            subject=(
                "Per-class mean shift-direction cosine, refusal-direction loading, and "
                "through-origin predicted-over-observed shift size for one-word safety swaps, "
                "with the raw context-shift baseline and shuffled-pair null"
            ),
            include_frac=refusal_frac,
            sources=[SVMP_DIR / "perpair.jsonl", SVMP_DIR / "summary.json"],
            displayed_data=refusal,
        )
        plt.close(refusal_fig)
        report.append(("refusal_by_class", refusal_outputs))

    if wanted("directions_and_pairs"):
        spectrum = json.loads(SPECTRUM_SOURCE.read_text())
        combined_fig, combined_frac = make_directions_and_pairs_figure(spectrum, pair_shifts)
        combined_outputs = _save(
            combined_fig,
            args.out_dir,
            "c3_directions_and_pairs",
            title="Useful directions on the answer-PCA spectrum and minimal-pair shift sizes",
            subject=(
                "Per-direction held-out R2 of the context-to-answer map at layer 19 against "
                "variance rank (labeled behavior/persona/correctness directions), alongside "
                "predicted over observed answer-shift size for six minimal-pair elements "
                "with 95% bootstrap CIs"
            ),
            include_frac=combined_frac,
            sources=[
                SPECTRUM_SOURCE,
                MINPAIR_SOURCE,
                PERSONA_SOURCE,
                ONEWORD_SOURCE,
                ONEWORD_PAIRS,
            ],
            displayed_data={
                "panel_a": _spectrum_displayed(spectrum),
                "panel_b": {
                    "elements": pair_shifts,
                    "reference_line": 1.0,
                    "order": "descending slope",
                },
            },
        )
        plt.close(combined_fig)
        report.append(("directions_and_pairs", combined_outputs))

    if wanted("directions_and_features", "failures_and_shifts"):
        info = _information_data()

    if wanted("directions_and_features"):
        daf_fig, daf_frac = make_directions_and_features_figure(info)
        daf_outputs = _save(
            daf_fig,
            args.out_dir,
            "c3_directions_and_features",
            title="Which parts of an answer the context-to-answer map predicts",
            subject=(
                "Held-out R2 of the projection onto a direction against its answer-variance "
                "rank, and the conditional association between an SAE feature property and the "
                "held-out R2 of its decoder direction"
            ),
            include_frac=daf_frac,
            sources=[SPECTRUM_SOURCE, SAE_SOURCE],
            displayed_data={"panel_a": info["panel_a"], "panel_b": info["panel_b"]},
        )
        plt.close(daf_fig)
        report.append(("directions_and_features", daf_outputs))

    if wanted("failures_and_shifts"):
        fas_fig, fas_frac = make_failures_and_shifts_figure(info)
        # The panel-A background is 60,000 scatter points.  The sidecar records
        # its shape and points at the sha-pinned source instead of restating a
        # megabyte of coordinates the source file already carries verbatim.
        background = dict(info["panel_c"]["background"])
        panel_a = {
            **info["panel_c"],
            "background": {
                key: value for key, value in background.items() if key not in ("ctx", "ans")
            }
            | {"stored_in": _display_path(SECTION42_PANELS)},
        }
        fas_outputs = _save(
            fas_fig,
            args.out_dir,
            "c3_failures_and_shifts",
            title="Where the context-to-answer map fails and how well it sizes a change",
            subject=(
                "Retrieval-failure shift sizes against the candidate-pool background, and "
                "variance explained per controlled change before and after correcting the "
                "predicted shift size"
            ),
            include_frac=fas_frac,
            sources=[SECTION42_PANELS],
            displayed_data={"panel_a": panel_a, "panel_b": info["panel_d"]},
        )
        plt.close(fas_fig)
        report.append(("failures_and_shifts", fas_outputs))

    if wanted("direction_r2_spectrum"):
        appendix_spectrum = json.loads(SPECTRUM_SOURCE.read_text())
        spec_fig, spec_frac = make_direction_spectrum_figure(appendix_spectrum)
        spec_outputs = _save(
            spec_fig,
            args.out_dir,
            "c3_direction_r2_spectrum",
            title="Held-out R2 of a direction's projection against its answer-variance rank",
            subject=(
                "Per-direction held-out R2 of the context-to-answer map at layer 19 against "
                "answer-variance rank, with the random-direction band and the labeled "
                "behavior, persona and correctness directions"
            ),
            include_frac=spec_frac,
            sources=[SPECTRUM_SOURCE],
            displayed_data=_spectrum_displayed(appendix_spectrum),
        )
        plt.close(spec_fig)
        report.append(("direction_r2_spectrum", spec_outputs))

    elements: dict | None = None
    if wanted("element_shifts", "element_shifts_by_slot", "features_and_shifts"):
        elements = _element_shift_data()

    if wanted("features_and_shifts"):
        fs_sae = _sae_data()
        fs_tier = _tier_concordance_group()
        fs_rows, _fs_bands = _fs_grouped_rows(elements, _FEATURES_AND_SHIFTS_GROUPS)
        fs_fig, fs_frac = make_features_and_shifts_figure(fs_sae, elements, fs_tier)
        fs_outputs = _save(
            fs_fig,
            args.out_dir,
            "c3_features_and_shifts",
            title=(
                "SAE feature properties and what the context-to-answer map keeps "
                "per changed context element"
            ),
            subject=(
                "Conditional association between an SAE feature property and the held-out R2 "
                "of its decoder direction, beside the mean cosine between predicted and "
                "observed answer shift, the median ratio of predicted to observed shift size, "
                "and the two-way discrimination rate per controlled context element, for the "
                "seven rows left after the two refusal-holds rows are dropped, with 95% "
                "pair-bootstrap intervals"
            ),
            include_frac=fs_frac,
            sources=[SAE_SOURCE, ELEMENT_SHIFT_SOURCE, TIER_CONCORDANCE_SOURCE],
            displayed_data={
                "panel_a": {
                    "properties": fs_sae["properties"],
                    "property_label_wraps": _FS_PROPERTY_LABEL_WRAP,
                    "dv": fs_sae["dv"],
                    "separate_group": {
                        "kicker": fs_tier["kicker"],
                        "rows": fs_tier["rows"],
                        "universe": fs_tier["universe"],
                        "matching": fs_tier["matching"],
                        "bootstrap": fs_tier["bootstrap"],
                        "property_rows_universe": fs_tier["property_rows_universe"],
                        "caption_note": fs_tier["caption_note"],
                        "separation": (
                            "one empty row, a shaded stripe, a lighter bar fill and the "
                            "group's own kicker; the stripe, the gap and the fill survive "
                            "the grayscale audit"
                        ),
                        "error_bars": (
                            "none drawn; the panel shows no interval on any row, so the "
                            "tier interval is recorded here instead"
                        ),
                    },
                },
                "panels_b_c_d": {
                    "rows": fs_rows,
                    "groups": [list(labels) for _group, labels in _FEATURES_AND_SHIFTS_GROUPS],
                    "rows_not_drawn": [
                        row["row"]
                        for row in elements["elements"]
                        if row["row"] not in {drawn["row"] for drawn in fs_rows}
                    ],
                    "reference_line": {"magnitude": 1.0, "twoway": 0.5},
                    "twoway_axis": {
                        "segments": [list(segment) for segment in _FS_TWOWAY_SEGMENTS],
                        "segment_ticks": [list(ticks) for ticks in _FS_TWOWAY_SEGMENT_TICKS],
                        "rule": (
                            "two linear segments, plotted widths proportional to their data "
                            "spans, diagonal cut marks on the bottom seam; the omitted range "
                            "carries no value or interval endpoint"
                        ),
                    },
                    "row_labels": {row["row"]: L.tick_label(row["row"]) for row in fs_rows},
                    "pair_counts_in_labels": False,
                    "order": "grouped: identity, format, content, refusal reverses",
                    "bootstrap": elements["bootstrap"],
                    "map": elements["map"],
                    "metrics": elements["metrics"],
                    "caveat": elements["caveat"],
                },
            },
        )
        plt.close(fs_fig)
        report.append(("features_and_shifts", fs_outputs))

    if wanted("element_shifts"):
        elem_fig, elem_frac = make_element_shifts_figure(elements)
        elem_outputs = _save(
            elem_fig,
            args.out_dir,
            "c3_element_shifts",
            title="What the context-to-answer map keeps per changed context element",
            subject=(
                "Mean cosine between predicted and observed answer shift, and median ratio of "
                "predicted to observed shift size, per controlled context element, with 95% "
                "pair-bootstrap intervals"
            ),
            include_frac=elem_frac,
            sources=[ELEMENT_SHIFT_SOURCE],
            displayed_data={
                "rows": elements["elements"],
                "groups": [list(labels) for _group, labels in _ELEMENT_SHIFT_GROUPS],
                "reference_line": {"magnitude": 1.0},
                "order": "grouped: identity, format, content, word refusal, framing refusal",
                "bootstrap": elements["bootstrap"],
                "map": elements["map"],
                "metrics": elements["metrics"],
                "caveat": elements["caveat"],
            },
        )
        plt.close(elem_fig)
        report.append(("element_shifts", elem_outputs))

    if wanted("element_shifts_by_slot"):
        slot_fig, slot_frac = make_element_shifts_by_slot_figure(elements)
        slot_outputs = _save(
            slot_fig,
            args.out_dir,
            "c3_element_shifts_by_slot",
            title="One-word topic change by grammatical slot",
            subject=(
                "Answer separation, two-way retrieval, shift direction and shift size for a "
                "one-word query change pinned to the subject, verb or object slot, beside the "
                "pooled one-word row of the element figure, with 95% pair-bootstrap intervals"
            ),
            include_frac=slot_frac,
            sources=[ELEMENT_SHIFT_SOURCE],
            displayed_data={
                "rows": elements["slots"],
                "row_labels": _SLOT_ROW_LABELS,
                "reference_line": {"twoway": 0.5, "magnitude": 1.0},
                "order": "pooled one-word row, then subject, verb, object",
                "bootstrap": elements["bootstrap"],
                "map": elements["map"],
                "metrics": elements["metrics"],
                "caveat": elements["caveat"],
            },
        )
        plt.close(slot_fig)
        report.append(("element_shifts_by_slot", slot_outputs))

    for name, outputs in report:
        for kind, path in outputs.items():
            if isinstance(path, Path):
                print(f"{name}.{kind}: {path}")


if __name__ == "__main__":
    main()
