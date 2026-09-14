"""Combine the manuscript's CoT and capability panels from frozen plotted data.

No inference, fitting, aggregation, or statistical tests run here. The data
snapshot records the original producer commits and verified source hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    MUTED,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
    style_score_axis,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures/paper/c1_cot_capability_data.json"
STEM = ROOT / "figures/paper/c1_cot_capability"
MID_COT = "#E1AA9E"  # 50% white tint of the CoT-end red: same family, earlier state
CELL_ORDER = ["p7_Aoff", "p7_A", "p7_traj_t20", "p7_D"]
CONDITION_COLORS = dict(zip(CELL_ORDER, [MUTED, ROLES["linear"].color, MID_COT, ROLES["nonlinear"].color], strict=True))
BORDERS = dict(zip(CELL_ORDER, [":", "-", "-.", "--"], strict=True))
OFFSETS = {
    "q35_0p8b": (40, 65),
    "q35_2b": (12, 5),
    "q35_4b": (40, -24),
    "q35_9b": (12, 28),
    "q35_27b": (12, 0),
    "q36_27b": (12, -28),
    "q38_27b": (-12, 9),
    "o3_7b_i": (12, -5),
    "o31_32b_i": (12, -10),
    "q3_32b": (-12, 4),
}


def bar_with_interval(ax, x, value, bounds, width, cell, *, retrieval=False):
    """Draw a saved estimate and its endpoints, including non-enclosing intervals."""
    lo, hi = bounds
    if not np.isfinite([value, lo, hi]).all() or lo > hi:
        raise ValueError(f"Invalid estimate/interval: {value}, {bounds}")
    color = CONDITION_COLORS[cell]
    ax.bar(
        x,
        value,
        width=width * 0.86,
        facecolor="white" if retrieval else color,
        edgecolor=color,
        linestyle=BORDERS[cell],
        linewidth=1.8,
        hatch="///" if retrieval else None,
        zorder=2,
    )
    ax.vlines(x, lo, hi, color=MUTED, linewidth=1.8, zorder=4)


def draw_prediction(ax, data):
    """Panel A: the three shipped conditions plus the best interior thinking-span state."""
    conditions = data["panels"]["A"]["conditions"]
    assert [row["cell"] for row in conditions] == CELL_ORDER
    assert data["panels"]["A"]["n_evaluated"] == 33810
    width = 0.27
    for j, key in enumerate(["r2_corpus", "acc1"]):
        for i, row in enumerate(conditions):
            scores = row["metrics"]
            assert scores["n"] == 33810
            bar_with_interval(
                ax,
                j * 1.35 + (i - 1.5) * width,
                scores[key],
                scores[f"{key}_ci"],
                width,
                row["cell"],
                retrieval=j == 1,
            )
    ax.set_xticks([0, 1.35], [r"$R^2$", "Top-1\nretrieval"])
    ax.set_xlim(-0.72, 2.07)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(better_label("Score"))
    style_axis(ax)
    panel_header(ax, "", "", "A Answer prediction", kicker_y=1.025, title_y=1.025)


def draw_correctness(ax, data):
    """Preserve equal-dataset-weight estimates and intervals in panel B."""
    pooled = data["panels"]["B"]["readouts"]
    counts = {"necessary": 4522, "both_correct": 17693}
    width = 0.30
    readout_cells = {"context": "p7_A", "end_of_thought": "p7_D"}
    for i, readout in enumerate(["context", "end_of_thought"]):
        for j, (group, count) in enumerate(counts.items()):
            row = pooled[readout][group]
            assert row["n"] == sum(row["n_by_corpus"].values()) == count
            assert len(row["weights"]) == 7
            assert np.allclose(list(row["weights"].values()), 1 / 7)
            bar_with_interval(
                ax,
                j + (i - 0.5) * width,
                row["r2_corpus_mean"],
                row["r2_corpus_mean_ci"],
                width,
                readout_cells[readout],
            )
    ax.set_xticks([0, 1], ["Only with\nthinking", "Both\nmodes"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 0.61)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_ylabel(better_label(r"Held-out $R^2$"))
    style_axis(ax)
    panel_header(ax, "", "", "B Correctness groups", kicker_y=1.025, title_y=1.025)


def draw_capability(ax, data):
    """Retain all ten model labels, coordinates, and saved statistics in panel C."""
    rows = data["rows"]
    assert len(rows) == len(OFFSETS) == data["plotted_panel"]["n"] == 10
    assert {row["model_key"] for row in rows} == set(OFFSETS)
    assert np.isfinite([[r["aa_index"], r["test_r2"]] for r in rows]).all()
    style_score_axis(ax, y_min=0.59, y_max=0.756, y_step=0.04)
    ax.set_yticks([0.60, 0.65, 0.70, 0.75])
    ax.set_xlim(-3, 64)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel(better_label("AA Intelligence Index"))
    ax.set_ylabel(better_label(r"Held-out $R^2$"))
    role = ROLES["linear"]
    ax.scatter(
        [r["aa_index"] for r in rows],
        [r["test_r2"] for r in rows],
        s=110,
        marker=role.marker,
        color=role.color,
        linewidths=1.9,
        zorder=3,
    )
    for row in rows:
        dx, dy = OFFSETS[row["model_key"]]
        key = row["model_key"]
        arrow = None
        if key in {"q35_0p8b", "q35_4b", "q35_9b", "q36_27b"}:
            arrow = {"arrowstyle": "-", "color": MUTED, "linewidth": 0.9}
        ax.annotate(
            row["label"],
            (row["aa_index"], row["test_r2"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center" if key == "q35_0p8b" else ("left" if dx > 0 else "right"),
            va="center",
            color=INK,
            arrowprops=arrow,
            fontsize=15,
        )
    stats = data["plotted_panel"]
    ax.text(
        0.025,
        0.95,
        rf"Spearman $\rho = {stats['rho']:.3f}$, $p = {stats['p_uncorrected']:.3f}$",
        transform=ax.transAxes,
        va="top",
    )
    panel_header(ax, "", "", "C Model capability", kicker_y=1.025, title_y=1.025)


def main():
    """Export the combined figure with unchanged values and complete provenance."""
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    cot, capability = [source[key]["data"] for key in ("cot", "capability")]
    set_c2a_style()
    fig, frac = c2a_figure("full", aspect=0.35)
    axes = [
        fig.add_axes([0.080, 0.20, 0.190, 0.62]),
        fig.add_axes([0.345, 0.20, 0.195, 0.62]),
        fig.add_axes([0.625, 0.20, 0.355, 0.62]),
    ]
    fig.legend(
        handles=[
            Patch(
                facecolor=color,
                edgecolor=color,
                linestyle=border,
                linewidth=2,
                label=row["label"],
            )
            for row, color, border in (
                (row, CONDITION_COLORS[row["cell"]], BORDERS[row["cell"]])
                for row in cot["panels"]["A"]["conditions"]
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        handlelength=1.1,
        columnspacing=0.7,
        handletextpad=0.5,
    )
    draw_prediction(axes[0], cot)
    draw_correctness(axes[1], cot)
    draw_capability(axes[2], capability)
    exported = save_c2a_figure(
        fig,
        STEM,
        include_width=frac,
        title="Answer predictability, chain of thought, and model capability",
        subject="Existing Qwen3-8B CoT comparisons and ten-model capability association.",
        creator="scripts/paper_fig_cot_capability.py",
    )
    record = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "original_panels": source,
        "panel_mapping": {"A": "cot.A", "B": "cot.B", "C": "capability"},
        "layout": "One horizontal row: CoT prediction, correctness groups, model capability",
        "changes": (
            "Single-row layout with one-line panel headings and tighter vertical spacing. "
            "Panel C uses AA Intelligence Index, expanded in the caption. "
            "All estimates, intervals, model labels, and font sizes preserved. "
            "2026-09-14: panel A gains a fourth bar, the best interior thinking-span state "
            "(t=0.2, cell p7_traj_t20 from p7_traj__a3.json); no map refit."
        ),
        "maps_refit": False,
        "render": exported["record"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_sha256": hashlib.sha256(
            (ROOT / "src/explore_persona_space/analysis/c2a_plot_style.py").read_bytes()
        ).hexdigest(),
        "output_sha256": {
            key: hashlib.sha256(Path(exported[key]).read_bytes()).hexdigest()
            for key in ("pdf", "png", "grayscale")
        },
    }
    STEM.with_suffix(".meta.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    plt.close(fig)
    print(exported["pdf"])


if __name__ == "__main__":
    main()
