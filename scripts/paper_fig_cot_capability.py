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
CONDITION_COLORS = [MUTED, ROLES["linear"].color, ROLES["nonlinear"].color]
BORDERS = [":", "-", "--"]
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


def bar_with_interval(ax, x, value, bounds, width, condition, *, retrieval=False):
    """Draw a saved estimate and its endpoints, including non-enclosing intervals."""
    lo, hi = bounds
    if not np.isfinite([value, lo, hi]).all() or lo > hi:
        raise ValueError(f"Invalid estimate/interval: {value}, {bounds}")
    color = CONDITION_COLORS[condition]
    ax.bar(
        x,
        value,
        width=width * 0.86,
        facecolor="white" if retrieval else color,
        edgecolor=color,
        linestyle=BORDERS[condition],
        linewidth=1.8,
        hatch="///" if retrieval else None,
        zorder=2,
    )
    ax.vlines(x, lo, hi, color=MUTED, linewidth=1.8, zorder=4)


def draw_prediction(ax, data):
    """Preserve the all-question comparison and its two metrics in panel A."""
    conditions = data["panels"]["A"]["conditions"]
    assert [row["cell"] for row in conditions] == ["p7_Aoff", "p7_A", "p7_D"]
    assert data["panels"]["A"]["n_evaluated"] == 33810
    width = 0.34
    for j, key in enumerate(["r2_corpus", "acc1"]):
        for i, row in enumerate(conditions):
            scores = row["metrics"]
            assert scores["n"] == 33810
            bar_with_interval(
                ax,
                j * 1.25 + (i - 1) * width,
                scores[key],
                scores[f"{key}_ci"],
                width,
                i,
                retrieval=j == 1,
            )
    ax.set_xticks([0, 1.25], [r"$R^2$", "Top-1\nretrieval"])
    ax.set_xlim(-0.52, 1.77)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(better_label("Score"))
    style_axis(ax)
    panel_header(ax, "A", "All questions", "Answer prediction", kicker_y=1.18, title_y=1.035)


def draw_correctness(ax, data):
    """Preserve equal-dataset-weight estimates and intervals in panel B."""
    pooled = data["panels"]["B"]["readouts"]
    counts = {"necessary": 4522, "both_correct": 17693}
    width = 0.30
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
                i + 1,
            )
    ax.set_xticks([0, 1], ["Only with\nthinking", "Both\nmodes"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 0.61)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_ylabel(better_label(r"Held-out $R^2$"))
    style_axis(ax)
    panel_header(ax, "B", "Thinking on", "Correctness groups", kicker_y=1.18, title_y=1.035)


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
    ax.set_xlabel(better_label("Artificial Analysis\nIntelligence Index"))
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
    panel_header(
        ax,
        "C",
        "10 models · thinking off",
        "Model capability",
        kicker_y=1.18,
        title_y=1.035,
    )


def main():
    """Export the combined figure with unchanged values and complete provenance."""
    raw = SOURCE.read_bytes()
    source = json.loads(raw)
    cot, capability = [source[key]["data"] for key in ("cot", "capability")]
    set_c2a_style()
    fig, frac = c2a_figure("full", aspect=0.42)
    axes = [
        fig.add_axes([0.080, 0.19, 0.190, 0.55]),
        fig.add_axes([0.345, 0.19, 0.195, 0.55]),
        fig.add_axes([0.625, 0.19, 0.355, 0.55]),
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
            for row, color, border in zip(
                cot["panels"]["A"]["conditions"],
                CONDITION_COLORS,
                BORDERS,
                strict=True,
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 1.0),
        ncol=3,
        handlelength=1.4,
        columnspacing=1.4,
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
            "Single-row layout, compact headings, and relocated model labels; "
            "bar-value text omitted from narrow panels. All estimates and intervals preserved."
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
