"""Parent-vs-gpu2 per-family gap scatter for issue #2094 round `gpu2_mq_replacement_prefix`.

The parent grid's matched-query context-end arm was re-run on the 5 pairs re-formed
with the `conv2` replacement conversation prefix (4 of 5 recovered by the anchor
gate). This figure compares, per cell family (layer variant x dose x metric), the
steered-minus-null mean-F gap measured on the parent's 10 well-separated pairs
(x axis, 1024-token cap) against the same family's gap on the 4 recovered conv2
pairs (y axis, 2048-token cap) — the previously-EXCLUDED pair class. The 17
families the parent screen kept as clean-separating are drawn filled and labeled;
every family is a visible point (the low-level per-unit view).

All values come verbatim from the committed
eval_results/issue_2094/f_metrics/gpu2/gpu2_summary.json `parent_comparison`
block (never recomputed); the per-metric Spearman rank correlations and grid
mean gaps are re-derived here and asserted against the persisted values.

Writes figures/issue_2094/gpu2_parent_gap_scatter.{png,pdf,meta.json}.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

SUMMARY_PATH = Path("eval_results/issue_2094/f_metrics/gpu2/gpu2_summary.json")

_METRIC_PANEL = {
    "f_beh_prefix": "behavior F (prefix rubric)",
    "f_act": "activation F",
}
_DOSE_LABEL = {"a0.5": "0.5x", "a1": "1x", "a2": "2x", "a4": "4x", "replace": "patch"}
_LV_LABEL = {"joint_mid": "mid", "joint_all": "all"}


def _point_label(row: dict) -> str:
    lv = _LV_LABEL.get(row["layer_variant"], row["layer_variant"])
    return f"{lv} {_DOSE_LABEL[row['dose']]}"


# Hand-placed label anchors (data coords) for the 17 parent-clean families:
# the behavior panel's low-gap band is too dense for offset labels, so those
# labels sit in a ladder above the band with thin leader lines.
_LABEL_POS: dict[tuple[str, str, str], tuple[float, float, str, str]] = {
    # behavior panel
    ("f_beh_prefix", "L16", "a2"): (0.26, 0.32, "right", "center"),
    ("f_beh_prefix", "L17", "a1"): (0.26, 0.245, "right", "center"),
    ("f_beh_prefix", "L15", "replace"): (0.26, 0.17, "right", "center"),
    ("f_beh_prefix", "L14", "a1"): (0.50, 0.32, "left", "center"),
    ("f_beh_prefix", "L16", "a1"): (0.50, 0.245, "left", "center"),
    ("f_beh_prefix", "L13", "a2"): (0.50, 0.17, "left", "center"),
    ("f_beh_prefix", "L12", "a4"): (0.431, -0.07, "center", "top"),
    ("f_beh_prefix", "L20", "replace"): (0.505, 0.135, "left", "center"),
    ("f_beh_prefix", "joint_all", "replace"): (0.672, -0.03, "center", "top"),
    ("f_beh_prefix", "joint_mid", "a0.5"): (0.594, 0.82, "center", "bottom"),
    ("f_beh_prefix", "L16", "a4"): (0.286, 0.82, "center", "bottom"),
    ("f_beh_prefix", "L17", "a2"): (0.345, 0.441, "left", "center"),
    # activation panel
    ("f_act", "L16", "a1"): (0.155, 0.100, "left", "center"),
    ("f_act", "L16", "a2"): (0.129, 0.175, "center", "bottom"),
    ("f_act", "L18", "a1"): (0.155, 0.045, "left", "center"),
    ("f_act", "joint_all", "replace"): (0.285, 0.008, "right", "center"),
    ("f_act", "joint_mid", "a0.5"): (0.270, 0.218, "left", "center"),
}


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    comp = summary["parent_comparison"]
    rows = comp["rows"]
    assert len(rows) == 300, len(rows)

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))

    colors = {
        "f_beh_prefix": paper_palette_role("primary"),
        "f_act": paper_palette_role("control"),
    }
    neutral = paper_palette_role("neutral")

    for ax, metric in zip(axes, ["f_beh_prefix", "f_act"]):
        mrows = [r for r in rows if r["metric"] == metric]
        assert len(mrows) == 150, (metric, len(mrows))
        pm = comp["per_metric"][metric]

        clean = [r for r in mrows if r["parent"]["verdict"] == "clean_separating"]
        other = [r for r in mrows if r["parent"]["verdict"] != "clean_separating"]
        assert len(clean) == pm["n_parent_clean_separating"], (metric, len(clean))
        # the headline direction-replication read: every parent-clean family matches
        assert all(r["direction_match"] for r in clean), metric
        assert (
            sum(r["direction_match"] for r in clean) == pm["n_parent_clean_gpu2_direction_match"]
        ), metric

        px = np.array([r["parent"]["gap"] for r in mrows])
        py = np.array([r["gpu2"]["gap"] for r in mrows])
        rho, pval = spearmanr(px, py)
        assert abs(rho - pm["spearman_gap_parent_vs_gpu2"]) < 1e-9, (metric, rho)
        assert abs(px.mean() - pm["mean_gap_parent"]) < 1e-9, metric
        assert abs(py.mean() - pm["mean_gap_gpu2"]) < 1e-9, metric

        color = colors[metric]
        ax.scatter(
            [r["parent"]["gap"] for r in other],
            [r["gpu2"]["gap"] for r in other],
            s=16,
            color=color,
            alpha=0.35,
            linewidths=0,
            label="other family",
        )
        ax.scatter(
            [r["parent"]["gap"] for r in clean],
            [r["gpu2"]["gap"] for r in clean],
            s=54,
            color=color,
            edgecolors="#222222",
            linewidths=1.0,
            label="parent clean-separating family",
            zorder=3,
        )
        lims = [
            min(px.min(), py.min()) - 0.06,
            max(px.max(), py.max()) + 0.06,
        ]

        # Hand-placed labels for every parent-clean family (_LABEL_POS): the
        # behavior panel's dense low-gap band gets a two-column ladder of
        # labels above the band, tied to points by thin leader lines.
        for r in clean:
            x, y = r["parent"]["gap"], r["gpu2"]["gap"]
            lx, ly, ha, va = _LABEL_POS[(metric, r["layer_variant"], r["dose"])]
            if abs(lx - x) > 0.03 or abs(ly - y) > 0.06:
                ax.annotate(
                    _point_label(r),
                    xy=(x, y),
                    xytext=(lx, ly),
                    ha=ha,
                    va=va,
                    fontsize=6.0,
                    color="#333333",
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#aaaaaa"},
                    zorder=2,
                )
            else:
                ax.text(lx, ly, _point_label(r), fontsize=6.0, ha=ha, va=va, color="#333333")
        ax.plot(lims, lims, ls="--", lw=1.0, color=neutral, label="identity", zorder=1)
        ax.axhline(0.0, lw=0.6, color=neutral, alpha=0.4, zorder=0)
        ax.axvline(0.0, lw=0.6, color=neutral, alpha=0.4, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(_METRIC_PANEL[metric], loc="left")
        ax.set_xlabel("parent-grid gap: steered - null mean F\n(10 well-separated pairs)")
        ax.set_ylabel("conv2 re-run gap: steered - null mean F\n(4 recovered pairs)")
        ax.annotate(
            f"Spearman rho = {rho:.2f}, p = {pval:.1e} (150 families)",
            xy=(0.03, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=8,
        )

    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    savefig_paper(fig, "issue_2094/gpu2_parent_gap_scatter", dir="figures/")
    plt.close(fig)
    print("wrote figures/issue_2094/gpu2_parent_gap_scatter.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
