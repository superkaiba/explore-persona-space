"""Issue #472 follow-up figures: negative budget vs source implantation + controlled nulls.

Figure 1 (negatives_source_implantation): (a) source marker log-prob shift vs the
total contrastive-negative row budget; (b) bystander leakage vs source implantation
per cell x seed, with the constant-fraction reference line.

Figure 2 (leakage_controlled_nulls): leakage NORMALIZED by source implantation
(bystander shift / source shift) across (a) placement arms, (b) persona splits at a
fixed total budget, (c) per-bystander distance to the nearest trained negative at
layer 20.

Data: terminal checkpoint of eval_results/issue_472/c472_*/trajectory.json.
Panels are reconstructed with the realized layer-10 selector (how the arms were
actually trained); panel-(c) distances are read from the layer-20 centroid bundle.
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472 import CELL_SPECS
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    cos_to_source as load_cts,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    load_cos_matrix,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
    d_nearest_neg,
    d_source,
    negatives_for_cell,
)

RESULTS = Path("eval_results/issue_472")
TOTALS = {
    "c472_noneg": 0,
    "c472_negex_100": 400,
    "c472_negp_2": 400,
    "c472_single_near": 400,
    "c472_single_far": 400,
    "c472_anchor": 800,
    "c472_near": 800,
    "c472_far": 800,
    "c472_negex_400": 1600,
    "c472_negp_8": 1600,
}
ARM_LABELS = {"c472_near": "Near", "c472_anchor": "Spread", "c472_far": "Far"}


def load_cells() -> list[dict]:
    """Terminal-checkpoint reads: per cell x seed, source shift + per-persona bystander shifts."""
    cells = []
    for path in sorted(glob(str(RESULTS / "c472_*" / "trajectory.json"))):
        d = json.load(open(path))
        ck = d["checkpoints"][-1]
        per_persona: dict[str, float] = {}
        for persona, qs in ck["held_out"].items():
            vals = [
                r["delta_g"]
                for r in qs.values()
                if isinstance(r, dict) and r.get("delta_g") is not None and not r.get("r_collapsed")
            ]
            if vals:
                per_persona[persona] = float(np.mean(vals))
        cells.append(
            dict(
                cell=d["cell"],
                seed=d["seed"],
                total=TOTALS[d["cell"]],
                src=ck["source_self"]["delta_g_mean"],
                per_persona=per_persona,
            )
        )
    return cells


def boot_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI of the mean over personas."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def figure_1(cells: list[dict]) -> None:
    set_paper_style("blog")
    fig, ax_a = plt.subplots(figsize=(6.8, 4.4))
    colors = paper_palette(4)
    total_levels = [0, 400, 800, 1600]
    total_color = dict(zip(total_levels, colors))

    # ---- (a) source implantation vs total negative rows ----
    rng = np.random.default_rng(0)
    for c in cells:
        x = total_levels.index(c["total"]) + rng.uniform(-0.10, 0.10)
        ax_a.scatter(x, c["src"], s=42, color=total_color[c["total"]], alpha=0.85, zorder=3)
    for i, t in enumerate(total_levels):
        vals = [c["src"] for c in cells if c["total"] == t]
        ax_a.hlines(np.mean(vals), i - 0.22, i + 0.22, color="#1A1A1A", lw=1.8, zorder=4)
    ax_a.set_xticks(range(4), [str(t) for t in total_levels])
    ax_a.set_xlabel("Total contrastive-negative rows (positives fixed at 200)")
    ax_a.set_ylabel("Source marker log-prob shift,\ntrained − base (nats)")
    set_title_subtitle(
        ax_a,
        "More negatives, much stronger source implantation",
        "Each point is one training cell × seed; black bar = mean per budget",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_472/negatives_source_implantation", dir="figures/")
    plt.close(fig)


def figure_2(cells: list[dict]) -> None:
    set_paper_style("blog")
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.5, 4.5))
    arm_colors = dict(zip(["c472_near", "c472_anchor", "c472_far"], paper_palette(3)))

    by_cell: dict[tuple[str, int], dict] = {(c["cell"], c["seed"]): c for c in cells}

    def norm_per_persona(cell: str) -> np.ndarray:
        """Per-persona normalized leakage (bystander shift / source shift), seeds pooled."""
        per = {}
        for seed in (42, 137):
            c = by_cell[(cell, seed)]
            for p, v in c["per_persona"].items():
                per.setdefault(p, []).append(v / c["src"])
        return np.array([np.mean(v) for v in per.values()])

    # ---- (a) placement of negatives relative to source ----
    arms = ["c472_near", "c472_anchor", "c472_far"]
    means, los, his = [], [], []
    for a in arms:
        vals = norm_per_persona(a)
        m = vals.mean()
        lo, hi = boot_ci(vals)
        means.append(m), los.append(m - lo), his.append(hi - m)
    grand = float(np.mean(means))
    # run-noise reference: the two identical-mix runs' normalized-leakage gap
    rep_gap = abs(
        norm_per_persona("c472_negp_2").mean() - norm_per_persona("c472_single_near").mean()
    )
    ax_a.axhspan(grand - rep_gap / 2, grand + rep_gap / 2, color="#000000", alpha=0.07, zorder=1)
    ax_a.bar(
        range(3),
        means,
        yerr=[los, his],
        color=[arm_colors[a] for a in arms],
        width=0.62,
        capsize=3,
        zorder=3,
    )
    ax_a.text(
        2.42,
        grand + rep_gap / 2 - 0.004,
        "identical-mix\nrerun gap",
        fontsize=8.5,
        color="#555555",
        ha="right",
        va="top",
    )
    ax_a.set_xticks(range(3), [ARM_LABELS[a] for a in arms])
    ax_a.set_xlabel("Negative placement relative to source")
    ax_a.set_ylabel("Bystander shift ÷ source shift")
    ax_a.set_ylim(0.40, 0.62)
    set_title_subtitle(
        ax_a,
        "Placement of negatives: no usable effect",
        "All arms at 800 rows; grey band = run-noise scale",
    )

    # ---- (b) persona split at fixed total budget ----
    rep_gap_b = rep_gap  # identical-mix rerun gap, reused as the run-noise band per group
    groups = [
        (
            "total = 400 rows",
            [
                ("c472_negex_100", "4p × 100"),
                ("c472_negp_2", "2p × 200"),
            ],
        ),
        (
            "total = 1600 rows",
            [
                ("c472_negex_400", "4p × 400"),
                ("c472_negp_8", "8p × 200"),
            ],
        ),
    ]
    xpos, xticklabels, x = [], [], 0.0
    neutral = paper_palette(5)[4]
    for glabel, cellspecs in groups:
        gx, gmeans = [], []
        for cell, label in cellspecs:
            vals = norm_per_persona(cell)
            m = vals.mean()
            lo, hi = boot_ci(vals)
            ax_b.bar(
                x, m, yerr=[[m - lo], [hi - m]], width=0.62, capsize=3, zorder=3, color=neutral
            )
            xpos.append(x), xticklabels.append(label)
            gx.append(x), gmeans.append(m)
            x += 1.3
        gmean = float(np.mean(gmeans))
        ax_b.fill_between(
            [min(gx) - 0.45, max(gx) + 0.45],
            gmean - rep_gap_b / 2,
            gmean + rep_gap_b / 2,
            color="#000000",
            alpha=0.07,
            zorder=1,
        )
        x += 0.9
    ax_b.set_xticks(xpos, xticklabels, fontsize=9)
    ax_b.text(0.65, 0.835, "total = 400 rows", ha="center", fontsize=9.5, color="#555555")
    ax_b.text(4.15, 0.835, "total = 1600 rows", ha="center", fontsize=9.5, color="#555555")
    ax_b.set_ylabel("Bystander shift ÷ source shift")
    ax_b.set_ylim(0.40, 0.86)
    set_title_subtitle(
        ax_b,
        "Split between personas: no effect",
        "Grey band = identical-mix rerun gap (run-noise scale)",
    )

    # ---- (c) bystander distance to nearest negative, layer 20: binned trend ----
    cts10 = load_cts(10, "villain")  # panels are realized layer-10 objects
    panels = {s[0]: negatives_for_cell(s[0], cts10) for s in CELL_SPECS}
    cosm20, _ = load_cos_matrix(20)
    cts20 = load_cts(20, "villain")
    pts_x, pts_y, pts_ds = [], [], []
    for a in arms:
        per: dict[str, list[float]] = {}
        for seed in (42, 137):
            c = by_cell[(a, seed)]
            for p, v in c["per_persona"].items():
                per.setdefault(p, []).append(v / c["src"])
        for p, vs in per.items():
            pts_x.append(d_nearest_neg(p, panels[a], cosm20))
            pts_y.append(float(np.mean(vs)))
            pts_ds.append(d_source(p, cts20))
    pts_x, pts_y, pts_ds = np.array(pts_x), np.array(pts_y), np.array(pts_ds)
    ax_c.scatter(pts_x, pts_y, s=22, color="#999999", alpha=0.55, zorder=2)
    # stats: bivariate Spearman + partial Spearman controlling distance-to-source
    rho_biv, p_biv = spearmanr(pts_x, pts_y)

    def _resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        b1, b0 = np.polyfit(x, y, 1)
        return y - (b1 * x + b0)

    rho_par, p_par = spearmanr(_resid(pts_x, pts_ds), _resid(pts_y, pts_ds))
    ax_c.annotate(
        f"Spearman rho = {rho_biv:+.2f}, p = {p_biv:.3f}\n"
        f"controlling distance to source:\nrho = {rho_par:+.2f}, p = {p_par:.2f}",
        xy=(0.97, 0.96),
        va="top",
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="#555555",
    )
    ax_c.set_xlabel("Distance to nearest trained negative\n(1 − cosine, layer 20)")
    ax_c.set_ylabel("Bystander shift ÷ source shift")
    set_title_subtitle(
        ax_c,
        "Distance to negatives: no effect net of source distance",
        "One point per held-out persona per arm, seeds pooled",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_472/leakage_controlled_nulls", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    cells = load_cells()
    figure_1(cells)
    figure_2(cells)
    print("done")
