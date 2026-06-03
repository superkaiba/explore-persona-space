# Qwen marker token " ※" + em-dash intentional
"""Task #472 clean-result figures (analyzer). Blog style, plain-English labels.

Figures:
  hero_implant_drives_leakage  — held-out leakage vs source-implant strength (master).
  count_more_negatives_more_leakage — count axis bars (negex + negp), held-out ΔG.
  emission_floor — P(marker) on source vs bystander across cells (measurement validity).
  geometry_source_proximity — within-arm leakage vs distance-to-source (the gradient
        that survives) + identification-gate annotation (barrier/bubble indeterminate).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472 import CELL_SPECS
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    cos_to_source as lcts,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    load_cos_matrix,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
    d_source,
    held_out_panel,
)

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-472")
SLAB = WT / "eval_results" / "issue_472"
CDIR = WT / "data" / "issue_472"
FIG = WT / "figures" / "issue_472"
SEEDS = [42, 137]
POOLED = [c[0] for c in CELL_SPECS if c[5]]

CELL_LABEL = {
    "c472_noneg": "No negatives",
    "c472_negex_100": "Few negatives (1:2)",
    "c472_negp_2": "Few negatives (2 personas)",
    "c472_single_near": "Single near negative",
    "c472_single_far": "Single far negative",
    "c472_anchor": "Baseline (1:4)",
    "c472_near": "Near negatives",
    "c472_far": "Far negatives",
    "c472_negex_400": "Many negatives (1:8)",
    "c472_negp_8": "Many negatives (8 personas)",
}


def traj(cell, seed):
    return json.loads((SLAB / f"{cell}_seed{seed}" / "trajectory.json").read_text())


def earliest(cell, seed):
    return sorted(traj(cell, seed)["checkpoints"], key=lambda c: c["frac"])[0]


def held_mean_dg(cell, seed):
    ck = earliest(cell, seed)
    return float(np.mean([r["delta_g"] for per in ck["held_out"].values() for r in per.values()]))


def src_max_dg(cell, seed):
    return max(c["source_self"]["delta_g_mean"] for c in traj(cell, seed)["checkpoints"])


def src_pmarker(cell, seed):
    # P(marker) on the source's own response at end of training.
    last = sorted(traj(cell, seed)["checkpoints"], key=lambda c: c["frac"])[-1]
    return float(np.exp(last["source_self"]["g_logp_mean"]))


def held_argmax_share(cell, seed):
    ck = earliest(cell, seed)
    return float(
        np.mean([r["argmax_marker"] for per in ck["held_out"].values() for r in per.values()])
    )


def main():
    set_paper_style("blog")
    FIG.mkdir(parents=True, exist_ok=True)
    all_cells = [c[0] for c in CELL_SPECS]

    # ── HERO: held-out leakage vs source-implant strength. ──
    xs, ys, labs = [], [], []
    for cell in all_cells:
        for seed in SEEDS:
            xs.append(src_max_dg(cell, seed))
            ys.append(held_mean_dg(cell, seed))
            labs.append(cell)
    rho = spearmanr(xs, ys).correlation
    pe = pearsonr(xs, ys)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(
        xs,
        ys,
        s=46,
        alpha=0.85,
        color=paper_palette_role("primary"),
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    # mark the no-neg points distinctly
    nx = [x for x, l in zip(xs, labs) if l == "c472_noneg"]
    ny = [y for y, l in zip(ys, labs) if l == "c472_noneg"]
    ax.scatter(
        nx,
        ny,
        s=70,
        color=paper_palette_role("control"),
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="No-negatives arm",
    )
    coef = np.polyfit(xs, ys, 1)
    xl = np.linspace(min(xs), max(xs), 20)
    ax.plot(xl, np.polyval(coef, xl), color=paper_palette_role("baseline"), lw=1.6, zorder=2)
    ax.set_xlabel("How hard the marker was implanted on the source (ΔG, nats)")
    ax.set_ylabel("Bystander marker leakage (ΔG, nats)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    set_title_subtitle(
        ax,
        "Bystander leakage tracks source-implant strength, not recipe geometry",
        f"Each point = 1 cell × seed (n=20). Spearman {rho:.2f}, p = {pe[1]:.0e}.",
        source="Task #472 · 47 held-out probes · layer 10",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_472/hero_implant_drives_leakage", dir=str(FIG.parent))
    plt.close(fig)

    # ── COUNT: more negatives -> more leakage. ──
    def cell_mean_seedavg(cell):
        return float(np.mean([held_mean_dg(cell, s) for s in SEEDS]))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.2, 4.0), sharey=True)
    nx_cells = ["c472_negex_100", "c472_anchor", "c472_negex_400"]
    nx_lab = ["100", "200", "400"]
    np_cells = ["c472_negp_2", "c472_anchor", "c472_negp_8"]
    np_lab = ["2", "4", "8"]
    col = paper_palette_role("primary")
    axl.bar(nx_lab, [cell_mean_seedavg(c) for c in nx_cells], color=col, width=0.6)
    axl.set_xlabel("Negative examples per persona")
    axl.set_ylabel("Bystander leakage (ΔG, nats)")
    axl.set_title("More examples", fontsize=10)
    axr.bar(
        np_lab,
        [cell_mean_seedavg(c) for c in np_cells],
        color=paper_palette_role("accent"),
        width=0.6,
    )
    axr.set_xlabel("Number of negative personas")
    axr.set_title("More personas", fontsize=10)
    for ax, cells in ((axl, nx_cells), (axr, np_cells)):
        for i, c in enumerate(cells):
            v = cell_mean_seedavg(c)
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    fig.suptitle(
        "Adding negatives raises bystander leakage — the opposite of suppression",
        fontsize=11,
        fontweight="semibold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_paper(fig, "issue_472/count_more_negatives_more_leakage", dir=str(FIG.parent))
    plt.close(fig)

    # ── EMISSION FLOOR: P(marker) source vs bystander across cells (measurement validity). ──
    order = [
        "c472_noneg",
        "c472_negp_2",
        "c472_negex_100",
        "c472_anchor",
        "c472_near",
        "c472_far",
        "c472_negp_8",
        "c472_negex_400",
    ]
    src_p = [float(np.mean([src_pmarker(c, s) for s in SEEDS])) for c in order]
    by_p = [float(np.mean([held_argmax_share(c, s) for s in SEEDS])) for c in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    w = 0.38
    ax.bar(x - w / 2, src_p, w, label="Source persona", color=paper_palette_role("primary"))
    ax.bar(x + w / 2, by_p, w, label="Bystanders (mean)", color=paper_palette_role("control"))
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABEL[c] for c in order], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Marker emission probability  P(※)")
    ax.set_ylim(0, 0.20)
    ax.legend(frameon=False, fontsize=8)
    set_title_subtitle(
        ax,
        "The marker never actually emits — not on bystanders, barely on the source",
        "Even the strongest cell reaches only P(※) ≈ 0.17 on the source; bystander emission ≈ 0% everywhere.",
        source="Task #472 · greedy next-token marker probability at the post-response slot",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_472/emission_floor", dir=str(FIG.parent))
    plt.close(fig)

    # ── GEOMETRY: within-arm leakage vs distance-to-source (the surviving gradient). ──
    cts = lcts(10, "villain", CDIR)
    cm, _ = load_cos_matrix(10, CDIR)
    panel = held_out_panel(cts, source="villain")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    arm_lab = {"c472_near": "Near", "c472_anchor": "Spread", "c472_far": "Far"}
    arm_col = {
        "c472_near": paper_palette_role("primary"),
        "c472_anchor": paper_palette_role("accent"),
        "c472_far": paper_palette_role("control"),
    }
    allx, ally = [], []
    for slug in POOLED:
        xs2, ys2 = [], []
        for seed in SEEDS:
            ck = earliest(slug, seed)
            for p in panel:
                if p in ck["held_out"]:
                    dg = float(np.mean([r["delta_g"] for r in ck["held_out"][p].values()]))
                    xs2.append(d_source(p, cts))
                    ys2.append(dg)
        ax.scatter(xs2, ys2, s=22, alpha=0.45, color=arm_col[slug], label=arm_lab[slug])
        allx += xs2
        ally += ys2
    coef = np.polyfit(allx, ally, 1)
    xl = np.linspace(min(allx), max(allx), 20)
    ax.plot(xl, np.polyval(coef, xl), color=paper_palette_role("baseline"), lw=1.6)
    rho2 = spearmanr(allx, ally).correlation
    ax.set_xlabel("Distance from bystander to source  (1 − cosine, layer 10)")
    ax.set_ylabel("Bystander leakage (ΔG, nats)")
    ax.legend(frameon=False, fontsize=8, title="Negative placement", title_fontsize=8)
    set_title_subtitle(
        ax,
        "Bystanders closer to the source leak more — placement makes no difference",
        f"Spearman(leakage, distance-to-source) = {rho2:.2f}. Near/Spread/Far arms overlap entirely.",
        source="Task #472 · pooled placement arms · n = 282 probe×arm×seed",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_472/geometry_source_proximity", dir=str(FIG.parent))
    plt.close(fig)

    print("All figures written to", FIG)


if __name__ == "__main__":
    main()
