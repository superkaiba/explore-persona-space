# ruff: noqa: RUF001  # Qwen marker token, em-dash, multiplication sign all intentional
"""Task #472 clean-result figures (analyzer). Blog style, plain-English labels.

Figures:
  hero_implant_drives_leakage  — held-out leakage vs source-implant strength, BOTH read
        at the earliest checkpoint (matched slice), colored by training step to disclose
        the step confound (round-2 fix; round-1 used x=max / y=earliest).
  count_more_negatives_more_leakage — count axis bars (negex + negp), held-out ΔG.
  emission_floor — TWO panels: source P(marker) (probability) and bystander argmax-marker
        rate (% of slots), kept on separate axes so probability and rate are not mixed.
  geometry_source_proximity — within-arm leakage vs distance-to-source (the gradient
        that survives, holds at L10/15/20); placement arms overlap (barrier/bubble
        indeterminate, see reanalysis_earliest_slice.json + reanalysis_multilayer.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

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


def src_earliest_dg(cell, seed):
    # Source-self implant strength read at the SAME earliest checkpoint as the
    # held-out leakage (matched-slice; avoids the round-1 x=max / y=earliest gap).
    return earliest(cell, seed)["source_self"]["delta_g_mean"]


def earliest_step(cell, seed):
    ck = earliest(cell, seed)
    return ck.get("step", ck.get("global_step"))


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

    # ── HERO: held-out leakage vs source-implant strength (BOTH read at the same
    #    earliest checkpoint; color encodes absolute training step at that ckpt to
    #    disclose the step confound). ──
    xs, ys, labs, steps = [], [], [], []
    for cell in all_cells:
        for seed in SEEDS:
            xs.append(src_earliest_dg(cell, seed))
            ys.append(held_mean_dg(cell, seed))
            labs.append(cell)
            steps.append(earliest_step(cell, seed))
    rho = spearmanr(xs, ys).correlation
    # The read checkpoint sits at one of four absolute step counts (2/4/6/10);
    # encode it as discrete marker shades so the step confound is visible without
    # a continuous colorbar (which conflicts with the blog constrained layout).
    step_levels = sorted(set(steps))
    shades = ["#cfe3f2", "#8ec3e6", "#3f8fd0", "#0d4f8b"]
    step_color = {s: shades[min(i, len(shades) - 1)] for i, s in enumerate(step_levels)}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for s in step_levels:
        sx = [x for x, st in zip(xs, steps, strict=False) if st == s]
        sy = [y for y, st in zip(ys, steps, strict=False) if st == s]
        ax.scatter(
            sx,
            sy,
            s=54,
            alpha=0.9,
            color=step_color[s],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=f"{s} steps",
        )
    coef = np.polyfit(xs, ys, 1)
    xl = np.linspace(min(xs), max(xs), 20)
    ax.plot(xl, np.polyval(coef, xl), color=paper_palette_role("baseline"), lw=1.5, zorder=2)
    ax.set_xlabel("Source-implant strength at the read checkpoint (ΔG, nats)")
    ax.set_ylabel("Bystander marker leakage (ΔG, nats)")
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
        title="Training steps at read",
        title_fontsize=8,
    )
    set_title_subtitle(
        ax,
        "Held-out leakage drifts together with source implant and training step",
        f"Each point = 1 cell × seed (n=20), both axes at the earliest checkpoint. "
        f"Spearman {rho:.2f}; the read step (shade) co-moves — see body.",
        source="Task #472 · 47 held-out probes · layer 10",
    )
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

    # ── EMISSION FLOOR (measurement validity): TWO panels with distinct y-units so
    #    probability and rate are never mixed on one axis (round-1 fix).
    #    Left  = source-self P(marker) (probability, terminal seed-avg).
    #    Right = bystander argmax-marker rate (% of held-out probe slots, earliest ck). ──
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
    by_rate = [100.0 * float(np.mean([held_argmax_share(c, s) for s in SEEDS])) for c in order]
    x = np.arange(len(order))
    labels = [CELL_LABEL[c] for c in order]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.6, 4.4))

    axl.bar(x, src_p, 0.62, color=paper_palette_role("primary"))
    axl.set_xticks(x)
    axl.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    axl.set_ylabel("Source P(※)  (emission probability)")
    axl.set_ylim(0, 0.20)
    axl.set_title("Source persona: marker probability", fontsize=10)
    for i, v in enumerate(src_p):
        if v > 0.005:
            axl.text(i, v + 0.004, f"{v:.2f}", ha="center", fontsize=7.5)

    axr.bar(x, by_rate, 0.62, color=paper_palette_role("control"))
    axr.set_xticks(x)
    axr.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    axr.set_ylabel("Bystander argmax-marker rate  (% of probe slots)")
    axr.set_ylim(0, 2.6)
    axr.set_title("Bystanders: marker-is-greedy rate", fontsize=10)
    for i, v in enumerate(by_rate):
        if v > 0.01:
            axr.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=7.5)

    fig.suptitle(
        "The marker stays sub-emission: barely on the source, near-zero on bystanders",
        fontsize=11,
        fontweight="semibold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_paper(fig, "issue_472/emission_floor", dir=str(FIG.parent))
    plt.close(fig)

    # ── GEOMETRY: within-arm leakage vs distance-to-source (the surviving gradient). ──
    cts = lcts(10, "villain", CDIR)
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
        "Bystanders closer to the source show higher ΔG — placement makes no difference",
        f"Spearman(ΔG, distance-to-source) = {rho2:.2f} (holds at layers 10/15/20). "
        f"Near/Spread/Far arms overlap entirely.",
        source="Task #472 · pooled placement arms · n = 282 probe×arm×seed",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_472/geometry_source_proximity", dir=str(FIG.parent))
    plt.close(fig)

    print("All figures written to", FIG)


if __name__ == "__main__":
    main()
