"""Round-5 (second-behavior-rank-replication) figures for issue #642.

Reads eval_results/issue_642/refusal_v9/analysis.json (refusal) and
eval_results/issue_642_v4_analysis_workdir/sycophancy/analysis_v4.json
(round-4 sycophancy) and produces a single 2-panel figure for the new
finding:

  (left)  cross-behavior adapter-vs-dense bar: refusal Delta_rank_matched
          (+0.027, this round) beside sycophancy Delta_rank_matched
          (+0.063, round 4), both at s*=0.50 with bootstrap CIs and the
          +/-0.04 separation band.
  (right) per-persona profile scatter at s*=0.50: cmftRefOP vs loraRefOP
          per-bystander refusal-leakage delta, identity line, rho=0.94.

Both panels are RAW reads (per-persona deltas, no residualization), so no
separate raw-sibling figure is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[2]
REFUSAL = REPO / "eval_results/issue_642/refusal_v9/analysis.json"
_SYCO_REL = "eval_results/issue_642_v4_analysis_workdir/sycophancy/analysis_v4.json"
# The worktree is sparse for eval_results/; round-4 sycophancy analysis lives at
# the repo-root checkout. Resolve from the worktree first, fall back to repo root.
SYCO = REPO / _SYCO_REL
if not SYCO.exists():
    SYCO = Path.home() / "explore-persona-space" / _SYCO_REL
OUT_DIR = REPO / "figures/"  # savefig_paper prepends "figures/" -> save under figures/issue_642/


def main() -> None:
    ref = json.loads(REFUSAL.read_text())
    syco = json.loads(SYCO.read_text())

    rh = ref["headline"]["delta_rank_matched"]
    sh = syco["headline"]["contrasts"]["delta_rank_matched"]

    # --- cross-behavior bar data ---
    ref_point = rh["gap_plugin"]
    ref_ci = rh["gap_ci95"]
    syco_point = sh["gap_plugin"]
    syco_ci = sh["gap_ci95"]

    # --- per-persona profile (refusal) ---
    hi = rh["per_persona_hi"]  # cmftRefOP per-persona delta
    lo = rh["per_persona_lo"]  # loraRefOP per-persona delta
    personas = sorted(hi)
    x = [lo[p] for p in personas]  # LoRA
    y = [hi[p] for p in personas]  # dense (cmft)
    rho, _ = spearmanr(x, y)
    n_above = sum(1 for xi, yi in zip(x, y) if yi > xi)

    set_paper_style("blog")
    # The blog style turns on constrained_layout via rcParams, which conflicts with
    # the explicit subplots_adjust this 2-panel figure needs (memory:
    # feedback_set_title_subtitle_breaks_subplot_grids). Disable the rcParam BEFORE
    # creating the figure so no layout engine is attached, then use plain
    # left-aligned ax.set_title + explicit subplots_adjust.
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ===== Panel A: cross-behavior bar =====
    c_ref = paper_palette_role("primary")
    c_syco = paper_palette_role("baseline")
    xs = [0, 1]
    pts = [ref_point, syco_point]
    cis = [ref_ci, syco_ci]
    cols = [c_ref, c_syco]
    yerr = [
        [p - ci[0] for p, ci in zip(pts, cis)],
        [ci[1] - p for p, ci in zip(pts, cis)],
    ]
    axb.bar(xs, pts, width=0.55, color=cols, zorder=3)
    axb.errorbar(xs, pts, yerr=yerr, fmt="none", ecolor="#1A1A1A", capsize=5, lw=1.4, zorder=4)
    # +/-0.04 separation band
    axb.axhspan(-0.04, 0.04, color="#CCCCCC", alpha=0.35, zorder=0)
    axb.axhline(0.0, color="#7A7A7A", lw=0.8, zorder=1)
    axb.set_xticks(xs)
    axb.set_xticklabels(["Refusal\n(this round)", "Sycophancy\n(prior round)"])
    axb.set_ylabel("Dense FT − LoRA bystander leakage\n(trained − base)")
    axb.set_ylim(-0.06, 0.13)
    for xi, p, ci in zip(xs, pts, cis):
        axb.annotate(
            f"+{p:.3f}",
            (xi, ci[1]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=10,
            fontweight="semibold",
        )
    axb.annotate(
        "±0.04 band",
        (1.46, 0.012),
        ha="right",
        va="center",
        fontsize=8,
        color="#5A5A5A",
    )
    axb.set_title(
        "Adapter-vs-dense gap is positive for both behaviors",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=10,
    )

    # ===== Panel B: per-persona profile scatter (refusal) =====
    lim = max(max(x), max(y)) * 1.08
    lo_lim = min(min(x), min(y), 0) - 0.01
    axs.plot([lo_lim, lim], [lo_lim, lim], ls="--", color="#7A7A7A", lw=1.0, zorder=1)
    axs.scatter(
        x,
        y,
        s=34,
        color=paper_palette_role("control"),
        edgecolors="#1A1A1A",
        linewidths=0.6,
        alpha=0.9,
        zorder=3,
    )
    axs.set_xlabel("LoRA per-persona refusal leakage (trained − base)")
    axs.set_ylabel("Dense FT per-persona refusal leakage (trained − base)")
    axs.set_xlim(lo_lim, lim)
    axs.set_ylim(lo_lim, lim)
    axs.annotate(
        f"Spearman ρ = {rho:.2f}\n{n_above} of {len(personas)} above identity",
        (0.04, 0.96),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        color="#1A1A1A",
    )
    axs.set_title(
        "Same personas leak most under both fine-tunes",
        fontsize=11,
        fontweight="semibold",
        loc="left",
        pad=10,
    )

    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.14, wspace=0.42)

    saved = savefig_paper(fig, "issue_642/refusal_r5_cross_behavior", dir=str(OUT_DIR))
    plt.close(fig)
    print("saved:", {k: str(v) for k, v in saved.items()})
    print(
        f"refusal Δ_rank = {ref_point:.4f} CI {ref_ci}; syco Δ_rank = {syco_point:.4f} CI {syco_ci}"
    )
    print(f"profile rho = {rho:.4f}; {n_above}/{len(personas)} above identity")


if __name__ == "__main__":
    main()
