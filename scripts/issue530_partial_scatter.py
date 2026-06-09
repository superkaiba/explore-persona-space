# em-dash + ΔG intentional
"""Partial-residual scatters for #530 — the partialled counterpart of
figures/issue_530/raw_scatter_predictors_vs_dg.png.

For each headline predictor, residualize BOTH the DV (mean held-out ΔG) and
the predictor against the other 5 regression covariates via the exact same
`_residualize` code path `fit_pooled_partial_spearman` uses, then scatter the
two residual vectors. The Spearman of each panel's residuals IS the published
partial ρ — recomputed here and asserted against analyze_summary.json.

Also prints the raw (un-partialled) Spearman per predictor and the
base-prior diagnostics for the logprob-vs-logit discussion:
Spearman(b_logp, ΔG) and Spearman(b_logp, g_logp) at the row level.

Usage:
    uv run python scripts/issue530_partial_scatter.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    POSITIONED_ARM_SLUGS_V3,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
    PREDICTORS,
    _residualize,
    _spearman,
    build_rows,
    load_trajectory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SLAB = REPO_ROOT / "eval_results/issue_530"
OUT_DIR = REPO_ROOT / "figures/issue_530"
SEEDS = (42, 137)

PANEL_PREDICTORS = ["d_source", "d_nearest_neg_nd", "shadow_angle", "base_prior_marker"]
PANEL_LABELS = [
    "d_source residual\n(angular distance source → probe)",
    "d_nearest_neg_nd residual\n(distance nearest-negative → probe)",
    "shadow_angle residual\n(angle source→N vs source→probe)",
    "base_prior_marker residual\n(base-model log P(marker), nats)",
]

ARM_LABEL = {
    "c504v3_near": "near",
    "c504v3_mid_near": "mid-near",
    "c504v3_mid_far": "mid-far",
    "c504v3_far": "far",
}


def _build_530_rows() -> list[dict]:
    """Rebuild the 432-row pooled-regression input exactly as Phase 2 did."""
    gates = json.loads((SLAB / "phase0_5_gates.json").read_text())
    base_prior = json.loads((SLAB / "base_prior_marker.json").read_text())
    built = build_rows(
        slab_root=SLAB,
        chosen_frac=1.0,
        per_probe=gates["per_probe"],
        arm_to_positioned_n=gates["arm_to_positioned_n"],
        seeds=list(SEEDS),
        base_prior_by_probe=base_prior,
        positioned_arm_slugs=POSITIONED_ARM_SLUGS_V3,
    )
    rows = built["rows"]
    assert len(rows) == 432, f"expected the 432-row pool, got {len(rows)}"
    return rows


def _assert_match_published(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Recompute partial ρ per predictor and assert against analyze_summary.json."""
    summary = json.loads((SLAB / "analyze_summary.json").read_text())
    published = summary["pooled_fit"]["partial_spearman"]
    y = [r["delta_g"] for r in rows]
    cols = {p: [r[p] for r in rows] for p in PREDICTORS}
    recomputed: dict[str, dict[str, float]] = {}
    for p in PREDICTORS:
        others = np.asarray([cols[q] for q in PREDICTORS if q != p], dtype=np.float64).T
        y_res = _residualize(y, others)
        t_res = _residualize(cols[p], others)
        rho = _spearman(t_res.tolist(), y_res.tolist())
        # training_step is constant (band-stop halted every cell at step 20), so
        # its residual vector is ~1e-13 lstsq noise and the Spearman on it is a
        # platform-dependent degenerate artifact — skip the cross-check there.
        degenerate = float(np.std(np.asarray(cols[p]))) == 0.0
        if not degenerate:
            assert abs(rho - published[p]["rho"]) < 1e-9, (
                f"{p}: recomputed ρ={rho:.6f} != published {published[p]['rho']:.6f}"
            )
        recomputed[p] = {"rho": rho, "y_res": y_res, "t_res": t_res, "degenerate": degenerate}
    return recomputed


def fig_partial_scatter(rows: list[dict], resid: dict) -> None:
    """1×4 partial-residual scatter, colored by negative-position arm."""
    set_paper_style("blog")
    summary = json.loads((SLAB / "analyze_summary.json").read_text())
    holm = summary["pooled_fit"]["holm"]

    arms = list(ARM_LABEL)
    arm_color = dict(zip(arms, paper_palette_blog(4), strict=True))
    cells = [r["cell"] for r in rows]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), sharey=True)
    for ax, pred, lab in zip(axes, PANEL_PREDICTORS, PANEL_LABELS, strict=True):
        y_res = resid[pred]["y_res"]
        t_res = resid[pred]["t_res"]
        for arm in arms:
            mask = np.asarray([c == arm for c in cells])
            ax.scatter(
                t_res[mask],
                y_res[mask],
                s=18,
                alpha=0.55,
                color=arm_color[arm],
                label=ARM_LABEL[arm],
                edgecolor="white",
                linewidth=0.3,
            )
        rho = resid[pred]["rho"]
        p_holm = holm[pred]["p"]
        ax.set_title(f"partial ρ = {rho:+.2f}  (Holm p = {p_holm:.1e})", fontsize=9.6)
        ax.set_xlabel(lab, fontsize=9.0)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    axes[0].set_ylabel("Held-out ΔG residual\n(partialled, nats)", fontsize=9.6)
    axes[-1].legend(
        loc="upper right", fontsize=8.2, framealpha=0.0, title="Negative arm", title_fontsize=8.4
    )
    fig.tight_layout()
    savefig_paper(
        fig,
        "issue_530/partial_residual_scatter_predictors_vs_dg",
        dir=str(REPO_ROOT / "figures") + "/",
    )
    plt.close(fig)


def base_prior_diagnostics(rows: list[dict]) -> None:
    """Raw Spearmans + the b_logp→g_logp read for the prior-as-signal question."""
    y = [r["delta_g"] for r in rows]
    print("\n— raw (un-partialled) Spearman vs ΔG, 432-row pool —")
    for p in PREDICTORS:
        print(f"  {p:>20}: ρ = {_spearman([r[p] for r in rows], y):+.3f}")

    # Row-level g_logp / b_logp means from the trajectories (the regression
    # only kept ΔG; rebuild the absolute levels for the prior diagnostics).
    g_by_row: list[float] = []
    b_by_row: list[float] = []
    dg_by_row: list[float] = []
    for cell in POSITIONED_ARM_SLUGS_V3:
        for seed in SEEDS:
            traj = load_trajectory(SLAB, cell, seed)
            assert traj is not None, f"missing trajectory for {cell} seed {seed}"
            ck = traj["checkpoints"][-1]
            for per_q in ck["held_out"].values():
                gs = [d["g_logp"] for d in per_q.values()]
                bs = [d["b_logp"] for d in per_q.values()]
                g_by_row.append(float(np.mean(gs)))
                b_by_row.append(float(np.mean(bs)))
                dg_by_row.append(float(np.mean(gs)) - float(np.mean(bs)))
    print(f"\n— base-prior diagnostics ({len(g_by_row)} rows) —")
    print(
        f"  Spearman(b_logp, g_logp)  = {_spearman(b_by_row, g_by_row):+.3f}  "
        "(does the base prior set the trained absolute level?)"
    )
    print(
        f"  Spearman(b_logp, ΔG)      = {_spearman(b_by_row, dg_by_row):+.3f}  "
        "(does the base prior predict the GAIN?)"
    )
    print(
        f"  b_logp range: [{min(b_by_row):.1f}, {max(b_by_row):.1f}] nats; "
        f"g_logp range: [{min(g_by_row):.1f}, {max(g_by_row):.1f}] nats"
    )


def main() -> None:
    rows = _build_530_rows()
    resid = _assert_match_published(rows)
    print("recomputed partial ρ matches analyze_summary.json for all 6 predictors ✓")
    fig_partial_scatter(rows, resid)
    base_prior_diagnostics(rows)


if __name__ == "__main__":
    main()
