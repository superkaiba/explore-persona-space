"""Issue #1895 pure-residual-path-b follow-up figures (VM, CPU-only).

Inputs: committed eval_results/issue_1895/pure-residual-path-b/*.json +
perdirection_profiles.npz, plus the HF-staged round null_bands.npz
(issue1895_subspaces/pure_residual_path_b/eval_results_issue_1895/) for the
10,000 paired bootstrap draws.

Outputs (figures/issue_1895/, paper-plots conventions):
  pure_residual_delta_dark.{png,pdf,meta.json}    round headline: observed
      R^2(pure e_bar) vs its variance-profile plug-in + the paired-bootstrap
      delta distribution (all 10,000 draws <= 0).
  pure_residual_perdirection.{png,pdf,meta.json}  low-level per-direction view
      behind the aggregate: pure-error per-direction R^2 vs the plug-in
      profile g(u), sized by residual-energy share.

Run: uv run python scripts/issue1895_pure_residual_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + credentials BEFORE matplotlib/numpy (shared-VM harvest; #847)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

EVAL = Path("eval_results/issue_1895/pure-residual-path-b")
FIGDIR = Path("figures/issue_1895")
STAGE = Path("data/issue_1895/hf_dl/pure_residual_path_b")
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_NULLBANDS = "issue1895_subspaces/pure_residual_path_b/eval_results_issue_1895/null_bands.npz"


def _stage_nullbands() -> Path:
    out = STAGE / HF_NULLBANDS
    if not out.exists():
        from explore_persona_space.orchestrate.hub import stage_hub_file

        stage_hub_file(HF_REPO, HF_NULLBANDS, out, repo_type="dataset")
    assert out.exists(), "staging failed for round null_bands.npz"
    return out


def fig_delta_dark(ang: dict, delta_draws: np.ndarray, pal: list[str]) -> None:
    boot = ang["h3_plugin_bootstrap"]
    pure_pilot = ang["dark_spot_pilot_pure_ebar"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: observed pure-error R^2 vs plug-in, full holdout + pilot-row subset.
    reads = [
        (
            "full holdout\n(n=20,000)",
            boot["observed"]["r2_ebar"],
            boot["all_ci"]["r2_ebar"],
            boot["observed"]["plugin_e"],
        ),
        (
            "pilot-row subset\n(n=512)",
            pure_pilot["r2_pure_ebar"],
            None,
            pure_pilot["plugin_pure_ebar"],
        ),
    ]
    for i, (lab, r2, ci, pv) in enumerate(reads):
        a.bar(i, r2, color=pal[2], width=0.5)
        if ci is not None:
            a.errorbar(i, r2, yerr=[[r2 - ci[0]], [ci[1] - r2]], color="black", capsize=3, lw=1)
        a.text(i - 0.33, r2, f"{r2:.3f}", ha="right", va="center", fontsize=9)
        a.hlines(
            pv,
            i - 0.25,
            i + 0.25,
            color="black",
            ls="--",
            lw=1.4,
            label="variance-profile plug-in" if i == 0 else None,
        )
        a.text(i + 0.28, pv, f"{pv:.3f}", va="center", fontsize=8)
    a.set_xticks(range(2), [r[0] for r in reads])
    a.set_xlim(-0.6, 1.6)
    a.set_ylim(0, 0.34)
    a.set_ylabel("held-out pooled R$^2$ (pure SAE error $\\bar e$)")
    a.legend(fontsize=8, loc="lower right")

    # Right: the paired bootstrap delta distribution (10,000 draws).
    obs = boot["observed"]["delta_dark"]
    b.hist(delta_draws, bins=60, color=pal[2], alpha=0.85)
    b.axvline(0.0, color="black", lw=1.2)
    b.axvline(obs, color="black", ls="--", lw=1.4)
    b.text(obs, b.get_ylim()[1] * 0.95, f"observed {obs:+.4f}", ha="right", va="top", fontsize=9)
    b.set_xlim(min(delta_draws.min(), obs) - 0.002, 0.004)
    b.set_xlabel(r"$\Delta$ = R$^2$(pure $\bar e$) $-$ plug-in (paired draw)")
    b.set_ylabel("bootstrap draws (of 10,000)")
    savefig_paper(fig, "pure_residual_delta_dark", dir=FIGDIR)
    plt.close(fig)


def fig_perdirection(g: np.ndarray, r2e: np.ndarray, energy: np.ndarray, pal: list[str]) -> None:
    n = len(g)
    rank = np.arange(1, n + 1)
    share = energy / energy.sum()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        g,
        r2e,
        s=6 + 4000 * share,
        c=np.log10(rank),
        cmap="viridis",
        alpha=0.45,
        linewidths=0,
    )
    lim = (-0.05, 1.0)
    ax.plot(lim, lim, color="black", ls="--", lw=1.2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    offsets = {
        1: (0.015, -0.03),
        10: (-0.015, 0.03),
        100: (0.02, 0.0),
        1000: (0.02, 0.0),
        3584: (0.02, -0.03),
    }
    for r in (1, 10, 100, 1000, 3584):
        i = r - 1
        dx, dy = offsets[r]
        ha = "left" if dx > 0 else "right"
        ax.text(g[i] + dx, r2e[i] + dy, f"rank {r}", fontsize=8, va="center", ha=ha)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log10 eigenvalue rank")
    ax.set_xlabel("plug-in profile g(u): raw-target per-direction R$^2$")
    ax.set_ylabel("pure-error map per-direction R$^2$")
    savefig_paper(fig, "pure_residual_perdirection", dir=FIGDIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    pal = paper_palette_blog(5)
    ang = json.loads((EVAL / "angles_summary.json").read_text())
    with np.load(_stage_nullbands(), allow_pickle=False) as z:
        delta_draws = z["boot__delta_dark"].astype(np.float64)
        energy = z["bootprof__energy_e"].astype(np.float64).mean(axis=0)
    assert delta_draws.size == 10_000 and float((delta_draws <= 0).mean()) == 1.0
    with np.load(EVAL / "perdirection_profiles.npz", allow_pickle=False) as z:
        g = z["r2u__t_vA_ctx"].astype(np.float64)
        r2e = z["r2u__t_ebar_ctx"].astype(np.float64)
    # figure-input integrity: the per-direction data reproduce the committed aggregates
    plugin = float((energy * g).sum() / energy.sum())
    pooled = float((energy * r2e).sum() / energy.sum())
    assert abs(plugin - ang["h3_plugin_bootstrap"]["observed"]["plugin_e"]) < 1e-4
    assert abs(pooled - ang["h3_plugin_bootstrap"]["observed"]["r2_ebar"]) < 1e-4
    fig_delta_dark(ang, delta_draws, pal)
    fig_perdirection(g, r2e, energy, pal)
    print("done:", sorted(p.name for p in FIGDIR.glob("pure_residual_*")))


if __name__ == "__main__":
    main()
