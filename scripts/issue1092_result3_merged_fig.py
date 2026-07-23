"""Merged Result-3 figure: generated (constructed) + natural prefixes, one image.

Plot-only, banked inputs:
  panel A — generated/constructed conditions (#658 A3.5a substrate, layer 14):
            whitened within-condition spread vs held-out LOCO residual
            (`eval_results/issue_658/inline_a3_5a_coherence/per_condition_layer.npz`,
            spread_W[14] vs Rlin_loco_max[14], 50 conditions);
  panels B/C — natural WildChat/LMSYS prefixes (#1092, layer 14, ambient):
            whitened within-prefix spread vs the averaged read's per-prefix
            held-out error (strata + deepdive per-prefix arrays, 996 prefixes),
            instruct and base cells.
Each panel keeps its own axes (different substrates/whitening corpora and
error definitions — stated in the doc caption); per-panel Spearman rho + p.
Order proof for the natural panels: strata and deepdive n_turns identical.
"""

from __future__ import annotations

from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
C658 = PROJECT_ROOT / "eval_results/issue_658/inline_a3_5a_coherence/per_condition_layer.npz"
STRATA = PROJECT_ROOT / "eval_results/issue_1092/inline_spread_whitened_strata"
DEEPDIVE = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison_deepdive"
LAYER_658 = 14
CELLS = ["cell_inst_own", "cell_pre_own"]
CELL_LABELS = {"cell_inst_own": "instruct", "cell_pre_own": "base"}


def _annot(ax, rho: float, p: float) -> None:
    ptxt = "p < 1e-200" if p < 1e-200 else f"p = {p:.1e}"
    ax.text(
        0.03,
        0.95,
        f"Spearman ρ = +{rho:.2f}, {ptxt}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )


def main() -> int:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    pp.set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), layout="constrained")
    c_gen = pp.paper_palette_role("accent")
    c_nat = pp.paper_palette_role("primary")

    z = np.load(C658, allow_pickle=True)
    xg = np.asarray(z["spread_W"][LAYER_658], dtype=np.float64)
    yg = np.asarray(z["Rlin_loco_max"][LAYER_658], dtype=np.float64)
    rho, p = spearmanr(xg, yg)
    ax = axes[0]
    ax.scatter(xg, yg, s=26, alpha=0.75, color=c_gen, linewidths=0)
    _annot(ax, rho, p)
    ax.set_xlabel("within-condition context-vector spread (whitened)")
    ax.set_ylabel("held-out LOCO residual")
    ax.set_title("Generated prefixes (50 constructed conditions)", loc="left")

    for ax, cell in zip(axes[1:], CELLS, strict=True):
        spread = np.load(STRATA / f"per_prefix_whitened_{cell}.npz")["spread_whitened"]
        dd = np.load(DEEPDIVE / f"per_prefix_arrays_{cell}.npz")
        err = dd["err_ctx"]
        nt_strata = np.load(STRATA / f"per_prefix_whitened_{cell}.npz")["n_turns"]
        assert np.array_equal(np.asarray(nt_strata), np.asarray(dd["n_turns"])), (
            f"{cell}: per-prefix ordering differs between strata and deepdive arrays"
        )
        rho, p = spearmanr(spread, err)
        ax.scatter(spread, err, s=10, alpha=0.35, color=c_nat, linewidths=0)
        _annot(ax, rho, p)
        ax.set_xlabel("within-prefix context-vector spread (whitened)")
        ax.set_ylabel("averaged-map per-prefix held-out error")
        ax.set_title(f"Natural prefixes ({CELL_LABELS[cell]}, 996)", loc="left")

    pp.savefig_paper(
        fig,
        "summaries/prefix_vs_context_map/spread_vs_error_generated_and_natural",
        dir=str(PROJECT_ROOT / "figures"),
    )
    plt.close(fig)
    print("figure written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
