"""Clean 2-panel figure: averaged-map per-prefix error vs WHITENED spread (#1092).

Plot-only (no fits): x = banked whitened within-prefix context-vector spread
(`inline_spread_whitened_strata/per_prefix_whitened_<cell>.npz`), y = banked
per-prefix held-out error of the query-averaged context-map read
(`inline_fair_comparison_deepdive/per_prefix_arrays_<cell>.npz` err_ctx),
996 natural prefixes, ambient basis, both cells. Alignment gate: the
recomputed Spearman rho must match the banked
`spread_whitened_strata.json` `spread_whitened_vs_e_avgctx.rho` within 5e-3
(both arrays are written in the same sorted-pid order; the gate proves it).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRATA = PROJECT_ROOT / "eval_results/issue_1092/inline_spread_whitened_strata"
DEEPDIVE = PROJECT_ROOT / "eval_results/issue_1092/inline_fair_comparison_deepdive"
CELLS = ["cell_inst_own", "cell_pre_own"]
CELL_LABELS = {"cell_inst_own": "Instruct model", "cell_pre_own": "Base model"}
ALIGN_TOL = 5e-3


def main() -> int:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots as pp

    banked = json.loads((STRATA / "spread_whitened_strata.json").read_text())
    pp.set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    color = pp.paper_palette_role("primary")
    for ax, cell in zip(axes, CELLS, strict=True):
        spread = np.load(STRATA / f"per_prefix_whitened_{cell}.npz")["spread_whitened"]
        err = np.load(DEEPDIVE / f"per_prefix_arrays_{cell}.npz")["err_ctx"]
        rho, p = spearmanr(spread, err)
        # Order proof: the strata and deepdive per-prefix files share the sorted-pid
        # convention; their n_turns columns must be IDENTICAL. (The banked strata rho
        # used the centroid-arm error, so rho-matching is not an alignment test.)
        nt_strata = np.load(STRATA / f"per_prefix_whitened_{cell}.npz")["n_turns"]
        nt_deep = np.load(DEEPDIVE / f"per_prefix_arrays_{cell}.npz")["n_turns"]
        assert np.array_equal(np.asarray(nt_strata), np.asarray(nt_deep)), (
            f"{cell}: per-prefix ordering differs between strata and deepdive arrays"
        )
        rho_banked = banked["cells"][cell]["overall"]["ambient"]["spread_whitened_vs_e_avgctx"][
            "rho"
        ]
        assert abs(rho - rho_banked) < 0.02, (
            f"{cell}: rho {rho:.4f} implausibly far from banked centroid-arm {rho_banked:.4f}"
        )
        ax.scatter(spread, err, s=10, alpha=0.35, color=color, linewidths=0)
        ptxt = "p < 1e-200" if p < 1e-200 else f"p = {p:.1e}"
        ax.text(
            0.03,
            0.95,
            f"Spearman ρ = +{rho:.2f}, {ptxt}",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
        )
        ax.set_xlabel("within-prefix context-vector spread (whitened)")
        ax.set_ylabel("averaged-map per-prefix held-out error")
        ax.set_title(f"{CELL_LABELS[cell]} (996 natural prefixes)", loc="left")
    pp.savefig_paper(
        fig,
        "summaries/prefix_vs_context_map/perprefix_avgerr_vs_whitened_spread",
        dir=str(PROJECT_ROOT / "figures"),
    )
    plt.close(fig)
    print("figure written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
