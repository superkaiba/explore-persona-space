"""Figure for the #1092 averaged-context-map spread-delta test (analysis-only).

Reads the per-unit npz written by scripts/issue1092_avgctx_spread_delta.py and
plots, per cell (ambient basis), within-prefix context-vector spread vs the two
averaging-specific error gaps:

  d_pe  = err(averaged-context map) - err(prefix-end map)
  d_ctx = err(averaged-context map) - err(query-averaged per-row context preds)

One 2x2 grid: rows = gap type, cols = cell. Spearman rho + p per panel in the
legend label (the perprefix_error_vs_spread convention).
"""

from __future__ import annotations

from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "eval_results/issue_1092/inline_avgctx_spread_delta"
FIGDIR = PROJECT_ROOT / "figures/summaries/prefix_vs_context_map"
CELLS = ["cell_inst_own", "cell_pre_own"]
CELL_LABEL = {
    "cell_inst_own": "Instruct model (own answers)",
    "cell_pre_own": "Pretrained-base model (own answers)",
}
GAPS = [
    ("d_pe", "Averaged-context map err − prefix-end map err"),
    ("d_ctx", "Averaged-context map err − avg-of-context-preds err"),
]


def main() -> int:
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex="col")
    for col, cell in enumerate(CELLS):
        z = np.load(SRC / f"per_prefix_avgctx_{cell}_ambient.npz", allow_pickle=True)
        s = z["spread"]
        for row, (key, ylabel) in enumerate(GAPS):
            ax = axes[row, col]
            d = z[key]
            rho, p = stats.spearmanr(s, d)
            p_txt = f"p = {p:.1g}" if p >= 1e-200 else "p < 1e-200"
            ax.scatter(
                s,
                d,
                s=8,
                alpha=0.35,
                edgecolor="none",
                color=colors[row],
                label=f"Spearman ρ = {rho:+.2f}, {p_txt}",
            )
            ax.axhline(0.0, color="0.35", lw=1.0, ls="--")
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(CELL_LABEL[cell])
            if row == 1:
                ax.set_xlabel("Within-prefix context-vector spread (raw L2)")
            ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        "Averaging-specific error gap vs within-prefix spread (996 natural prefixes, ambient)"
    )
    fig.tight_layout()
    savefig_paper(fig, "perprefix_avgctx_delta_vs_spread", dir=FIGDIR)
    plt.close(fig)
    print(f"wrote {FIGDIR}/perprefix_avgctx_delta_vs_spread.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
