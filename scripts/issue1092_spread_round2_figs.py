"""Figures for the #1092 round-2 spread reads (whitened metric + MLP Jensen).

Reads the cross-join npz (inline_spread_crossjoin) and writes two figures:

  perprefix_whitened_spread_vs_error : spread_w vs e_avgctx (top) and vs d_pe
      (bottom), per cell — the metric-flip headline (raw was null/inverted).
  perprefix_jensen_vs_spread : Jensen gap vs RAW spread (top) and vs WHITENED
      spread (bottom), per cell — curvature tracks ambient dispersion, not the
      whitened difficulty axis.
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
SRC = PROJECT_ROOT / "eval_results/issue_1092/inline_spread_crossjoin"
FIGDIR = PROJECT_ROOT / "figures/summaries/prefix_vs_context_map"
CELLS = ["cell_inst_own", "cell_pre_own"]
CELL_LABEL = {
    "cell_inst_own": "Instruct model (own answers)",
    "cell_pre_own": "Pretrained-base model (own answers)",
}


def _panel(ax, x, y, xlabel, ylabel, color, title=None) -> None:
    rho, p = stats.spearmanr(x, y)
    p_txt = f"p = {p:.1g}" if p >= 1e-200 else "p < 1e-200"
    ax.scatter(
        x,
        y,
        s=8,
        alpha=0.35,
        edgecolor="none",
        color=color,
        label=f"Spearman ρ = {rho:+.2f}, {p_txt}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9)


def fig_whitened_error() -> None:
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex="col")
    for col, cell in enumerate(CELLS):
        z = np.load(SRC / f"per_prefix_crossjoin_{cell}.npz")
        _panel(
            axes[0, col],
            z["spread_whitened"],
            z["e_avgctx"],
            "",
            "Averaged-context map per-prefix error",
            colors[0],
            title=CELL_LABEL[cell],
        )
        _panel(
            axes[1, col],
            z["spread_whitened"],
            z["d_pe"],
            "Within-prefix context-vector spread (whitened, #658 metric)",
            "Averaged-context err − prefix-end err",
            colors[1],
        )
        axes[1, col].axhline(0.0, color="0.35", lw=1.0, ls="--")
    fig.suptitle("WHITENED within-prefix spread vs error (996 natural prefixes, ambient)")
    fig.tight_layout()
    savefig_paper(fig, "perprefix_whitened_spread_vs_error", dir=FIGDIR)
    plt.close(fig)


def fig_jensen() -> None:
    set_paper_style("blog")
    colors = paper_palette(2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, cell in enumerate(CELLS):
        z = np.load(SRC / f"per_prefix_crossjoin_{cell}.npz")
        _panel(
            axes[0, col],
            z["spread_raw"],
            z["jensen"],
            "Within-prefix spread (raw L2)",
            "MLP Jensen gap ‖mean h(x) − h(x̄)‖",
            colors[0],
            title=CELL_LABEL[cell],
        )
        _panel(
            axes[1, col],
            z["spread_whitened"],
            z["jensen"],
            "Within-prefix spread (whitened)",
            "MLP Jensen gap ‖mean h(x) − h(x̄)‖",
            colors[1],
        )
    fig.suptitle("Nonlinear (MLP) Jensen gap vs spread metric (996 natural prefixes)")
    fig.tight_layout()
    savefig_paper(fig, "perprefix_jensen_vs_spread", dir=FIGDIR)
    plt.close(fig)


def main() -> int:
    fig_whitened_error()
    fig_jensen()
    print(f"wrote 2 figures to {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
