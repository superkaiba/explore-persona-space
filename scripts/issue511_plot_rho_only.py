# ruff: noqa: RUF001  # Greek rho + multiplication sign in figure text intentional
"""Issue #511: headline-cell |Spearman rho| vs N probes per persona, RHO ONLY
(the left panel of convergence_fixed_cell.png, without the CV R^2 panel).

Usage: uv run python scripts/issue511_plot_rho_only.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

try:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "eval_results/issue_511/probe_count_sweep_results.json"
OUT = REPO / "figures/issue_511/convergence_rho_only.png"
CELL = "last_prompt__L22__gauss_kl__raw"
NS = [25, 50, 100, 200, 350, 500]


def main():
    agg = json.loads(SWEEP.read_text())["aggregates"]
    mean = [agg[f"{CELL}|loc|1|{n}"]["abs_rho_mean"] for n in NS]
    std = [agg[f"{CELL}|loc|1|{n}"]["abs_rho_std"] for n in NS]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(NS, mean, yerr=std, marker="o", ms=7, color="#1f4e79", lw=2, capsize=4, zorder=3)
    for n, m in zip(NS, mean, strict=False):
        ax.annotate(
            f"{m:.3f}",
            (n, m),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("N probes per persona", fontsize=11)
    ax.set_ylabel("|length-partial Spearman ρ| vs ΔG", fontsize=11)
    ax.set_xticks(NS)
    ax.grid(alpha=0.25, zorder=0)

    fig.suptitle(
        "#511 — headline cell |ρ| vs probe count (last-prompt × L22 × Gaussian-KL)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    for n, m, s in zip(NS, mean, std, strict=False):
        print(f"  N={n:>3}: |rho| = {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
