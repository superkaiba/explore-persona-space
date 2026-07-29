"""#1774 follow-up figure — persona-averaged per-trait R² heatmap (zero-GPU round).

Renders ``figures/issue_1774/averaged_grain_trait_r2_heatmap`` from the committed
``eval_results/issue_1774/averaged_grain/averaged_grain_trait_table.json``
(fold-accumulated values): rows = the three trait directions plus the overall
persona-averaged read, columns = the four conditioning arms. Same viridis/imshow
conventions as the hero per-answer heatmap
(``issue1774_figures.fig_hero_trait_arm_heatmap``).

Usage: uv run python scripts/issue1774_averaged_grain_figure.py [--layer 14]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARMS = ("arm_context", "arm_prefix_end", "arm_bare_query", "arm_query_avg")
ARM_LABELS = {
    "arm_context": "Full context",
    "arm_prefix_end": "Prefix end",
    "arm_bare_query": "Bare query",
    "arm_query_avg": "Query-averaged prefix",
}
TRAITS = ("evil", "sycophancy", "hallucination")
ROW_LABELS = ["Evil", "Sycophancy", "Hallucination", "Overall (all coords)"]
TABLE_PATH = PROJECT_ROOT / "eval_results/issue_1774/averaged_grain/averaged_grain_trait_table.json"


def main(argv: list[str] | None = None) -> int:
    """Render the averaged-grain trait heatmap from the committed table; returns 0 on success."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures/issue_1774"))
    args = ap.parse_args(argv)
    per = json.loads(TABLE_PATH.read_text())["per_arm_layer"]
    M = np.full((len(ROW_LABELS), len(ARMS)), np.nan)
    for j, arm in enumerate(ARMS):
        blk = per[f"{arm}_L{args.layer}"]
        for i, t in enumerate(TRAITS):
            M[i, j] = blk["per_trait_r2_averaged_foldacc"][t]
        M[len(TRAITS), j] = blk["r2_averaged_overall_fold_mean"]
    assert np.isfinite(M).all(), M
    n_prefixes = per[f"{ARMS[0]}_L{args.layer}"]["n_prefixes"]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.6), layout="constrained")
    im = ax.imshow(M, cmap="viridis", vmin=min(0.0, float(np.nanmin(M))), vmax=float(np.nanmax(M)))
    ax.set_xticks(range(len(ARMS)), [ARM_LABELS[a] for a in ARMS], rotation=20, ha="right")
    ax.set_yticks(range(len(ROW_LABELS)), ROW_LABELS)
    for i in range(len(ROW_LABELS)):
        for j in range(len(ARMS)):
            ax.text(
                j,
                i,
                f"{M[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if im.norm(M[i, j]) < 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, label="Held-out persona-averaged R²")
    ax.set_title(
        f"Persona-averaged trait readability by arm (L{args.layer}, {n_prefixes} prefixes)"
    )
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "averaged_grain_trait_r2_heatmap", dir=fig_dir)
    plt.close(fig)
    print(f"[averaged-grain-fig] written -> {fig_dir}/averaged_grain_trait_r2_heatmap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
