"""Issue #224 hero figures.

Two figures, both saved via paper_plots:

1) `trained_vs_base_librarian.{png,pdf}` — per-layer marker-minus-C1 system_B
   delta, trained vs base on the SAME force-fed librarian token sequences.
   The visual takeaway: the two curves overlap → LoRA training did not induce
   the system-attention rise. Hero figure for the clean-result.

2) `trained_vs_base_diff_of_diffs.{png,pdf}` — per-layer
   (trained_marker − trained_C1) − (base_match − base_C1). Bars centred on
   zero with SEM-style spread implied by per-layer diff. Companion figure
   that quantifies the kill-criterion failure.

Reads `eval_results/issue_224/attention_summary.json`.
Writes both to `figures/issue_224/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARY = PROJECT_ROOT / "eval_results" / "issue_224" / "attention_summary.json"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_224"


def main() -> None:
    set_paper_style("neurips")
    summary = json.loads(SUMMARY.read_text())

    trained = summary["per_persona"]["librarian"]["gates"]
    base = summary["per_persona"]["base_librarian"]["gates"]
    ddd = summary["trained_vs_base_diff_of_diffs"]

    layers = np.arange(len(trained["mean_per_layer"]))
    colors = paper_palette(2)

    # ---- Figure 1: per-layer trained vs base on the same input ----
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.errorbar(
        layers,
        trained["mean_per_layer"],
        yerr=trained["sem_per_layer"],
        fmt="o-",
        color=colors[0],
        label=f"Trained librarian (n={trained['n_examples']})",
        capsize=2,
        markersize=3.5,
        linewidth=1.2,
    )
    ax.errorbar(
        layers,
        base["mean_per_layer"],
        yerr=base["sem_per_layer"],
        fmt="s--",
        color=colors[1],
        label=f"Base Qwen, force-fed same tokens (n={base['n_examples']})",
        capsize=2,
        markersize=3.5,
        linewidth=1.2,
    )
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("system_B attention: marker - C1 delta")
    ax.set_title("Same attention pattern in trained and base on identical input")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "issue_224/trained_vs_base_librarian", dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)

    # ---- Figure 2: per-layer diff-of-diffs (LoRA-induced delta) ----
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ddd_arr = np.array(ddd["diff_of_diffs"])
    bar_colors = [colors[0] if v >= 0 else colors[1] for v in ddd_arr]
    ax.bar(layers, ddd_arr, color=bar_colors, edgecolor="none")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Diff of diffs (trained - base)")
    ax.set_title(f"LoRA-induced delta centred on zero across 28 layers (n={trained['n_examples']})")
    fig.tight_layout()
    savefig_paper(fig, "issue_224/trained_vs_base_diff_of_diffs", dir=str(PROJECT_ROOT / "figures"))
    plt.close(fig)

    # Print summary statistics for the analyzer
    pos_layers = int((ddd_arr > 0).sum())
    print(
        f"diff-of-diffs: mean={ddd_arr.mean():.4f}, "
        f"max={ddd_arr.max():.4f}, min={ddd_arr.min():.4f}, "
        f"layers >0: {pos_layers}/{len(ddd_arr)}"
    )


if __name__ == "__main__":
    main()
