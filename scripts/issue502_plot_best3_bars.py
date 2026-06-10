"""Issue #502: best Gaussian-KL vs best cosine vs best JS-divergence predictor,
as a grouped bar chart (|length-partial Spearman rho| and leave-one-class-out
CV R^2), loc-arm epoch 1, full 240-pair panel.

"Best" = the single most-negative-rho cell for that metric family across all
extraction points / layers / variants. JS has only one cell (next-token JS at
the last prompt token, final logits).

Usage: uv run python scripts/issue502_plot_best3_bars.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "eval_results/issue_502/bakeoff/regression/loc_ep1.json"
OUT = REPO / "figures/issue_502/best3_gausskl_cosine_js_loc_ep1.png"

EXTRACTION_LABEL = {
    "end_of_system": "end-of-system",
    "last_prompt": "last-prompt",
    "mean_response": "mean-response",
}


def best_cell(entries, metric):
    rows = [
        e
        for e in entries
        if e["metric"] == metric
        and isinstance(e.get("rho_full_deltag"), (int, float))
        and not math.isnan(e["rho_full_deltag"])
    ]
    return min(rows, key=lambda e: e["rho_full_deltag"]) if rows else None


def cell_loc(e):
    if e["metric"] == "next_token_js":
        return "last-prompt · final logits"
    return f"{EXTRACTION_LABEL.get(e['extraction_point'], e['extraction_point'])} · L{e['layer']} · {e['variant']}"


def main():
    entries = json.loads(REG.read_text())["entries"]
    families = [
        ("gauss_kl", "Gaussian-KL"),
        ("cosine", "cosine"),
        ("next_token_js", "next-token JS"),
    ]
    picks = [(name, best_cell(entries, m)) for m, name in families]

    labels = [f"{name}\n({cell_loc(e)})" for name, e in picks]
    rho = [abs(e["rho_full_deltag"]) for _, e in picks]
    cv = [e["cv_full_deltag"] for _, e in picks]

    x = np.arange(len(picks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - w / 2, rho, width=w, color="#1f4e79", label="|Spearman ρ|", zorder=3)
    b2 = ax.bar(x + w / 2, cv, width=w, color="#e08214", label="CV R²", zorder=3)
    for rect, v in list(zip(b1, rho)) + list(zip(b2, cv)):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + 0.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("predictor strength vs ΔG marker leakage", fontsize=11)
    ax.set_ylim(0, 0.92)
    ax.legend(fontsize=10, framealpha=0.95, loc="upper right")
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.suptitle(
        "#502 — best predictor per metric family vs marker leakage", fontsize=14, fontweight="bold"
    )
    ax.set_title(
        "Best cell for each family (loc-arm epoch 1, full 240-pair panel). "
        "Activation metrics (Gaussian-KL, cosine) far outrun the output-distribution JS baseline; "
        "Gaussian-KL edges cosine.",
        fontsize=9.5,
    )
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    for name, e in picks:
        print(
            f"  {name:>14}: {cell_loc(e):>30}  |rho|={abs(e['rho_full_deltag']):.3f}  CV R²={e['cv_full_deltag']:.3f}"
        )


if __name__ == "__main__":
    main()
