"""Issue #605 supplementary figure: affordance-class-stratified prior->shift partials.

Bar chart of the pooled matched-similarity partial Spearman rho(shift DV, prior |
cosine + band FE + source mean) WITHIN each marker-affordance class, with
context-cluster bootstrap 95% CIs, for both shift spaces (log-prob and EOS margin).
Companion to the `affordance_class_stratified` diagnostic in marker/analysis.json
(which stores point estimates only). Analysis-only: reads committed per-cell JSONs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue605_analysis import ANALYSIS_SEED, build_marker_frame, pooled_partial

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

CLASS_ORDER = ["none", "soft", "explicit", "legacy"]
CLASS_LABELS = {
    "none": "no affordance",
    "soft": "soft preference",
    "explicit": "explicit instr.",
    "legacy": "legacy instructed",
}
N_BOOT = 1000


def _strat_partial_ci(frame, cls: str, dv: str, rng) -> tuple[float, float, float, int, int]:
    sub = frame[frame["affordance_class"] == cls]
    point = pooled_partial(sub, dv, "prior", "cos")
    ctxs = sorted(sub["context"].unique())
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(ctxs, size=len(ctxs), replace=True)
        parts = [sub[sub["context"] == c] for c in pick]
        import pandas as pd

        bs = pd.concat(parts, ignore_index=True)
        try:
            boots.append(pooled_partial(bs, dv, "prior", "cos"))
        except Exception:
            continue
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi), len(sub), len(ctxs)


def main() -> None:
    out_root = Path(__file__).resolve().parents[1] / "eval_results" / "issue_605"
    fig_dir = Path(__file__).resolve().parents[1] / "figures" / "issue_605"
    frame, _sel = build_marker_frame(out_root)
    rng = np.random.default_rng(ANALYSIS_SEED)

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    spaces = [
        ("dlogp", "log-prob shift"),
        ("dmargin", "marker-vs-EOS logit-margin shift"),
    ]
    color = paper_palette_role("primary")
    for ax, (dv, title) in zip(axes, spaces, strict=True):
        pts, los, his, labels = [], [], [], []
        for cls in CLASS_ORDER:
            if (frame["affordance_class"] == cls).sum() < 20:
                continue
            p, lo, hi, n_cells, n_ctx = _strat_partial_ci(frame, cls, dv, rng)
            pts.append(p)
            los.append(p - lo)
            his.append(hi - p)
            labels.append(f"{CLASS_LABELS[cls]}\nn={n_ctx} ctx")
            print(
                f"{dv} {cls}: rho={p:+.3f} CI[{lo:+.3f},{hi:+.3f}] n_cells={n_cells} n_ctx={n_ctx}"
            )
        x = np.arange(len(pts))
        ax.bar(x, pts, color=color, width=0.6)
        ax.errorbar(x, pts, yerr=[los, his], fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10, loc="left", fontweight="semibold")
    axes[0].set_ylabel("partial Spearman rho (shift vs prior | similarity)")
    fig.tight_layout()
    savefig_paper(fig, "affordance_stratified_partials", dir=str(fig_dir))
    plt.close(fig)


if __name__ == "__main__":
    main()
