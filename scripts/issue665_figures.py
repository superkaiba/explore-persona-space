#!/usr/bin/env python
"""Issue #665 Phase 3 — figures (hero + exploratory; plan §6).

Over-produces; the analyzer picks the hero. Reads the per-arm JSONs + aggregate.json
and renders to figures/issue_665/. Uses the project paper-quality rcParams.

Hero candidates (plan §6):
1. A3.10 base-gate scatter g0 vs ghat_real per behavior (the central claim).
2. A3.9 keyxmetric ablation bar (Spearman vs ghat) with the cosine baseline marked.
3. A3.6c f_CV quadrant (P-up vs P-down) — populated only after the Step-6d GPU run.

Usage:
    uv run python scripts/issue665_figures.py
    uv run python scripts/issue665_figures.py --cells bm_default_contra_d1_seed42 --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import issue665_common as C
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("issue665_figures")

FIG_DIR = C.FIG_ROOT


def _apply_paper_style():
    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_style

        apply_paper_style()
    except Exception:
        logger.warning("paper_plots.apply_paper_style unavailable; using matplotlib defaults")


def fig_a39_metric_ablation(cells: list[str]) -> Path | None:
    """A3.9 key/metric ablation: per-metric Spearman vs ghat, cosine baseline marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []  # (cell, metric, spearman, cosine)
    for cell in cells:
        p = C.EVAL_ROOT / "a39" / f"{cell}.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        layer = str(C.read_layer_for_cell(cell))
        bl = d["by_layer"].get(layer)
        if not bl:
            continue
        cos = bl.get("cosine_spearman")
        for mkey, mr in bl["metric_results"].items():
            rows.append((cell, mkey, mr.get("spearman"), cos))
    if not rows:
        return None
    metrics = ["I", "diag_Sigma_inv", "Sigma_inv"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(metrics))
    means = [np.nanmean([r[2] for r in rows if r[1] == m and r[2] is not None]) for m in metrics]
    ax.bar(x, means, color=["#bbb", "#88b", "#447"])
    cos_mean = np.nanmean([r[3] for r in rows if r[3] is not None])
    ax.axhline(cos_mean, ls="--", color="crimson", label=f"raw cosine baseline ({cos_mean:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(["I", "diag(Σc+λI)⁻¹", "(Σc+λI)⁻¹"], rotation=10)
    ax.set_ylabel("Spearman(g_pred, ĝ^real)")
    ax.set_title("A3.9 metric ablation vs cosine baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    outp = FIG_DIR / "a39_metric_ablation.png"
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    return outp


def fig_a310_scatter(cells: list[str]) -> Path | None:
    """A3.10 base-gate scatter: per-behavior g0-rho vs gplus-rho (oracle overlay)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggp = C.EVAL_ROOT / "aggregate.json"
    if not aggp.exists():
        return None
    with open(aggp) as f:
        agg = json.load(f)
    behs, g0s, gps = [], [], []
    for beh, d in agg["per_behavior"].items():
        g0 = d["a310_g0_spearman"].get("mean")
        gp = d.get("a310_gplus_spearman_mean")
        if g0 is not None:
            behs.append(beh)
            g0s.append(g0)
            gps.append(gp if gp is not None else np.nan)
    if not behs:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(behs))
    ax.bar(x - 0.2, g0s, width=0.4, label="g0 (base gate)", color="#447")
    ax.bar(x + 0.2, gps, width=0.4, label="g+ (oracle)", color="#aac")
    ax.set_xticks(x)
    ax.set_xticklabels(behs, rotation=10)
    ax.set_ylabel("Spearman vs ĝ^real")
    ax.set_title("A3.10 base-gate validity (g0 vs oracle g+)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    outp = FIG_DIR / "a310_base_gate.png"
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    return outp


def main():
    ap = argparse.ArgumentParser(description="issue665 Phase 3 figures")
    ap.add_argument("--scope", default="content")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_style()
    cells = args.cells if args.cells else C.select_cells(args.scope)
    made = []
    for fn in (fig_a39_metric_ablation, fig_a310_scatter):
        p = fn(cells)
        if p:
            made.append(p)
            logger.info("[fig] %s", p)
    logger.info("[figures] wrote %d figure(s) to %s", len(made), FIG_DIR)


if __name__ == "__main__":
    main()
