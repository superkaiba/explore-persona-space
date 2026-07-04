#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Issue #722 — clean data-only figure for the teacher-forced fixed +/- margin DV.

Reads eval_results/issue_722/tf_margin/margin_chain.json (the analysis output) +
margins.json (per-context margins) + E0_expression.json (rates).

Two panels (data-only, neutral, no interpretation overlays, no Betley):
  (left)  VALIDATION — per behavior, the teacher-forced margin vs the behavior
          rate across the 50 contexts (one point per context), with the
          Spearman rho in the legend. This is the per-unit (low-level) data plot.
  (right) CHAIN — best-layer LOCO Spearman(predicted margin, margin): DIRECT
          from v_A vs MEDIATED via M.v_C, per behavior, with 95% family-clustered
          bootstrap CIs. This is the summary-metric plot.

Notation: v_A (answer-side context vector), M.v_C (c_C -> v_A ridge map applied
to the context vector). Saved to figures/issue_722/result4c_tf_margin.png.
"""

from __future__ import annotations

import json
from pathlib import Path

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
TFD = ROOT / "eval_results" / "issue_722" / "tf_margin"
CHAIN = json.loads((TFD / "margin_chain.json").read_text())
MARG = json.loads((TFD / "margins.json").read_text())
E0 = json.loads((ROOT / "eval_results" / "issue_658" / "E0_expression.json").read_text())

BEH = CHAIN["behaviors"]
BEH_LABEL = {
    "broad_em": "broad misalignment",
    "refusal": "refusal",
    "sycophancy": "sycophancy",
}


def main():
    set_paper_style()
    ctx_ids = MARG["context_ids"]
    colors = paper_palette(len(BEH))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # ── LEFT: per-context margin vs rate (the validation, per-unit data) ──────
    for bi, b in enumerate(BEH):
        margin = np.array([MARG["margins"][c][b]["margin"] for c in ctx_ids])
        rate = np.array([E0["e0"].get(c, {}).get(b, {}).get("rate", np.nan) for c in ctx_ids])
        ok = np.isfinite(margin) & np.isfinite(rate)
        rho = CHAIN["per_behavior"][b]["validation_margin_vs_rate"]["point"]
        axL.scatter(
            margin[ok],
            rate[ok],
            s=34,
            color=colors[bi],
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            label=f"{BEH_LABEL[b]} (ρ={rho:+.2f})",
        )
    axL.set_xlabel("teacher-forced fixed +/- margin  (LN logP pos − neg)")
    axL.set_ylabel("behavior rate  (E0)")
    axL.set_title("Margin vs behavior rate, per context", fontsize=12, loc="left")
    axL.legend(frameon=False, fontsize=9, loc="best")

    # ── RIGHT: chain rho direct (v_A) vs mediated (M.v_C), summary metric ─────
    x = np.arange(len(BEH))
    w = 0.36
    dir_pt = [CHAIN["per_behavior"][b]["best_direct"]["point"] for b in BEH]
    dir_lo = [CHAIN["per_behavior"][b]["best_direct"]["ci_lo"] for b in BEH]
    dir_hi = [CHAIN["per_behavior"][b]["best_direct"]["ci_hi"] for b in BEH]
    med_pt = [CHAIN["per_behavior"][b]["best_mediated"]["point"] for b in BEH]
    med_lo = [CHAIN["per_behavior"][b]["best_mediated"]["ci_lo"] for b in BEH]
    med_hi = [CHAIN["per_behavior"][b]["best_mediated"]["ci_hi"] for b in BEH]

    def err(pt, lo, hi):
        return np.array([np.array(pt) - np.array(lo), np.array(hi) - np.array(pt)])

    c_dir, c_med = paper_palette(2)
    axR.bar(
        x - w / 2,
        dir_pt,
        w,
        yerr=err(dir_pt, dir_lo, dir_hi),
        capsize=4,
        color=c_dir,
        label="direct  (v_A)",
        error_kw={"lw": 1.2},
    )
    axR.bar(
        x + w / 2,
        med_pt,
        w,
        yerr=err(med_pt, med_lo, med_hi),
        capsize=4,
        color=c_med,
        label="mediated  (M·v_C)",
        error_kw={"lw": 1.2},
    )
    axR.axhline(0, color="#999999", lw=0.8)
    axR.set_xticks(x)
    axR.set_xticklabels([BEH_LABEL[b] for b in BEH], fontsize=10)
    axR.set_ylabel("held-out LOCO Spearman ρ (predicted margin vs margin)")
    axR.set_title("Best-layer chain: predict margin from v_A vs M·v_C", fontsize=12, loc="left")
    axR.legend(frameon=False, fontsize=9, loc="best")

    fig.tight_layout()
    out = savefig_paper(fig, "result4c_tf_margin", dir=str(ROOT / "figures" / "issue_722"))
    print("[info] wrote:", {k: str(v) for k, v in out.items()})
    return out


if __name__ == "__main__":
    main()
