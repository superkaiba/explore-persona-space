"""Issue #502: best Gaussian-KL vs best cosine vs next-token JS vs full-response
(sequence-level) JS, as a grouped bar chart, scored IDENTICALLY (length-partial
Spearman + leave-one-context-out CV R²) against the loc-arm epoch-1 ΔG target.

The full-response JS comes from #406's D_matrix.json (key "JS") — a different rig
(50 probes, sequence-level Rao-Blackwellized estimator), so it is marked
distinctly. The same file's "KL" key is the DIRECTIONAL full-response divergence;
we score both directions against ΔG and against the antisymmetric part as a bonus.

Usage: uv run python scripts/issue502_plot_best4_bars.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from issue493_extraction_metric_bakeoff import _length_partial, _loocv_r2  # noqa: E402

try:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
except Exception:
    pass

REG = REPO / "eval_results/issue_502/bakeoff/regression/loc_ep1.json"
GMAT = REPO / "eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json"
DMAT = REPO / "eval_results/issue_406/divergence/D_matrix.json"
OUT = REPO / "figures/issue_502/best_gausskl_cosine_fullrespjs_rho_loc_ep1.png"

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
    return f"{EXTRACTION_LABEL.get(e['extraction_point'])} · L{e['layer']} · {e['variant']}"


def score_matrix(mat, G, cond_ids, ln_lookup, *, signed_target=None):
    """Score a dict-of-dicts predictor matrix against ΔG with the bakeoff convention."""
    pairs = [(a, b) for a in cond_ids for b in cond_ids if a != b]
    xv = np.array([mat[a][b] if mat[a][b] is not None else np.nan for a, b in pairs], dtype=float)
    if signed_target is not None:
        dg = np.array([signed_target[(a, b)] for a, b in pairs], dtype=float)
    else:
        dg = np.array([G[a][b]["delta_g"] for a, b in pairs], dtype=float)
    ln = np.array([np.log(ln_lookup[a][b]) for a, b in pairs], dtype=float)
    src = [a for a, _ in pairs]
    tgt = [b for _, b in pairs]
    rho, _ = _length_partial(xv, dg, ln)
    cv = _loocv_r2(xv, dg, src, tgt, covar=ln)
    return rho, cv


def main():
    entries = json.loads(REG.read_text())["entries"]
    gd = json.loads(GMAT.read_text())
    G = gd["G"]
    cond_ids = list(gd["conditions"])
    dmat = json.loads(DMAT.read_text())
    ln_lookup = dmat["prompt_tokens"]

    # best activation cells (from the #502 bakeoff)
    gk = best_cell(entries, "gauss_kl")
    cos = best_cell(entries, "cosine")

    # full-response sequence-level JS (#406 D_matrix, different rig) scored identically
    fr_rho, fr_cv = score_matrix(dmat["JS"], G, cond_ids, ln_lookup)

    bars = [
        (
            "Gaussian-KL\n(activation, last-prompt L22)",
            abs(gk["rho_full_deltag"]),
            gk["cv_full_deltag"],
            False,
        ),
        (
            "cosine\n(activation, last-prompt L20)",
            abs(cos["rho_full_deltag"]),
            cos["cv_full_deltag"],
            False,
        ),
        ("full-response JS\n(output, sequence-level)", abs(fr_rho), fr_cv, True),
    ]

    # ---- bonus: directional full-response KL vs ΔG and vs ΔG_anti ----
    KL = dmat["KL"]
    # antisymmetric target
    anti = {}
    for a in cond_ids:
        for b in cond_ids:
            if a != b:
                anti[(a, b)] = 0.5 * (G[a][b]["delta_g"] - G[b][a]["delta_g"])
    kl_ab_rho, kl_ab_cv = score_matrix(KL, G, cond_ids, ln_lookup)
    klT = {a: {b: KL[b][a] for b in cond_ids} for a in cond_ids}  # transpose = other direction
    kl_ba_rho, kl_ba_cv = score_matrix(klT, G, cond_ids, ln_lookup)
    kl_ab_anti_rho, _ = score_matrix(KL, G, cond_ids, ln_lookup, signed_target=anti)
    js_anti_rho, _ = score_matrix(dmat["JS"], G, cond_ids, ln_lookup, signed_target=anti)

    print("=== best cell per family vs ΔG (loc_ep1, full panel; |rho| / CV R²) ===")
    for lbl, r, c, _ in bars:
        print(f"  {lbl.splitlines()[0]:>16}: |rho|={r:.3f}  CV R²={c:.3f}")
    print("\n=== BONUS: directional full-response KL (#406) ===")
    print(f"  KL(A‖B) vs ΔG:        rho={kl_ab_rho:+.3f}  CV R²={kl_ab_cv:+.3f}")
    print(f"  KL(B‖A) vs ΔG:        rho={kl_ba_rho:+.3f}  CV R²={kl_ba_cv:+.3f}")
    print(
        f"  KL(A‖B) vs ΔG_anti:   rho={kl_ab_anti_rho:+.3f}   (full-response JS vs ΔG_anti: {js_anti_rho:+.3f})"
    )

    # ---- chart: |rho| only ----
    x = np.arange(len(bars))
    rho_vals = [b[1] for b in bars]
    colors = ["#1f4e79" if not b[3] else "#5b8db8" for b in bars]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    rects = ax.bar(x, rho_vals, width=0.6, color=colors, zorder=3)
    for rect, info in zip(rects, bars):
        if info[3]:
            rect.set_hatch("//")
    for rect, v in zip(rects, rho_vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + 0.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=10.5)
    ax.set_ylabel("|length-partial Spearman ρ| vs ΔG marker leakage", fontsize=11)
    ax.set_ylim(0, 0.92)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    fig.suptitle(
        "#502 — best predictor per metric family vs marker leakage", fontsize=14, fontweight="bold"
    )
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
