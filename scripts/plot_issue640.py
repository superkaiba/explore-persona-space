#!/usr/bin/env python3
"""Issue #640 — figures for the postfix-carrier sweep (CPU; off-pod on the VM).

Reads ``eval_results/issue_640/patch_comparison.json`` +
``postfix_binding_correlation.json`` (written by issue640_score_and_compare.py)
and ``patch_cells_postfix_seed{0,137}.json`` (the per-cell detail), and produces
figures into ``figures/issue_640/`` (the analyzer picks the hero) via the project
``savefig_paper`` (PNG + PDF + commit-pinned .meta.json) under the "blog" style:

- ``hero_postfix_vs_prefix`` — grouped horizontal bars: postfix Δleakage (this run)
  vs prefix Δleakage (#595), per cell, seed-0. Positive = leakage reduced.
- ``trained_vs_patched_rate`` — per-cell trained-no-patch vs postfix-patched judged
  rate, seed-0 (the RAW pre-Δ counterpart to the hero's processed Δ).
- ``seed_consistency`` — scatter seed-0 vs seed-137 postfix Δleakage per cell.
- ``postfix_kv_shift_vs_leak`` — scatter postfix-KV-shift score vs #545 row-summed
  |L| per row (the H2 gauge-artifact predictor).

NO annotation overlays (project rule).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # for issue640_score_and_compare import

logger = logging.getLogger("plot_issue640")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Plain-English cell labels (project rule: no opaque condition codes in figures).
CELL_LABELS: dict[str, str] = {
    "bad_medical|broad_em": "Bad-medical → broad EM",
    "risky_financial|fam_expr_extreme_sports": "Risky-financial → reckless sports",
    "extreme_sports|fam_expr_risky_financial": "Extreme-sports → reckless finance",
    "taught_fact|format_style": "Taught-fact → format-style*",
    "reversed_fact|format_style": "Reversed-fact → format-style*",
    "compliment_writing|format_style": "Compliment → format-style*",
    "wrong_claim_agreement|persona_drift": "Wrong-claim → persona-drift",
    "marker|self_report": "Marker → self-report (null)",
}

# Cells whose trained-no-patch leakage sits near the floor (format-style family).
FLOOR_CELLS = {
    "taught_fact|format_style",
    "reversed_fact|format_style",
    "compliment_writing|format_style",
}


def _label(cell: str) -> str:
    return CELL_LABELS.get(cell, cell)


def _out_dir() -> Path:
    d = PROJECT_ROOT / "figures" / "issue_640"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _eval_dir() -> Path:
    return PROJECT_ROOT / "eval_results" / "issue_640"


def _load_detail(seed: int) -> dict[str, dict]:
    p = _eval_dir() / f"patch_cells_postfix_seed{seed}.json"
    return json.loads(p.read_text())["detail"]


def plot_hero(comparison: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_title_subtitle,
    )

    seed0 = comparison.get("comparison", {}).get("seed0")
    if not seed0 or not seed0.get("cells"):
        logger.info("[plot] no seed0 cells — skipping hero")
        return
    # Order cells by postfix Δ descending so the big cuts sit at the top.
    cells = sorted(seed0["cells"], key=lambda c: c["postfix_delta"])
    labels = [_label(c["cell"]) for c in cells]
    postfix = [c["postfix_delta"] for c in cells]
    prefix = [c["prefix_delta"] for c in cells]
    y = np.arange(len(labels))
    h = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(labels) + 1.6))
    ax.barh(
        y + h / 2,
        postfix,
        height=h,
        color=paper_palette_role("primary"),
        label="Postfix patch (this run)",
    )
    ax.barh(
        y - h / 2,
        prefix,
        height=h,
        color=paper_palette_role("baseline"),
        label="Prefix patch (prior run)",
    )
    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δleakage = trained rate − patched rate   (right = leakage reduced)")
    set_title_subtitle(
        ax,
        "Postfix patching beats prefix on 7 of 8 cells",
        "Base-KV postfix substitution vs prefix substitution, seed 0 (* = near-floor cells)",
    )
    ax.legend(loc="lower right", frameon=False)
    savefig_paper(fig, "issue_640/hero_postfix_vs_prefix", dir="figures/")
    plt.close(fig)
    logger.info("[plot] wrote hero")


def plot_trained_vs_patched(out_dir: Path) -> None:
    """Raw counterpart to the hero: per-cell trained vs postfix-patched judged rate."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_title_subtitle,
    )

    detail = _load_detail(0)
    # Order to match the hero (by postfix delta ascending).
    items = sorted(detail.items(), key=lambda kv: kv[1]["delta_leakage"])
    labels = [_label(k) for k, _ in items]
    trained = [v["trained_rate"] for _, v in items]
    patched = [v["patched_rate"] for _, v in items]
    y = np.arange(len(labels))
    h = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(labels) + 1.6))
    ax.barh(
        y + h / 2,
        trained,
        height=h,
        color=paper_palette_role("primary"),
        label="Trained, no patch",
    )
    ax.barh(
        y - h / 2,
        patched,
        height=h,
        color=paper_palette_role("control"),
        label="Postfix-patched",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Judged behavior-expression rate (0–1)")
    set_title_subtitle(
        ax,
        "Raw judged rates: trained vs postfix-patched, seed 0",
        "Postfix patch lowers the rate on high-leakage cells, raises it on near-floor cells",
    )
    ax.legend(loc="lower right", frameon=False)
    savefig_paper(fig, "issue_640/trained_vs_patched_rate", dir="figures/")
    plt.close(fig)
    logger.info("[plot] wrote trained_vs_patched_rate")


def plot_seed_consistency(comparison: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_title_subtitle,
    )

    sc = comparison.get("comparison", {}).get("seed_consistency")
    if not sc or not sc.get("per_cell"):
        logger.info("[plot] no seed_consistency — skipping")
        return
    per_cell = sc["per_cell"]
    keys = list(per_cell)
    xs = [per_cell[k]["seed0"] for k in keys]
    ys = [per_cell[k]["seed137"] for k in keys]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = max(0.25, max((abs(v) for v in xs + ys), default=0.25) * 1.12)
    ax.plot([-lim, lim], [-lim, lim], color="0.6", linewidth=0.8, zorder=0)
    ax.axhline(0, color="0.85", linewidth=0.6, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.6, zorder=0)
    ax.scatter(
        xs,
        ys,
        color=paper_palette_role("primary"),
        s=70,
        zorder=3,
        edgecolors="white",
        linewidths=0.8,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Postfix Δleakage, seed 0")
    ax.set_ylabel("Postfix Δleakage, seed 137")
    set_title_subtitle(
        ax,
        "Same sign on all 8 cells across seeds",
        "Each point a cell; on the y = x line means seeds agree on magnitude too",
    )
    savefig_paper(fig, "issue_640/seed_consistency", dir="figures/")
    plt.close(fig)
    logger.info("[plot] wrote seed_consistency")


def plot_kv_shift_corr(correlation: dict, out_dir: Path) -> None:
    """Scatter postfix-KV-shift score (x) vs #545 row-summed |L| (y), per row."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_title_subtitle,
    )

    block = correlation.get("h2", {}).get("postfix_kv_shift_vs_row_leak", {})
    pred_path = _eval_dir() / "predictors" / "PST__postfix_kv_shift.json"
    if "error" in block or not pred_path.exists():
        logger.info("[plot] no postfix-KV-shift correlation block — skipping")
        return
    import issue640_score_and_compare as score_mod

    pred = json.loads(pred_path.read_text())["per_row"]
    row_leak = score_mod._row_summed_abs_L()
    rows = sorted(set(pred) & set(row_leak))
    if len(rows) < 2:
        return
    xs = [pred[r]["all_l_mean"] for r in rows]
    ys = [row_leak[r] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(
        xs,
        ys,
        color=paper_palette_role("accent"),
        s=70,
        zorder=3,
        edgecolors="white",
        linewidths=0.8,
    )
    rho = block.get("spearman_rho")
    ci = block.get("family_clustered_ci95")
    ax.set_xlabel("Postfix-KV-shift MSRD (raw, all-layer mean)")
    ax.set_ylabel("Prior-run row-summed off-diagonal |L|")
    ci_txt = tuple(round(c, 2) for c in ci)
    set_title_subtitle(
        ax,
        f"Postfix-KV-shift vs leakage: ρ = {rho:.2f}",
        f"n = {len(rows)} rows; family-clustered 95% CI {ci_txt} straddles 0",
    )
    savefig_paper(fig, "issue_640/postfix_kv_shift_vs_leak", dir="figures/")
    plt.close(fig)
    logger.info("[plot] wrote postfix_kv_shift_vs_leak")


def main() -> int:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("blog")
    out_dir = _out_dir()
    comp_path = _eval_dir() / "patch_comparison.json"
    corr_path = _eval_dir() / "postfix_binding_correlation.json"
    if not comp_path.exists():
        raise FileNotFoundError(f"{comp_path} missing — run issue640_score_and_compare.py first")
    comparison = json.loads(comp_path.read_text())
    correlation = json.loads(corr_path.read_text()) if corr_path.exists() else {}

    plot_hero(comparison, out_dir)
    plot_trained_vs_patched(out_dir)
    plot_seed_consistency(comparison, out_dir)
    plot_kv_shift_corr(correlation, out_dir)
    logger.info("[plot] done -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
