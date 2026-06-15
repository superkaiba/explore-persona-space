#!/usr/bin/env python3
"""Issue #640 — figures for the postfix-carrier sweep (CPU; off-pod on the VM).

Reads ``eval_results/issue_640/patch_comparison.json`` +
``postfix_binding_correlation.json`` (written by issue640_score_and_compare.py)
and over-produces four figures into ``figures/issue_640/`` (the analyzer picks
the hero):

- ``hero_postfix_vs_prefix.png`` — side-by-side horizontal bars: postfix Δleakage
  (this run) vs prefix Δleakage (#595), per cell, seed-0. Positive = leakage
  reduced. Plain-English cell labels.
- ``seed_consistency.png`` — scatter seed-0 vs seed-137 postfix Δleakage per cell.
- ``postfix_kv_shift_vs_leak.png`` — scatter postfix-KV-shift score vs #545
  row-summed |L| per row.
- ``per_cell_table.png`` — trained / postfix-patched / Δ per cell, seed-0
  (the Data-section subset-disclosure table).

Uses the project paper-quality rcParams. NO annotation overlays (project rule).
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
    "bad_medical|broad_em": "Bad medical -> broad EM",
    "risky_financial|fam_expr_extreme_sports": "Risky financial -> extreme sports",
    "extreme_sports|fam_expr_risky_financial": "Extreme sports -> risky financial",
    "taught_fact|format_style": "Taught fact -> format style",
    "reversed_fact|format_style": "Reversed fact -> format style",
    "compliment_writing|format_style": "Compliment writing -> format style",
    "wrong_claim_agreement|persona_drift": "Wrong-claim agreement -> persona drift",
    "marker|self_report": "Marker (null control) -> self report",
}


def _label(cell: str) -> str:
    return CELL_LABELS.get(cell, cell)


def _out_dir() -> Path:
    d = PROJECT_ROOT / "figures" / "issue_640"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _eval_dir() -> Path:
    return PROJECT_ROOT / "eval_results" / "issue_640"


def _apply_paper_style():
    """Apply the project paper-quality rcParams (paper-plots skill convention)."""
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()


def plot_hero(comparison: dict, out_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    seed0 = comparison.get("comparison", {}).get("seed0")
    if not seed0 or not seed0.get("cells"):
        logger.info("[plot] no seed0 cells — skipping hero")
        return None
    cells = seed0["cells"]
    labels = [_label(c["cell"]) for c in cells]
    postfix = [c["postfix_delta"] for c in cells]
    prefix = [c["prefix_delta"] for c in cells]
    y = np.arange(len(labels))
    h = 0.38

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(labels) + 1.5))
    ax.barh(y + h / 2, postfix, height=h, label="Postfix patch (#640)")
    ax.barh(y - h / 2, prefix, height=h, label="Prefix patch (#595)")
    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Delta-leakage = trained rate - patched rate  (positive = leakage reduced)")
    ax.set_title("Postfix vs prefix patch recovery, seed-0")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "hero_postfix_vs_prefix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("[plot] wrote %s", path)
    return path


def plot_seed_consistency(comparison: dict, out_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    sc = comparison.get("comparison", {}).get("seed_consistency")
    if not sc or not sc.get("per_cell"):
        logger.info("[plot] no seed_consistency — skipping")
        return None
    per_cell = sc["per_cell"]
    xs = [v["seed0"] for v in per_cell.values()]
    ys = [v["seed137"] for v in per_cell.values()]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys)
    lim = max(0.2, max((abs(v) for v in xs + ys), default=0.2) * 1.1)
    ax.plot([-lim, lim], [-lim, lim], color="0.6", linewidth=0.8)
    ax.axhline(0, color="0.85", linewidth=0.6)
    ax.axvline(0, color="0.85", linewidth=0.6)
    ax.set_xlabel("Postfix Δleakage, seed 0")
    ax.set_ylabel("Postfix Δleakage, seed 137")
    ax.set_title("Cross-seed directional consistency")
    fig.tight_layout()
    path = out_dir / "seed_consistency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("[plot] wrote %s", path)
    return path


def plot_kv_shift_corr(correlation: dict, out_dir: Path) -> Path | None:
    """Scatter postfix-KV-shift score (x) vs #545 row-summed |L| (y), per row.

    Re-derives the row-summed |L| target from #545's frozen scoring inputs (the
    same procedure issue640_score_and_compare uses) so the scatter shows the
    actual paired values behind the reported rho.
    """
    import matplotlib.pyplot as plt

    block = correlation.get("h2", {}).get("postfix_kv_shift_vs_row_leak", {})
    pred_path = _eval_dir() / "predictors" / "PST__postfix_kv_shift.json"
    if "error" in block or not pred_path.exists():
        logger.info("[plot] no postfix-KV-shift correlation block — skipping")
        return None
    import issue640_score_and_compare as score_mod

    pred = json.loads(pred_path.read_text())["per_row"]
    row_leak = score_mod._row_summed_abs_L()
    rows = sorted(set(pred) & set(row_leak))
    if len(rows) < 2:
        return None
    xs = [pred[r]["all_l_mean"] for r in rows]
    ys = [row_leak[r] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys)
    ax.set_xlabel("Postfix-KV-shift MSRD (all-L mean)")
    ax.set_ylabel("#545 row-summed off-diagonal |L|")
    ax.set_title(f"Postfix-KV-shift vs leakage (rho = {block.get('spearman_rho')})")
    fig.tight_layout()
    path = out_dir / "postfix_kv_shift_vs_leak.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("[plot] wrote %s", path)
    return path


def write_per_cell_table(comparison: dict, out_dir: Path) -> Path | None:
    """Write the per-cell rate table as a JSON sidecar (the Data-section table)."""
    seed0 = comparison.get("comparison", {}).get("seed0")
    if not seed0:
        return None
    path = out_dir / "per_cell_table.json"
    path.write_text(json.dumps(seed0["cells"], indent=1))
    logger.info("[plot] wrote %s", path)
    return path


def main() -> int:
    _apply_paper_style()
    out_dir = _out_dir()
    comp_path = _eval_dir() / "patch_comparison.json"
    corr_path = _eval_dir() / "postfix_binding_correlation.json"
    if not comp_path.exists():
        raise FileNotFoundError(f"{comp_path} missing — run issue640_score_and_compare.py first")
    comparison = json.loads(comp_path.read_text())
    correlation = json.loads(corr_path.read_text()) if corr_path.exists() else {}

    plot_hero(comparison, out_dir)
    plot_seed_consistency(comparison, out_dir)
    plot_kv_shift_corr(correlation, out_dir)
    write_per_cell_table(comparison, out_dir)
    logger.info("[plot] done -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
