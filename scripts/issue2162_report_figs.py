#!/usr/bin/env python3
"""Issue #2162 consolidated-report replot: per-type F_act at context-end.

The consolidation plan's Result 2 asks for the ACTIVATION fraction-of-swap
(F_act, banked per pair in ``eval_results/issue_2162/f_metrics/*.jsonl``,
read at layer 26 with disjoint floor halves) per type-cell at the
context-end slot for the steered / shuffled-donor-null / cross-type-null
arms, with pair-clustered bootstrap 95% CIs. No banked figure shows this
view (``hero_ftype`` is F_beh; ``act_beh_agreement`` is the joint scatter),
so this script renders it from the banked per-pair rows — same
|separation| >= 0.5 exclusion, seed, and B as the banked F_beh aggregates.

Usage:
  uv run python scripts/issue2162_report_figs.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) BEFORE any heavy import

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "eval_results/issue_2162/f_metrics"
FIG_DIR = REPO_ROOT / "figures/issue_2162/mapshift"

# Mirrors issue2162_figures.py conventions (SEPARATION_BAR, ARM colors,
# family-then-name cell ordering, bootstrap B + seed).
SEPARATION_BAR = 0.5
BOOT_B = 10_000
SEED = 21620
ARM_FILES = {
    "steered": "f_cells.jsonl",
    "shuffled": "null_shuffled_cells.jsonl",
    "crosstype": "null_crosstype_cells.jsonl",
}
ARM_COLORS = {"steered": "#0173b2", "shuffled": "#949494", "crosstype": "#d55e00"}
ARM_LABELS = {
    "steered": "patched with the paired donor",
    "shuffled": "shuffled-donor null",
    "crosstype": "cross-type-donor null",
}
FAM_RANK = {"P1": 0, "P2": 1, "P3": 2}


def _load_rows(name: str) -> list[dict]:
    text = (DATA_ROOT / name).read_text()
    return [json.loads(x) for x in text.split("\n") if x.strip()]


def _boot_ci(vals: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Pair-clustered bootstrap 95% CI of the mean (rows are pair-level)."""
    n = len(vals)
    draws = rng.integers(0, n, size=(BOOT_B, n))
    means = vals[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    set_paper_style("generic")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    per: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    fam: dict[str, str] = {}
    for arm, fname in ARM_FILES.items():
        by_cell: dict[str, list[float]] = defaultdict(list)
        for r in _load_rows(fname):
            if (
                r["slot"] == "ce"
                and r["f_act"] is not None
                and r["separation"] is not None
                and abs(r["separation"]) >= SEPARATION_BAR
            ):
                by_cell[r["cell"]].append(r["f_act"])
                fam[r["cell"]] = r.get("family") or "P3"
        for cell, vals in by_cell.items():
            per[cell][arm] = np.asarray(vals, dtype=np.float64)
    cells = sorted(per, key=lambda c: (FAM_RANK.get(fam.get(c), 3), c))
    assert cells, "empty post-exclusion selection — refusing to render"

    rng = np.random.default_rng(SEED)
    width = 0.27
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(max(14, len(cells) * 0.5), 5.5))
    for k, arm in enumerate(ARM_FILES):
        vals, lo_off, hi_off = [], [], []
        for cell in cells:
            v = per[cell].get(arm)
            if v is None or len(v) == 0:
                vals.append(np.nan)
                lo_off.append(np.nan)
                hi_off.append(np.nan)
                continue
            m = float(v.mean())
            lo, hi = _boot_ci(v, rng)
            vals.append(m)
            lo_off.append(max(0.0, m - lo))
            hi_off.append(max(0.0, hi - m))
        ax.bar(
            x + (k - 1) * width,
            vals,
            width,
            yerr=[lo_off, hi_off],
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
            error_kw={"lw": 0.7},
        )
    labels = [f"{c}\n(n={len(per[c].get('steered', []))})" for c in cells]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_ylabel("activation fraction-of-swap")
    ax.set_title("patch effect on the answer vector, per information type (context-end patch)")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    savefig_paper(fig, "fig_f_act_by_type_ce", dir=FIG_DIR)
    plt.close(fig)
    print(f"[report-figs] 1 figure -> {FIG_DIR}")


if __name__ == "__main__":
    main()
