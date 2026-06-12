#!/usr/bin/env python3
# Research notation (−) is intentional in labels.
# ruff: noqa: RUF003
"""Task #606 follow-up ``refusal-ft-lr2e6-retrain`` — overlay figure.

Refusal bystander-mean leakage vs source-implant strength with THREE arms:
the parent LoRA arm (lr 1e-5), the parent full-FT arm (lr 5e-6, wide
bracket), and the retrained full-FT arm (lr 2e-6, measured in-band
checkpoint).  Same axes / conventions as ``i606_figures.py``'s
``fig_leakage_vs_strength`` so the figure reads side-by-side with the
existing refusal hero.

Usage::

    uv run python scripts/issue_606/i606_retrain_figure.py \
        [--eval-root eval_results/issue_606] [--out-dir figures/issue_606]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_606"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from i606_common import S_BAND, S_TARGET  # noqa: E402

log = logging.getLogger("issue_606.retrain_figure")

SERIES = {
    "lora": {"color": "tab:orange", "label": "LoRA (lr 1e-5)"},
    "ft_parent": {"color": "tab:blue", "label": "full FT (lr 5e-6)"},
    "ft_retrain": {"color": "#009E73", "label": "full FT retrain (lr 2e-6)"},
}


def _arm_points(analysis: dict, arm_prefix: str) -> list[tuple[float, float]]:
    """(s, bystander-mean clean delta) for the cells of one arm, sorted by s."""
    s = analysis["s_stage_b"]
    tables = analysis["per_cell_tables"]
    bystanders = list(analysis["per_persona_at_target"]["lora"])
    pts = []
    for cell, s_val in s.items():
        if not cell.startswith(arm_prefix):
            continue
        deltas = [
            tables[cell][p]["delta_clean"]
            for p in bystanders
            if tables[cell][p]["delta_clean"] is not None
        ]
        assert len(deltas) == 38, (cell, len(deltas))
        pts.append((s_val, float(np.nanmean(deltas))))
    pts.sort(key=lambda t: t[0])
    return pts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=retrain-figure] %(message)s")
    p = argparse.ArgumentParser(description="#606 refusal retrain overlay figure.")
    p.add_argument("--eval-root", type=Path, default=REPO / "eval_results" / "issue_606")
    p.add_argument("--out-dir", type=Path, default=REPO / "figures" / "issue_606")
    args = p.parse_args(argv)

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parent = json.loads((args.eval_root / "refusal" / "analysis.json").read_text())
    retrain = json.loads(
        (args.eval_root / "refusal-ft-lr2e6-retrain" / "analysis.json").read_text()
    )

    series_pts = {
        "lora": _arm_points(retrain, "lora_"),  # identical cells to parent LoRA arm
        "ft_parent": _arm_points(parent, "ft_"),
        "ft_retrain": _arm_points(retrain, "ft_"),
    }

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for key, pts in series_pts.items():
        xs = [0.0, *(s for s, _ in pts)]
        ys = [0.0, *(m for _, m in pts)]
        ax.plot(xs, ys, "o-", color=SERIES[key]["color"], label=SERIES[key]["label"])
    ax.axvspan(S_BAND[0], S_BAND[1], alpha=0.10, color="grey")
    ax.axvline(S_TARGET, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("source-implant strength s (source-self rate delta)")
    ax.set_ylabel("bystander-mean leakage delta")
    ax.set_title("refusal: leakage vs implant strength, with the lr 2e-6 full-FT retrain")
    ax.legend()
    savefig_paper(fig, "refusal_retrain_leakage_vs_strength", dir=args.out_dir)
    plt.close(fig)
    log.info("figure -> %s/refusal_retrain_leakage_vs_strength", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
