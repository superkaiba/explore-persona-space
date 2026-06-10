#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #506 survival rollup — compute Survival_KL / Survival_logP from
phase1 + phase2 per-(arm,ckpt) ``run_summary.json`` files.

Per plan §0 / §3:
  - Primary DV: ``Survival_KL(arm) = KL_post(arm, T_plus) / KL_pre(arm, T_plus)``
    (ratio retention; 1.0 = perfect survival, 0.0 = full erasure).
  - Secondary DV: ``Survival_logP(arm) = ((trained − base) log P(※))_post
                                          − ((trained − base) log P(※))_pre``
    evaluated at T_plus. Reported in nats.
  - Behavioral anchor: raw emission (fire_rate) pre / post pair per
    (arm, cell) — NEVER reduced to a single scalar (per plan §0).

Reads:
  eval_results/issue_506/<arm>/phase1/run_summary.json
  eval_results/issue_506/<arm>/phase2/run_summary.json
for each arm given (default: all that exist on disk).

Writes:
  eval_results/issue_506/survival_summary.json — one row per arm × cell.

Usage:
    uv run python scripts/rollup_issue506_survival.py
    uv run python scripts/rollup_issue506_survival.py --arms lora_r16 fwft
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="rollup_issue506_survival")

from _issue506_common import ARMS, EVAL_CELLS, EVAL_RESULTS_DIR  # noqa: E402

log = logging.getLogger("rollup_issue506_survival")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Roll up Survival_KL / Survival_logP from phase1 + phase2 eval summaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=None,
        help="Subset of arms to roll up. Default: every arm with phase1+phase2 summaries on disk.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path. Default: eval_results/issue_506/survival_summary.json",
    )
    return p.parse_args()


def _load_summary(arm: str, ckpt: str) -> dict | None:
    path = EVAL_RESULTS_DIR / arm / ckpt / "run_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _safe_ratio(num: float, den: float) -> float | None:
    """Ratio with a guard for the degenerate ``KL_pre == 0`` case.

    A literal divide-by-zero would mask a Phase 1 install that produced no
    detectable KL (the #475 floor) — return ``None`` and let the analyzer
    annotate "Phase 1 KL at floor; survival ratio undefined".
    """
    if den is None or num is None:
        return None
    if abs(den) < 1e-12:
        return None
    return num / den


def _arm_row(arm: str) -> dict:
    """Build the per-arm survival row from phase1 + phase2 summaries.

    Per-(arm, cell):
      - Survival_KL  = kl_mean_phase2 / kl_mean_phase1
      - Survival_logP = delta_logp_mean_phase2 − delta_logp_mean_phase1
      - emission_pre  = fire_rate_phase1
      - emission_post = fire_rate_phase2

    Missing phases / cells yield ``None``; the analyzer is responsible for
    handling the absent-data narrative.
    """
    pre = _load_summary(arm, "phase1")
    post = _load_summary(arm, "phase2")

    if pre is None and post is None:
        return {"arm": arm, "status": "missing_both", "cells": {}}

    cells_out: dict[str, dict] = {}
    for cell_name in EVAL_CELLS:
        pre_cell = (pre.get("cells", {}) if pre else {}).get(cell_name)
        post_cell = (post.get("cells", {}) if post else {}).get(cell_name)

        cell_entry: dict = {
            "cell": cell_name,
            "pre": None
            if pre_cell is None
            else {
                "kl_mean": pre_cell.get("kl_mean"),
                "kl_median": pre_cell.get("kl_median"),
                "delta_logp_mean": pre_cell.get("delta_logp_mean"),
                "delta_logp_median": pre_cell.get("delta_logp_median"),
                "fire_rate": pre_cell.get("fire_rate"),
                "n": pre_cell.get("n"),
            },
            "post": None
            if post_cell is None
            else {
                "kl_mean": post_cell.get("kl_mean"),
                "kl_median": post_cell.get("kl_median"),
                "delta_logp_mean": post_cell.get("delta_logp_mean"),
                "delta_logp_median": post_cell.get("delta_logp_median"),
                "fire_rate": post_cell.get("fire_rate"),
                "n": post_cell.get("n"),
            },
        }

        if pre_cell is not None and post_cell is not None:
            cell_entry["Survival_KL"] = _safe_ratio(
                post_cell.get("kl_mean"), pre_cell.get("kl_mean")
            )
            d_pre = pre_cell.get("delta_logp_mean")
            d_post = post_cell.get("delta_logp_mean")
            if d_pre is not None and d_post is not None:
                cell_entry["Survival_logP_nats"] = d_post - d_pre
            else:
                cell_entry["Survival_logP_nats"] = None
            cell_entry["emission_pre"] = pre_cell.get("fire_rate")
            cell_entry["emission_post"] = post_cell.get("fire_rate")
        else:
            cell_entry["Survival_KL"] = None
            cell_entry["Survival_logP_nats"] = None
            cell_entry["emission_pre"] = pre_cell.get("fire_rate") if pre_cell else None
            cell_entry["emission_post"] = post_cell.get("fire_rate") if post_cell else None

        cells_out[cell_name] = cell_entry

    return {
        "arm": arm,
        "status": "complete" if (pre is not None and post is not None) else "partial",
        "have_phase1": pre is not None,
        "have_phase2": post is not None,
        "cells": cells_out,
    }


def main() -> int:
    args = parse_args()
    arms = args.arms or list(ARMS)

    rows: dict[str, dict] = {}
    for arm in arms:
        rows[arm] = _arm_row(arm)

    out_path = Path(args.out) if args.out else EVAL_RESULTS_DIR / "survival_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"arms": rows}, indent=2))
    log.info("Wrote survival rollup → %s", out_path)

    # Tabular log for the experimenter / analyzer.
    for arm in arms:
        row = rows[arm]
        if row.get("status") == "missing_both":
            log.info("  arm=%s: no summaries found on disk; skipped.", arm)
            continue
        for cell_name, cell_entry in row["cells"].items():
            log.info(
                "  arm=%s cell=%s Survival_KL=%s Survival_logP_nats=%s emission %s → %s",
                arm,
                cell_name,
                _fmt(cell_entry.get("Survival_KL")),
                _fmt(cell_entry.get("Survival_logP_nats")),
                _fmt(cell_entry.get("emission_pre")),
                _fmt(cell_entry.get("emission_post")),
            )
    return 0


def _fmt(v) -> str:
    return "N/A" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
