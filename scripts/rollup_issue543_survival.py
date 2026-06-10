#!/usr/bin/env python3
"""Issue #543 rollup — aggregate per-cell eval summaries into one survival table.

Reads every ``eval_results/issue_543/<arm>/seed<S>/<phase>/run_summary.json``
plus the ``phase1_stop_record.json`` manipulation-check records, and writes
``eval_results/issue_543/rollup.json`` with:

  - per (arm x seed x phase x cell): emission rate + the three-space slot
    means (delta log-prob PRIMARY, delta EOS-margin logit SECONDARY,
    probability sanity);
  - per arm: pooled pre/post trigger emission with Wilson 95% CIs, per-seed
    values, Phase-1 stop steps (the dose covariate), excluded cells;
  - the §7 pre-registered criteria readouts (computed, NOT auto-verdicted —
    the analyzer owns interpretation).

CPU-only; safe to re-run any time. The driver invokes it after the fan-out.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="rollup_issue543_survival")

from _issue543_common import (  # noqa: E402
    ARM_PLAIN_NAMES,
    ARMS,
    EVAL_RESULTS_DIR,
    PHASES,
    SEEDS,
    repro_metadata,
)

log = logging.getLogger("rollup_issue543_survival")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issue #543 rollup (CPU; idempotent re-runs).")
    p.add_argument(
        "--out",
        type=Path,
        default=EVAL_RESULTS_DIR / "rollup.json",
        help=(
            "Output JSON path (default: the parent sweep's eval_results/issue_543/rollup.json; "
            "follow-up runs pass eval_results/issue_543/<followup_label>/rollup.json so the "
            "parent's 12-cell rollup is never clobbered)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rollup: dict = {**repro_metadata(), "cells": {}, "arms": {}}

    pooled: dict[str, dict[str, list[tuple[int, int]]]] = {
        a: {ph: [] for ph in PHASES} for a in ARMS
    }
    per_seed_emission: dict[str, dict[str, dict[int, float]]] = {
        a: {ph: {} for ph in PHASES} for a in ARMS
    }

    for arm in ARMS:
        for seed in SEEDS:
            cell_key = f"{arm}_seed{seed}"
            cell_dir = EVAL_RESULTS_DIR / arm / f"seed{seed}"
            stop_path = cell_dir / "phase1_stop_record.json"
            stop = json.loads(stop_path.read_text()) if stop_path.exists() else None
            entry: dict = {
                "stop_record_present": stop is not None,
                "install_excluded": bool(stop and stop.get("install_excluded")),
                "match_failure": bool(stop and stop.get("match_failure")),
                "phase1_total_steps": stop.get("phase1_total_steps") if stop else None,
                "phases": {},
            }
            for phase in PHASES:
                rs_path = cell_dir / phase / "run_summary.json"
                if not rs_path.exists():
                    entry["phases"][phase] = None
                    continue
                rs = json.loads(rs_path.read_text())
                entry["phases"][phase] = rs["cells"]
                trig = rs["cells"].get("trigger")
                if trig:
                    n = trig["n"]
                    k = round(trig["emission_rate"] * n)
                    pooled[arm][phase].append((k, n))
                    per_seed_emission[arm][phase][seed] = trig["emission_rate"]
            rollup["cells"][cell_key] = entry

    for arm in ARMS:
        arm_entry: dict = {"plain_name": ARM_PLAIN_NAMES[arm]}
        for phase in PHASES:
            ks = sum(k for k, _ in pooled[arm][phase])
            ns = sum(n for _, n in pooled[arm][phase])
            lo, hi = wilson_ci(ks, ns)
            arm_entry[phase] = {
                "pooled_k": ks,
                "pooled_n": ns,
                "pooled_emission_rate": (ks / ns) if ns else None,
                "wilson_95ci": [lo, hi],
                "per_seed_trigger_emission": per_seed_emission[arm][phase],
            }
        rollup["arms"][arm] = arm_entry

    # §7 criteria readouts (descriptive; analyzer owns the verdict).
    post = {
        a: rollup["arms"][a]["phase2"]["pooled_emission_rate"]
        for a in ARMS
        if rollup["arms"][a]["phase2"]["pooled_n"]
    }
    crit: dict = {"post_sft_pooled_trigger_emission": post}
    if "r05" in post and "r50" in post:
        crit["r05_minus_r50_pp"] = 100 * (post["r05"] - post["r50"])
        r05_seeds = per_seed_emission["r05"]["phase2"]
        r50_seeds = per_seed_emission["r50"]["phase2"]
        if r05_seeds and r50_seeds:
            crit["worst_r05_seed"] = min(r05_seeds.values())
            crit["best_r50_seed"] = max(r50_seeds.values())
            crit["worst_r05_gt_best_r50"] = crit["worst_r05_seed"] > crit["best_r50_seed"]
        ci05 = rollup["arms"]["r05"]["phase2"]["wilson_95ci"]
        ci50 = rollup["arms"]["r50"]["phase2"]["wilson_95ci"]
        crit["pooled_cis_disjoint"] = ci05[0] > ci50[1] or ci50[0] > ci05[1]
    rollup["criteria_readouts"] = crit

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rollup, indent=2))
    log.info("Rollup -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
