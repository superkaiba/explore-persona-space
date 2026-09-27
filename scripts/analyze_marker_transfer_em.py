#!/usr/bin/env python3
"""Aggregation + analysis for marker-transfer EM (issue #80).

STUB — the analyzer agent owns this step.

The complete aggregation plan (per issue-80 plan §4.5 and §6):
  * Walk /workspace/marker_transfer/{c1,c2,c3,c4,c5}_*/run_result_*.json
    (pulled down from pod5 to eval_results/marker_transfer/ via
    `python scripts/pod.py sync results --all`).
  * For each (condition, seed): load assistant_strict_rate,
    villain_strict_rate, max_bystander_strict_rate, and ARC-C + Betley
    alignment from em_sub_run_result (for G4 gate).
  * G4 filter: drop any seed whose post-EM alignment is > 40. Re-run (or
    flag for re-run) before including in the primary conjunction.
  * Primary conjunction (one-sided Welch t-test across 3 seeds per pair):
      1. C1 - C2 >= 0.15, p < 0.05
      2. C1 - C5 >= 0.15, p < 0.05
      3. C1 post-EM villain rate >= 0.50
      4. C1 degeneracy_audit.degenerate_fraction < 0.20 (averaged across
         C1 seeds)
  * Hero figure: grouped bar chart with error bars — per-condition
    assistant strict_rate (x-axis = condition, bars = mean across seeds,
    error bars = SD). Include villain rate as secondary colour. Label
    C4 as pre-EM baseline. Use paper-plots skill + paper_plots rcParams.
  * Secondary table: per-persona strict_rate for C1 seed-42 (sanity
    check for feature dilution / bystander bleed).

This script is left as a stub so the analyzer's fresh context runs the
aggregation from scratch — per the plan, the experimenter owns launching
the runs and writing run_result.json, not the cross-cell analysis.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "analyze_marker_transfer_em.py is a stub. The analyzer agent owns "
        "aggregation + hero figure; the experimenter's job is to ensure "
        "every cell's run_result_<cond>_seed<seed>.json is on disk.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
