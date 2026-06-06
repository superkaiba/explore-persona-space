#!/usr/bin/env python3
"""Task #505 §13 — wrapper for the analysis pipeline (similarity matrix +
mixed-model fit + per-arm slopes + §13.3 partial)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.leave_one_out_505.analyze import (  # noqa: E402
    analyze_505,
)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Task #505 §13 analysis driver.")
    p.add_argument(
        "--panel-gate",
        type=Path,
        default=Path(os.environ.get("EPM_OUTPUT_ROOT", "eval_results/issue_505"))
        / "panel_coverage.json",
    )
    p.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path(os.environ.get("EPM_OUTPUT_ROOT", "eval_results/issue_505")) / "sweep",
    )
    p.add_argument(
        "--centroid-l10-dir",
        type=Path,
        default=Path(os.environ.get("EPM_I472_DATA_ROOT", "data/issue_472")),
    )
    p.add_argument(
        "--centroid-pv-dir",
        type=Path,
        default=Path(os.environ.get("EPM_DATA_ROOT", "data/issue_505")) / "centroids_pv",
    )
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(os.environ.get("EPM_OUTPUT_ROOT", "eval_results/issue_505")) / "analysis",
    )
    args = p.parse_args(argv)
    summary = analyze_505(
        panel_gate_path=args.panel_gate,
        sweep_dir=args.sweep_dir,
        centroid_dir_l10=args.centroid_l10_dir,
        centroid_dir_pv=args.centroid_pv_dir,
        analysis_dir=args.analysis_dir,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
