#!/usr/bin/env python3
"""Task #505 followup `expanded-predictor-reanalysis` — CLI driver.

Zero-GPU, analysis-only: rebuilds the 936-row pooled leave-one-out frame from
the existing trajectory JSONs, fits the expanded-covariate per-arm + pooled
regressions (cos(b, source) + base prior + shadow angle + nearest-remaining
negative), and writes JSONs + figures. See
``explore_persona_space.experiments.leave_one_out_505.analyze_expanded``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.leave_one_out_505.analyze_expanded import (  # noqa: E402
    figure_forest_cos_bj,
    figure_pooled_geometry,
    run_expanded_analysis,
)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    root = Path(os.environ.get("EPM_OUTPUT_ROOT", "eval_results/issue_505"))
    p = argparse.ArgumentParser(
        description="Task #505 expanded-predictor reanalysis driver (CPU-only)."
    )
    p.add_argument("--panel-gate", type=Path, default=root / "panel_coverage.json")
    p.add_argument("--sweep-dir", type=Path, default=root / "sweep")
    p.add_argument(
        "--original-analysis-dir",
        type=Path,
        default=root / "analysis",
        help="The original #505 analysis outputs (read-only; cross-checks + comparison).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "expanded-predictor-reanalysis",
        help="Followup artifact dir (new JSONs land here).",
    )
    p.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figures/issue_505/expanded-predictor-reanalysis"),
    )
    p.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    args = p.parse_args(argv)

    results = run_expanded_analysis(
        panel_gate_path=args.panel_gate,
        sweep_dir=args.sweep_dir,
        original_analysis_dir=args.original_analysis_dir,
        out_dir=args.out_dir,
    )
    if not args.no_figures:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        figure_forest_cos_bj(results, args.fig_dir)
        figure_pooled_geometry(results, args.fig_dir)

    print(json.dumps(results["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
