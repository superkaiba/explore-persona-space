#!/usr/bin/env python
"""Aggregate per-cell extractions into ``eval_results/issue_520/analysis.json``.

Reads every ``cells/<arm_slug>_seed<S>.json`` produced by
``run_issue520_train.py`` and runs DV1-DV5 (analysis.py) across them.

Smoke usage (1 cell)::

  uv run python scripts/run_issue520_aggregate.py \\
      --out-dir eval_results/issue_520 --far-pair paramedic comedian
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("issue520.aggregate")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate task #520 per-cell extractions into analysis.json"
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="eval_results/issue_520",
        help="Directory containing cells/ subdirectory.",
    )
    p.add_argument(
        "--far-pair",
        nargs=2,
        default=["paramedic", "comedian"],
        help="Far pair (default: paramedic comedian).",
    )
    p.add_argument(
        "--near-pair",
        nargs=2,
        default=None,
        help="Near pair (optional, e.g. surgeon medical_doctor).",
    )
    p.add_argument(
        "--include-b2-far",
        action="store_true",
        default=True,
        help="Include beta-2 robustness arm on the far pair (default ON).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    out_dir = Path(args.out_dir)
    cells_dir = out_dir / "cells"
    if not cells_dir.exists():
        logger.error("No cells dir at %s; nothing to aggregate", cells_dir)
        return 2
    cell_paths = sorted(cells_dir.glob("*.json"))
    if not cell_paths:
        logger.error("No per-cell JSONs in %s; nothing to aggregate", cells_dir)
        return 2

    from explore_persona_space.experiments.issue520.analysis import (
        aggregate_all,
        write_analysis,
    )

    payload = aggregate_all(
        cell_paths=cell_paths,
        far_pair=tuple(args.far_pair),
        near_pair=tuple(args.near_pair) if args.near_pair else None,
        include_b2_far=args.include_b2_far,
    )
    write_analysis(out_dir / "analysis.json", payload)
    logger.info(
        "Aggregated %d cells; H1 verdict=%s; H2 verdict=%s",
        len(cell_paths),
        payload.get("H1_b1_verdict", "n/a"),
        payload.get("H2_verdict", "n/a"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
