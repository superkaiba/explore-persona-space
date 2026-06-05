#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, →) in scientific docstrings + logs.
"""Issue #503 — build the broad-sycophancy training pool (plan §3.2.2).

Generates 200 positives + 200 contrastive negatives per seed via the
Claude Sonnet 4.5 pool builder in
``src/explore_persona_space/experiments/issue503/broad_syco_dataset.py``,
plus a 50-row held-out positive panel used for (a) the install kill
criterion (plan §3.2.2: ≥+0.30 above base on the held-out claim panel)
and (b) the broad-syco target persona-vector pool (plan §3.3.2).

Outputs (idempotent — skips if files already present at correct counts):

    data/issue503/broad_syco/topics_seed{S}.json
    data/issue503/broad_syco/train_seed{S}.jsonl      (400 rows)
    data/issue503/broad_syco/heldout_seed{S}.jsonl    (50 rows)

Usage::

    uv run python scripts/issue503_build_broad_syco_dataset.py --seeds 0 137
    uv run python scripts/issue503_build_broad_syco_dataset.py --seeds 0  # smoke
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue503_build_broad_syco_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 137], help="Seeds to materialize."
    )
    parser.add_argument(
        "--judge-model", default="claude-sonnet-4-5", help="Claude model for generation."
    )
    args = parser.parse_args()

    from explore_persona_space.experiments.issue503.broad_syco_dataset import (
        build_broad_syco_dataset,
    )

    for seed in args.seeds:
        logger.info("==> seed=%d", seed)
        paths = build_broad_syco_dataset(PROJECT_ROOT, seed=seed, judge_model=args.judge_model)
        logger.info(
            "  wrote train=%s heldout=%s topics=%s",
            paths["train"],
            paths["heldout"],
            paths["topics"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
