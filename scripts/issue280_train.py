#!/usr/bin/env python3
"""Phase-1 SFT launcher for issue #280 (sequential subprocess loop).

Iterates over the 4 sources x 3 NEW arms = 12 cells plus the 1 librarian-only
``generic-cot-correct`` control = **13 (source, arm) cells**, each at 3 seeds:
13 x 3 = **39 LoRA-7B SFT runs**.

Mirrors ``scripts/run_issue186_train.py`` verbatim except for the cell list
and the WandB tag prefix. Each cell shells out to ``scripts/train.py`` with
the right Hydra ``condition=...`` override; the existing in-process LoRA path
handles HF Hub upload of the resulting adapter.

CLI::

    uv run python scripts/issue280_train.py
    uv run python scripts/issue280_train.py --only-source librarian
    uv run python scripts/issue280_train.py --only-arm contradicting_cot
    uv run python scripts/issue280_train.py --seeds 42 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("issue280_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SOURCES: list[str] = ["software_engineer", "librarian", "comedian", "police_officer"]
NEW_MAIN_ARMS: list[str] = [
    "garbage_cot",
    "scrambled_english_cot",
    "contradicting_cot",
]
LIBRARIAN_ONLY_CELL: tuple[str, str] = ("librarian", "generic_cot_correct")
DEFAULT_SEEDS: list[int] = [42, 137, 256]


def _build_cells(only_source: str | None, only_arm: str | None) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for src in SOURCES:
        for arm in NEW_MAIN_ARMS:
            cells.append((src, arm))
    cells.append(LIBRARIAN_ONLY_CELL)
    if only_source:
        cells = [c for c in cells if c[0] == only_source]
    if only_arm:
        cells = [c for c in cells if c[1] == only_arm]
    return cells


def _condition_name(source: str, arm: str) -> str:
    return f"issue280/i280_{source}_{arm}"


def _adapter_already_on_hf(repo_id: str, condition: str, seed: int) -> bool:
    """Return True if the merged checkpoint dir for this (condition, seed) is
    already on HF Hub. Mirrors ``run_issue186_train.py`` (be17b42)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False
    try:
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        logger.debug("HF Hub listing failed (treating as not-uploaded): %s", e)
        return False
    cell_id = condition.split("/")[-1]
    needle = f"{cell_id}_seed{seed}_post_em/"
    return any(f.startswith(needle) for f in files)


def _run_one_cell(condition: str, seed: int, dry_run: bool) -> int:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train.py",
        f"condition={condition}",
        f"seed={seed}",
    ]
    env = os.environ.copy()
    parts = condition.split("/")
    cell_id = parts[-1]
    pretty = cell_id.removeprefix("i280_")
    env["WANDB_NAME"] = f"issue280_{pretty}_seed{seed}"
    env.setdefault("WANDB_TAGS", "issue280")

    logger.info("→ %s seed=%d (run name=%s)", condition, seed, env["WANDB_NAME"])
    logger.info("   cmd: %s", " ".join(cmd))
    if dry_run:
        return 0

    started = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    elapsed = time.time() - started
    if proc.returncode != 0:
        logger.error("FAILED: %s seed=%d (rc=%d, %.1fs)", condition, seed, proc.returncode, elapsed)
    else:
        logger.info("DONE: %s seed=%d (%.1fs)", condition, seed, elapsed)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Seeds to run (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--only-source",
        type=str,
        default=None,
        help="Restrict to a single source persona (e.g. 'librarian').",
    )
    parser.add_argument(
        "--only-arm",
        type=str,
        default=None,
        help=(
            "Restrict to a single arm (underscore form, e.g. 'contradicting_cot' or "
            "'generic_cot_correct')."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="superkaiba1/explore-persona-space",
        help="HF Hub model repo (used to short-circuit already-uploaded cells).",
    )
    parser.add_argument(
        "--no-skip-uploaded",
        action="store_true",
        help="Disable the HF Hub presence check; always run every cell.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cells but don't shell out.",
    )
    args = parser.parse_args()

    cells = _build_cells(args.only_source, args.only_arm)
    runs = [(c[0], c[1], s) for c in cells for s in args.seeds]
    logger.info("Plan: %d cells x %d seeds = %d runs", len(cells), len(args.seeds), len(runs))

    failures: list[tuple[str, int, int]] = []
    for source, arm, seed in runs:
        condition = _condition_name(source, arm)
        if not args.no_skip_uploaded and _adapter_already_on_hf(args.hf_repo, condition, seed):
            logger.info("SKIP (already on HF): %s seed=%d", condition, seed)
            continue
        rc = _run_one_cell(condition, seed, args.dry_run)
        if rc != 0:
            failures.append((condition, seed, rc))

    if failures:
        logger.error("Failures: %s", failures)
        sys.exit(1)
    logger.info("All runs completed successfully.")


if __name__ == "__main__":
    main()
