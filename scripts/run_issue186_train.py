#!/usr/bin/env python3
"""Phase-1 SFT launcher for issue #186 (sequential subprocess loop).

Iterates over the 4x3 main grid (12 (source, arm) cells) plus the
1-cell ``persona-cot-correct`` librarian control, each run with 3 seeds:
4x3x3 + 1x3 = **39 LoRA-7B SFT runs**. Each cell shells out to
``scripts/train.py`` with the right Hydra ``condition=...`` override and
seed override; the existing in-process LoRA path handles HF Hub upload of
the resulting adapter.

The launcher is intentionally a thin orchestrator -- all training logic
lives in ``scripts/train.py`` / ``orchestrate.runner.run_single``. We just:

* compose the Hydra command line correctly,
* tag the WandB run name as ``issue186_{source}_{arm}_seed{S}``,
* skip cells whose adapter is already on HF Hub (idempotent re-runs),
* bail loudly on any non-zero exit.

CLI::

    uv run python scripts/run_issue186_train.py
    uv run python scripts/run_issue186_train.py --only librarian persona_cot
    uv run python scripts/run_issue186_train.py --seeds 42
    uv run python scripts/run_issue186_train.py --dry-run
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

logger = logging.getLogger("run_issue186_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SOURCES: list[str] = ["software_engineer", "librarian", "comedian", "police_officer"]
MAIN_ARMS: list[str] = ["no_cot", "generic_cot", "persona_cot"]
CORRECT_ARM = ("librarian", "persona_cot_correct")
DEFAULT_SEEDS: list[int] = [42, 137, 256]


def _build_cells(only_source: str | None, only_arm: str | None) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    for src in SOURCES:
        for arm in MAIN_ARMS:
            cells.append((src, arm))
    cells.append(CORRECT_ARM)
    if only_source:
        cells = [c for c in cells if c[0] == only_source]
    if only_arm:
        cells = [c for c in cells if c[1] == only_arm]
    return cells


def _condition_name(source: str, arm: str) -> str:
    return f"issue186/i186_{source}_{arm}"


def _wandb_run_name(source: str, arm: str, seed: int) -> str:
    return f"issue186_{source}_{arm}_seed{seed}"


def _adapter_already_on_hf(repo_id: str, condition: str, seed: int) -> bool:
    """Return True if the adapter folder for this (condition, seed) is on HF Hub.

    Lazy: the runner uploads to ``i186_{source}_{arm}/seed{seed}`` (matching
    the existing folder convention used by orchestrate.runner). We only do
    a quick listing -- on import error or auth error we conservatively return
    False so the run isn't skipped accidentally.
    """
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
    needle = f"{cell_id}/seed{seed}/"
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
    # cell_id is like ``i186_librarian_persona_cot``; strip the leading ``i186_``
    # to recover the source/arm tuple for the WandB tag.
    pretty = cell_id.removeprefix("i186_")
    env["WANDB_NAME"] = f"issue186_{pretty}_seed{seed}"
    env.setdefault("WANDB_TAGS", "issue186")

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
            "Restrict to a single arm (e.g. 'persona_cot' or 'persona_cot_correct'). "
            "Underscore-form (matches the YAML stem)."
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
