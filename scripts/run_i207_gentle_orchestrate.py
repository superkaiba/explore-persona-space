#!/usr/bin/env python3
"""Orchestrate the 12-adapter gentler-recipe sweep for issue #343.

Spawns 4 parallel workers per batch (one per H100 GPU), cycling through 3
batches to cover all 12 (family, seed) cells. Each worker invokes
``scripts/run_i207_gentle_worker.py`` which trains+merges+uploads the
adapter then runs the panel eval.

Stages run in order:
    Batch 1: (task, 42), (instruction, 42), (context, 42), (format, 42)
    Batch 2: (task, 137), (instruction, 137), (context, 137), (format, 137)
    Batch 3: (task, 256), (instruction, 256), (context, 256), (format, 256)

Designed to be idempotent: re-running picks up existing merged checkpoints
and panel evals (workers skip already-done work).

Usage:
    nohup uv run python scripts/run_i207_gentle_orchestrate.py \
        > /workspace/logs/i343_gentle_orchestrate.log 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = Path("/workspace/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = ["task", "instruction", "context", "format"]
SEEDS = [42, 137, 256]


def run_one_batch(
    seed: int, n_gpus: int = 4, skip_eval: bool = False, skip_upload: bool = False
) -> list[tuple[str, int, int]]:
    """Spawn n_gpus parallel workers (one per family) for this seed.

    Returns list of (family, gpu_id, return_code).
    """
    procs = []
    log_files = []
    for gpu_id, family in enumerate(FAMILIES[:n_gpus]):
        run_name = f"i181_gentle_{family}_seed{seed}_train"
        log_path = LOG_DIR / f"i343_worker_{run_name}.log"
        log_files.append(log_path)
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_i207_gentle_worker.py",
            "--family",
            family,
            "--seed",
            str(seed),
            "--gpu",
            str(gpu_id),
        ]
        if skip_eval:
            cmd.append("--skip-eval")
        if skip_upload:
            cmd.append("--skip-upload")
        logger.info("Spawning %s on GPU %d -> %s", run_name, gpu_id, log_path)
        f = open(log_path, "w")
        p = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT)
        procs.append((family, gpu_id, p, f))

    # Wait for all to complete
    results = []
    for family, gpu_id, p, f in procs:
        rc = p.wait()
        f.close()
        results.append((family, gpu_id, rc))
        if rc == 0:
            logger.info("  OK: %s (seed=%d, gpu=%d)", family, seed, gpu_id)
        else:
            logger.error("  FAIL: %s (seed=%d, gpu=%d) rc=%d", family, seed, gpu_id, rc)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS, help="Which seeds to run (default: all 3)"
    )
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ISSUE #343 ORCHESTRATOR — gentler-recipe sweep")
    logger.info("=" * 60)
    logger.info(
        "seeds=%s, n_gpus=%d, skip_eval=%s, skip_upload=%s",
        args.seeds,
        args.n_gpus,
        args.skip_eval,
        args.skip_upload,
    )

    t_start = time.time()
    all_results = {}
    for seed in args.seeds:
        logger.info("\n" + "=" * 60)
        logger.info("BATCH seed=%d", seed)
        logger.info("=" * 60)
        t_batch = time.time()
        batch_results = run_one_batch(
            seed=seed,
            n_gpus=args.n_gpus,
            skip_eval=args.skip_eval,
            skip_upload=args.skip_upload,
        )
        all_results[seed] = batch_results
        logger.info("Batch seed=%d done in %.1f min", seed, (time.time() - t_batch) / 60)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("ORCHESTRATOR SUMMARY")
    logger.info("=" * 60)
    n_ok = 0
    n_fail = 0
    for seed, results in all_results.items():
        for family, gpu_id, rc in results:
            tag = "OK" if rc == 0 else "FAIL"
            logger.info("  seed=%d family=%-12s gpu=%d  rc=%d  %s", seed, family, gpu_id, rc, tag)
            if rc == 0:
                n_ok += 1
            else:
                n_fail += 1
    logger.info("Total: %d OK, %d FAIL, %.1f min wall", n_ok, n_fail, (time.time() - t_start) / 60)
    logger.info("Done at %s", datetime.now(UTC).isoformat())


if __name__ == "__main__":
    main()
