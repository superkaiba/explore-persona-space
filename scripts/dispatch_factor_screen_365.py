#!/usr/bin/env python3
"""Pod-side dispatcher for the task #365 2^5 factor screen.

Two stages:

  1. **Pool stage** (one process per source, sequential). For each source in
     ``--sources``, calls
     ``python -m explore_persona_space.experiments.factor_screen_365
     --mode dispatch --source <src> --pool-dir <dir>`` to:

       - render every ``(A, B, C)`` system prompt under Qwen2.5-7B-Instruct,
       - run the C-axis preflight (Jaccard / role-adoption / token equality),
       - generate the on-policy (D=0) and off-policy (D=1) completion pools,
       - emit ``prompt_manifest.json`` and ``persona_panel_manifest.csv``.

     Pools are SHARED across the E-axis flip — one (source, A, B, C) tuple
     yields one on-policy JSONL + one off-policy JSONL, reused for E=0 / E=1.

  2. **Training stage** (96 jobs fanned across 8 GPUs by
     ``CUDA_VISIBLE_DEVICES``). For each ``(cell, source, seed)`` triple,
     launches ``python -m explore_persona_space.experiments.factor_screen_365
     --cell <ABCDE> --source <src> --seed <N> --pool-dir <dir>
     --output-dir <slab_root>/cell_<key>/source_<src>/seed_<N>/``.

The librarian-only gate (plan §7) is implemented by ``--sources librarian``
on the first run; once the gate clears the user re-runs the dispatcher with
the full ``--sources librarian,surgeon,programmer`` set.

Aggregation is a separate one-shot:

    uv run python -m explore_persona_space.experiments.factor_screen_365 \\
        --mode aggregate --slab-root <slab_root> --output-dir <agg_dir>
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("dispatch_factor_screen_365")

SOURCES_DEFAULT = ("librarian", "surgeon", "programmer")


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )


def _parse_sources(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_seeds(raw: str) -> list[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    p.add_argument(
        "--sources",
        type=_parse_sources,
        default=list(SOURCES_DEFAULT),
        help="Comma-separated source personas (default: librarian,surgeon,programmer).",
    )
    p.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=[42],
        help="Comma-separated baseline seeds (default: 42).",
    )
    p.add_argument(
        "--pool-dir",
        type=Path,
        default=Path("data/issue_365/pools"),
        help="Where to materialise the per-source on/off-policy completion pools.",
    )
    p.add_argument(
        "--slab-root",
        type=Path,
        default=Path("eval_results/issue_365"),
        help="Per-cell metrics + adapters land here (cell_<key>/source_<src>/seed_<N>/).",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="GPU pool size for the training stage (default: 8).",
    )
    p.add_argument(
        "--skip-pool-stage",
        action="store_true",
        help="Skip the pool-generation stage (use when pools already exist on disk).",
    )
    p.add_argument(
        "--skip-off-policy",
        action="store_true",
        help="Skip Claude D1 off-policy generation (D=1 cells will fail at prepare_cell).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command list without launching anything.",
    )
    return p


def _pool_stage(args: argparse.Namespace) -> int:
    """Sequential per-source dispatch mode call."""
    args.pool_dir.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        cmd = [
            sys.executable,
            "-m",
            "explore_persona_space.experiments.factor_screen_365",
            "--mode",
            "dispatch",
            "--source",
            source,
            "--pool-dir",
            str(args.pool_dir),
        ]
        if args.skip_off_policy:
            cmd.append("--skip-off-policy")
        log.info("Pool stage: %s", " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            log.error("Pool stage failed for source=%s (rc=%d)", source, rc)
            return rc
    return 0


def _training_jobs(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    """Enumerate the (cell_key, source, seed) jobs for the training stage."""
    cells = ["".join(map(str, bits)) for bits in itertools.product((0, 1), repeat=5)]
    return [
        (cell_key, source, seed)
        for cell_key in cells
        for source in args.sources
        for seed in args.seeds
    ]


def _training_cmd(
    *,
    cell_key: str,
    source: str,
    seed: int,
    pool_dir: Path,
    slab_root: Path,
) -> list[str]:
    output_dir = slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"
    return [
        sys.executable,
        "-m",
        "explore_persona_space.experiments.factor_screen_365",
        "--cell",
        cell_key,
        "--source",
        source,
        "--seed",
        str(seed),
        "--pool-dir",
        str(pool_dir),
        "--output-dir",
        str(output_dir),
    ]


def _wait_for_free_gpu(running: dict[int, subprocess.Popen], gpu_pool: list[int]) -> int:
    while True:
        for gpu in gpu_pool:
            proc = running.get(gpu)
            if proc is None:
                return gpu
            if proc.poll() is not None:
                # Process finished.
                running.pop(gpu, None)
                if proc.returncode != 0:
                    log.warning("Job on GPU %d exited with rc=%d", gpu, proc.returncode)
                return gpu
        time.sleep(2)


def _training_stage(args: argparse.Namespace) -> int:
    jobs = _training_jobs(args)
    gpu_pool = list(range(args.num_gpus))
    running: dict[int, subprocess.Popen] = {}

    for cell_key, source, seed in jobs:
        cmd = _training_cmd(
            cell_key=cell_key,
            source=source,
            seed=seed,
            pool_dir=args.pool_dir,
            slab_root=args.slab_root,
        )
        if args.dry_run:
            log.info("DRYRUN: %s", " ".join(cmd))
            continue
        gpu = _wait_for_free_gpu(running, gpu_pool)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log.info("Launching cell=%s source=%s seed=%d on GPU %d", cell_key, source, seed, gpu)
        running[gpu] = subprocess.Popen(cmd, env=env)

    if args.dry_run:
        return 0

    # Drain.
    while running:
        gpu = _wait_for_free_gpu(running, gpu_pool)
        running.pop(gpu, None)
    log.info("Training stage complete: %d jobs", len(jobs))
    return 0


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = _build_arg_parser().parse_args(argv)

    if not args.skip_pool_stage:
        rc = _pool_stage(args)
        if rc != 0:
            log.error("Pool stage failed (rc=%d); aborting before training", rc)
            return rc
    else:
        log.info("Skipping pool stage as requested")

    return _training_stage(args)


if __name__ == "__main__":
    sys.exit(main())
