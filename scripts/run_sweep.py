#!/usr/bin/env python3
"""Local multi-GPU sweep runner.

NOTE: This script predates the ephemeral one-pod-per-issue execution model.
It assigns conditions round-robin across local GPUs and is appropriate only
for local multi-GPU development on a workstation. For production runs, file
one GitHub issue per condition and let `/issue` (interactive) or
`/auto-experiment-runner` (overnight queue) provision a dedicated pod per
issue. See CLAUDE.md "Ephemeral Pod Lifecycle" for the supported flow.

Usage (local dev only):
    # Full sweep with pilot:
    python scripts/run_sweep.py

    # Skip pilot:
    python scripts/run_sweep.py --no-pilot

    # Eval only (models already trained):
    python scripts/run_sweep.py --eval-only

    # Training only:
    python scripts/run_sweep.py --train-only

    # Check status:
    python scripts/run_sweep.py --status

    # Set parallelism:
    python scripts/run_sweep.py --parallel 8
"""

import argparse

# Setup paths and env
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run the Make Evil Dumb experiment sweep")
    parser.add_argument("--config-dir", default="configs", help="Root config directory")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: auto-detect)"
    )
    parser.add_argument("--parallel", type=int, default=4, help="Max parallel jobs")
    parser.add_argument("--no-pilot", action="store_true", help="Skip pilot run")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only run eval")
    parser.add_argument("--train-only", action="store_true", help="Skip eval, only train")
    parser.add_argument("--status", action="store_true", help="Print sweep status and exit")
    args = parser.parse_args()

    from explore_persona_space.orchestrate.sweep import ExperimentSweep

    sweep = ExperimentSweep(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        max_parallel=args.parallel,
    )

    if args.status:
        sweep.print_status()
        return

    sweep.run_sweep(
        skip_training=args.eval_only,
        skip_eval=args.train_only,
        pilot_first=not args.no_pilot,
    )


if __name__ == "__main__":
    main()
