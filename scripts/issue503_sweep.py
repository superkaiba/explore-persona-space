#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, →) in scientific docstrings + logs.
"""Issue #503 — top-level dispatcher (plan §3.6 architectural parity).

ONE dispatcher. Smoke is sweep with one cell: ``--cells <one>
--max-prompts 8``. Identical code path; same env injection; same teardown;
same logging. No in-process vs subprocess divergence.

Per (source, seed) the dispatcher runs:
1. Source training is delegated to ``scripts/train.py`` — this script
   does NOT train sources; it sweeps over already-trained adapters.
2. Cross-eval generation (vLLM) + judging (Claude Batch) — calls
   ``scripts/issue503_cross_eval.py`` per source via subprocess.
3. Predictor extraction (base model forward) — calls
   ``scripts/issue503_extract_predictors.py`` per cell.
4. Regression (pooled, binomial mixed model) — calls
   ``scripts/issue503_regression.py``.

Each subprocess is launched with explicit ``env={**os.environ}`` and the
dispatcher calls ``load_dotenv()`` at entry so HF_TOKEN / WANDB_API_KEY
/ ANTHROPIC_API_KEY are inherited explicitly per the experiment-
implementer subprocess-env-explicit checklist.

Per CLAUDE.md checkpoint-per-phase: per-source per-target completions +
verdicts + predictor records are written immediately as each phase
completes. A crash never loses earlier phases.

Usage (smoke, 1 cell)::

    uv run python scripts/issue503_sweep.py \\
        --cells insecure_code--T1_medical --seeds 0 --max-prompts 8 \\
        --skip-source-training

Usage (sweep, full matrix — requires source adapters trained
externally and source-adapter-path provided)::

    uv run python scripts/issue503_sweep.py --all-cells --seeds 0 137
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue503_sweep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:  # noqa: C901 — dispatcher with argument-parser branches, intentionally flat
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Cells as 'source--target_id' pairs.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 137], help="Seeds per cell.")
    parser.add_argument(
        "--all-cells",
        action="store_true",
        help="Enumerate every cell (98 off-diagonal + 10 install-QC).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Smoke: cap prompts per target.",
    )
    parser.add_argument(
        "--n-rollouts-override",
        type=int,
        default=None,
        help="Smoke: override n_rollouts per prompt.",
    )
    parser.add_argument(
        "--adapter-repo",
        default="superkaiba1/explore-persona-space",
        help="HF model repo where source adapters live.",
    )
    parser.add_argument(
        "--skip-cross-eval",
        action="store_true",
        help="Skip the vLLM cross-eval phase (assume completions already written).",
    )
    parser.add_argument(
        "--bucket",
        choices=("A", "B", "D", "E"),
        default=None,
        help=(
            "Round-2 Rec 2: forward --bucket A|B|D|E down to the cross_eval "
            "subprocess so the per-source enumeration in cross_eval uses the "
            "right Bucket's targets when this sweep was invoked without an "
            "explicit --targets list. The dispatch still groups cells by "
            "(source, seed); the bucket only affects the cross_eval's "
            "fallback target list when --cells/--all-cells did not name "
            "Bucket-A/D/E target ids explicitly."
        ),
    )
    parser.add_argument(
        "--skip-kl",
        action="store_true",
        help=(
            "MF-J round-3: skip the KL-secondary-DV phase inside the cross_eval "
            "subprocess. Disabled by default so the full sweep records "
            "kl_secondary_dv per cell. Smoke parity flag."
        ),
    )
    parser.add_argument(
        "--skip-predictors",
        action="store_true",
        help="Skip the predictor extraction phase.",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip the final regression phase.",
    )
    parser.add_argument(
        "--skip-missing-adapter",
        action="store_true",
        help=(
            "Round-6 GAP-5: forward --skip-missing-adapter to the per-source "
            "issue503_cross_eval.py subprocess so cells whose adapter never "
            "trained (e.g. failed Bucket D selectors) record a deviation and "
            "skip rather than crashing the whole sweep."
        ),
    )
    args = parser.parse_args()

    from explore_persona_space.experiments.issue503.behaviors import (
        adapter_subfolder_for_source,
        enumerate_all_cells_as_tuples,
    )

    if args.all_cells:
        # Round-3 in-line fix (post-cap-3 orchestrator patch): --all-cells
        # enumerates ALL 5 buckets (A/B/C/D/E), not just v1's enumerate_cells()
        # which is B/C-only. The v1 path is reachable via --bucket B + --bucket C
        # (or explicit --cells). Production sweep needs the 5-bucket union for the
        # H8 calibration headline; previously a launcher --all-cells would
        # silently launch a 4-bucket-of-5 sweep that cannot produce the headline.
        target_cells = enumerate_all_cells_as_tuples(seeds_v1=tuple(args.seeds))
    elif args.cells:
        target_cells = []
        for pair in args.cells:
            src, tid = pair.split("--", 1)
            for seed in args.seeds:
                target_cells.append((src, tid, seed))
    else:
        parser.error("Provide --cells or --all-cells.")

    env = {**os.environ}
    sources = sorted({src for src, _, _ in target_cells})
    seeds = sorted({seed for _, _, seed in target_cells})

    # Phase 1: cross-eval per (source, seed). Each invocation handles
    # all targets per source per the §3.4 "one model load, all targets"
    # spec.
    if not args.skip_cross_eval:
        # Group target_ids per (source, seed) so each subprocess sees
        # the right subset.
        for source in sources:
            for seed in seeds:
                src_seed_targets = sorted(
                    {tid for s, tid, sd in target_cells if s == source and sd == seed}
                )
                if not src_seed_targets:
                    continue
                logger.info(
                    "[phase=cross_eval] source=%s seed=%d targets=%s",
                    source,
                    seed,
                    src_seed_targets,
                )
                cmd = [
                    "uv",
                    "run",
                    "python",
                    str(PROJECT_ROOT / "scripts" / "issue503_cross_eval.py"),
                    "--source",
                    source,
                    "--seed",
                    str(seed),
                    "--adapter-path",
                    args.adapter_repo,
                    "--adapter-subfolder",
                    # MF-F round-2 revision: source-family-aware mapping.
                    # narrow → issue458_pair_{source}_seed{seed}/sft_narrow_adapter
                    # broad_em → issue458_pair_turner_risky_financial_seed{seed}/sft_narrow_adapter
                    # broad_syco → issue503_broad_syco_seed{seed}/adapter
                    adapter_subfolder_for_source(source, seed),
                    "--targets",
                    *src_seed_targets,
                ]
                if args.max_prompts is not None:
                    cmd += ["--max-prompts", str(args.max_prompts)]
                if args.n_rollouts_override is not None:
                    cmd += ["--n-rollouts-override", str(args.n_rollouts_override)]
                # MF-J round-3: forward --skip-kl down to the cross_eval
                # subprocess. Without forwarding it the sweep would
                # silently run KL even when --skip-kl was passed at the
                # sweep level (smoke parity bug).
                if args.skip_kl:
                    cmd += ["--skip-kl"]
                if args.skip_missing_adapter:
                    cmd += ["--skip-missing-adapter"]
                if args.bucket is not None:
                    cmd += ["--bucket", args.bucket]
                subprocess.run(cmd, env=env, check=True, cwd=PROJECT_ROOT)

    # Phase 2: predictor extraction per cell. Base-model forward only;
    # one model load (re-used across cells).
    if not args.skip_predictors:
        logger.info("[phase=predictors] %d cells", len(target_cells))
        cell_args: list[str] = []
        for src, tid, _seed in target_cells:
            cell_args.append(f"{src}--{tid}")
        cell_args = sorted(set(cell_args))
        cmd = [
            "uv",
            "run",
            "python",
            str(PROJECT_ROOT / "scripts" / "issue503_extract_predictors.py"),
            "--cells",
            *cell_args,
            "--seeds",
            *[str(s) for s in seeds],
        ]
        if args.max_prompts is not None:
            cmd += ["--n-probes", str(min(args.max_prompts, 8))]
        subprocess.run(cmd, env=env, check=True, cwd=PROJECT_ROOT)

    # Phase 3: regression.
    if not args.skip_regression:
        logger.info("[phase=regression]")
        cmd = [
            "uv",
            "run",
            "python",
            str(PROJECT_ROOT / "scripts" / "issue503_regression.py"),
        ]
        if args.max_prompts is not None:
            cmd += ["--smoke"]
        subprocess.run(cmd, env=env, check=True, cwd=PROJECT_ROOT)

    logger.info("[phase=done] sweep complete; %d cells", len(target_cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
