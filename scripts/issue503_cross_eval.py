#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, →) in scientific docstrings + logs.
"""Issue #503 — cross-evaluation rig dispatcher (plan §3.4).

Per (source, seed), runs vLLM batched generation across all targets,
then judges via Claude Sonnet 4.5 (or static check + Sonnet for T2),
emits per-target (k, n, rate) verdict JSONs.

CLAUDE.md vLLM rule: never sequential ``model.generate``; this uses
``LLM.generate(..., sampling)`` with ``SamplingParams(n=K)``. Per-phase
checkpoint per CLAUDE.md: each target's completions JSONL is written
IMMEDIATELY after generation; each target's verdict JSON is written
IMMEDIATELY after judging — a downstream crash never loses earlier
phases.

Smoke vs sweep (plan §3.6 architectural parity): smoke = same script
with ``--max-prompts 8 --n-rollouts-override 1``. No in-process vs
subprocess divergence.

Usage (full sweep, per source)::

    uv run python scripts/issue503_cross_eval.py \\
        --source insecure_code --seed 0 \\
        --adapter-path superkaiba1/explore-persona-space \\
        --adapter-subfolder issue458_pair_insecure_code_seed0/sft_narrow_adapter

Smoke (one cell, ~5 min)::

    uv run python scripts/issue503_cross_eval.py \\
        --source insecure_code --seed 0 \\
        --targets T1_medical --max-prompts 8 --n-rollouts-override 1 \\
        --adapter-path superkaiba1/explore-persona-space \\
        --adapter-subfolder issue458_pair_insecure_code_seed0/sft_narrow_adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue503_cross_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ALL_TARGETS = ("T1_medical", "T2_code", "T3_legal", "B1_broad_em", "B2_broad_syco")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Source cell name.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model id (default: Qwen-2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help=(
            "LoRA adapter path or HF model repo id. If repo id, --adapter-subfolder must be set."
        ),
    )
    parser.add_argument(
        "--adapter-subfolder",
        default=None,
        help="Subfolder inside an HF model repo where the adapter lives.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help=f"Subset of targets to score (default: all 5: {', '.join(ALL_TARGETS)}).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Smoke: cap prompts per target to this many.",
    )
    parser.add_argument(
        "--n-rollouts-override",
        type=int,
        default=None,
        help="Smoke: override n_rollouts per prompt.",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-5",
        help="Claude judge model id (default: Sonnet 4.5).",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip vLLM generation; only run the judge phase on existing completions.",
    )
    parser.add_argument(
        "--skip-judging",
        action="store_true",
        help="Skip judging; only run vLLM generation.",
    )
    parser.add_argument(
        "--skip-kl",
        action="store_true",
        help=(
            "MF-J round-3: skip the KL-secondary-DV phase (per CLAUDE.md "
            "marker-leakage-measurement rule's saturation-fallback DV). "
            "Smoke parity flag — disabled in full sweeps so every cell records "
            "kl_secondary_dv."
        ),
    )
    args = parser.parse_args()

    from explore_persona_space.experiments.issue503.behaviors import (
        BROAD_TARGETS,
        NARROW_TARGETS,
    )
    from explore_persona_space.experiments.issue503.cross_eval import (
        compute_kl_secondary_dv_for_source,
        cross_eval_dir,
        generate_completions_for_source,
        score_completions_for_source,
    )

    all_target_objs = list(NARROW_TARGETS) + list(BROAD_TARGETS)
    if args.targets:
        target_objs = tuple(t for t in all_target_objs if t.target_id in args.targets)
        if len(target_objs) != len(args.targets):
            missing = set(args.targets) - {t.target_id for t in target_objs}
            raise ValueError(f"unknown target ids: {missing}")
    else:
        target_objs = tuple(all_target_objs)

    # Resolve adapter path. If --adapter-path is an HF repo id, download.
    adapter_path = args.adapter_path
    if "/" in args.adapter_path and not Path(args.adapter_path).exists():
        # Treat as HF repo id; download subfolder.
        from huggingface_hub import snapshot_download

        if args.adapter_subfolder is None:
            raise ValueError(
                "--adapter-path looks like an HF repo id but --adapter-subfolder is unset"
            )
        adapter_path = snapshot_download(
            repo_id=args.adapter_path,
            allow_patterns=[f"{args.adapter_subfolder}/*"],
        )
        adapter_path = str(Path(adapter_path) / args.adapter_subfolder)
        logger.info("Downloaded adapter to %s", adapter_path)

    if not args.skip_generation:
        logger.info(
            "Generation phase: source=%s seed=%d targets=%s",
            args.source,
            args.seed,
            [t.target_id for t in target_objs],
        )
        generate_completions_for_source(
            source_adapter_path=adapter_path,
            source=args.source,
            seed=args.seed,
            base_model_id=args.base_model,
            repo_root=PROJECT_ROOT,
            targets=target_objs,
            max_prompts_per_target=args.max_prompts,
            n_rollouts_override=args.n_rollouts_override,
        )

    # MF-J round-3 revision: production wire-up of the KL secondary DV
    # phase. ``compute_kl_secondary_dv_for_source`` was added in round-2
    # as the saturation-fallback DV (plan §5.1 +
    # ``.claude/rules/marker-leakage-measurement.md``) but no production
    # caller invoked it, so every cell's verdict recorded
    # ``kl_secondary_dv: None``. The KL phase runs after generation (its
    # per-target completions are already on disk per checkpoint-per-phase)
    # and BEFORE the judge phase so the verdict merge picks up the
    # ``<target>.kl.json`` files. The function loads base + LoRA on the
    # single GPU; its own ``finally`` block releases both models — see
    # the helper itself for the empty-cache + gc.collect call sequence
    # that closes the round-2 analyzer-Minor note.
    if not args.skip_kl:
        logger.info("KL secondary DV phase: source=%s seed=%d", args.source, args.seed)
        compute_kl_secondary_dv_for_source(
            source_adapter_path=adapter_path,
            source=args.source,
            seed=args.seed,
            base_model_id=args.base_model,
            repo_root=PROJECT_ROOT,
            targets=target_objs,
        )

    if not args.skip_judging:
        logger.info("Judging phase: source=%s seed=%d", args.source, args.seed)
        cells = score_completions_for_source(
            source=args.source,
            seed=args.seed,
            repo_root=PROJECT_ROOT,
            targets=target_objs,
            judge_model=args.judge_model,
        )
        out_path = cross_eval_dir(PROJECT_ROOT, args.source, args.seed) / "cells_summary.json"
        out_path.write_text(
            json.dumps(
                [
                    {
                        "source": c.source,
                        "target_id": c.target_id,
                        "seed": c.seed,
                        "k": c.k,
                        "n": c.n,
                        "rate": c.rate,
                        "n_errors": c.n_errors,
                        "n_static_positive": c.n_static_positive,
                        "median_tokens": c.median_tokens,
                        "truncation_rate": c.truncation_rate,
                    }
                    for c in cells
                ],
                indent=2,
            )
        )
        logger.info("Wrote cells summary to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
