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
# Bucket B targets (the v1 default — narrow + broad).
B_TARGETS_IDS = ("T1_medical", "T2_code", "T3_legal", "B1_broad_em", "B2_broad_syco")
# Bucket A targets (plan v2 §4.2 cross-lingual).
A_TARGETS_IDS = ("A1_es_syco", "A1_prime_es_honest_correction", "A2_it_syco")
# Bucket D target (plan v2 §4.5 benign-data → AdvBench).
D_TARGETS_IDS = ("D_advbench",)
# Bucket E "synthetic" target ids (plan v2 §4.6 non-transfer; share T1/T2
# judges but bucket-tagged 'E'). Round-2 Rec 1 introduced these.
E_TARGETS_IDS = ("T1_medical_E", "T2_code_E", "T1_medical_E_alt")
# Round-2 Rec 2: all targets across buckets A/B/D/E. ``--bucket`` selects a
# subset; ``--targets`` overrides explicitly.
ALL_TARGETS = B_TARGETS_IDS + A_TARGETS_IDS + D_TARGETS_IDS + E_TARGETS_IDS


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
        help=(
            f"Subset of targets to score. "
            f"Bucket B (default): {', '.join(B_TARGETS_IDS)}. "
            f"Bucket A: {', '.join(A_TARGETS_IDS)}. "
            f"Bucket D: {', '.join(D_TARGETS_IDS)}. "
            f"Bucket E (synthetic ids; uses T1/T2 judges with E source adapters): "
            f"{', '.join(E_TARGETS_IDS)}."
        ),
    )
    parser.add_argument(
        "--bucket",
        choices=("A", "B", "D", "E"),
        default=None,
        help=(
            "Round-2 Rec 2 shorthand: enumerate all targets in a single bucket. "
            "Mutually-exclusive with --targets (if both are given, --targets wins). "
            "Bucket B = the v1 default narrow + broad target matrix."
        ),
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
    parser.add_argument(
        "--skip-missing-adapter",
        action="store_true",
        help=(
            "Round-6 GAP-5 graceful path: when the HF snapshot_download cannot "
            "find the adapter subfolder (training failed or never ran), write a "
            "deviation marker to ``eval_results/issue503/cross_eval/<source>_seed"
            "<seed>/_missing_adapter.json`` and exit 0 instead of crashing. The "
            "sweep records this as a per-cell deviation but the rest of the "
            "Bucket runs unaffected."
        ),
    )
    args = parser.parse_args()

    from explore_persona_space.experiments.issue503.behaviors import (
        A_TARGETS,
        BROAD_TARGETS,
        D_TARGETS,
        E_TARGETS,
        NARROW_TARGETS,
    )
    from explore_persona_space.experiments.issue503.cross_eval import (
        compute_kl_secondary_dv_for_source,
        cross_eval_dir,
        generate_completions_for_source,
        score_completions_for_source,
    )

    # Round-2 Rec 2: include Bucket A/D/E targets in the enumerable pool,
    # not just Bucket B. ``--bucket A|D|E`` is the shorthand that picks one
    # bucket end-to-end; ``--targets <ids>`` is the explicit override that
    # crosses buckets if needed.
    all_target_objs = (
        list(NARROW_TARGETS)
        + list(BROAD_TARGETS)
        + list(A_TARGETS)
        + list(D_TARGETS)
        + list(E_TARGETS)
    )
    if args.targets:
        target_objs = tuple(t for t in all_target_objs if t.target_id in args.targets)
        if len(target_objs) != len(args.targets):
            missing = set(args.targets) - {t.target_id for t in target_objs}
            raise ValueError(f"unknown target ids: {missing}")
    elif args.bucket is not None:
        if args.bucket == "A":
            target_objs = tuple(A_TARGETS)
        elif args.bucket == "B":
            target_objs = tuple(list(NARROW_TARGETS) + list(BROAD_TARGETS))
        elif args.bucket == "D":
            target_objs = tuple(D_TARGETS)
        else:  # E
            target_objs = tuple(E_TARGETS)
    else:
        # No --targets, no --bucket: keep the v1 Bucket-B default for
        # back-compat. Smokes for A/D/E must pass --bucket explicitly.
        target_objs = tuple(list(NARROW_TARGETS) + list(BROAD_TARGETS))

    # Resolve adapter path. If --adapter-path is an HF repo id, download.
    adapter_path = args.adapter_path
    if "/" in args.adapter_path and not Path(args.adapter_path).exists():
        # Treat as HF repo id; download subfolder.
        from huggingface_hub import hf_hub_download, list_repo_files

        if args.adapter_subfolder is None:
            raise ValueError(
                "--adapter-path looks like an HF repo id but --adapter-subfolder is unset"
            )
        try:
            # snapshot_download relies on model_info.siblings which is capped at
            # ~1000 files; the explore-persona-space model repo has thousands
            # of cumulative-experiment adapter files, so newly-uploaded
            # adapters fall outside the visible window and snapshot_download
            # returns zero files. Round-7 fix: use list_repo_files (paginated)
            # + per-file hf_hub_download (file-by-file metadata), which has no
            # repo-size cap.
            all_files = list_repo_files(repo_id=args.adapter_path)
            subfolder_prefix = args.adapter_subfolder.rstrip("/") + "/"
            matching = [f for f in all_files if f.startswith(subfolder_prefix)]
            if not matching:
                raise FileNotFoundError(
                    f"adapter subfolder {args.adapter_subfolder!r} has no files in "
                    f"{args.adapter_path!r} per list_repo_files; adapter never uploaded"
                )
            # Download each matching file individually. hf_hub_download
            # creates the directory structure under the cache root.
            local_root: Path | None = None
            for fname in matching:
                fp = hf_hub_download(repo_id=args.adapter_path, filename=fname)
                if local_root is None:
                    # Snapshot root is everything before the first occurrence
                    # of subfolder_prefix in the cached path.
                    parts = Path(fp).as_posix()
                    idx = parts.find(subfolder_prefix)
                    if idx >= 0:
                        local_root = Path(parts[:idx])
            adapter_path_final = (local_root or Path()) / args.adapter_subfolder
            if not adapter_path_final.exists() or not any(adapter_path_final.iterdir()):
                raise FileNotFoundError(
                    f"adapter subfolder {args.adapter_subfolder!r} is empty after "
                    f"file-by-file download in {args.adapter_path!r}"
                )
            adapter_path = str(adapter_path_final)
            logger.info("Downloaded %d files for adapter to %s", len(matching), adapter_path)
        except Exception as exc:
            # Round-6 GAP-5: gracefully skip cells whose adapter wasn't trained.
            # The sweep drives sources like Bucket D (D0_random/D1_representation
            # /...) where training may have failed; without this branch the
            # downstream vLLM load crashes on missing adapter_config.json and
            # all downstream cells go with it.
            if not args.skip_missing_adapter:
                raise
            from explore_persona_space.experiments.issue503.cross_eval import (
                cross_eval_dir,
            )

            out_dir = cross_eval_dir(PROJECT_ROOT, args.source, args.seed)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "_missing_adapter.json").write_text(
                json.dumps(
                    {
                        "source": args.source,
                        "seed": args.seed,
                        "adapter_repo": args.adapter_path,
                        "adapter_subfolder": args.adapter_subfolder,
                        "reason": "snapshot_download failed or returned empty",
                        "exception": str(exc),
                    },
                    indent=2,
                )
            )
            logger.warning(
                "Adapter missing for source=%s seed=%d (subfolder=%s); skipping cell. "
                "Deviation written to %s/_missing_adapter.json",
                args.source,
                args.seed,
                args.adapter_subfolder,
                out_dir,
            )
            return 0

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
