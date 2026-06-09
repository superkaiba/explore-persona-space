# ruff: noqa: RUF003  # em-dash + Qwen marker " ※" + × + − intentional
#!/usr/bin/env python3
"""Task #504 Phase 2 — analyze subprocess entrypoint (plan §4.4 + §6).

Runs the pooled partial-Spearman regression with the 6 covariates from the
per-cell trajectory.json + phase0_calibration.json + phase0_5_gates.json
artifacts. Subprocess for clean import isolation (numpy + statsmodels).

Writes:
    eval_results/issue_504/analyze_summary.json

Usage (driven by the dispatcher):
    uv run python scripts/i504_phase_analyze.py \
        --slab-root eval_results/issue_504 \
        --phase0-path eval_results/issue_504/phase0_calibration.json \
        --phase05-path eval_results/issue_504/phase0_5_gates.json \
        --seeds 42,137
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("i504.phase_analyze")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_504"))
    ap.add_argument(
        "--phase0-path", type=Path, default=Path("eval_results/issue_504/phase0_calibration.json")
    )
    ap.add_argument(
        "--phase05-path", type=Path, default=Path("eval_results/issue_504/phase0_5_gates.json")
    )
    ap.add_argument(
        "--base-prior-path",
        type=Path,
        default=None,
        help=(
            "Optional JSON {probe: base_prior_marker_logp_mean} from Phase 1c. "
            "If absent the regression covariate runs with 0.0 placeholder."
        ),
    )
    ap.add_argument("--seeds", default="42,137")
    ap.add_argument("--sentinel-path", type=Path, default=None)
    ap.add_argument(
        "--positioned-arms",
        choices=("v1", "v2", "v3"),
        default="v2",
        help=(
            "Round-2 fix (BLOCKER #1, concern_id `analyze-v2-slug-iteration`): "
            "which 4-arm slug set Phase 2 iterates over. `v2` (default) reads "
            "`c504v2_<arm>_seed<S>/trajectory.json` produced by the v2 dispatcher "
            "(`--phase phase1`). `v1` reads the legacy `c504_<arm>_seed<S>` slugs "
            "for archived-result re-analysis. `v3` reads the EPOCHS-ladder / v5 "
            "main-sweep `c504v3_<arm>_seed<S>` slugs."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=analyze] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        POSITIONED_ARM_SLUGS,
        POSITIONED_ARM_SLUGS_V2,
        POSITIONED_ARM_SLUGS_V3,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.analyze import (
        aggregate_base_prior_from_trajectories,
        run_phase2_analysis,
        write_analyze_summary,
        write_base_prior_marker,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase0 import (
        load_phase0_pick,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase05 import (
        load_phase05,
    )

    # Round-2 fix (BLOCKER #1): pick which 4-arm slug set Phase 2 iterates over.
    # v5 hot-fix: add v3 branch for the EPOCHS-ladder / main-sweep pipeline.
    if args.positioned_arms == "v3":
        positioned_arm_slugs: tuple[str, ...] = POSITIONED_ARM_SLUGS_V3
    elif args.positioned_arms == "v2":
        positioned_arm_slugs = POSITIONED_ARM_SLUGS_V2
    else:
        positioned_arm_slugs = POSITIONED_ARM_SLUGS
    log.info(
        "[positioned-arms] iterating %s arms: %s",
        args.positioned_arms,
        list(positioned_arm_slugs),
    )

    pick = load_phase0_pick(args.phase0_path)
    gates = load_phase05(args.phase05_path)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Round-2 fix (blocker #2): wire the per-probe base_prior_marker covariate.
    # If --base-prior-path was provided AND the file exists, use it. Otherwise,
    # aggregate b_logp per probe from the trajectory artifacts in-place (the
    # trajectory eval rig writes `b_logp` per (probe, q, ckpt); aggregating
    # across cells × seeds × ckpts × q gives the per-probe base-model marker
    # prior on the eval distribution — the #500 sign-flip discipline reads
    # this covariate, so 0.0-default was operationally disabling that check).
    base_prior: dict[str, float] | None = None
    if args.base_prior_path is not None and args.base_prior_path.exists():
        base_prior = json.loads(args.base_prior_path.read_text())
        log.info(
            "[base_prior] loaded base_prior_marker for %d probes from %s",
            len(base_prior),
            args.base_prior_path,
        )
    else:
        agg = aggregate_base_prior_from_trajectories(
            slab_root=args.slab_root,
            seeds=seeds,
            positioned_arm_slugs=positioned_arm_slugs,
        )
        if agg:
            base_prior = agg
            # Persist the aggregated map so downstream consumers (re-analyze,
            # robustness panels, the analyzer agent's body) can read it.
            target = (
                args.base_prior_path
                if args.base_prior_path is not None
                else args.slab_root / "base_prior_marker.json"
            )
            write_base_prior_marker(base_prior, target)
        else:
            log.warning(
                "[base_prior] no b_logp values found in trajectories under %s — "
                "base_prior_marker covariate falls back to 0.0 placeholder (the "
                "regression will run but the #500 sign-flip check is disabled).",
                args.slab_root,
            )

    summary = run_phase2_analysis(
        slab_root=args.slab_root,
        phase0_calibration=pick,
        phase05_gates=gates,
        seeds=seeds,
        base_prior_by_probe=base_prior,
        positioned_arm_slugs=positioned_arm_slugs,
    )
    out_path = args.slab_root / "analyze_summary.json"
    write_analyze_summary(summary, out_path)
    if not out_path.exists():
        raise RuntimeError(f"analyze wrote no summary at {out_path} — silent failure.")

    if args.sentinel_path is not None:
        args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        args.sentinel_path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": "epm:progress",
                    "version": 1,
                    "task_id": 504,
                    "phase": "analyze",
                    "by": "i504_phase_analyze",
                    "ts": datetime.now(UTC).isoformat(),
                    "note": json.dumps(
                        {
                            "analyze_summary_path": str(out_path),
                            "n_rows_pooled": summary.get("n_rows_pooled", 0),
                            "chosen_checkpoint_fraction": summary.get("chosen_checkpoint_fraction"),
                            "notes": summary.get("notes", []),
                        }
                    ),
                },
                indent=2,
            )
        )
    log.info(
        "[phase=analyze] n_rows_pooled=%d, notes=%s",
        summary.get("n_rows_pooled", 0),
        summary.get("notes", []),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
