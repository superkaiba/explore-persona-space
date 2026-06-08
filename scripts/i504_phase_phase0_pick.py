# em-dash + Qwen marker " ※" intentional
#!/usr/bin/env python3
"""Task #504 Phase 0 — anchor pick from smoke trajectories (plan §4.1).

CPU-only. Reads the smoke trajectories produced by the dispatcher and writes
the Phase 0 calibration artifact.

Two modes (selected by ``--mode``):

* ``v1`` (default for backwards-compat) — reads the v1 rank-ladder smokes
  (``c504_smoke_r{4,8,16}_seed42``) and writes ``phase0_calibration.json``
  with the pinned (chosen_rank, chosen_alpha, chosen_checkpoint_fraction).

* ``v2`` (plan v2 §4.1, the lr-ladder redesign) — reads the v2 lr-ladder
  smokes (``c504v2_smoke_lr{1e5,3e5,1e4}_seed42``) and writes
  ``phase0_calibration_v2.json`` with the pinned
  (chosen_lr, chosen_checkpoint_fraction). chosen_rank is pinned at 8 and
  chosen_alpha at 32 in v2 — neither is swept. On any of the §4.1 fallback
  triggers (A: floor; B: saturated; C: empty in-band set), the artifact
  carries ``fallback_triggered=True`` + ``fallback_reason=...`` and the
  dispatcher reroutes to the §4.2 fallback (easier source) phase.

Usage:
    # v1 (rank ladder, default)
    uv run python scripts/i504_phase_phase0_pick.py \\
        --slab-root eval_results/issue_504 \\
        --out-path eval_results/issue_504/phase0_calibration.json

    # v2 (lr ladder)
    uv run python scripts/i504_phase_phase0_pick.py \\
        --mode v2 \\
        --slab-root eval_results/issue_504 \\
        --out-path eval_results/issue_504/phase0_calibration_v2.json
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

log = logging.getLogger("i504.phase_phase0_pick")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("v1", "v2"),
        default="v1",
        help=(
            "Phase 0 picker mode: v1=rank-ladder (legacy, default for backwards-"
            "compat); v2=lr-ladder (plan v2 §4.1 — the anchor-recipe redesign)."
        ),
    )
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_504"))
    ap.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help=(
            "Where to write the calibration artifact. Default depends on --mode: "
            "v1=phase0_calibration.json, v2=phase0_calibration_v2.json under the "
            "--slab-root."
        ),
    )
    ap.add_argument("--smoke-seed", type=int, default=42)
    ap.add_argument(
        "--source",
        default=None,
        help=(
            "Source persona name (recorded in the v2 artifact). Default = villain "
            "(matches the plan default in v2 §4.2). Pass --source <name> when the "
            "Phase 0 fallback fires on an easier persona."
        ),
    )
    ap.add_argument("--sentinel-path", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase0_pick] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    if args.mode == "v1":
        return _run_v1(args)
    return _run_v2(args)


def _run_v1(args: argparse.Namespace) -> int:
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        PHASE0_SMOKE_SLUGS,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase0 import (
        pick_anchor_from_smoke,
        write_phase0_artifact,
    )

    out_path = args.out_path or args.slab_root / "phase0_calibration.json"

    smoke_trajs: dict[str, dict] = {}
    for slug in PHASE0_SMOKE_SLUGS:
        p = args.slab_root / f"{slug}_seed{args.smoke_seed}" / "trajectory.json"
        if not p.exists():
            raise FileNotFoundError(
                f"smoke trajectory missing at {p} — Phase 0 smoke {slug} must complete first."
            )
        smoke_trajs[slug] = json.loads(p.read_text())
        log.info(
            "[load] %s trajectory: %d checkpoints", slug, len(smoke_trajs[slug]["checkpoints"])
        )

    pick = pick_anchor_from_smoke(smoke_trajs)
    write_phase0_artifact(pick, out_path)

    _maybe_write_sentinel(args, pick, out_path, "phase0_pick")

    log.info(
        "[phase=phase0_pick mode=v1] verdict=%s, chosen_rank=%s, chosen_alpha=%s, chosen_frac=%s",
        pick.get("verdict"),
        pick.get("chosen_rank"),
        pick.get("chosen_alpha"),
        pick.get("chosen_checkpoint_fraction"),
    )
    if pick.get("verdict") != "pass":
        log.error("[phase=phase0_pick mode=v1] FAIL — see smoke_table in %s", out_path)
        return 2
    return 0


def _run_v2(args: argparse.Namespace) -> int:
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        PHASE0_SMOKE_SLUGS_V2,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase0 import (
        pick_anchor_from_lr_smoke,
        write_phase0_v2_artifact,
    )

    out_path = args.out_path or args.slab_root / "phase0_calibration_v2.json"
    source = args.source or SOURCE_PERSONA

    smoke_trajs: dict[str, dict] = {}
    for slug in PHASE0_SMOKE_SLUGS_V2:
        p = args.slab_root / f"{slug}_seed{args.smoke_seed}" / "trajectory.json"
        if not p.exists():
            raise FileNotFoundError(
                f"smoke trajectory missing at {p} — Phase 0 v2 smoke {slug} must complete first."
            )
        smoke_trajs[slug] = json.loads(p.read_text())
        log.info(
            "[load] %s trajectory: %d checkpoints", slug, len(smoke_trajs[slug]["checkpoints"])
        )

    pick = pick_anchor_from_lr_smoke(smoke_trajs, source=source)
    write_phase0_v2_artifact(pick, out_path)

    _maybe_write_sentinel(args, pick, out_path, "phase0_pick_v2")

    log.info(
        "[phase=phase0_pick mode=v2] verdict=%s, chosen_lr=%s, chosen_frac=%s, "
        "fallback_triggered=%s",
        pick.get("verdict"),
        pick.get("chosen_lr"),
        pick.get("chosen_checkpoint_fraction"),
        pick.get("fallback_triggered"),
    )
    # NOTE: v2 verdict != "pass" is NOT a hard CLI failure — the dispatcher
    # interprets `fallback_triggered=True` and reroutes to §4.2 fallback.
    # We still return 2 on non-pass for parity with v1, so the dispatcher's
    # subprocess.run(check=True) raises and the fallback path is taken in a
    # caller-controlled try/except (see scripts/dispatch_neg_geometry_504.py).
    if pick.get("verdict") != "pass":
        log.error(
            "[phase=phase0_pick mode=v2] non-pass verdict=%s, fallback_reason=%s "
            "— see smoke_table in %s",
            pick.get("verdict"),
            pick.get("fallback_reason"),
            out_path,
        )
        return 2
    return 0


def _maybe_write_sentinel(
    args: argparse.Namespace,
    pick: dict,
    out_path: Path,
    phase_name: str,
) -> None:
    if args.sentinel_path is None:
        return
    args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    args.sentinel_path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:progress",
                "version": 1,
                "task_id": 504,
                "phase": phase_name,
                "by": "i504_phase_phase0_pick",
                "ts": datetime.now(UTC).isoformat(),
                "note": json.dumps(
                    {
                        "mode": args.mode,
                        "verdict": pick.get("verdict"),
                        "chosen_rank": pick.get("chosen_rank"),
                        "chosen_alpha": pick.get("chosen_alpha"),
                        "chosen_lr": pick.get("chosen_lr"),
                        "chosen_checkpoint_fraction": pick.get("chosen_checkpoint_fraction"),
                        "source_delta_g_at_pick_nats": pick.get("source_delta_g_at_pick_nats"),
                        "source_emission_at_pick": pick.get("source_emission_at_pick"),
                        "fallback_triggered": pick.get("fallback_triggered"),
                        "fallback_reason": pick.get("fallback_reason"),
                        "source": pick.get("source"),
                        "out_path": str(out_path),
                    }
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
