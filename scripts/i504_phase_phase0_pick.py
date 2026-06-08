# em-dash + Qwen marker " ※" intentional
#!/usr/bin/env python3
"""Task #504 Phase 0 — anchor pick from smoke trajectories (plan §4.1).

CPU-only. Reads the smoke trajectories produced by the dispatcher and writes
the Phase 0 calibration artifact.

Three modes (selected by ``--mode``):

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

* ``v3`` (plan v3 §4.1, the EPOCHS-ladder redesign) — reads the v3
  EPOCHS-ladder smokes (``c504v3_smoke_eps{2,3}_seed42``) and writes
  ``phase0_calibration_v3.json`` with the pinned
  (chosen_epochs, chosen_checkpoint_fraction). chosen_lr is FIXED at 1e-4
  (v2 evidence). chosen_rank=8 / chosen_alpha=32 pinned. On Trigger A or C
  the picker ALSO writes ``phase0_v3_exit_to_v4.json`` (exit-to-v4 signal).
  On Trigger B the picker emits a recovery signal and exits non-zero so the
  dispatcher can launch the in-plan finer-fraction recovery on EPOCHS=2.

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

    # v3 (EPOCHS ladder at fixed lr=1e-4)
    uv run python scripts/i504_phase_phase0_pick.py \\
        --mode v3 \\
        --slab-root eval_results/issue_504 \\
        --out-path eval_results/issue_504/phase0_calibration_v3.json
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
        choices=("v1", "v2", "v3", "v4"),
        default="v1",
        help=(
            "Phase 0 picker mode: v1=rank-ladder (legacy, default for backwards-"
            "compat); v2=lr-ladder (plan v2 §4.1); v3=EPOCHS-ladder at fixed lr=1e-4 "
            "(plan v3 §4.1 — the EPOCHS-anchor redesign after v2 lr-ladder refutation); "
            "v4=bystander-resolution gate at the pinned EPOCHS=3 anchor (plan v5 §4.1 "
            "fix #2 — drops the source-emission gate that contaminated v3's pick)."
        ),
    )
    ap.add_argument(
        "--v4-trajectory-path",
        type=Path,
        default=None,
        help=(
            "v4 only: path to the EPOCHS=3 anchor re-eval trajectory JSON "
            "produced by `i504_eval_trajectory.py` (the Phase 0 v4 §4.1 step "
            "input). Default: <slab_root>/c504v4_smoke_eps3_reread_seed42/"
            "trajectory.json."
        ),
    )
    ap.add_argument(
        "--fixed-lr",
        type=float,
        default=None,
        help=(
            "v3 only: fixed lr value (default FIXED_LR_V3 = 1e-4). Recorded "
            "in the artifact as `fixed_lr` and `chosen_lr` (NOT swept in v3)."
        ),
    )
    ap.add_argument(
        "--chosen-epochs",
        type=int,
        default=3,
        help=(
            "v4 only: EPOCHS at which the re-eval trajectory was produced. "
            "Default 3 (the v4 primary anchor). Pass --chosen-epochs 2 from "
            "the dispatcher's bisection path (§4.2 Step 1) — the bisection "
            "re-trains EPOCHS=2 on the finer-fraction grid and the picker "
            "must record `chosen_epochs=2` in the bisection artifact so the "
            "downstream Phase 1 scheduler trains the main arms at EPOCHS=2 "
            "(matching the recipe whose `chosen_frac` was selected)."
        ),
    )
    ap.add_argument(
        "--exit-to-v4-path",
        type=Path,
        default=None,
        help=(
            "v3 only: where to write `phase0_v3_exit_to_v4.json` when Trigger "
            "A or C fires. Defaults to <slab_root>/phase0_v3_exit_to_v4.json."
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
    ap.add_argument(
        "--include-finer-recovery",
        action="store_true",
        help=(
            "v3 only (plan v3 §4.1 trigger B + §4.2): MERGE the finer-grid "
            "recovery trajectory into the coarse EPOCHS=2 trajectory and "
            "re-apply the pick rule over the augmented (epochs, frac) table. "
            "Used by the dispatcher after the recovery cell completes. Reads "
            "the recovery trajectory from "
            "`<slab_root>/c504v3_smoke_eps2_seed42<recovery-traj-suffix>/"
            "trajectory.json`. When unset, runs the pick rule over the "
            "coarse-only trajectories (byte-identical pre-recovery behavior)."
        ),
    )
    ap.add_argument(
        "--recovery-traj-suffix",
        default="__recovery_finer",
        help=(
            "v3 only: subdir suffix used by the recovery cell when writing "
            "its finer-grid trajectory under --slab-root. Only consulted "
            "when --include-finer-recovery is set. Must MATCH the "
            "dispatcher's `trajectory_suffix` value for the recovery cell."
        ),
    )
    ap.add_argument(
        "--recovery-cell-slug",
        default="c504v3_smoke_eps2",
        help=(
            "v3 only: the smoke slug whose trajectory the recovery augments "
            "(canonical EPOCHS=2; the lower-epochs cell, since Trigger B "
            "fires when both ladder rungs saturate and EPOCHS=2 is the "
            "cheaper rung to retrain). Only consulted when "
            "--include-finer-recovery is set."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s [phase=phase0_pick] %(name)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    if args.mode == "v1":
        return _run_v1(args)
    if args.mode == "v2":
        return _run_v2(args)
    if args.mode == "v3":
        return _run_v3(args)
    return _run_v4(args)


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
                        "chosen_epochs": pick.get("chosen_epochs"),
                        "chosen_checkpoint_fraction": pick.get("chosen_checkpoint_fraction"),
                        "chosen_checkpoint_steps": pick.get("chosen_checkpoint_steps"),
                        "source_delta_g_at_pick_nats": pick.get("source_delta_g_at_pick_nats"),
                        "source_emission_at_pick": pick.get("source_emission_at_pick"),
                        "fallback_triggered": pick.get("fallback_triggered"),
                        "fallback_reason": pick.get("fallback_reason"),
                        "in_plan_recovery_triggered": pick.get("in_plan_recovery_triggered"),
                        "source": pick.get("source"),
                        "out_path": str(out_path),
                    }
                ),
            },
            indent=2,
        )
    )


def _run_v3(args: argparse.Namespace) -> int:
    """v3 EPOCHS-ladder picker (plan v3 §4.1).

    Reads ``c504v3_smoke_eps{2,3}_seed42/trajectory.json``, applies the v3
    pick rule + Trigger A/B/C fallback logic, writes
    ``phase0_calibration_v3.json``. On Trigger A or C ALSO writes
    ``phase0_v3_exit_to_v4.json`` (the explicit exit-to-v4 signal). Returns:

      * rc=0 when verdict=="pass" (in-band anchor found).
      * rc=2 when verdict in {"no_in_band_anchor", "all_saturated"} —
        Trigger A/B/C fired. The dispatcher catches this via
        subprocess.run(check=True) and decides whether to launch the
        in-plan finer-fraction recovery (Trigger B) or emit
        `epm:failure v1 failure_class=methodology` (Trigger A or C).
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        FIXED_LR_V3,
        PHASE0_SMOKE_SLUGS_V3,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase0 import (
        merge_recovery_into_v3_pick,
        pick_anchor_from_epochs_smoke,
        write_phase0_v3_artifact,
        write_phase0_v3_exit_to_v4_artifact,
    )

    out_path = args.out_path or args.slab_root / "phase0_calibration_v3.json"
    exit_to_v4_path = args.exit_to_v4_path or args.slab_root / "phase0_v3_exit_to_v4.json"
    source = args.source or SOURCE_PERSONA
    fixed_lr = args.fixed_lr if args.fixed_lr is not None else FIXED_LR_V3

    smoke_trajs: dict[str, dict] = {}
    for slug in PHASE0_SMOKE_SLUGS_V3:
        p = args.slab_root / f"{slug}_seed{args.smoke_seed}" / "trajectory.json"
        if not p.exists():
            raise FileNotFoundError(
                f"smoke trajectory missing at {p} — Phase 0 v3 smoke {slug} must complete first."
            )
        smoke_trajs[slug] = json.loads(p.read_text())
        log.info(
            "[load] %s trajectory: %d checkpoints", slug, len(smoke_trajs[slug]["checkpoints"])
        )

    # v3 in-plan recovery (plan §4.1 trigger B + §4.2): when the dispatcher
    # invokes the picker with --include-finer-recovery, the recovery cell has
    # ALREADY completed and written its finer trajectory at a suffix-decorated
    # subdir. Load + merge into the EPOCHS=2 cell's checkpoint list, then
    # re-apply the pick rule.
    if args.include_finer_recovery:
        recovery_path = (
            args.slab_root
            / f"{args.recovery_cell_slug}_seed{args.smoke_seed}{args.recovery_traj_suffix}"
            / "trajectory.json"
        )
        if not recovery_path.exists():
            raise FileNotFoundError(
                f"--include-finer-recovery set but recovery trajectory missing "
                f"at {recovery_path}. The dispatcher must have written it via "
                f"`--phase phase0_v3-recovery` BEFORE invoking this picker with "
                f"--include-finer-recovery."
            )
        recovery_traj = json.loads(recovery_path.read_text())
        log.info(
            "[load] recovery trajectory: %d checkpoints (from %s); merging into %r",
            len(recovery_traj["checkpoints"]),
            recovery_path,
            args.recovery_cell_slug,
        )
        pick = merge_recovery_into_v3_pick(
            smoke_trajs,
            recovery_traj,
            source=source,
            fixed_lr=fixed_lr,
            recovery_slug=args.recovery_cell_slug,
        )
        log.info(
            "[phase=phase0_pick mode=v3 merged] verdict=%s, chosen_epochs=%s, "
            "chosen_frac=%s, in_plan_recovery=%s",
            pick.get("verdict"),
            pick.get("chosen_epochs"),
            pick.get("chosen_checkpoint_fraction"),
            pick.get("in_plan_recovery_triggered"),
        )
    else:
        pick = pick_anchor_from_epochs_smoke(smoke_trajs, source=source, fixed_lr=fixed_lr)
    write_phase0_v3_artifact(pick, out_path)

    # Trigger A or C fired → ALSO write the exit-to-v4 artifact. Trigger B
    # is the in-plan recovery path (the dispatcher launches finer fractions
    # on EPOCHS=2 — NOT an exit-to-v4 yet).
    if pick.get("fallback_triggered") and not pick.get("in_plan_recovery_triggered"):
        write_phase0_v3_exit_to_v4_artifact(pick, exit_to_v4_path)

    _maybe_write_sentinel(args, pick, out_path, "phase0_pick_v3")

    log.info(
        "[phase=phase0_pick mode=v3] verdict=%s, chosen_epochs=%s, chosen_lr=%s, "
        "chosen_frac=%s, fallback_triggered=%s, in_plan_recovery=%s",
        pick.get("verdict"),
        pick.get("chosen_epochs"),
        pick.get("chosen_lr"),
        pick.get("chosen_checkpoint_fraction"),
        pick.get("fallback_triggered"),
        pick.get("in_plan_recovery_triggered"),
    )
    # Non-pass verdict (Trigger A/B/C) returns rc=2 for parity with v1/v2 so
    # the dispatcher's subprocess.run(check=True) raises and the caller's
    # try/except routes to exit-to-v4 (A/C) or in-plan recovery (B).
    if pick.get("verdict") != "pass":
        log.error(
            "[phase=phase0_pick mode=v3] non-pass verdict=%s, fallback_reason=%s "
            "— see smoke_table in %s",
            pick.get("verdict"),
            pick.get("fallback_reason"),
            out_path,
        )
        return 2
    return 0


def _run_v4(args: argparse.Namespace) -> int:
    """v4 bystander-resolution picker (plan v5 §4.1 fix #2).

    Reads ONE trajectory (the EPOCHS=3 anchor re-eval) and applies the v4
    bystander-resolution gate (≥ 20% of held-out probes in the open interval
    (+0.5 nats floor, log(0.9) ≈ -0.105 ceiling)). Drops the v3 source-
    emission gate that contaminated the v3 pick. Writes
    ``phase0_calibration_v4.json``. Returns:

      * rc=0 when verdict=="pass" (in-band anchor found).
      * rc=2 when verdict=="no_in_band_anchor" — bystander layer is saturated
        at every fraction; dispatcher invokes the EPOCHS=2 bisection
        (§4.2 Step 1) before declaring a hard exit-to-v5 (rank bump).
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        FIXED_LR_V3,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase0 import (
        pick_anchor_v4_bystander_resolution,
        write_phase0_v4_artifact,
    )

    out_path = args.out_path or args.slab_root / "phase0_calibration_v4.json"
    source = args.source or SOURCE_PERSONA
    fixed_lr = args.fixed_lr if args.fixed_lr is not None else FIXED_LR_V3
    traj_path = args.v4_trajectory_path or (
        args.slab_root / "c504v4_smoke_eps3_reread_seed42" / "trajectory.json"
    )
    if not traj_path.exists():
        raise FileNotFoundError(
            f"v4 trajectory missing at {traj_path} — Phase 0 v4 §4.1 must "
            f"re-evaluate the EPOCHS=3 anchor through the fixed reader before "
            f"this picker runs. Run `--phase phase0_v4_reeval` first."
        )

    trajectory = json.loads(traj_path.read_text())
    log.info(
        "[load] v4 trajectory: %d checkpoints (from %s)",
        len(trajectory.get("checkpoints", [])),
        traj_path,
    )

    chosen_epochs = int(args.chosen_epochs)
    if chosen_epochs not in (2, 3):
        raise ValueError(
            f"--chosen-epochs must be 2 (bisection §4.2 Step 1) or 3 (v4 "
            f"primary anchor); got {chosen_epochs}. Other values are not "
            f"supported in the v4 picker."
        )
    pick = pick_anchor_v4_bystander_resolution(
        trajectory,
        source=source,
        fixed_lr=fixed_lr,
        chosen_epochs=chosen_epochs,
    )
    write_phase0_v4_artifact(pick, out_path)

    _maybe_write_sentinel(args, pick, out_path, "phase0_pick_v4")

    log.info(
        "[phase=phase0_pick mode=v4] verdict=%s, chosen_epochs=%s, chosen_lr=%s, "
        "chosen_frac=%s, chosen_checkpoint_steps=%s, "
        "bystander_resolution_at_pick=%s, fallback=%s",
        pick.get("verdict"),
        pick.get("chosen_epochs"),
        pick.get("chosen_lr"),
        pick.get("chosen_checkpoint_fraction"),
        pick.get("chosen_checkpoint_steps"),
        pick.get("bystander_resolution_at_pick"),
        pick.get("fallback_triggered"),
    )
    if pick.get("verdict") != "pass":
        log.error(
            "[phase=phase0_pick mode=v4] non-pass verdict=%s, fallback_reason=%s "
            "— see smoke_table in %s",
            pick.get("verdict"),
            pick.get("fallback_reason"),
            out_path,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
