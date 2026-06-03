# ruff: noqa: RUF002
"""Phase A analyzer -- read BOTH cond1 trajectories, pick the anchor step.

Plan v3 §4.3.

Inputs:
  * WandB run history for the two Phase A runs
    (i471_route_a_cond1_withneg + i471_route_a_cond1_posonly).
  * Fallback: parse the per-step trajectory log lines from
    `logs/issue_471/train_<run_name>.log` when WandB is unavailable.

For each step, compute per-shape mean log P(`※`) trained-base:
  * source          (in_trained_shape)
  * default         (demo_free_default)
  * trained_neg     (trained_neg_medical_doctor)
  * bystander       (bystander_software_engineer)

Anchor rule (cond1_withneg, plan §4.3):
  Latest step s* with source >= +5 nats AND (source - default) >= +3 nats
  AND a saved checkpoint exists at or before s*.

Matched control checkpoint (cond1_posonly):
  Nearest cond1_posonly saved step to s*. If s* > cond1_posonly's final
  saved step, use its final step.

Outputs `eval_results/issue_471/route_a/phaseA_anchor.json` with:
  * `anchor_step` (int or null on lockstep)
  * `matched_posonly_step` (int)
  * per-run trajectory tables (source/default/trained_neg/bystander vs step)
  * per-run (source - default) gap per step (H_disentangle precursor)
  * threshold-sensitivity sweep at (+3,+5,+7) source x (+1,+3,+5) gap
  * lockstep flag + reason if no step satisfies the cond1_withneg rule

Usage:
    uv run python scripts/i471_phaseA_analyze.py \
        --withneg-run i471_route_a_cond1_withneg \
        --posonly-run i471_route_a_cond1_posonly
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("i471.phaseA_analyze")

OUT_DIR = Path("eval_results/issue_471/route_a")
OUT_FILE = OUT_DIR / "phaseA_anchor.json"
LOCKSTEP_FILE = OUT_DIR / "lockstep_finding.json"

# Anchor thresholds (plan §4.3).
SOURCE_NATS_FLOOR = 5.0
GAP_NATS_FLOOR = 3.0

# Threshold-sensitivity sweep grids.
SOURCE_THRESHOLDS = (3.0, 5.0, 7.0)
GAP_THRESHOLDS = (1.0, 3.0, 5.0)

# Default save_steps interval used by Phase A's dispatcher; the post-hoc
# anchor analyzer asserts a saved checkpoint at or before the chosen step
# by floor-dividing.
DEFAULT_SAVE_STEPS_INTERVAL = 10

# Trajectory log line pattern emitted by MarkerLogprobKLTrajectoryCallback
# (see src/explore_persona_space/train/i471_trajectory.py).
#   trajectory step=N cond=COND shape=SHAPE mean_logp=X.XXX emission=Y.YY
#   mean_kl=Z.ZZZ n=...
TRAJECTORY_LINE_RE = re.compile(
    r"trajectory step=(?P<step>\d+) cond=(?P<cond>\S+) shape=(?P<shape>\S+) "
    r"mean_logp=(?P<logp>-?\d+\.\d+) emission=(?P<emission>\d+\.\d+) "
    r"mean_kl=(?P<kl>-?\d+\.\d+)"
)


SHAPE_TO_BUCKET = {
    "in_trained_shape": "source",
    "demo_free_default": "default",
    "trained_neg_medical_doctor": "trained_neg",
    "bystander_software_engineer": "bystander",
}


def _parse_trajectory_log(log_path: Path) -> dict[int, dict[str, float]]:
    """Parse per-step per-shape mean log P(`※`) from a training log.

    Returns {step: {bucket: mean_logp}} where bucket in
    {source, default, trained_neg, bystander}. Missing shapes per step
    are simply absent from the inner dict; callers must check membership.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Trajectory log not found: {log_path}")
    out: dict[int, dict[str, float]] = {}
    with open(log_path) as f:
        for line in f:
            m = TRAJECTORY_LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group("step"))
            shape = m.group("shape")
            bucket = SHAPE_TO_BUCKET.get(shape)
            if bucket is None:
                # Unknown shape (logs from prior runs may have other names);
                # skip rather than fail so the analyzer is forward-compatible.
                continue
            logp = float(m.group("logp"))
            out.setdefault(step, {})[bucket] = logp
    if not out:
        raise RuntimeError(
            f"No trajectory lines parsed from {log_path}. Either the run "
            "crashed before any probe fired or the log path is wrong."
        )
    return out


def _base_logp_at_step_zero(traj: dict[int, dict[str, float]]) -> dict[str, float]:
    """Return per-bucket base log-prob from the step=0 probe (genuine pre-train).

    The trajectory callback (MarkerLogprobKLTrajectoryCallback) fires
    `on_train_begin` BEFORE any optimizer step and emits a step=0 trajectory
    line for every shape — that IS the true base log-prob (adapter weights
    are zero-deltas at training start, so the trained pass equals the base
    pass to machine precision).

    Plan v3 §4.3 anchor rule reads `(source − base) ≥ +5 nats` and
    `(source − default) ≥ +3 nats`, both as trained-minus-base deltas.
    Earlier versions of this analyzer fell back to `min(traj.keys())`
    (typically step=5 ≈ 80 training examples in) which biased base LOW
    and pushed the chosen anchor LATER than the thresholds intend. We now
    fail loud if step=0 is missing — the callback's `on_train_begin` hook
    SHOULD always emit it, so its absence means a regression worth blocking
    on rather than silently substituting a partially-trained proxy.
    """
    if 0 not in traj:
        raise RuntimeError(
            "phaseA_analyze: no step=0 trajectory row found. The trajectory "
            "callback (MarkerLogprobKLTrajectoryCallback) must emit a step=0 "
            "probe via on_train_begin BEFORE any optimizer step; that row IS "
            "the genuine base log-prob used as the analyzer's anchor floor. "
            "Re-train with the current i471_trajectory.py (which writes step=0) "
            "and re-run this analyzer; do NOT silently substitute step="
            f"{min(traj.keys())} as a 'base proxy' — it would push s* late by "
            "~1 save_steps interval."
        )
    return dict(traj[0])


def _build_step_table(traj: dict[int, dict[str, float]]) -> dict:
    """Build the per-step table + (source - default) gap series for one run."""
    steps_sorted = sorted(traj.keys())
    base = _base_logp_at_step_zero(traj)
    rows = []
    for s in steps_sorted:
        row = {"step": s}
        for bucket in ("source", "default", "trained_neg", "bystander"):
            absolute = traj[s].get(bucket)
            row[f"{bucket}_logp_trained"] = absolute
            base_for_bucket = base.get(bucket)
            if absolute is None or base_for_bucket is None:
                row[f"{bucket}_logp_delta"] = None
            else:
                row[f"{bucket}_logp_delta"] = absolute - base_for_bucket
        # H_disentangle precursor: (source - default) gap per step.
        src = row.get("source_logp_delta")
        defv = row.get("default_logp_delta")
        if src is not None and defv is not None:
            row["source_minus_default_gap"] = src - defv
        else:
            row["source_minus_default_gap"] = None
        rows.append(row)
    return {
        # step=0 row IS the genuine base anchor (trajectory callback emits it
        # via on_train_begin BEFORE any optimizer step). This field used to
        # name "earliest_step_used_as_base_proxy" — kept for downstream
        # compatibility but the value now always points at the step=0 row.
        "base_anchor_step": 0,
        "step_rows": rows,
    }


def _pick_anchor(
    table: dict,
    *,
    source_thr: float = SOURCE_NATS_FLOOR,
    gap_thr: float = GAP_NATS_FLOOR,
    save_steps_interval: int = DEFAULT_SAVE_STEPS_INTERVAL,
) -> int | None:
    """Latest step satisfying (source - base) >= source_thr AND gap >= gap_thr.

    Floors to the nearest saved checkpoint step (multiple of save_steps_interval).
    Returns None if no step satisfies (the lockstep finding).
    """
    candidates = []
    for row in table["step_rows"]:
        src = row.get("source_logp_delta")
        gap = row.get("source_minus_default_gap")
        if src is None or gap is None:
            continue
        if src >= source_thr and gap >= gap_thr:
            candidates.append(row["step"])
    if not candidates:
        return None
    s_star = max(candidates)
    # Floor to the saved checkpoint at-or-before s_star.
    if save_steps_interval > 0:
        s_star = (s_star // save_steps_interval) * save_steps_interval
        if s_star == 0:
            # If the first probe (typically step=save_steps_interval) already
            # satisfies, the floor would land at 0 (no checkpoint) — use the
            # first saved step instead.
            s_star = save_steps_interval
    return int(s_star)


def _pick_matched_posonly_step(
    posonly_table: dict,
    *,
    target_step: int,
    save_steps_interval: int = DEFAULT_SAVE_STEPS_INTERVAL,
) -> int:
    """Nearest cond1_posonly saved step to target_step (<= target_step if available).

    If target_step exceeds cond1_posonly's final saved step, return the final.
    """
    posonly_steps = [r["step"] for r in posonly_table["step_rows"]]
    if not posonly_steps:
        raise ValueError("cond1_posonly trajectory has zero steps; cannot match.")
    final = max(posonly_steps)
    if target_step >= final:
        # Floor `final` to the nearest checkpoint (typically already aligned).
        return int((final // save_steps_interval) * save_steps_interval or final)
    # Floor target_step to the nearest checkpoint at-or-below.
    floored = (target_step // save_steps_interval) * save_steps_interval
    return int(floored or save_steps_interval)


def _threshold_sensitivity_sweep(table: dict) -> list[dict]:
    """Anchor-pick under (source x gap) threshold grids — robustness check."""
    out = []
    for s_thr in SOURCE_THRESHOLDS:
        for g_thr in GAP_THRESHOLDS:
            anchor = _pick_anchor(table, source_thr=s_thr, gap_thr=g_thr)
            out.append(
                {
                    "source_threshold_nats": s_thr,
                    "gap_threshold_nats": g_thr,
                    "anchor_step": anchor,
                }
            )
    return out


def _final_saved_step(
    table: dict,
    *,
    save_steps_interval: int = DEFAULT_SAVE_STEPS_INTERVAL,
) -> int:
    """Largest saved-checkpoint step (multiple of save_steps_interval) <= max trajectory step.

    Trajectory rows are emitted at ``log_every`` (=5) intervals so the max
    trajectory step is typically a multiple of 5 — but checkpoints are saved
    only at ``save_steps`` (=10) intervals, so loading "the final checkpoint"
    requires flooring to a multiple of ``save_steps_interval``. The lockstep
    branch of the route-(a) launcher needs this exact value: it mirrors
    ``adapters/<run>/checkpoint-<step>/`` to a stepped adapter dir, and the
    checkpoint dir only exists at multiples of 10.

    Returns 0 if no usable step exists (e.g. training crashed before the first
    save). Callers must check for 0 + fail loud rather than silently mirroring
    a non-existent checkpoint.
    """
    steps = [r["step"] for r in table.get("step_rows", [])]
    if not steps:
        return 0
    max_step = max(steps)
    if save_steps_interval <= 0:
        return int(max_step)
    return int((max_step // save_steps_interval) * save_steps_interval)


def _analyze(
    *,
    withneg_log: Path,
    posonly_log: Path,
) -> dict:
    """Read both training logs, pick anchor, return the full result dict."""
    withneg_traj = _parse_trajectory_log(withneg_log)
    posonly_traj = _parse_trajectory_log(posonly_log)
    withneg_table = _build_step_table(withneg_traj)
    posonly_table = _build_step_table(posonly_traj)

    # Anchor rule applied to cond1_withneg (H_A1).
    withneg_anchor = _pick_anchor(withneg_table)
    # ALSO applied to cond1_posonly (H_A1').
    posonly_anchor_rule = _pick_anchor(posonly_table)

    if withneg_anchor is None:
        matched_posonly = _pick_matched_posonly_step(
            posonly_table,
            target_step=max(r["step"] for r in withneg_table["step_rows"]),
        )
    else:
        matched_posonly = _pick_matched_posonly_step(posonly_table, target_step=withneg_anchor)

    # Plan v3 §4.3 lockstep escape: under lockstep the launcher mirrors BOTH
    # cond1 variants' FINAL SAVED checkpoint (not their final trajectory step,
    # since trajectory rows fire at log_every=5 but checkpoints are saved
    # only at save_steps=10). Emit the floored final-saved step for both so
    # the shell layer can read it WITHOUT re-implementing the floor.
    withneg_final_saved_step = _final_saved_step(withneg_table)
    posonly_final_saved_step = _final_saved_step(posonly_table)

    return {
        "anchor_step": withneg_anchor,
        "matched_posonly_step": matched_posonly,
        "withneg_final_saved_step": withneg_final_saved_step,
        "posonly_final_saved_step": posonly_final_saved_step,
        "posonly_localizes_under_same_rule": posonly_anchor_rule,
        "lockstep_in_this_regime": withneg_anchor is None,
        "withneg_table": withneg_table,
        "posonly_table": posonly_table,
        "withneg_threshold_sensitivity": _threshold_sensitivity_sweep(withneg_table),
        "posonly_threshold_sensitivity": _threshold_sensitivity_sweep(posonly_table),
        "regime": {
            "lr": 5e-6,
            "epochs_ceiling": 2,
            "neg_set": ["default", "medical_doctor", "police_officer"],
            "ratio_pos_to_total_neg": "1:1",
            "marker_token": " ※",
            "marker_id": 83399,
            "collator_flag": "suppress_at_post_response_slot=True",
            "save_steps_interval": DEFAULT_SAVE_STEPS_INTERVAL,
            "anchor_source_floor_nats": SOURCE_NATS_FLOOR,
            "anchor_gap_floor_nats": GAP_NATS_FLOOR,
        },
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--withneg-run",
        default="i471_route_a_cond1_withneg",
        help="run_name of the cond1_withneg Phase A run (default: %(default)s).",
    )
    ap.add_argument(
        "--posonly-run",
        default="i471_route_a_cond1_posonly",
        help="run_name of the cond1_posonly Phase A run (default: %(default)s).",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/issue_471"),
        help="Directory holding train_<run_name>.log files (default: %(default)s).",
    )
    args = ap.parse_args(argv)

    withneg_log = args.log_dir / f"train_{args.withneg_run}.log"
    posonly_log = args.log_dir / f"train_{args.posonly_run}.log"

    result = _analyze(withneg_log=withneg_log, posonly_log=posonly_log)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s (anchor_step=%s)", OUT_FILE, result["anchor_step"])

    if result["lockstep_in_this_regime"]:
        LOCKSTEP_FILE.write_text(
            json.dumps(
                {
                    "lockstep_in_this_regime": True,
                    "reason": (
                        "No cond1_withneg step satisfied source >= +5 nats AND "
                        "(source - default) >= +3 nats. Phase B does NOT fire. "
                        "Regime-scoped finding per plan v3 §6.2.13: lr=5e-6 x "
                        "<=2ep x 3 close negatives at 1:1 x single-token  marker "
                        "x slot-aligned EOS-only loss x this Q x seed=42."
                    ),
                    "phaseA_anchor_summary": result,
                },
                indent=2,
            )
        )
        logger.info("Wrote lockstep finding to %s", LOCKSTEP_FILE)
        logger.info("Phase B should NOT fire. Phase 4 will still run on both cond1 variants.")
    else:
        logger.info(
            "Phase B can fire with --anchor-step %d. Matched cond1_posonly step: %d.",
            result["anchor_step"],
            result["matched_posonly_step"],
        )


if __name__ == "__main__":
    main()
