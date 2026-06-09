# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG + × intentional
"""Task #504 Phase 0 — anchor calibration pick rule (plan §4.1).

Given 3 smoke trajectories at LoRA r ∈ {4, 8, 16} (each at lr=2e-6, count=2,
1 epoch ≈ 25 steps with batch 16, seed 42), pick the (rank, fraction) where
source ΔG lands in [5, 12] nats AND on-policy emission ∈ [0.1, 0.8]. The
chosen `chosen_checkpoint_fraction` is applied UNIFORMLY across every Phase 1
arm for the headline read.

Plan §4.1 pick rule:
  1. For each rank, find the LATEST checkpoint fraction where source ΔG and
     source emission BOTH fall in band.
  2. Across the 3 ranks, pick the rank where the in-band checkpoint exists
     AND sits at the midpoint of the trajectory (so we have band-width on
     both sides). Tie-break to lower rank.
  3. If NO rank lands in-band at ANY checkpoint: fall back to 2 epochs at r=4
     or r=8.
  4. If EVERY rank saturates at every checkpoint: abort + re-plan.

CPU-only (consumes the smoke trajectories produced by the dispatcher).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    CHECKPOINT_FRACTIONS,
    EMISSION_BAND_HIGH,
    EMISSION_BAND_LOW,
    EPOCHS_FROM_V3_SMOKE_SLUG,
    EPOCHS_LADDER_V3,
    FIXED_LR_V3,
    LR_FROM_V2_SMOKE_SLUG,
    PHASE0_CALIB_RANKS,
    PHASE0_SMOKE_SLUGS,
    PHASE0_SMOKE_SLUGS_V2,
    PHASE0_SMOKE_SLUGS_V3,
    SOURCE_DG_BAND_HIGH,
    SOURCE_DG_BAND_LOW,
    SOURCE_PERSONA,
    alpha_for_rank,
)

# Force-reference the v2/v3-only imports so ruff's `F401` auto-strip does not
# remove them under the formatter's pre-commit pass. All are used INSIDE the
# pick functions below; the references here ensure the imports survive the
# auto-fixer per `feedback_ruff_strips_unused_imports`.
_V2_IMPORT_REFS = (LR_FROM_V2_SMOKE_SLUG, PHASE0_SMOKE_SLUGS_V2)
_V3_IMPORT_REFS = (
    EPOCHS_FROM_V3_SMOKE_SLUG,
    EPOCHS_LADDER_V3,
    FIXED_LR_V3,
    PHASE0_SMOKE_SLUGS_V3,
)

log = logging.getLogger("issue_504.phase0")


def _rank_from_smoke_slug(slug: str) -> int:
    """Extract the rank int from a smoke slug like 'c504_smoke_r8' → 8."""
    if not slug.startswith("c504_smoke_r"):
        raise ValueError(f"Not a #504 smoke slug: {slug!r}")
    return int(slug.removeprefix("c504_smoke_r"))


def _per_frac_source_diagnostics(
    trajectory: dict, source: str = SOURCE_PERSONA
) -> dict[float, dict[str, float]]:
    """Extract per-fraction (source_dg, source_emission) from a trajectory JSON.

    Trajectory shape (eval_trajectory.run_trajectory_eval output):
      {"cell": ..., "seed": ..., "source": ...,
       "checkpoints": [
         {"frac": 0.08, "step": 12,
          "source_self": {"g_logp": .., "b_logp": .., "delta_g": ..,
                          "emission_rate": .., ...},
          "held_out": {...}},
         ...]}

    Returns: {frac: {"source_dg": nats, "source_emission": rate}}.
    """
    out: dict[float, dict[str, float]] = {}
    for ck in trajectory["checkpoints"]:
        frac = float(ck["frac"])
        src = ck.get("source_self", {})
        # eval_trajectory stores either delta_g_mean (mean over Q_eval) or
        # delta_g (per-q dict); prefer the mean if present.
        dg = src.get("delta_g_mean")
        if dg is None and "delta_g" in src:
            v = src["delta_g"]
            if isinstance(v, dict):
                vals = [float(x) for x in v.values() if x is not None]
                dg = sum(vals) / len(vals) if vals else float("nan")
            else:
                dg = float(v)
        if dg is None:
            dg = float("nan")
        # #472's eval_trajectory writes 'emission_p' (#477 v6 DV-C field name).
        # The older 'emission_rate' name is read as a backward-compat alias.
        emit = src.get("emission_p")
        if emit is None:
            emit = src.get("emission_rate")
        if emit is None and "argmax_marker" in src:
            v = src["argmax_marker"]
            if isinstance(v, dict):
                emit = sum(1 for x in v.values() if x) / max(len(v), 1)
            else:
                emit = float(v)
        if emit is None:
            emit = float("nan")
        out[frac] = {"source_dg": float(dg), "source_emission": float(emit)}
    return out


def _latest_in_band_frac(
    per_frac: dict[float, dict[str, float]],
    *,
    dg_low: float,
    dg_high: float,
    emit_low: float,
    emit_high: float,
) -> float | None:
    """Return the LATEST fraction whose source ΔG + emission BOTH land in band.

    None when no fraction in `per_frac` lands in both bands (the smoke cell
    is either fully sub-band — too sub-ceiling — or fully super-band —
    saturated — at every checkpoint).
    """
    candidates = [
        frac
        for frac, v in per_frac.items()
        if dg_low <= v["source_dg"] <= dg_high and emit_low <= v["source_emission"] <= emit_high
    ]
    if not candidates:
        return None
    return max(candidates)


def _frac_midpoint_distance(frac: float, all_fracs: tuple[float, ...]) -> float:
    """How close `frac` is to the trajectory midpoint (closer = better).

    Distance is |index(frac) - midpoint_index|; lower is better. Used as the
    rank tie-breaker after the "latest in-band" pick per rank.
    """
    sorted_fracs = sorted(all_fracs)
    midpoint_idx = (len(sorted_fracs) - 1) / 2.0
    if frac not in sorted_fracs:
        return float("inf")
    return abs(sorted_fracs.index(frac) - midpoint_idx)


def pick_anchor_from_smoke(
    smoke_trajectories: dict[str, dict],
    *,
    dg_band: tuple[float, float] = (SOURCE_DG_BAND_LOW, SOURCE_DG_BAND_HIGH),
    emit_band: tuple[float, float] = (EMISSION_BAND_LOW, EMISSION_BAND_HIGH),
    checkpoint_fractions: tuple[float, ...] = CHECKPOINT_FRACTIONS,
) -> dict[str, Any]:
    """Pick the anchor (chosen_rank, chosen_checkpoint_fraction) per plan §4.1.

    Args:
        smoke_trajectories: {smoke_slug: trajectory_dict} for the 3 smoke
            cells.  smoke_slug must be in PHASE0_SMOKE_SLUGS.
        dg_band: (low, high) for source ΔG in nats (default [5, 12]).
        emit_band: (low, high) for source on-policy emission (default [0.1, 0.8]).
        checkpoint_fractions: the full trajectory cadence (used for midpoint
            tie-break).

    Returns:
        {
          "chosen_rank": int | None,
          "chosen_alpha": int | None,
          "chosen_checkpoint_fraction": float | None,
          "source_delta_g_at_pick_nats": float | None,
          "source_emission_at_pick": float | None,
          "verdict": "pass" | "no_in_band_anchor" | "all_saturated",
          "smoke_table": [
            {"slug": ..., "rank": ..., "alpha": ...,
             "per_frac": {frac: {source_dg, source_emission, in_band}}},
            ...],
        }

    On `verdict != "pass"` the caller MUST NOT advance to Phase 1; the
    dispatcher fails loud with the verdict message.
    """
    dg_low, dg_high = dg_band
    emit_low, emit_high = emit_band

    smoke_table: list[dict] = []
    candidates: list[tuple[int, float]] = []  # (rank, latest_in_band_frac)
    all_saturated = True
    for slug in PHASE0_SMOKE_SLUGS:
        if slug not in smoke_trajectories:
            raise KeyError(
                f"pick_anchor_from_smoke: smoke trajectory missing for {slug!r}; "
                f"got slugs: {sorted(smoke_trajectories)}"
            )
        rank = _rank_from_smoke_slug(slug)
        alpha = alpha_for_rank(rank) if rank in PHASE0_CALIB_RANKS and rank != 16 else 32
        per_frac = _per_frac_source_diagnostics(smoke_trajectories[slug])
        # Tag in-band status for the table.
        per_frac_tagged = {
            frac: {
                **v,
                "in_band": (
                    dg_low <= v["source_dg"] <= dg_high
                    and emit_low <= v["source_emission"] <= emit_high
                ),
            }
            for frac, v in per_frac.items()
        }
        latest_in_band = _latest_in_band_frac(
            per_frac,
            dg_low=dg_low,
            dg_high=dg_high,
            emit_low=emit_low,
            emit_high=emit_high,
        )
        smoke_table.append(
            {
                "slug": slug,
                "rank": rank,
                "alpha": alpha,
                "per_frac": per_frac_tagged,
                "latest_in_band_frac": latest_in_band,
            }
        )
        # "Saturated" = every fraction sits ABOVE the upper ΔG band.
        all_above = all(v["source_dg"] > dg_high for v in per_frac.values())
        if not all_above:
            all_saturated = False
        if latest_in_band is not None:
            candidates.append((rank, latest_in_band))

    if not candidates:
        verdict = "all_saturated" if all_saturated else "no_in_band_anchor"
        return {
            "chosen_rank": None,
            "chosen_alpha": None,
            "chosen_checkpoint_fraction": None,
            "source_delta_g_at_pick_nats": None,
            "source_emission_at_pick": None,
            "verdict": verdict,
            "smoke_table": smoke_table,
        }

    # Pick: the rank whose latest-in-band fraction is CLOSEST to the trajectory
    # midpoint. Tie-break to lower rank (plan §4.1 step 2).
    candidates.sort(
        key=lambda rk_frac: (
            _frac_midpoint_distance(rk_frac[1], checkpoint_fractions),
            rk_frac[0],
        )
    )
    chosen_rank, chosen_frac = candidates[0]
    chosen_alpha = alpha_for_rank(chosen_rank) if chosen_rank != 16 else 32
    # Look up the diagnostics at the chosen point.
    row = next(r for r in smoke_table if r["rank"] == chosen_rank)
    per = row["per_frac"][chosen_frac]
    return {
        "chosen_rank": int(chosen_rank),
        "chosen_alpha": int(chosen_alpha),
        "chosen_checkpoint_fraction": float(chosen_frac),
        "source_delta_g_at_pick_nats": float(per["source_dg"]),
        "source_emission_at_pick": float(per["source_emission"]),
        "verdict": "pass",
        "smoke_table": smoke_table,
    }


def write_phase0_artifact(pick: dict[str, Any], out_path: Path) -> Path:
    """Write phase0_calibration.json (plan §4.1 output format)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pick, indent=2))
    log.info("[phase0] wrote %s (verdict=%s)", out_path, pick.get("verdict"))
    return out_path


def load_phase0_pick(path: Path) -> dict[str, Any]:
    """Load a phase0_calibration.json; raise on missing file (fail-loud)."""
    if not path.exists():
        raise FileNotFoundError(
            f"phase0_calibration.json missing at {path} — Phase 0 must complete "
            f"BEFORE Phase 1 can spawn. Re-run with `--smoke` and inspect the "
            f"smoke trajectories before advancing."
        )
    pick = json.loads(path.read_text())
    if pick.get("verdict") != "pass":
        raise RuntimeError(
            f"phase0_calibration.json verdict={pick.get('verdict')!r}; Phase 1 "
            f"cannot proceed without a passing anchor. See smoke_table in {path}."
        )
    return pick


# ── v2 lr-ladder picker (plan v2 §4.1) ──────────────────────────────────────


def _lr_from_v2_smoke_slug(slug: str) -> float:
    """Return the lr value associated with a v2 smoke slug.

    Raises:
        KeyError: `slug` is not a v2 smoke slug.
    """
    if slug not in LR_FROM_V2_SMOKE_SLUG:
        raise KeyError(
            f"Not a v2 smoke slug: {slug!r}; expected one of {sorted(LR_FROM_V2_SMOKE_SLUG)}"
        )
    return LR_FROM_V2_SMOKE_SLUG[slug]


def pick_anchor_from_lr_smoke(
    smoke_trajectories: dict[str, dict],
    *,
    dg_band: tuple[float, float] = (SOURCE_DG_BAND_LOW, SOURCE_DG_BAND_HIGH),
    emit_band: tuple[float, float] = (EMISSION_BAND_LOW, EMISSION_BAND_HIGH),
    checkpoint_fractions: tuple[float, ...] = CHECKPOINT_FRACTIONS,
    source: str = SOURCE_PERSONA,
) -> dict[str, Any]:
    """Pick the anchor (chosen_lr, chosen_checkpoint_fraction) per plan v2 §4.1.

    Walks the 3 v2 smoke trajectories (one per lr in {1e-5, 3e-5, 1e-4}), tags
    each (lr, frac) pair as in-band when source ΔG ∈ [5,12] nats AND on-policy
    emission ∈ [0.1, 0.8], and applies the plan v2 §4.1 pick rule:

      1. Latest in-band fraction (per #479: ΔG plateaus by frac=0.08 in some
         cells; the latest in-band fraction gives the most stable read).
      2. Tie-break: source ΔG closest to 8.0 nats (band midpoint).
      3. Tie-break: lower lr (smaller hyperparameter footprint).

    Fallback triggers (plan v2 §4.1 step 5):

      - Trigger A (floor): max source ΔG across all 18 (lr, frac) pairs < 5 nats.
      - Trigger B (saturated): min source ΔG > 12 nats.
      - Trigger C (empty band): in_band set is empty (even with max ≥ 5 and
        min ≤ 12 individually — a non-overlapping floor + saturation regime).

    On any fallback trigger, `verdict` is set to "no_in_band_anchor",
    `fallback_triggered` is True, and `fallback_reason` names the trigger so
    the dispatcher can fire the Phase 0 fallback (§4.2 easier source).

    Args:
        smoke_trajectories: {smoke_slug: trajectory_dict} keyed by v2 smoke
            slug (`c504v2_smoke_lr{1e5,3e5,1e4}`).
        dg_band: (low, high) for source ΔG in nats (default [5, 12]).
        emit_band: (low, high) for source on-policy emission (default [0.1, 0.8]).
        checkpoint_fractions: full trajectory cadence (used for midpoint tie-break).
        source: source persona name (recorded in the artifact; default villain).

    Returns:
        {
          "version": 2,
          "lr_ladder": [1e-5, 3e-5, 1e-4],
          "chosen_lr": float | None,
          "chosen_rank": 8,            # pinned in v2, NOT swept
          "chosen_alpha": 32,          # pinned in v2
          "chosen_checkpoint_fraction": float | None,
          "chosen_checkpoint_steps": int | None,  # ≈ round(frac × 25)
          "source": str,
          "source_delta_g_at_pick_nats": float | None,
          "source_emission_at_pick": float | None,
          "fallback_triggered": bool,
          "fallback_reason": str | None,
          "verdict": "pass" | "no_in_band_anchor" | "all_saturated",
          "smoke_table": [
            {"slug": ..., "lr": ..., "per_frac": {frac: {source_dg, source_emission, in_band}},
             "latest_in_band_frac": ...},
            ...],
        }

    On `verdict != "pass"` the caller MUST NOT proceed to Phase 1 with this
    source; the dispatcher reroutes to Phase 0 fallback (§4.2) on a fallback
    trigger, or aborts on a clean "all_saturated" with no in-band cell.
    """
    dg_low, dg_high = dg_band
    emit_low, emit_high = emit_band

    smoke_table: list[dict] = []
    candidates: list[tuple[float, float, float]] = []  # (lr, latest_in_band_frac, source_dg)
    all_pairs_dg: list[float] = []
    for slug in PHASE0_SMOKE_SLUGS_V2:
        if slug not in smoke_trajectories:
            raise KeyError(
                f"pick_anchor_from_lr_smoke: smoke trajectory missing for {slug!r}; "
                f"got slugs: {sorted(smoke_trajectories)}"
            )
        lr = _lr_from_v2_smoke_slug(slug)
        per_frac = _per_frac_source_diagnostics(smoke_trajectories[slug], source=source)
        per_frac_tagged = {
            frac: {
                **v,
                "in_band": (
                    dg_low <= v["source_dg"] <= dg_high
                    and emit_low <= v["source_emission"] <= emit_high
                ),
            }
            for frac, v in per_frac.items()
        }
        latest_in_band = _latest_in_band_frac(
            per_frac,
            dg_low=dg_low,
            dg_high=dg_high,
            emit_low=emit_low,
            emit_high=emit_high,
        )
        smoke_table.append(
            {
                "slug": slug,
                "lr": lr,
                "per_frac": per_frac_tagged,
                "latest_in_band_frac": latest_in_band,
            }
        )
        for v in per_frac.values():
            all_pairs_dg.append(v["source_dg"])
        if latest_in_band is not None:
            candidates.append(
                (
                    lr,
                    latest_in_band,
                    per_frac[latest_in_band]["source_dg"],
                )
            )

    # ── Fallback trigger detection (plan v2 §4.1 step 5). ───────────────────
    fallback_triggered = False
    fallback_reason: str | None = None
    if all_pairs_dg:
        max_dg = max(all_pairs_dg)
        min_dg = min(all_pairs_dg)
    else:
        max_dg = float("nan")
        min_dg = float("nan")

    if not candidates:
        # Some kind of fallback. Distinguish the trigger.
        if all_pairs_dg and max_dg < dg_low:
            fallback_triggered = True
            fallback_reason = (
                f"trigger_A_floor: max(source_dg) over the {len(all_pairs_dg)} "
                f"(lr, frac) pairs = {max_dg:.3f} nats < {dg_low} nats — marker "
                f"won't install at any lr in the ladder for {source!r}."
            )
            verdict = "no_in_band_anchor"
        elif all_pairs_dg and min_dg > dg_high:
            fallback_triggered = True
            fallback_reason = (
                f"trigger_B_saturated: min(source_dg) over the {len(all_pairs_dg)} "
                f"(lr, frac) pairs = {min_dg:.3f} nats > {dg_high} nats — every lr "
                f"saturates instantly; no dynamic range for the placement sweep."
            )
            verdict = "all_saturated"
        else:
            fallback_triggered = True
            fallback_reason = (
                f"trigger_C_empty_band: in_band set is empty (max_dg={max_dg:.3f}, "
                f"min_dg={min_dg:.3f}; some pairs above {dg_high} nats, some below "
                f"{dg_low} nats, but no (lr, frac) lands BOTH in the ΔG band AND "
                f"in the emission band [{emit_low}, {emit_high}])."
            )
            verdict = "no_in_band_anchor"
        return {
            "version": 2,
            "lr_ladder": list(LR_FROM_V2_SMOKE_SLUG.values()),
            "chosen_lr": None,
            "chosen_rank": 8,
            "chosen_alpha": 32,
            "chosen_checkpoint_fraction": None,
            "chosen_checkpoint_steps": None,
            "source": source,
            "source_delta_g_at_pick_nats": None,
            "source_emission_at_pick": None,
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "verdict": verdict,
            "smoke_table": smoke_table,
        }

    # ── In-band pick (plan v2 §4.1 step 3). ─────────────────────────────────
    # 1. Latest in-band frac (DESC).
    # 2. Tie-break: source_dg closest to the literal plan target = 8.0 nats
    #    (plan v2 §4.1 step 3(b) "closest to source_ΔG = 8.0"). Round-2 fix
    #    (Concern A): the round-17 code used (dg_low + dg_high) / 2.0 = 8.5
    #    which disagrees with the plan literal; reconciling to the plan value.
    #    A synthetic tie at the same latest fraction picked 8.4 over 7.9 under
    #    the midpoint rule, opposite the 8.0-target rule. The 0.5-nat shift is
    #    operationally small, but the plan target IS the contract.
    # 3. Lower lr (ASC).
    _TIE_BREAK_TARGET_NATS = 8.0
    candidates.sort(
        key=lambda lr_frac_dg: (
            -lr_frac_dg[1],  # latest fraction first (DESC)
            abs(lr_frac_dg[2] - _TIE_BREAK_TARGET_NATS),  # closest to 8.0 (ASC)
            lr_frac_dg[0],  # lower lr (ASC)
        )
    )
    chosen_lr, chosen_frac, chosen_dg = candidates[0]
    # Look up emission at the chosen point.
    chosen_slug = next(
        slug
        for slug in PHASE0_SMOKE_SLUGS_V2
        if abs(_lr_from_v2_smoke_slug(slug) - chosen_lr) < 1e-12
    )
    chosen_row = next(r for r in smoke_table if r["slug"] == chosen_slug)
    chosen_emit = chosen_row["per_frac"][chosen_frac]["source_emission"]

    # Steps at the picked fraction. The Phase 1 composition (1 epoch / 400
    # rows / batch 16) yields max_steps = 25 (plan v2 §4.1 + §11). The
    # picker rounds frac × max_steps but the trainer's
    # CheckpointAtFractionsCallback recomputes the actual saved-step itself;
    # this field is informational only.
    chosen_steps = max(1, round(chosen_frac * 25))

    return {
        "version": 2,
        "lr_ladder": list(LR_FROM_V2_SMOKE_SLUG.values()),
        "chosen_lr": float(chosen_lr),
        "chosen_rank": 8,
        "chosen_alpha": 32,
        "chosen_checkpoint_fraction": float(chosen_frac),
        "chosen_checkpoint_steps": int(chosen_steps),
        "source": source,
        "source_delta_g_at_pick_nats": float(chosen_dg),
        "source_emission_at_pick": float(chosen_emit),
        "fallback_triggered": False,
        "fallback_reason": None,
        "verdict": "pass",
        "smoke_table": smoke_table,
    }


def write_phase0_v2_artifact(pick: dict[str, Any], out_path: Path) -> Path:
    """Write phase0_calibration_v2.json (plan v2 §4.1 output format)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pick, indent=2))
    log.info(
        "[phase0_v2] wrote %s (verdict=%s, chosen_lr=%s, fallback=%s)",
        out_path,
        pick.get("verdict"),
        pick.get("chosen_lr"),
        pick.get("fallback_triggered"),
    )
    return out_path


def load_phase0_v2_pick(path: Path) -> dict[str, Any]:
    """Load a phase0_calibration_v2.json; raise on missing file (fail-loud).

    The verdict check is the caller's responsibility — Phase 0 fallback
    (§4.2) is triggered when `fallback_triggered=True` AND the dispatcher
    proceeds to a second smoke on an easier source, NOT a hard abort.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"phase0_calibration_v2.json missing at {path} — Phase 0 v2 must "
            f"complete BEFORE Phase 1 can spawn."
        )
    return json.loads(path.read_text())


# ── v3 EPOCHS-ladder picker (plan v3 §4.1) ──────────────────────────────────


def _epochs_from_v3_smoke_slug(slug: str) -> int:
    """Return the EPOCHS value associated with a v3 smoke slug.

    Raises:
        KeyError: `slug` is not a v3 smoke slug.
    """
    if slug not in EPOCHS_FROM_V3_SMOKE_SLUG:
        raise KeyError(
            f"Not a v3 smoke slug: {slug!r}; expected one of {sorted(EPOCHS_FROM_V3_SMOKE_SLUG)}"
        )
    return EPOCHS_FROM_V3_SMOKE_SLUG[slug]


def pick_anchor_from_epochs_smoke(
    smoke_trajectories: dict[str, dict],
    *,
    dg_band: tuple[float, float] = (SOURCE_DG_BAND_LOW, SOURCE_DG_BAND_HIGH),
    emit_band: tuple[float, float] = (EMISSION_BAND_LOW, EMISSION_BAND_HIGH),
    checkpoint_fractions: tuple[float, ...] = CHECKPOINT_FRACTIONS,
    source: str = SOURCE_PERSONA,
    fixed_lr: float = FIXED_LR_V3,
    expected_smoke_slugs: tuple[str, ...] = PHASE0_SMOKE_SLUGS_V3,
) -> dict[str, Any]:
    """Pick the anchor (chosen_epochs, chosen_checkpoint_fraction) per plan v3 §4.1.

    Walks the 2 v3 smoke trajectories (one per EPOCHS in {2, 3} at fixed
    lr=1e-4), tags each (epochs, frac) pair as in-band when source ΔG ∈
    [5, 12] nats AND on-policy emission ∈ [0.1, 0.8], and applies the plan v3
    §4.1 step 3 pick rule:

      1. Latest in-band fraction (DESC, per #479 — ΔG plateaus by frac=0.08
         in some cells; the latest in-band fraction gives the most stable read).
      2. Tie-break: source ΔG closest to 8.0 nats (band midpoint).
      3. Tie-break: LOWER EPOCHS (cheaper Phase 1 wall-time — preferred when
         either bracket lands in band).

    Fallback triggers (plan v3 §4.1 step 5):

      - Trigger A (floor): max source ΔG across all (epochs, frac) pairs
        < 5 nats. The EPOCHS lever doesn't unstick emission → exit to v4.
      - Trigger B (saturated on EITHER axis — Codex methodology REVISE
        binding): EITHER `min(source_ΔG) > 12` OR `max(source_emission) > 0.8`.
        Re-run EPOCHS=2 at finer fraction grid {0.02, 0.04, 0.06, 0.08}
        (in-plan recovery; caller handles).
      - Trigger C (empty band): in_band set is empty AND neither A nor B
        fires individually — the band is bracketed but no cell lands in it
        at any fraction. Exit to v4.

    On any fallback trigger, `verdict` carries the failure mode and
    `fallback_triggered` is True; `fallback_reason` names the trigger so
    the dispatcher (or the picker CLI) can route appropriately.

    Args:
        smoke_trajectories: {smoke_slug: trajectory_dict} keyed by v3 smoke
            slug (default: ``c504v3_smoke_eps{2,3}``).
        dg_band: (low, high) for source ΔG in nats (default [5, 12]).
        emit_band: (low, high) for source on-policy emission (default [0.1, 0.8]).
        checkpoint_fractions: full trajectory cadence (informational).
        source: source persona name recorded in the artifact (default villain).
        fixed_lr: pinned lr (always 1e-4 in v3, but exposed for tests).
        expected_smoke_slugs: which slugs to iterate (defaults to canonical
            v3 ladder; tests pass the synthesized off-canonical ladder slugs).

    Returns:
        {
          "version": 3,
          "epochs_ladder": [2, 3],
          "fixed_lr": 1e-4,
          "fixed_rank": 8,
          "fixed_alpha": 32,
          "chosen_epochs": int | None,
          "chosen_lr": 1e-4,                # pinned, NOT swept
          "chosen_rank": 8,                  # pinned
          "chosen_alpha": 32,                # pinned
          "chosen_checkpoint_fraction": float | None,
          "chosen_checkpoint_steps": int | None,
          "chosen_source": str,
          "source": str,                     # alias for chosen_source, parity
          "source_delta_g_at_pick_nats": float | None,
          "source_emission_at_pick": float | None,
          "fallback_triggered": bool,
          "fallback_reason": str | None,
          "in_plan_recovery_triggered": bool,
          "verdict": "pass" | "no_in_band_anchor" | "all_saturated",
          "smoke_table": [{slug, epochs, per_frac, latest_in_band_frac}, ...],
        }
    """
    dg_low, dg_high = dg_band
    emit_low, emit_high = emit_band

    smoke_table: list[dict] = []
    # candidates: (epochs, latest_in_band_frac, source_dg)
    candidates: list[tuple[int, float, float]] = []
    all_pairs_dg: list[float] = []
    all_pairs_emit: list[float] = []
    for slug in expected_smoke_slugs:
        if slug not in smoke_trajectories:
            raise KeyError(
                f"pick_anchor_from_epochs_smoke: smoke trajectory missing for "
                f"{slug!r}; got slugs: {sorted(smoke_trajectories)}"
            )
        epochs = _epochs_from_v3_smoke_slug(slug)
        per_frac = _per_frac_source_diagnostics(smoke_trajectories[slug], source=source)
        per_frac_tagged = {
            frac: {
                **v,
                "in_band": (
                    dg_low <= v["source_dg"] <= dg_high
                    and emit_low <= v["source_emission"] <= emit_high
                ),
            }
            for frac, v in per_frac.items()
        }
        latest_in_band = _latest_in_band_frac(
            per_frac,
            dg_low=dg_low,
            dg_high=dg_high,
            emit_low=emit_low,
            emit_high=emit_high,
        )
        smoke_table.append(
            {
                "slug": slug,
                "epochs": epochs,
                "per_frac": per_frac_tagged,
                "latest_in_band_frac": latest_in_band,
            }
        )
        for v in per_frac.values():
            all_pairs_dg.append(v["source_dg"])
            all_pairs_emit.append(v["source_emission"])
        if latest_in_band is not None:
            candidates.append(
                (
                    epochs,
                    latest_in_band,
                    per_frac[latest_in_band]["source_dg"],
                )
            )

    # ── Fallback trigger detection (plan v3 §4.1 step 5). ───────────────────
    # NB: trigger detection runs over ALL pairs, not just in-band ones; the
    # plan distinguishes A (floor) / B (saturated on EITHER axis) / C (empty
    # band) as three separate exit paths.
    fallback_triggered = False
    in_plan_recovery_triggered = False
    fallback_reason: str | None = None
    if all_pairs_dg:
        max_dg = max(all_pairs_dg)
        min_dg = min(all_pairs_dg)
    else:
        max_dg = float("nan")
        min_dg = float("nan")
    max_emit = max(all_pairs_emit) if all_pairs_emit else float("nan")

    if not candidates:
        # Distinguish A / B / C. Note Trigger B is OR'd across BOTH axes
        # per plan v3 §4.1 step 5 (Codex methodology REVISE binding): the
        # B clause fires when EITHER the source-ΔG axis OR the emission
        # axis is saturated, even when the other axis is in band — without
        # OR-on-emission, the picker would treat (high emission, in-band
        # ΔG) as the empty-band Trigger C and falsely exit to v4 instead
        # of attempting the cheap finer-fraction in-plan recovery.
        if all_pairs_dg and max_dg < dg_low:
            fallback_triggered = True
            fallback_reason = (
                f"trigger_A_floor: max(source_dg) over the {len(all_pairs_dg)} "
                f"(epochs, frac) pairs = {max_dg:.3f} nats < {dg_low} nats — "
                f"EPOCHS lever alone is insufficient at lr={fixed_lr:g} for "
                f"{source!r}; exit to plan v4 (rank bump)."
            )
            verdict = "no_in_band_anchor"
        elif all_pairs_dg and (min_dg > dg_high or max_emit > emit_high):
            fallback_triggered = True
            in_plan_recovery_triggered = True
            fallback_reason = (
                f"trigger_B_saturated: EITHER min(source_dg) over the "
                f"{len(all_pairs_dg)} (epochs, frac) pairs = {min_dg:.3f} nats "
                f"> {dg_high} nats OR max(source_emission) = {max_emit:.3f} > "
                f"{emit_high} — anchor saturates on at least one axis at "
                f"every coarse fraction; in-plan recovery re-runs EPOCHS=2 at "
                f"finer fractions before exit-to-v4 is considered."
            )
            verdict = "all_saturated"
        else:
            fallback_triggered = True
            fallback_reason = (
                f"trigger_C_empty_band: in_band set is empty (max_dg={max_dg:.3f}, "
                f"min_dg={min_dg:.3f}, max_emit={max_emit:.3f}); band is "
                f"bracketed but no (epochs, frac) lands BOTH in the ΔG band "
                f"AND in the emission band [{emit_low}, {emit_high}]) — "
                f"EPOCHS lever has no sweet spot here; exit to plan v4."
            )
            verdict = "no_in_band_anchor"
        return {
            "version": 3,
            "epochs_ladder": list(EPOCHS_FROM_V3_SMOKE_SLUG.values()),
            "fixed_lr": float(fixed_lr),
            "fixed_rank": 8,
            "fixed_alpha": 32,
            "chosen_epochs": None,
            "chosen_lr": float(fixed_lr),
            "chosen_rank": 8,
            "chosen_alpha": 32,
            "chosen_checkpoint_fraction": None,
            "chosen_checkpoint_steps": None,
            "chosen_source": source,
            "source": source,
            "source_delta_g_at_pick_nats": None,
            "source_emission_at_pick": None,
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "in_plan_recovery_triggered": in_plan_recovery_triggered,
            "verdict": verdict,
            "smoke_table": smoke_table,
        }

    # ── In-band pick (plan v3 §4.1 step 3). ─────────────────────────────────
    # 1. Latest in-band frac (DESC).
    # 2. Tie-break: source_dg closest to 8.0 nats (band midpoint, per plan
    #    §4.1 step 3(b) "closest to source_ΔG = 8.0").
    # 3. Tie-break: LOWER EPOCHS (cheaper Phase 1 wall-time).
    _TIE_BREAK_TARGET_NATS = 8.0
    candidates.sort(
        key=lambda eps_frac_dg: (
            -eps_frac_dg[1],  # latest fraction first (DESC)
            abs(eps_frac_dg[2] - _TIE_BREAK_TARGET_NATS),  # closest to 8.0 (ASC)
            eps_frac_dg[0],  # LOWER epochs (ASC)
        )
    )
    chosen_epochs, chosen_frac, chosen_dg = candidates[0]
    chosen_slug = next(
        slug for slug in expected_smoke_slugs if _epochs_from_v3_smoke_slug(slug) == chosen_epochs
    )
    chosen_row = next(r for r in smoke_table if r["slug"] == chosen_slug)
    chosen_emit = chosen_row["per_frac"][chosen_frac]["source_emission"]

    # Steps at picked fraction: max_steps per epoch is ~25 (400 rows /
    # effective batch 16) — total steps over `chosen_epochs` epochs ≈
    # `25 × chosen_epochs`. Informational only; the trainer's
    # CheckpointAtFractionsCallback recomputes the actual saved-step itself.
    steps_per_epoch = 25
    chosen_steps = max(1, round(chosen_frac * steps_per_epoch * chosen_epochs))

    return {
        "version": 3,
        "epochs_ladder": list(EPOCHS_FROM_V3_SMOKE_SLUG.values()),
        "fixed_lr": float(fixed_lr),
        "fixed_rank": 8,
        "fixed_alpha": 32,
        "chosen_epochs": int(chosen_epochs),
        "chosen_lr": float(fixed_lr),
        "chosen_rank": 8,
        "chosen_alpha": 32,
        "chosen_checkpoint_fraction": float(chosen_frac),
        "chosen_checkpoint_steps": int(chosen_steps),
        "chosen_source": source,
        "source": source,
        "source_delta_g_at_pick_nats": float(chosen_dg),
        "source_emission_at_pick": float(chosen_emit),
        "fallback_triggered": False,
        "fallback_reason": None,
        "in_plan_recovery_triggered": False,
        "verdict": "pass",
        "smoke_table": smoke_table,
    }


def write_phase0_v3_artifact(pick: dict[str, Any], out_path: Path) -> Path:
    """Write phase0_calibration_v3.json (plan v3 §4.1 output format)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pick, indent=2))
    log.info(
        "[phase0_v3] wrote %s (verdict=%s, chosen_epochs=%s, fallback=%s, recovery=%s)",
        out_path,
        pick.get("verdict"),
        pick.get("chosen_epochs"),
        pick.get("fallback_triggered"),
        pick.get("in_plan_recovery_triggered"),
    )
    return out_path


def write_phase0_v3_exit_to_v4_artifact(pick: dict[str, Any], out_path: Path) -> Path:
    """Write phase0_v3_exit_to_v4.json (plan v3 §4.2; trigger A or C fired).

    The artifact carries the 12-row smoke table + a `next_plan: "v4_rank_bump"`
    field so the orchestrator routes back to `/adversarial-planner` for plan v4.
    Distinct from `phase0_calibration_v3.json` (which always exists when
    Phase 0 v3 ran, regardless of verdict).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**pick, "next_plan": "v4_rank_bump"}
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "[phase0_v3] wrote exit-to-v4 artifact → %s (reason=%s)",
        out_path,
        pick.get("fallback_reason"),
    )
    return out_path


def load_phase0_v3_pick(path: Path) -> dict[str, Any]:
    """Load a phase0_calibration_v3.json; raise on missing file (fail-loud).

    The verdict check is the caller's responsibility — exit-to-v4 path
    (§4.2) is triggered when `fallback_triggered=True` AND not the
    in-plan recovery path; the dispatcher decides how to route.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"phase0_calibration_v3.json missing at {path} — Phase 0 v3 must "
            f"complete BEFORE Phase 1 can spawn."
        )
    return json.loads(path.read_text())


# ── v3 in-plan finer-fraction recovery merge (plan v3 §4.1 trigger B + §4.2). ──


def _merge_trajectories(coarse: dict, finer: dict) -> dict:
    """Concatenate two trajectory dicts produced by run_trajectory_eval.

    The merged dict preserves the coarse trajectory's metadata (cell, seed,
    source) and concatenates the `checkpoints` lists. Duplicate fractions
    (same key at `frac_precision=2`) are resolved by preferring the FINER
    (recovery) row — recovery is the fresh measurement; the coarse value
    at that fraction was a saturated read the recovery was meant to refute.

    Args:
        coarse: trajectory.json dict from the original EPOCHS=2 coarse-grid
            run (6 checkpoints at fracs {0.08, 0.16, 0.33, 0.50, 0.75, 1.00}).
        finer: trajectory.json dict from the recovery EPOCHS=2 finer-grid
            run (4 checkpoints at fracs {0.02, 0.04, 0.06, 0.08}).

    Returns:
        Merged trajectory dict with `checkpoints` = union of both, sorted
        by `frac` ASC. Used by `merge_recovery_into_v3_pick` before
        re-applying `pick_anchor_from_epochs_smoke`.
    """
    by_frac: dict[float, dict] = {}
    for ck in coarse.get("checkpoints", []):
        by_frac[float(ck["frac"])] = ck
    for ck in finer.get("checkpoints", []):
        # Recovery wins on collision (e.g. both have frac=0.08).
        by_frac[float(ck["frac"])] = ck
    merged_checkpoints = [by_frac[k] for k in sorted(by_frac)]
    merged: dict[str, Any] = {**coarse, "checkpoints": merged_checkpoints}
    return merged


def merge_recovery_into_v3_pick(
    smoke_trajectories: dict[str, dict],
    recovery_trajectory: dict,
    *,
    dg_band: tuple[float, float] = (SOURCE_DG_BAND_LOW, SOURCE_DG_BAND_HIGH),
    emit_band: tuple[float, float] = (EMISSION_BAND_LOW, EMISSION_BAND_HIGH),
    checkpoint_fractions: tuple[float, ...] = CHECKPOINT_FRACTIONS,
    source: str = SOURCE_PERSONA,
    fixed_lr: float = FIXED_LR_V3,
    expected_smoke_slugs: tuple[str, ...] = PHASE0_SMOKE_SLUGS_V3,
    recovery_slug: str = "c504v3_smoke_eps2",
) -> dict[str, Any]:
    """Re-apply the v3 pick after merging the finer-grid recovery trajectory.

    Plan v3 §4.1 trigger B + §4.2: when the original coarse-grid Phase 0 v3
    fires Trigger B (saturated on either axis), the dispatcher re-runs the
    EPOCHS=2 cell at finer fractions {0.02, 0.04, 0.06, 0.08}. This helper
    merges the finer trajectory into the EPOCHS=2 cell's checkpoint list
    and re-runs `pick_anchor_from_epochs_smoke` over the merged table.

    The returned dict has the SAME shape as `pick_anchor_from_epochs_smoke`
    (the merged pick fields), PLUS two extra fields for audit:

      * `recovery_finer_trajectory` — the raw finer-grid trajectory dict
        (so a downstream analyzer can re-derive per-frac diagnostics
        without going back to the on-disk file).
      * `merged_from_coarse` — `True` (sentinel: this pick reflects a
        coarse+finer merge, not a fresh pick).

    Args:
        smoke_trajectories: ORIGINAL coarse-grid v3 smoke trajectories
            (keyed by `PHASE0_SMOKE_SLUGS_V3`).
        recovery_trajectory: trajectory.json dict from the finer-grid
            recovery run on the EPOCHS=2 cell.
        recovery_slug: which smoke slug the recovery augments (default
            `c504v3_smoke_eps2`, the canonical Trigger B recovery target).
        dg_band, emit_band, checkpoint_fractions, source, fixed_lr,
            expected_smoke_slugs: passed through to
            `pick_anchor_from_epochs_smoke`.

    Returns:
        The re-picked dict + `recovery_finer_trajectory` + `merged_from_coarse`.

    Raises:
        KeyError: `recovery_slug` not in `smoke_trajectories` — the
            caller passed a corrupt coarse-trajectory dict.
    """
    if recovery_slug not in smoke_trajectories:
        raise KeyError(
            f"merge_recovery_into_v3_pick: recovery_slug={recovery_slug!r} not "
            f"in smoke_trajectories keys={sorted(smoke_trajectories)}; the "
            f"recovery target must be present in the original v3 trajectories."
        )
    merged_traj = _merge_trajectories(smoke_trajectories[recovery_slug], recovery_trajectory)
    merged_trajectories = dict(smoke_trajectories)
    merged_trajectories[recovery_slug] = merged_traj
    pick = pick_anchor_from_epochs_smoke(
        merged_trajectories,
        dg_band=dg_band,
        emit_band=emit_band,
        checkpoint_fractions=checkpoint_fractions,
        source=source,
        fixed_lr=fixed_lr,
        expected_smoke_slugs=expected_smoke_slugs,
    )
    pick["recovery_finer_trajectory"] = recovery_trajectory
    pick["merged_from_coarse"] = True
    return pick


# ── v4 bystander-resolution picker (plan v5 §4.1 fix #2). ────────────────────
#
# Replaces v3's `source_emission ∈ [0.1, 0.8]` gate (which penalized the source
# for SUCCEEDING at the implant — it is the trained-on persona; saturation IS
# expected). The v4 gate scores saturation on the BYSTANDERS: a non-trivial
# fraction of held-out probes must sit below the marker-argmax ceiling so the
# geometry sweep has dynamic range to read. See plan v5 §11 + §4.1.
#
# DEFAULTS:
#   - FLOOR_DELTA_G_NATS = +0.5: probe ΔG must be at least 0.5 nats above base
#     to count as "the model has shifted toward emitting the marker here."
#   - CEILING_LOGP_NATS = log(0.9) ≈ -0.105: probe `log P(marker)` must be at
#     least this far below 0 (i.e. marker probability < 0.9) so the probe is
#     NOT pinned at marker-argmax (where ΔG saturates and the regression has
#     no dynamic range).
#   - GATE_FRACTION = 0.20: at least 20% (≥ 11 of 55) of probes in the open
#     interval. Defensible balance — too sparse (10%) leaves the regression
#     underpowered; too restrictive (50%) is unreachable at EPOCHS=3 where
#     saturation creeps in.

CEILING_LOGP_NATS_V4 = math.log(0.9)  # ≈ -0.10536
FLOOR_DELTA_G_NATS_V4 = 0.5
BYSTANDER_RESOLUTION_GATE_V4 = 0.20  # ≥ 20% of probes in the open interval


def compute_bystander_resolution_from_held_out(
    held_out: dict[str, dict[str, dict[str, float | bool]]],
    *,
    floor_delta_g_nats: float = FLOOR_DELTA_G_NATS_V4,
    ceiling_logp_nats: float = CEILING_LOGP_NATS_V4,
) -> tuple[float, int, int]:
    """Bystander resolution = fraction of (probe × q) pairs in the open interval.

    The v4 anchor gate (plan v5 §4.1 step 2). For each (persona, q) leaf in the
    trajectory's `held_out` dict, count the pair as "in-band" when ΔG ≥ floor
    AND ``log P(※ | trained slot) ≤ ceiling`` (i.e. the marker is NOT the
    argmax at probability ≥ 0.9). Returns the fraction in-band over the full
    panel × q grid, plus raw counts for diagnostics.

    Args:
        held_out: the trajectory's `held_out` dict shape
            ``{persona: {q: {"g_logp": float, "b_logp": float, "delta_g": float, ...}}}``.
        floor_delta_g_nats: probe ΔG must be ≥ this (default +0.5 nats).
        ceiling_logp_nats: probe ``g_logp`` (trained log P(※)) must be ≤ this
            (default log(0.9) ≈ -0.105) — excludes probes pinned at marker-argmax.

    Returns:
        ``(fraction_in_band, n_in_band, n_total)``.
    """
    n_in_band = 0
    n_total = 0
    for per_q in held_out.values():
        for leaf in per_q.values():
            n_total += 1
            dg = float(leaf.get("delta_g", float(leaf["g_logp"]) - float(leaf["b_logp"])))
            g_logp = float(leaf["g_logp"])
            if dg >= floor_delta_g_nats and g_logp <= ceiling_logp_nats:
                n_in_band += 1
    return (n_in_band / n_total if n_total else 0.0), n_in_band, n_total


def pick_anchor_v4_bystander_resolution(
    trajectory: dict,
    *,
    source: str = SOURCE_PERSONA,
    fixed_lr: float = FIXED_LR_V3,
    chosen_epochs: int = 3,
    floor_delta_g_nats: float = FLOOR_DELTA_G_NATS_V4,
    ceiling_logp_nats: float = CEILING_LOGP_NATS_V4,
    gate_fraction: float = BYSTANDER_RESOLUTION_GATE_V4,
) -> dict[str, Any]:
    """v4 picker (plan v5 §4.1 fix #2): bystander-resolution gate, no source-emission gate.

    Reads ONE trajectory (the v4 EPOCHS=3 anchor re-eval); for each of the 6
    checkpoint fractions, computes the bystander-resolution score on the
    held-out panel (excluding source). Tags each fraction `in_band` when the
    score ≥ `gate_fraction` (default 20%, plan v5 §11).

    Pick rule (plan v5 §4.1 step 4):
      (a) From the in_band set, prefer the fraction whose resolution is closest
          to the band midpoint (0.5 — maximum spread for the regression).
      (b) Tie-break: earlier fraction (cheaper Phase 1 wall-clock; less
          catastrophic-forgetting risk).

    Fallback trigger (plan v5 §4.1 step 6):
      - in_band set EMPTY at every fraction → exit to plan v5 (rank bump). The
        v4 dispatcher invokes the EPOCHS=2 finer-grid bisection (§4.2 Step 1)
        before declaring a hard exit.

    Args:
        trajectory: trajectory.json dict produced by
            ``i504_eval_trajectory.py`` (Phase 0 v4 re-eval of the EPOCHS=3
            anchor through the fixed reader).
        source: source persona name (recorded in the artifact).
        fixed_lr: pinned lr (always 1e-4 in v4; recorded only).
        chosen_epochs: pinned EPOCHS (3 in v4; recorded only).
        floor_delta_g_nats, ceiling_logp_nats, gate_fraction: thresholds.

    Returns:
        {
          "version": 4,
          "anchor_epochs": int,
          "fixed_lr": float,
          "fixed_rank": int,
          "fixed_alpha": int,
          "chosen_epochs": int,
          "chosen_lr": float,
          "chosen_rank": int,
          "chosen_alpha": int,
          "chosen_checkpoint_fraction": float | None,
          "chosen_checkpoint_steps": int | None,
          "source_delta_g_at_pick_nats": float | None,
          "source_emission_at_pick": float | None,
          "bystander_resolution_at_pick": float | None,
          "ceiling_logp": float,
          "floor_delta_g": float,
          "gate_fraction": float,
          "fallback_triggered": bool,
          "fallback_reason": str | None,
          "smoke_table": [
            {"epochs": 3, "ckpt_frac": float, "source_dg": float,
             "source_emission": float, "bystander_resolution": float,
             "n_in_band": int, "n_total": int, "in_band": bool},
            ...
          ],
          "verdict": "pass" | "no_in_band_anchor",
          "version_str": "v4_bystander_resolution",
        }
    """
    per_frac = _per_frac_source_diagnostics(trajectory, source=source)
    smoke_table: list[dict] = []
    in_band_candidates: list[tuple[float, float, float]] = []
    # tuple shape: (frac, resolution_distance_to_0.5, resolution)
    for ck in trajectory.get("checkpoints", []):
        frac = float(ck["frac"])
        held_out = ck.get("held_out", {})
        resolution, n_in_band, n_total = compute_bystander_resolution_from_held_out(
            held_out,
            floor_delta_g_nats=floor_delta_g_nats,
            ceiling_logp_nats=ceiling_logp_nats,
        )
        src_diag = per_frac.get(frac, {"source_dg": float("nan"), "source_emission": float("nan")})
        in_band = resolution >= gate_fraction
        row = {
            "epochs": chosen_epochs,
            "ckpt_frac": frac,
            "source_dg": float(src_diag["source_dg"]),
            "source_emission": float(src_diag["source_emission"]),
            "bystander_resolution": float(resolution),
            "n_in_band": int(n_in_band),
            "n_total": int(n_total),
            "in_band": bool(in_band),
        }
        smoke_table.append(row)
        if in_band:
            # Distance to band midpoint (0.5 — maximum spread). Round to 6dp
            # before sorting so float-precision ties (e.g. resolution=25/55
            # vs 30/55 against 0.5) collapse onto an integer-count distance
            # and the deterministic earlier-frac tie-break fires.
            rounded_dist = round(abs(resolution - 0.5), 6)
            in_band_candidates.append((frac, rounded_dist, resolution))

    if not in_band_candidates:
        # Fallback: bystander layer has no dynamic range at the EPOCHS=3 anchor.
        # Plan v5 §4.1 step 6 routes to EPOCHS=2 bisection (§4.2 Step 1); this
        # picker just surfaces the fallback signal.
        return {
            "version": 4,
            "anchor_epochs": int(chosen_epochs),
            "fixed_lr": float(fixed_lr),
            "fixed_rank": 8,
            "fixed_alpha": 32,
            "chosen_epochs": int(chosen_epochs),
            "chosen_lr": float(fixed_lr),
            "chosen_rank": 8,
            "chosen_alpha": 32,
            "chosen_checkpoint_fraction": None,
            "chosen_checkpoint_steps": None,
            "source_delta_g_at_pick_nats": None,
            "source_emission_at_pick": None,
            "bystander_resolution_at_pick": None,
            "ceiling_logp": float(ceiling_logp_nats),
            "floor_delta_g": float(floor_delta_g_nats),
            "gate_fraction": float(gate_fraction),
            "fallback_triggered": True,
            "fallback_reason": (
                f"bystander_resolution_unreachable: 0 of {len(smoke_table)} "
                f"checkpoint fractions have ≥ {gate_fraction:.0%} of probes in "
                f"the open interval (floor={floor_delta_g_nats:+.2f} nats, "
                f"ceiling={ceiling_logp_nats:.3f} nats). EPOCHS=3 anchor's "
                f"bystander layer has no dynamic range → §4.2 EPOCHS=2 "
                f"bisection per plan v5 §4.1 step 6."
            ),
            "smoke_table": smoke_table,
            "verdict": "no_in_band_anchor",
            "source": source,
            "version_str": "v4_bystander_resolution",
        }

    # Pick rule: closest to band midpoint (0.5) ASC, then earlier fraction ASC.
    in_band_candidates.sort(key=lambda x: (x[1], x[0]))
    chosen_frac, _dist, chosen_resolution = in_band_candidates[0]
    chosen_row = next(r for r in smoke_table if r["ckpt_frac"] == chosen_frac)

    # Steps at picked fraction: ~25 steps per epoch (400 rows / effective
    # batch 16) × chosen_epochs (3 in v4). Informational only; the trainer's
    # CheckpointAtFractionsCallback computes the actual saved-step itself.
    steps_per_epoch = 25
    chosen_steps = max(1, round(chosen_frac * steps_per_epoch * chosen_epochs))

    return {
        "version": 4,
        "anchor_epochs": int(chosen_epochs),
        "fixed_lr": float(fixed_lr),
        "fixed_rank": 8,
        "fixed_alpha": 32,
        "chosen_epochs": int(chosen_epochs),
        "chosen_lr": float(fixed_lr),
        "chosen_rank": 8,
        "chosen_alpha": 32,
        "chosen_checkpoint_fraction": float(chosen_frac),
        "chosen_checkpoint_steps": int(chosen_steps),
        "source_delta_g_at_pick_nats": float(chosen_row["source_dg"]),
        "source_emission_at_pick": float(chosen_row["source_emission"]),
        "bystander_resolution_at_pick": float(chosen_resolution),
        "ceiling_logp": float(ceiling_logp_nats),
        "floor_delta_g": float(floor_delta_g_nats),
        "gate_fraction": float(gate_fraction),
        "fallback_triggered": False,
        "fallback_reason": None,
        "smoke_table": smoke_table,
        "verdict": "pass",
        "source": source,
        "version_str": "v4_bystander_resolution",
    }


def write_phase0_v4_artifact(pick: dict[str, Any], out_path: Path) -> Path:
    """Write phase0_calibration_v4.json (plan v5 §4.1 output format)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pick, indent=2))
    return out_path
