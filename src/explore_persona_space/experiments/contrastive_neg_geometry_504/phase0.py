# em-dash + Qwen marker " ※" + Greek ΔG intentional
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
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    CHECKPOINT_FRACTIONS,
    EMISSION_BAND_HIGH,
    EMISSION_BAND_LOW,
    PHASE0_CALIB_RANKS,
    PHASE0_SMOKE_SLUGS,
    SOURCE_DG_BAND_HIGH,
    SOURCE_DG_BAND_LOW,
    SOURCE_PERSONA,
    alpha_for_rank,
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
