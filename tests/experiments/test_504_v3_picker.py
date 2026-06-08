# em-dash + Greek ΔG + Qwen marker intentional
"""Task #504 v3 Phase 0 picker — EPOCHS-ladder rule + Trigger A/B/C fallback.

Pins the contract for plan v3 §4.1 pick rule and fallback triggers (Codex
methodology REVISE binding: Trigger B fires on EITHER axis OR'd — source ΔG
> 12 nats OR emission > 0.8).

CPU-only, sub-second. Constructs synthetic trajectory dicts and exercises
`pick_anchor_from_epochs_smoke` directly; no GPU/HF/network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
    CHECKPOINT_FRACTIONS,
    EMISSION_BAND_HIGH,
    EMISSION_BAND_LOW,
    EPOCHS_FROM_V3_SMOKE_SLUG,
    FIXED_LR_V3,
    PHASE0_SMOKE_SLUGS_V3,
    SOURCE_DG_BAND_HIGH,
    SOURCE_DG_BAND_LOW,
)
from explore_persona_space.experiments.contrastive_neg_geometry_504.phase0 import (
    pick_anchor_from_epochs_smoke,
    write_phase0_v3_artifact,
    write_phase0_v3_exit_to_v4_artifact,
)


def _trajectory(
    *,
    per_frac: dict[float, tuple[float, float]],
) -> dict:
    """Build a trajectory.json shape that the picker can consume.

    Args:
        per_frac: {frac: (source_dg, source_emission)} for each checkpoint.
    """
    checkpoints = []
    for frac, (dg, emit) in per_frac.items():
        checkpoints.append(
            {
                "frac": frac,
                "step": round(frac * 50),
                "source_self": {
                    "delta_g_mean": dg,
                    "emission_p": emit,
                },
                "held_out": {},
            }
        )
    return {"checkpoints": checkpoints}


# ── Happy-path picks ────────────────────────────────────────────────────────


def test_happy_in_band_at_eps2_picks_eps2():
    """EPOCHS=2 has in-band cells; EPOCHS=3 floors → pick EPOCHS=2."""
    trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (1.0, 0.0),
                0.16: (3.0, 0.0),
                0.33: (6.0, 0.2),  # in-band
                0.50: (8.0, 0.4),  # in-band
                0.75: (10.0, 0.5),  # in-band
                1.00: (11.5, 0.7),  # in-band
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (0.5, 0.0),
                0.16: (1.5, 0.0),
                0.33: (2.5, 0.0),
                0.50: (3.0, 0.0),
                0.75: (3.5, 0.0),
                1.00: (4.0, 0.0),
            }
        ),
    }
    pick = pick_anchor_from_epochs_smoke(trajs)
    assert pick["verdict"] == "pass", pick.get("fallback_reason")
    assert pick["chosen_epochs"] == 2
    # Latest in-band fraction is 1.00 → that's the pick.
    assert pick["chosen_checkpoint_fraction"] == 1.00
    assert pick["fallback_triggered"] is False
    assert pick["chosen_lr"] == FIXED_LR_V3
    assert pick["chosen_rank"] == 8
    assert pick["chosen_alpha"] == 32
    assert pick["source"] == "villain"


def test_tie_break_lower_epochs_when_both_in_band_at_same_frac():
    """Both EPOCHS=2 AND EPOCHS=3 have in-band cells at the same latest frac;
    tie-break on closeness-to-8 then LOWER EPOCHS."""
    # Both at frac=1.00 land in band, equidistant from 8.0 (one at 7.5, one
    # at 8.5). Lower EPOCHS = 2 wins per plan §4.1 step 3(c).
    trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (1.0, 0.0),
                1.00: (8.5, 0.5),  # in-band, |8.5-8|=0.5
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (1.0, 0.0),
                1.00: (7.5, 0.5),  # in-band, |7.5-8|=0.5
            }
        ),
    }
    pick = pick_anchor_from_epochs_smoke(trajs)
    # Both at frac=1.00 with |Δ_dg-8|=0.5 → tied on (latest_frac, |dg-8|),
    # tie-broken by lower epochs.
    assert pick["verdict"] == "pass"
    assert pick["chosen_epochs"] == 2, "Lower EPOCHS should win on tie"


def test_tie_break_closer_to_8nats_when_two_epochs_in_band():
    """When two EPOCHS values both have in-band cells at the same latest frac
    but different ΔG values, pick the one closest to 8.0 nats."""
    trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                1.00: (5.5, 0.2),  # in-band, |5.5-8|=2.5
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                1.00: (8.0, 0.4),  # in-band, |8.0-8|=0.0 (closer)
            }
        ),
    }
    pick = pick_anchor_from_epochs_smoke(trajs)
    assert pick["verdict"] == "pass"
    assert pick["chosen_epochs"] == 3, "Closer to 8.0 nats should win"


# ── Trigger A — floor (max ΔG < 5 nats across all pairs) ────────────────────


def test_trigger_A_floor_when_max_dg_below_band_low():
    """All cells stuck below 5 nats → Trigger A (floor) → exit to v4."""
    trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (1.0, 0.0),
                0.16: (2.0, 0.0),
                0.33: (3.0, 0.0),
                0.50: (3.5, 0.0),
                0.75: (4.0, 0.0),
                1.00: (4.5, 0.0),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (1.0, 0.0),
                0.16: (2.0, 0.0),
                0.33: (3.0, 0.0),
                0.50: (3.5, 0.0),
                0.75: (4.0, 0.0),
                1.00: (4.5, 0.0),
            }
        ),
    }
    pick = pick_anchor_from_epochs_smoke(trajs)
    assert pick["verdict"] == "no_in_band_anchor"
    assert pick["fallback_triggered"] is True
    assert pick["in_plan_recovery_triggered"] is False
    assert "trigger_A_floor" in pick["fallback_reason"]
    assert pick["chosen_epochs"] is None


# ── Trigger B — saturated on EITHER axis (OR'd, Codex methodology REVISE) ──


def test_trigger_B_saturated_on_dg_axis_min_above_band_high():
    """min(source_ΔG) > 12 nats across all cells → Trigger B (dg saturated)."""
    trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (13.0, 0.95),
                0.16: (14.0, 0.95),
                0.33: (15.0, 0.98),
                0.50: (15.0, 0.99),
                0.75: (15.0, 0.99),
                1.00: (16.0, 1.0),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (13.0, 0.95),
                0.16: (14.0, 0.96),
                0.33: (15.0, 0.98),
                0.50: (15.0, 0.99),
                0.75: (15.0, 0.99),
                1.00: (16.0, 1.0),
            }
        ),
    }
    pick = pick_anchor_from_epochs_smoke(trajs)
    assert pick["verdict"] == "all_saturated"
    assert pick["fallback_triggered"] is True
    assert pick["in_plan_recovery_triggered"] is True
    assert "trigger_B_saturated" in pick["fallback_reason"]


def test_trigger_B_saturated_on_emission_axis_while_dg_in_band():
    """REGRESSION (Codex methodology REVISE binding): emission > 0.8 with
    source ΔG IN band must STILL fire Trigger B, not be treated as empty band
    (Trigger C). Without OR-on-emission the picker would route to v4 instead
    of attempting the cheap finer-fraction in-plan recovery.

    Specifically: every (eps, frac) pair has source ΔG ∈ [5, 12] (in band on
    that axis) but emission > 0.8 (saturated on the OTHER axis). Since at
    least one axis is saturated, NO pair lands in_band (which requires BOTH
    axes in band). Trigger B must fire — NOT Trigger A (floor) and NOT
    Trigger C (empty band, no axis saturated).
    """
    trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (6.0, 0.95),  # dg in band, emit saturated
                0.16: (7.0, 0.95),
                0.33: (8.0, 0.96),
                0.50: (8.5, 0.97),
                0.75: (9.0, 0.98),
                1.00: (9.5, 0.99),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (6.0, 0.95),
                0.16: (7.0, 0.96),
                0.33: (8.0, 0.97),
                0.50: (8.5, 0.97),
                0.75: (9.0, 0.98),
                1.00: (9.5, 0.99),
            }
        ),
    }
    pick = pick_anchor_from_epochs_smoke(trajs)
    assert pick["verdict"] == "all_saturated"
    assert pick["fallback_triggered"] is True
    assert pick["in_plan_recovery_triggered"] is True, (
        "in-plan recovery should be set for Trigger B (any-axis saturation), "
        "so the dispatcher fires finer-fraction recovery rather than exit-to-v4"
    )
    assert "trigger_B_saturated" in pick["fallback_reason"]


def test_trigger_B_OR_logic_explicit():
    """Explicit OR test: Trigger B fires when EITHER axis is saturated, even
    when the other axis is fully in band. This is the exact unit test the
    plan v3 §4.1 step 5 + reviewer's acceptance criteria require.
    """
    # Case A: source ΔG saturated, emission low → Trigger B (min ΔG > 12)
    trajs_a = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (13.0, 0.2),
                1.00: (14.0, 0.5),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (13.0, 0.2),
                1.00: (14.0, 0.5),
            }
        ),
    }
    pick_a = pick_anchor_from_epochs_smoke(trajs_a)
    assert pick_a["verdict"] == "all_saturated"
    assert pick_a["fallback_triggered"] is True
    assert pick_a["in_plan_recovery_triggered"] is True

    # Case B: source ΔG in band, emission saturated → Trigger B (max emit > 0.8)
    trajs_b = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (6.0, 0.85),
                1.00: (7.5, 0.95),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (6.0, 0.85),
                1.00: (7.5, 0.95),
            }
        ),
    }
    pick_b = pick_anchor_from_epochs_smoke(trajs_b)
    assert pick_b["verdict"] == "all_saturated"
    assert pick_b["fallback_triggered"] is True
    assert pick_b["in_plan_recovery_triggered"] is True


# ── Trigger C — empty in-band set (bracketed but no cell in band) ──────────


def test_trigger_C_empty_band_brackets_but_no_in_band_cell():
    """max(ΔG) ≥ 5 AND min(ΔG) ≤ 12 (band is bracketed) AND max(emit) ≤ 0.8
    (emission NOT saturated) AND no (eps, frac) cell happens to land BOTH in
    [5, 12] AND [0.1, 0.8] → Trigger C (empty band)."""
    # Some pairs > 12 nats, some < 5 nats — bracketed but no in-band cell.
    # All emissions ≤ 0.8 so trigger B (saturation) does NOT fire.
    trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (2.0, 0.0),  # below floor
                0.16: (3.0, 0.0),
                0.33: (4.0, 0.0),  # still < 5
                0.50: (13.0, 0.5),  # > 12, but emission < 0.8
                0.75: (14.0, 0.6),
                1.00: (15.0, 0.7),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (2.5, 0.0),
                0.16: (3.5, 0.0),
                0.33: (4.5, 0.0),  # < 5
                0.50: (13.0, 0.5),
                0.75: (14.0, 0.6),
                1.00: (15.0, 0.7),
            }
        ),
    }
    pick = pick_anchor_from_epochs_smoke(trajs)
    assert pick["verdict"] == "no_in_band_anchor"
    assert pick["fallback_triggered"] is True
    assert pick["in_plan_recovery_triggered"] is False, (
        "Trigger C should NOT trigger in-plan recovery (only Trigger B does)"
    )
    assert "trigger_C_empty_band" in pick["fallback_reason"]


# ── Artifact write/read round-trips ─────────────────────────────────────────


def test_write_phase0_v3_artifact_schema(tmp_path: Path):
    """`phase0_calibration_v3.json` carries chosen_epochs + chosen_source +
    verdict (the acceptance-criteria fields)."""
    trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.50: (8.0, 0.4),
                1.00: (10.0, 0.6),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.50: (1.0, 0.0),
                1.00: (1.5, 0.0),
            }
        ),
    }
    pick = pick_anchor_from_epochs_smoke(trajs, source="villain")
    out = tmp_path / "phase0_calibration_v3.json"
    write_phase0_v3_artifact(pick, out)
    payload = json.loads(out.read_text())
    # Acceptance criteria from the brief.
    assert "chosen_epochs" in payload
    assert payload["chosen_source"] == "villain"
    assert payload["verdict"] == "pass"
    # And the rest of the required fields per plan §4.1 output spec.
    assert payload["version"] == 3
    assert payload["fixed_lr"] == FIXED_LR_V3
    assert payload["fixed_rank"] == 8
    assert payload["fixed_alpha"] == 32
    assert payload["chosen_lr"] == FIXED_LR_V3
    assert payload["chosen_checkpoint_fraction"] is not None
    assert payload["chosen_checkpoint_steps"] is not None
    assert payload["fallback_triggered"] is False
    assert payload["in_plan_recovery_triggered"] is False
    assert "smoke_table" in payload and len(payload["smoke_table"]) == 2


def test_write_phase0_v3_exit_to_v4_artifact_includes_next_plan(tmp_path: Path):
    """The exit-to-v4 artifact carries `next_plan: 'v4_rank_bump'`."""
    trajs = {
        # All cells stuck below 5 → Trigger A floor.
        "c504v3_smoke_eps2": _trajectory(per_frac={0.50: (1.0, 0.0), 1.00: (2.0, 0.0)}),
        "c504v3_smoke_eps3": _trajectory(per_frac={0.50: (1.0, 0.0), 1.00: (2.0, 0.0)}),
    }
    pick = pick_anchor_from_epochs_smoke(trajs)
    assert pick["fallback_triggered"] is True
    out = tmp_path / "phase0_v3_exit_to_v4.json"
    write_phase0_v3_exit_to_v4_artifact(pick, out)
    payload = json.loads(out.read_text())
    assert payload["next_plan"] == "v4_rank_bump"
    assert payload["fallback_triggered"] is True
    assert "trigger_A_floor" in payload["fallback_reason"]


# ── Source persona threading ────────────────────────────────────────────────


def test_source_persona_recorded_in_artifact():
    """The `source` argument is recorded in the picker output."""
    trajs = {
        "c504v3_smoke_eps2": _trajectory(per_frac={1.00: (8.0, 0.4)}),
        "c504v3_smoke_eps3": _trajectory(per_frac={1.00: (1.0, 0.0)}),
    }
    pick = pick_anchor_from_epochs_smoke(trajs, source="medical_doctor")
    # In v3 we don't sweep source; just verify it's recorded.
    assert pick["source"] == "medical_doctor"
    assert pick["chosen_source"] == "medical_doctor"


# ── Missing smoke slug raises KeyError ──────────────────────────────────────


def test_missing_smoke_slug_raises():
    """If a required smoke slug is absent from the trajectory dict, the picker
    raises KeyError with a helpful message."""
    # Only EPOCHS=2 trajectory provided; EPOCHS=3 missing.
    trajs = {
        "c504v3_smoke_eps2": _trajectory(per_frac={1.00: (8.0, 0.4)}),
    }
    with pytest.raises(KeyError, match=r"c504v3_smoke_eps3"):
        pick_anchor_from_epochs_smoke(trajs)


# ── Band constants sanity ────────────────────────────────────────────────────


def test_band_constants_match_plan_v3():
    """Pin the band constants used by the v3 picker — plan §4.1 step 2."""
    assert SOURCE_DG_BAND_LOW == 5.0
    assert SOURCE_DG_BAND_HIGH == 12.0
    assert EMISSION_BAND_LOW == 0.1
    assert EMISSION_BAND_HIGH == 0.8
    # Canonical v3 ladder per plan §10.
    assert EPOCHS_FROM_V3_SMOKE_SLUG == {"c504v3_smoke_eps2": 2, "c504v3_smoke_eps3": 3}
    assert PHASE0_SMOKE_SLUGS_V3 == ("c504v3_smoke_eps2", "c504v3_smoke_eps3")
    assert FIXED_LR_V3 == 1e-4
    # Checkpoint cadence unchanged from v2.
    assert CHECKPOINT_FRACTIONS == (0.08, 0.16, 0.33, 0.50, 0.75, 1.00)
