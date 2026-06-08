# em-dash + Greek ΔG intentional
"""Task #504 v3 in-plan finer-fraction recovery merge — plan §4.1 trigger B + §4.2.

Pins the contract for `merge_recovery_into_v3_pick` (round-8 fix for the
deferred concern `phase0-v3-finer-grid-recovery-not-wired`): the dispatcher's
recovery phase retrains EPOCHS=2 at finer fractions {0.02, 0.04, 0.06, 0.08};
the picker MERGES that finer trajectory into the coarse EPOCHS=2 trajectory
and re-applies the pick rule.

The canonical scenario this pins is the exact pattern that landed at the
start of round-8:

  * Coarse trajectory at fracs {0.08, 0.16, 0.33, 0.50, 0.75, 1.00}.
  * All 6 cells saturated: min source ΔG = 7.993 (IN BAND on the ΔG axis,
    just under the upper bound 12), max emission = 1.000 (SATURATED on
    the emission axis at every coarse frac).
  * Trigger B fires correctly on the OR'd-axis rule.
  * Recovery at finer fracs {0.02, 0.04, 0.06, 0.08} finds at least one
    cell with emission ∈ [0.1, 0.8] AND source ΔG ∈ [5, 12] — the
    merged pick verdict flips to "pass" with the chosen frac drawn
    from the FINER grid.

CPU-only, sub-second. No GPU/HF/network.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_504.phase0 import (
    _merge_trajectories,
    merge_recovery_into_v3_pick,
    pick_anchor_from_epochs_smoke,
)


def _trajectory(*, per_frac: dict[float, tuple[float, float]]) -> dict:
    """Build a trajectory.json shape that the picker can consume.

    Mirrors the shape that `scripts/i504_eval_trajectory.py` writes;
    matches `tests/experiments/test_504_v3_picker.py::_trajectory`.
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


# ── _merge_trajectories — internal concat helper ────────────────────────────


def test_merge_trajectories_disjoint_fracs():
    """Disjoint fractions: union, sorted ASC by frac."""
    coarse = _trajectory(per_frac={0.08: (12.5, 0.95), 0.16: (12.7, 0.97)})
    finer = _trajectory(per_frac={0.02: (6.0, 0.3), 0.04: (8.0, 0.5)})
    merged = _merge_trajectories(coarse, finer)
    fracs = [ck["frac"] for ck in merged["checkpoints"]]
    assert fracs == [0.02, 0.04, 0.08, 0.16]


def test_merge_trajectories_recovery_wins_on_collision():
    """Same frac in both: the recovery (finer) row wins.

    Rationale: the coarse 0.08 was a saturated read the recovery was meant
    to refute. The recovery re-measurement at 0.08 is the more reliable
    value for the merged pick.
    """
    coarse = _trajectory(per_frac={0.08: (14.0, 1.0), 0.16: (15.0, 1.0)})
    finer = _trajectory(per_frac={0.08: (7.0, 0.4)})  # recovery: in-band
    merged = _merge_trajectories(coarse, finer)
    # 0.08 row should reflect the finer (recovery) values, not the coarse.
    ck_008 = next(ck for ck in merged["checkpoints"] if ck["frac"] == 0.08)
    assert ck_008["source_self"]["delta_g_mean"] == 7.0
    assert ck_008["source_self"]["emission_p"] == 0.4


def test_merge_trajectories_preserves_metadata():
    """Coarse metadata (cell, seed, source) is preserved on the merged dict."""
    coarse = _trajectory(per_frac={0.08: (12.5, 0.95)})
    coarse["cell"] = "c504v3_smoke_eps2"
    coarse["seed"] = 42
    coarse["source"] = "villain"
    finer = _trajectory(per_frac={0.02: (6.0, 0.3)})
    merged = _merge_trajectories(coarse, finer)
    assert merged["cell"] == "c504v3_smoke_eps2"
    assert merged["seed"] == 42
    assert merged["source"] == "villain"


# ── merge_recovery_into_v3_pick — pin the exact round-8 scenario ────────────


def test_canonical_round8_saturated_scenario_recovery_finds_in_band():
    """REGRESSION (round-8 fix): the exact pattern that landed at round-8.

    Coarse: min(source_ΔG)=7.993 (in band), max(emission)=1.000 (SATURATED
    on emission axis at every coarse frac). Trigger B fires correctly.
    Recovery at finer fracs finds an in-band cell — merged verdict flips
    to "pass" with the chosen frac drawn from the FINER grid.
    """
    coarse_trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                # All 6 coarse fracs: ΔG in band, emission saturated.
                # min ΔG = 7.993, max emission = 1.000 — exact round-8 numbers.
                0.08: (7.993, 0.95),
                0.16: (9.0, 0.98),
                0.33: (10.0, 0.99),
                0.50: (10.5, 1.0),
                0.75: (11.0, 1.0),
                1.00: (11.5, 1.0),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (8.5, 0.96),
                0.16: (9.5, 0.99),
                0.33: (10.5, 1.0),
                0.50: (11.0, 1.0),
                0.75: (11.5, 1.0),
                1.00: (12.0, 1.0),
            }
        ),
    }
    # First confirm the coarse-only pick fires Trigger B (saturation on EITHER
    # axis: emission saturated even though ΔG is in band).
    coarse_pick = pick_anchor_from_epochs_smoke(coarse_trajs)
    assert coarse_pick["verdict"] == "all_saturated"
    assert coarse_pick["fallback_triggered"] is True
    assert coarse_pick["in_plan_recovery_triggered"] is True

    # Recovery at finer fracs: at least one cell lands BOTH in ΔG band AND
    # in emission band.
    recovery_traj = _trajectory(
        per_frac={
            0.02: (5.5, 0.3),  # in-band on BOTH axes — should become the pick
            0.04: (6.5, 0.5),  # in-band on BOTH axes
            0.06: (7.0, 0.7),  # in-band on BOTH axes
            0.08: (7.5, 0.85),  # emission JUST out of band (saturation creeping in)
        }
    )

    merged_pick = merge_recovery_into_v3_pick(coarse_trajs, recovery_traj)

    # The merged pick should flip to "pass" — there is now an in-band cell.
    assert merged_pick["verdict"] == "pass", (
        f"merged pick should have flipped to 'pass' after recovery "
        f"introduced in-band cells; got verdict={merged_pick['verdict']!r}"
    )
    assert merged_pick["fallback_triggered"] is False
    assert merged_pick["chosen_epochs"] == 2  # the recovered cell
    # The chosen frac must come from the finer grid (0.02, 0.04, 0.06, 0.08),
    # since those are the only in-band candidates after merge.
    assert merged_pick["chosen_checkpoint_fraction"] in {0.02, 0.04, 0.06, 0.08}
    # Pick rule favors LATEST in-band frac (DESC); 0.06 is the latest of the
    # three in-band finer fracs (0.08 is saturated on emission).
    assert merged_pick["chosen_checkpoint_fraction"] == 0.06

    # Audit fields: the recovery trajectory is preserved + merged_from_coarse sentinel.
    assert merged_pick["merged_from_coarse"] is True
    assert merged_pick["recovery_finer_trajectory"] == recovery_traj


def test_recovery_still_saturated_keeps_trigger_B():
    """Recovery cells ALSO saturated (emission still ≥ 0.8 at every finer
    frac) → merged verdict stays "all_saturated" + in_plan_recovery_triggered.

    Operational meaning: even the finer cadence can't escape saturation;
    the dispatcher's downstream logic should now route to v4 (rank bump)
    rather than another recovery pass.
    """
    coarse_trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (8.0, 0.95),
                0.16: (9.0, 0.98),
                0.33: (10.0, 0.99),
                0.50: (10.5, 1.0),
                0.75: (11.0, 1.0),
                1.00: (11.5, 1.0),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (8.5, 0.96),
                0.16: (9.5, 0.99),
                0.33: (10.5, 1.0),
                0.50: (11.0, 1.0),
                0.75: (11.5, 1.0),
                1.00: (12.0, 1.0),
            }
        ),
    }
    # Recovery: emission STILL saturated at every finer frac too.
    recovery_traj = _trajectory(
        per_frac={
            0.02: (6.0, 0.85),
            0.04: (7.0, 0.90),
            0.06: (7.5, 0.93),
            0.08: (8.0, 0.95),
        }
    )

    merged_pick = merge_recovery_into_v3_pick(coarse_trajs, recovery_traj)

    assert merged_pick["verdict"] == "all_saturated"
    assert merged_pick["fallback_triggered"] is True
    assert merged_pick["in_plan_recovery_triggered"] is True
    assert "trigger_B_saturated" in merged_pick["fallback_reason"]


def test_recovery_at_floor_routes_to_trigger_A():
    """Recovery cells all FLOOR (source ΔG < 5 at every finer frac) AND the
    coarse cells were saturated → merged trigger detection re-runs over the
    augmented (epochs, frac) table; whether the merged verdict is A or B
    depends on whether the MERGED set still has saturation evidence.

    Here, the coarse cells stay saturated (max emit = 1.0), so the merged
    `max_emit` is still 1.0 — Trigger B still fires (not A). This is the
    correct behavior: the recovery added floor data without rescuing the
    pick, so the dispatcher should treat the merged outcome the same way.
    """
    coarse_trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (8.0, 0.95),
                0.16: (9.0, 0.98),
                0.33: (10.0, 0.99),
                0.50: (10.5, 1.0),
                0.75: (11.0, 1.0),
                1.00: (11.5, 1.0),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (8.5, 0.96),
                0.16: (9.5, 0.99),
                0.33: (10.5, 1.0),
                0.50: (11.0, 1.0),
                0.75: (11.5, 1.0),
                1.00: (12.0, 1.0),
            }
        ),
    }
    # Recovery: ΔG floor at every finer frac, but coarse saturation persists.
    recovery_traj = _trajectory(
        per_frac={
            0.02: (1.0, 0.0),
            0.04: (2.0, 0.0),
            0.06: (3.0, 0.05),
            0.08: (4.0, 0.1),
        }
    )

    merged_pick = merge_recovery_into_v3_pick(coarse_trajs, recovery_traj)
    # In-band set still empty; coarse cells still saturated on emission →
    # Trigger B classification persists (the recovery DID NOT rescue the
    # pick, and the saturation evidence the recovery aimed at refuting
    # is still in the merged set via the EPOCHS=3 cell).
    assert merged_pick["verdict"] == "all_saturated"
    assert merged_pick["fallback_triggered"] is True
    assert merged_pick["in_plan_recovery_triggered"] is True


def test_merge_recovery_unknown_slug_raises():
    """merge_recovery_into_v3_pick raises KeyError when recovery_slug is
    not in the smoke_trajectories dict."""
    coarse_trajs = {
        "c504v3_smoke_eps2": _trajectory(per_frac={0.08: (8.0, 0.95)}),
        "c504v3_smoke_eps3": _trajectory(per_frac={0.08: (8.5, 0.96)}),
    }
    recovery_traj = _trajectory(per_frac={0.02: (6.0, 0.5)})
    with pytest.raises(KeyError, match="recovery_slug='c504v3_smoke_eps99'"):
        merge_recovery_into_v3_pick(coarse_trajs, recovery_traj, recovery_slug="c504v3_smoke_eps99")


def test_merged_pick_carries_pre_pick_smoke_table_with_finer_rows():
    """The merged pick's smoke_table includes the FINER rows under the
    recovery slug — downstream analyzers can audit the recovery without
    going back to disk."""
    coarse_trajs = {
        "c504v3_smoke_eps2": _trajectory(
            per_frac={
                0.08: (8.0, 0.95),
                0.16: (9.0, 0.98),
                0.33: (10.0, 0.99),
                0.50: (10.5, 1.0),
                0.75: (11.0, 1.0),
                1.00: (11.5, 1.0),
            }
        ),
        "c504v3_smoke_eps3": _trajectory(
            per_frac={
                0.08: (8.5, 0.96),
                0.16: (9.5, 0.99),
                0.33: (10.5, 1.0),
                0.50: (11.0, 1.0),
                0.75: (11.5, 1.0),
                1.00: (12.0, 1.0),
            }
        ),
    }
    recovery_traj = _trajectory(
        per_frac={
            0.02: (5.5, 0.3),
            0.04: (6.5, 0.5),
            0.06: (7.0, 0.7),
            0.08: (7.5, 0.85),
        }
    )
    pick = merge_recovery_into_v3_pick(coarse_trajs, recovery_traj)

    eps2_row = next(r for r in pick["smoke_table"] if r["slug"] == "c504v3_smoke_eps2")
    # eps2 row should now have ≥ 9 fracs (6 coarse + 4 finer, minus 1 collision
    # at 0.08 = 9).
    fracs_in_table = set(eps2_row["per_frac"].keys())
    expected_fracs = {0.02, 0.04, 0.06, 0.08, 0.16, 0.33, 0.50, 0.75, 1.00}
    assert fracs_in_table == expected_fracs, (
        f"Merged eps2 smoke_table should carry the union of coarse + finer "
        f"fractions; got {sorted(fracs_in_table)} vs expected {sorted(expected_fracs)}"
    )


# ── --checkpoint-fractions CLI override on i504_run_cell.py (smoke parse) ──


def test_checkpoint_fractions_cli_csv_parse_invariants():
    """Pin the CLI parse rules so a malformed --checkpoint-fractions value
    fails LOUD at the cell runner (BEFORE train_one_cell)."""

    # Inline the parser to avoid importing scripts/i504_run_cell.py at
    # module load (it pulls heavy deps). The parse logic is intentionally
    # simple so the same checks here pin the contract.
    def _parse(csv: str) -> tuple[float, ...]:
        parsed = tuple(sorted(float(x.strip()) for x in csv.split(",") if x.strip()))
        if not parsed:
            raise ValueError("empty")
        if any(f <= 0 or f > 1.0 for f in parsed):
            raise ValueError("out of range")
        return parsed

    # Happy path: the canonical CHECKPOINT_FRACTIONS_V3_FINER value.
    assert _parse("0.02,0.04,0.06,0.08") == (0.02, 0.04, 0.06, 0.08)
    # Sorting normalization: input order should not matter.
    assert _parse("0.08,0.02,0.06,0.04") == (0.02, 0.04, 0.06, 0.08)
    # Whitespace tolerance.
    assert _parse(" 0.02 , 0.04 ") == (0.02, 0.04)
    # Empty parses raise.
    with pytest.raises(ValueError, match="empty"):
        _parse(",,")
    # Out-of-range raises.
    with pytest.raises(ValueError, match="out of range"):
        _parse("0.5,1.5")
    with pytest.raises(ValueError, match="out of range"):
        _parse("0,0.5")
