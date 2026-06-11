# ruff: noqa: RUF002  # Greek ΔG intentional
"""Task #534 round-2 — source-manifest adapter-applied guard unit tests.

Pins ``assert_source_delta_g_matches_manifest`` (eval_guard.py) against the
round-1 incident: vLLM served the FIRST loaded adapter at every fraction
(``lora_int_id`` reuse), the eval read a flat ≈0 trajectory, and neither the
B-matrix guard (structural — the dirs WERE trained) nor the byte-identical
guard (requires g == b exactly; a wrong-but-real adapter gives g ≠ b) could
catch it. The new guard cross-checks the eval's own on-policy source-self ΔG
against the selector manifest's teacher-forced read of the SAME snapshot.

Numbers in these tests are the real #534 c504v3_near_seed42 values:
manifest expectations {0.25: 0.033, 0.50: 0.324, 0.75: 2.204, 1.00: 6.277};
the broken eval read 0.00–0.07 at every fraction; #530's on-policy read at
the stop step was 5.82 vs 6.28 teacher-forced (~0.5-nat method noise).
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
    SourceDeltaGManifestMismatchError,
    assert_source_delta_g_matches_manifest,
)


def test_pass_within_tolerance_at_final_fraction():
    """#530-calibrated method noise (~0.5 nat) passes at the final fraction."""
    diag = assert_source_delta_g_matches_manifest(
        cell_label="c504v3_near_seed42_frac1.0",
        frac=1.0,
        eval_delta_g_nats=5.82,
        expected_delta_g_nats=6.277,
        is_final_frac=True,
        band_stop_fired=True,
    )
    assert diag["guard_verdict"] == "pass"
    assert diag["disagreement_nats"] == pytest.approx(0.457, abs=1e-3)


def test_round1_regression_raises_at_final_fraction():
    """THE round-1 bug: eval reads ≈0 at frac=1.00 while the manifest says 6.28."""
    with pytest.raises(SourceDeltaGManifestMismatchError, match="FINAL fraction"):
        assert_source_delta_g_matches_manifest(
            cell_label="c504v3_near_seed42_frac1.0",
            frac=1.0,
            eval_delta_g_nats=0.04,
            expected_delta_g_nats=6.277,
            is_final_frac=True,
            band_stop_fired=True,
        )


def test_nonfinal_mismatch_warns_but_does_not_raise():
    """frac=0.75 under the round-1 bug: 0.01 vs 2.204 — recorded, not fatal."""
    diag = assert_source_delta_g_matches_manifest(
        cell_label="c504v3_near_seed42_frac0.75",
        frac=0.75,
        eval_delta_g_nats=0.01,
        expected_delta_g_nats=2.204,
        is_final_frac=False,
        band_stop_fired=True,
    )
    assert diag["guard_verdict"] == "warn_nonfinal_mismatch"


def test_band_stop_floor_clause_fires_without_expected():
    """Manifest expectation missing (--skip-source-trajectory) but band fired:
    a < 1-nat final read is still physically inconsistent → raise."""
    with pytest.raises(SourceDeltaGManifestMismatchError, match="band-stop fired"):
        assert_source_delta_g_matches_manifest(
            cell_label="c504v3_far_seed137_frac1.0",
            frac=1.0,
            eval_delta_g_nats=0.4,
            expected_delta_g_nats=None,
            is_final_frac=True,
            band_stop_fired=True,
        )


def test_no_expected_nonfinal_passes():
    """No expectation + non-final fraction: nothing to check → pass."""
    diag = assert_source_delta_g_matches_manifest(
        cell_label="c504v3_near_seed42_frac0.25",
        frac=0.25,
        eval_delta_g_nats=0.02,
        expected_delta_g_nats=None,
        is_final_frac=False,
        band_stop_fired=True,
    )
    assert diag["guard_verdict"] == "pass"
    assert diag["disagreement_nats"] is None


def test_no_band_stop_final_floor_does_not_fire():
    """Cell never band-stopped (flagged in the manifest): the final-fraction
    floor clause must NOT fire — a genuinely-weak implant is a measurement."""
    diag = assert_source_delta_g_matches_manifest(
        cell_label="c504v3_default_only_seed42_frac1.0",
        frac=1.0,
        eval_delta_g_nats=0.4,
        expected_delta_g_nats=None,
        is_final_frac=True,
        band_stop_fired=False,
    )
    assert diag["guard_verdict"] == "pass"


def test_small_fraction_agreement_passes():
    """frac=0.25 with the adapter ACTUALLY applied: 0.02 vs 0.033 agrees."""
    diag = assert_source_delta_g_matches_manifest(
        cell_label="c504v3_near_seed42_frac0.25",
        frac=0.25,
        eval_delta_g_nats=0.02,
        expected_delta_g_nats=0.033,
        is_final_frac=False,
        band_stop_fired=True,
    )
    assert diag["guard_verdict"] == "pass"
