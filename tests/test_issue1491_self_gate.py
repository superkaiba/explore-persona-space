"""#1491 first-chunk self-gate predicate (plan §7 Gate 1) — crash-fix pins.

Two-round history, both pinned here against the REAL production functions:

ROUND 1 (commit 1c8b46d28a, line 741): ``(r2_fit - r2_null) > 0.05 and
abs(r2_null) < 0.05``. The two-sided bound is unsatisfiable against the gate's
OWN null construction — a shuffle-FIT null (beta refit on permuted (X, Y)
pairings, scored on val) has expected R² ~ -1, not ~ 0 (only a MEAN-PREDICTOR
null sits at 0). All 8 train_25k shards aborted on the first production run
(epm:failure v3, 2026-08-05) with r2_null in [-0.990, -0.833] and gap ~1.0.

ROUND 2 (commit ccc650f42e): one-sided cap + a BINDING constant floor
``-3.0 < r2_null < 0.05``. The floor is regime-WRONG: how deep a LEGITIMATE
shuffle-fit null sits is conditioning-dependent — at the fixed 2,000-row
mid-shard trigger (n_train=1600), n_train/h spans 1.79 (0.5B, h=896) down to
0.31 (14B/32B, h≈5120), and near the interpolation threshold n≈h the
legitimate null risk PEAKS (measured below on healthy synthetic data:
r2_null = -50.96 at n/h = 1.0, -21.98 at 0.96, -3.16 at 0.75) — so any fixed
floor false-aborts some rung, re-creating the round-1 incident class.

ROUND 3 (this file's pins): the GAP test stays binding + byte-unchanged (the
plan-registered gate — broken capture makes fit ≈ null so gap ≈ 0 at ANY
conditioning); the one-sided predictive-null cap ``r2_null < 0.05`` stays
binding (regime-INDEPENDENT: a shuffle-fit null's expectation is <= 0 in every
regime); the floor is demoted to an ADVISORY log line; and the diag discloses
conditioning per CLAUDE.md #1701 (``h``, ``n_train_over_h``,
``under_determined``) — an under-determined fit is disclosed, never silent.

Offline by construction: pure-numpy/torch synthetic rows, no Hub fetch, no GPU.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1491_ladder_generate_capture as D  # noqa: E402


def _pre_fix_predicate(r2_fit: float, r2_null: float) -> bool:
    """Round-1 predicate, verbatim (commit 1c8b46d28a, line ~741) — the
    two-sided |null| bound a shuffle-fit null can never satisfy."""
    return (r2_fit - r2_null) > 0.05 and abs(r2_null) < 0.05


def _round2_predicate(r2_fit: float, r2_null: float) -> bool:
    """Round-2 predicate, verbatim (commit ccc650f42e) — one-sided cap plus
    the BINDING -3.0 floor that false-aborts near-threshold rungs."""
    return (r2_fit - r2_null) > 0.05 and (-3.0 < r2_null < 0.05)


# The three representative OBSERVED shard diagnostics from the aborted
# production run (epm:failure v3 diagnosis; n_train=1600, n_val=400, h=896,
# lam=1.0 — consistent across all 8 train_25k shards).
OBSERVED_DIAGNOSTICS = [
    (0.094, -0.990),
    (0.108, -0.916),
    (0.158, -0.833),
]


def test_observed_production_diagnostics_pass():
    """Replay of the actual aborted-shard values through the real predicate."""
    for r2_fit, r2_null in OBSERVED_DIAGNOSTICS:
        assert D._self_gate_predicate(r2_fit, r2_null) is True, (r2_fit, r2_null)


def test_observed_diagnostics_fail_pre_fix_predicate():
    """Round-1 discriminator: identical inputs, pre-fix predicate False on
    every shard — the incident class could never have passed pre-fix."""
    for r2_fit, r2_null in OBSERVED_DIAGNOSTICS:
        assert _pre_fix_predicate(r2_fit, r2_null) is False, (r2_fit, r2_null)


def test_healthy_gap_under_determined_null_passes_now_failed_round2():
    """Round-3 discriminator: a healthy-gap rung whose legitimate
    deep-negative shuffle-fit null sits below the round-2 floor (e.g. 7B:
    n_train=1600 < h=3584 at the trigger; 1.5B: n/h ~ 1.04) must PASS now
    and provably FAILED the round-2 binding-floor predicate."""
    cases = [
        (-7.0, -8.0),  # the brief's shape: gap 1.0, null ~ -8
        (0.47, -50.96),  # the MEASURED legitimate null at n/h = 1.0 (e2e below)
        (0.73, -21.98),  # the MEASURED legitimate null at n/h = 0.96 (e2e below)
    ]
    for r2_fit, r2_null in cases:
        assert D._self_gate_predicate(r2_fit, r2_null) is True, (r2_fit, r2_null)
        assert _round2_predicate(r2_fit, r2_null) is False, (r2_fit, r2_null)


def test_predictive_null_fails():
    """The one-sided cap stays BINDING (regime-independent — a shuffle-fit
    null's expectation is <= 0 in every conditioning): a predictive null
    (~0.5) flags leakage/degeneracy and must fail."""
    # Small gap AND predictive null: fails both conditions.
    assert D._self_gate_predicate(0.52, 0.50) is False
    # Decisive gap but predictive null: the cap alone must reject.
    assert D._self_gate_predicate(0.90, 0.50) is False


def test_deep_negative_null_no_longer_aborts():
    """The floor is advisory: a deep-negative null with a healthy gap PASSES
    the binding predicate. (Round 2 called -50 'pathological'; the n/h=1.0
    e2e below measures a LEGITIMATE healthy-data null of -50.96 at the
    interpolation threshold — no constant separates the two, which is
    exactly why the floor cannot be a criterion.)"""
    assert D._self_gate_predicate(0.1, -50.0) is True
    assert D._self_gate_predicate(0.1, D.SELF_GATE_NULL_FLOOR) is True
    # But a deep-negative null with NO gap still fails — on the gap test,
    # the plan-registered criterion (broken capture: fit ≈ null => gap ≈ 0).
    assert D._self_gate_predicate(-50.0, -50.0) is False


def test_gap_condition_unchanged():
    """The gap arm is byte-unchanged: strict > 0.05."""
    assert D._self_gate_predicate(-0.81, -0.85) is False  # gap ~0.04 <= 0.05
    assert D._self_gate_predicate(-0.79, -0.85) is True  # gap ~0.06 > 0.05


def test_under_500_rows_auto_pass():
    """The tiny-split auto-pass (val_400 / test_1000 / tierB_3600 per-shard
    slices) is untouched."""
    passed, diag = D._first_chunk_self_gate([{} for _ in range(499)], 0)
    assert passed is True
    assert diag.get("skipped") is True


def _rows_from(X: np.ndarray, Y: np.ndarray) -> list[dict]:
    """Rows shaped as the capture loop builds them: per-row lists of
    per-layer tensors, indexed by layer_index_primary."""
    return [
        {"cx_last": [torch.from_numpy(X[i])], "v_x": [torch.from_numpy(Y[i])]}
        for i in range(len(X))
    ]


def _synthetic(n: int, h: int, seed: int, x_scale: float, noise: float) -> list[dict]:
    """Strong-linear-map synthetic rows at a chosen conditioning n/h."""
    rng = np.random.default_rng(seed)
    X = (x_scale * rng.standard_normal((n, h))).astype(np.float32)
    W = (rng.standard_normal((h, h)) / np.sqrt(h)).astype(np.float32)
    Y = (X @ W + noise * rng.standard_normal((n, h))).astype(np.float32)
    return _rows_from(X, Y)


def _assert_disclosure(diag: dict, n_train: int, h: int) -> None:
    """The #1701 conditioning-disclosure contract: fields present, correct,
    and self-consistent."""
    assert diag["n_train"] == n_train
    assert diag["h"] == h
    assert diag["n_train_over_h"] == round(n_train / h, 3)
    assert diag["under_determined"] is (n_train < h)
    assert diag["null_floor_advisory"] == D.SELF_GATE_NULL_FLOOR
    assert diag["null_floor_breached"] is (diag["r2_null"] <= D.SELF_GATE_NULL_FLOOR)


def test_full_gate_healthy_wellconditioned_passes_no_advisory(caplog):
    """e2e, well-over-determined (n/h = 60): passes, disclosure correct,
    under_determined False, NO advisory warning emitted."""
    rows = _synthetic(n=600, h=8, seed=0, x_scale=1.0, noise=0.1)
    with caplog.at_level(logging.WARNING, logger="issue1491_ladder_generate_capture"):
        passed, diag = D._first_chunk_self_gate(rows, 0)
    assert passed is True, diag
    assert diag["gap"] > 0.05, diag
    _assert_disclosure(diag, n_train=480, h=8)
    assert diag["under_determined"] is False
    assert diag["null_floor_breached"] is False, diag
    assert "ADVISORY" not in caplog.text


def test_full_gate_scale05_like_conditioning_passes():
    """e2e at the incident rung's conditioning, scaled down: n_train/h =
    480/256 ~ 1.9 (production scale05 trigger: 1600/896 ~ 1.8). The
    shuffle-fit null lands ~ -1 (the observed regime) and the gate passes."""
    rows = _synthetic(n=600, h=256, seed=1, x_scale=5.0, noise=0.5)
    passed, diag = D._first_chunk_self_gate(rows, 0)
    assert diag["r2_null"] < -0.3, diag  # shuffle-fit, not mean-predictor
    assert passed is True, diag
    _assert_disclosure(diag, n_train=480, h=256)
    assert diag["under_determined"] is False
    # Round-1 oracle: this healthy computation could never pass pre-fix.
    assert _pre_fix_predicate(diag["r2_fit"], diag["r2_null"]) is False


def test_full_gate_near_threshold_breach_is_advisory_not_abort(caplog):
    """e2e near the interpolation threshold (n_train/h = 480/500 = 0.96 —
    the 1.5B rung's regime, 1600/1536 ~ 1.04): HEALTHY data, healthy gap,
    yet the legitimate shuffle-fit null lands ~ -22, deep below the round-2
    floor. The gate must PASS, disclose the conditioning, flag + WARN the
    advisory breach — and the round-2 predicate provably false-aborts it."""
    rows = _synthetic(n=600, h=500, seed=7, x_scale=5.0, noise=0.5)
    with caplog.at_level(logging.WARNING, logger="issue1491_ladder_generate_capture"):
        passed, diag = D._first_chunk_self_gate(rows, 0)
    assert passed is True, diag
    assert diag["gap"] > 0.05, diag
    _assert_disclosure(diag, n_train=480, h=500)
    assert diag["under_determined"] is True
    # Deep legitimate breach (measured -21.98; wide margin below the floor).
    assert diag["r2_null"] < D.SELF_GATE_NULL_FLOOR - 5.0, diag
    assert diag["null_floor_breached"] is True, diag
    # The advisory fired as a LOG LINE, not an abort.
    assert "ADVISORY" in caplog.text
    assert "NOT a pass/fail criterion" in caplog.text
    # The real-data round-3 discriminator: round 2 would have false-aborted
    # this healthy rung; the round-1 predicate rejects it too.
    assert _round2_predicate(diag["r2_fit"], diag["r2_null"]) is False
    assert _pre_fix_predicate(diag["r2_fit"], diag["r2_null"]) is False
