"""#1491 first-chunk self-gate predicate (plan §7 Gate 1) — crash-fix pin.

The pre-fix predicate was ``(r2_fit - r2_null) > 0.05 and abs(r2_null) < 0.05``.
Its second condition is unsatisfiable against the gate's OWN null construction:
the null is a shuffle-FIT null (beta refit on permuted (X, Y) pairings, scored
on val), whose expected R² is ~ -1, not ~ 0 — ``abs(...) < 0.05`` encodes a
MEAN-PREDICTOR null's expectation (yhat == y_mu => SSE == SST). The gate
therefore aborted all 8 ``train_25k`` shards on their first production run
(epm:failure v3, 2026-08-05) with r2_null in [-0.990, -0.833] and gap ~1.0 —
condition 1 passed twenty-fold, condition 2 could never pass.

Fixed predicate (``_self_gate_predicate``): gap > 0.05 (byte-unchanged) AND
``SELF_GATE_NULL_FLOOR < r2_null < 0.05`` (one-sided cap + pathology floor).
These tests pin the table against the REAL production functions, replay the
ACTUAL observed shard diagnostics, and demonstrate the pre-fix predicate FAILS
those observed values (the pre/post discriminator).

Offline by construction: pure-numpy/torch synthetic rows, no Hub fetch, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1491_ladder_generate_capture as D  # noqa: E402


def _pre_fix_predicate(r2_fit: float, r2_null: float) -> bool:
    """The predicate as shipped pre-fix (commit 1c8b46d28a, line ~741).

    Kept verbatim as the regression oracle: two-sided |null| bound that a
    shuffle-fit null can never satisfy.
    """
    return (r2_fit - r2_null) > 0.05 and abs(r2_null) < 0.05


# The three representative OBSERVED shard diagnostics from the aborted
# production run (epm:failure v3 diagnosis; n_train=1600, n_val=400, h=896,
# lam=1.0 — consistent across all 8 train_25k shards).
OBSERVED_DIAGNOSTICS = [
    (0.094, -0.990),
    (0.108, -0.916),
    (0.158, -0.833),
]


def test_observed_production_diagnostics_now_pass():
    """Replay of the actual aborted-shard values through the real predicate."""
    for r2_fit, r2_null in OBSERVED_DIAGNOSTICS:
        assert D._self_gate_predicate(r2_fit, r2_null) is True, (r2_fit, r2_null)


def test_observed_diagnostics_fail_pre_fix_predicate():
    """The discriminator: same inputs, pre-fix predicate False on every shard.

    Proves the fix flips exactly the incident class (and that the incident
    class could never have passed pre-fix).
    """
    for r2_fit, r2_null in OBSERVED_DIAGNOSTICS:
        assert _pre_fix_predicate(r2_fit, r2_null) is False, (r2_fit, r2_null)


def test_predictive_null_fails():
    """A genuinely predictive null (~0.5) must FAIL — shuffled pairings should
    carry no signal; a predictive null flags leakage/degeneracy."""
    # Small gap AND predictive null: fails both conditions.
    assert D._self_gate_predicate(0.52, 0.50) is False
    # Decisive gap but predictive null: the one-sided cap alone must reject —
    # proves the cap still does real work post-fix.
    assert D._self_gate_predicate(0.90, 0.50) is False


def test_pathological_null_fails_on_floor():
    """gap = 50.1 >> 0.05, but the null is numerically pathological — the new
    floor (not the gap condition) must reject it."""
    assert D._self_gate_predicate(0.1, -50.0) is False


def test_floor_boundary_is_strict():
    assert D.SELF_GATE_NULL_FLOOR == -3.0
    # Exactly at the floor: FAIL (strict inequality).
    assert D._self_gate_predicate(0.1, D.SELF_GATE_NULL_FLOOR) is False
    # Just inside the floor: PASS (gap ~3.04 >> 0.05).
    assert D._self_gate_predicate(0.1, -2.99) is True


def test_gap_condition_unchanged():
    """The gap arm is byte-unchanged: strict > 0.05, evaluated against a
    healthy shuffle-fit-shaped null."""
    assert D._self_gate_predicate(-0.81, -0.85) is False  # gap ~0.04 <= 0.05
    assert D._self_gate_predicate(-0.79, -0.85) is True  # gap ~0.06 > 0.05


def test_under_500_rows_auto_pass():
    """The tiny-split auto-pass (val_400 / test_1000 / tierB_3600 per-shard
    slices) is untouched by the fix."""
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


def test_full_gate_healthy_wellconditioned_rows_pass():
    """End-to-end through the REAL _first_chunk_self_gate: well-conditioned
    regime (n_train >> h), strong linear map => high fit R², null near 0.
    Confirms the refactor did not break the fit/null computation."""
    rng = np.random.default_rng(0)
    n, h = 600, 8
    X = rng.standard_normal((n, h)).astype(np.float32)
    W = rng.standard_normal((h, h)).astype(np.float32)
    Y = (X @ W + 0.1 * rng.standard_normal((n, h))).astype(np.float32)
    passed, diag = D._first_chunk_self_gate(_rows_from(X, Y), 0)
    assert passed is True, diag
    assert diag["gap"] > 0.05, diag
    assert D.SELF_GATE_NULL_FLOOR < diag["r2_null"] < 0.05, diag


def test_full_gate_production_like_conditioning_passes():
    """End-to-end in the INCIDENT's conditioning regime, scaled down:
    n_train/h = 480/256 ~ 1.9 (production trigger: 1600/896 ~ 1.8), lam=1.0
    negligible vs feature scale. The shuffle-FIT null lands deep negative
    (the regime the pre-fix two-sided bound could never pass) and the gate
    must now PASS on strong real signal."""
    rng = np.random.default_rng(1)
    n, h = 600, 256
    X = (5.0 * rng.standard_normal((n, h))).astype(np.float32)
    W = (rng.standard_normal((h, h)) / np.sqrt(h)).astype(np.float32)
    Y = (X @ W + 0.5 * rng.standard_normal((n, h))).astype(np.float32)
    passed, diag = D._first_chunk_self_gate(_rows_from(X, Y), 0)
    # The shuffle-fit (NOT mean-predictor) regime: null well below 0 …
    assert diag["r2_null"] < -0.3, diag
    # … yet above the pathology floor, and the gate passes.
    assert diag["r2_null"] > D.SELF_GATE_NULL_FLOOR, diag
    assert passed is True, diag
    # Pre-fix, this same healthy computation could never pass.
    assert _pre_fix_predicate(diag["r2_fit"], diag["r2_null"]) is False
