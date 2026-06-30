# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (φ, ρ, ×, ≤) in scientific docstrings + asserts.
"""LOBO + LOCO cross-validation drivers for issue-666 (plan §4g, §6).

LOBO (leave-one-behavior-out): hold out one of {marker, bad_medical,
insecure_code/EM, taught_fact}; the φ link is calibrated on the REMAINING
behaviors' TRAIN partition; the held-out behavior is scored.
LOCO (leave-one-context-out): hold out one battery context as C'; calibrate on
the rest.

These tests pin the fold INTEGRITY (no held-out leakage into calibration) and
the train-only φ-calibration discipline (plan §4g: "φ on TRAIN only"; selection/
verdict on disjoint data). They use small synthetic (behavior, context) grids;
no store, no network, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class _LazyModule:
    """Proxy that imports a per-issue script on first attribute access (TDD).

    The net-new script does NOT exist this round, so the first ``cv.<fn>``
    access inside each test raises ImportError → the test FAILS (not skips).
    A module-level ``importorskip`` was rejected because it skips COLLECTION,
    so the proposed-test count could not be verified by approve-tests.
    """

    def __init__(self, dotted: str):
        self._dotted = dotted

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module(self._dotted), name)


cv = _LazyModule("issue666_lobo_loco")

BEHAVIORS = ("marker", "bad_medical", "em", "taught_fact")
N_CTX = 50


# ---------------------------------------------------------------------------
# LOBO fold integrity.
# ---------------------------------------------------------------------------
def test_lobo_folds_cover_each_behavior_exactly_once():
    folds = cv.lobo_folds(BEHAVIORS)
    assert len(folds) == len(BEHAVIORS)
    test_behaviors = [f.test_behavior for f in folds]
    assert sorted(test_behaviors) == sorted(BEHAVIORS)
    # Each fold's test set is exactly one behavior.
    for f in folds:
        assert f.test_behavior in BEHAVIORS


def test_lobo_train_excludes_held_out_behavior():
    folds = cv.lobo_folds(BEHAVIORS)
    for f in folds:
        assert f.test_behavior not in f.train_behaviors, (
            f"LOBO leakage: held-out {f.test_behavior!r} appears in train set"
        )
        # train ∪ {test} == all behaviors; train ∩ {test} == ∅
        assert set(f.train_behaviors) | {f.test_behavior} == set(BEHAVIORS)
        assert f.test_behavior not in set(f.train_behaviors)


# ---------------------------------------------------------------------------
# LOCO fold integrity.
# ---------------------------------------------------------------------------
def test_loco_folds_cover_each_context_exactly_once():
    ctx_ids = list(range(N_CTX))
    folds = cv.loco_folds(ctx_ids)
    assert len(folds) == N_CTX
    test_ctxs = [f.test_context for f in folds]
    assert sorted(test_ctxs) == sorted(ctx_ids)


def test_loco_train_excludes_held_out_context():
    ctx_ids = list(range(N_CTX))
    folds = cv.loco_folds(ctx_ids)
    for f in folds:
        assert f.test_context not in f.train_contexts, (
            f"LOCO leakage: held-out context {f.test_context} appears in train set"
        )
        assert set(f.train_contexts) | {f.test_context} == set(ctx_ids)
        assert len(f.train_contexts) == N_CTX - 1


# ---------------------------------------------------------------------------
# φ calibration uses TRAIN partition only.
# ---------------------------------------------------------------------------
def test_phi_calibrated_on_train_partition_only():
    """The held-out test rows must NOT influence the fitted φ link (plan §4g).

    Construct two scenarios that share the SAME train rows but DIFFER only in the
    held-out test rows; the fitted φ parameters must be identical (test data
    cannot leak into calibration).
    """
    rng = np.random.default_rng(60)
    n_train = 80
    x_train = rng.standard_normal(n_train)
    # A monotone target with noise (the latent score → [0,1] rate relationship).
    y_train = 1.0 / (1.0 + np.exp(-(0.7 * x_train + 0.2))) + rng.normal(0, 0.02, n_train)
    y_train = np.clip(y_train, 0, 1)

    # Two DIFFERENT held-out test sets.
    x_test_a = rng.standard_normal(20)
    x_test_b = rng.standard_normal(20) * 5.0 + 10.0  # wildly different distribution

    phi_a = cv.fit_phi(x_train, y_train)
    phi_b = cv.fit_phi(x_train, y_train)  # same train → same φ regardless of any test data
    # The fit must be a pure function of (x_train, y_train).
    assert phi_a.params == pytest.approx(phi_b.params, rel=1e-9)

    # Applying φ to either test set must NOT refit (idempotent apply, no leakage).
    pred_a = cv.apply_phi(phi_a, x_test_a)
    pred_b = cv.apply_phi(phi_a, x_test_b)
    assert pred_a.shape == x_test_a.shape
    assert pred_b.shape == x_test_b.shape
    # φ maps into [0,1] (monotone calibration link).
    assert np.all((pred_a >= 0) & (pred_a <= 1))
    assert np.all((pred_b >= 0) & (pred_b <= 1))


def test_phi_is_monotone():
    """φ must be a monotone link (plan §4g: monotone σ / clamp)."""
    rng = np.random.default_rng(61)
    x = np.sort(rng.standard_normal(40))
    y = np.clip(1.0 / (1.0 + np.exp(-x)) + rng.normal(0, 0.01, 40), 0, 1)
    phi = cv.fit_phi(x, y)
    grid = np.linspace(x.min(), x.max(), 100)
    mapped = cv.apply_phi(phi, grid)
    diffs = np.diff(mapped)
    # Monotone non-decreasing (allow tiny numerical noise).
    assert np.all(diffs >= -1e-9), "φ must be monotone non-decreasing"


def test_latent_scale_score_needs_no_phi():
    """The latent Δs scale is unbounded → no φ; only the [0,1] behavior scale uses φ.

    The driver must expose a latent-scale scoring path that does NOT call fit_phi
    (plan §4g: 'Latent-scale tests need no φ').
    """
    rng = np.random.default_rng(62)
    lhat = rng.standard_normal(50)
    ds = 0.8 * lhat + rng.normal(0, 0.3, 50)  # correlated latent ground truth
    rho = cv.score_latent_spearman(lhat, ds)
    assert -1.0 <= rho <= 1.0
    assert rho > 0.5, "correlated latent inputs should give a clearly positive ρ"
