"""Tests for scripts/issue1112_debiased_cosine.py (batched paired half-draws)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "issue1112_debiased_cosine", REPO_ROOT / "scripts" / "issue1112_debiased_cosine.py"
)
mod = importlib.util.module_from_spec(_SPEC)
sys.modules["issue1112_debiased_cosine"] = mod
_SPEC.loader.exec_module(mod)


def _toy_clouds(n_q: int = 4, n_ctx: int = 3, d: int = 8, seed: int = 0):
    """Two small clouds with a shared question x context row grid."""
    rng = np.random.default_rng(seed)
    n = n_q * n_ctx
    question_idx = np.tile(np.arange(n_q), n_ctx)
    mu_a = rng.standard_normal(d)
    mu_b = mu_a + 0.5 * rng.standard_normal(d)
    cloud_a = mu_a + 0.3 * rng.standard_normal((n, d))
    cloud_b = mu_b + 0.3 * rng.standard_normal((n, d))
    return cloud_a, cloud_b, question_idx


def test_masks_exact_half_without_replacement():
    masks = mod.half_partition_masks(12, 50, seed=1)
    assert masks.shape == (50, 12)
    assert (masks.sum(axis=1) == 6).all()  # exact halves, no replacement by construction


def test_question_aligned_masks_are_aligned():
    _, _, question_idx = _toy_clouds()
    masks = mod.half_partition_masks(12, 40, seed=2, question_idx=question_idx)
    assert (masks.sum(axis=1) == 6).all()
    for mask in masks:  # every question id fully inside or outside half A
        for q in np.unique(question_idx):
            rows = mask[question_idx == q]
            assert rows.all() or (~rows).all()


def test_identical_clouds_cross_cosine_is_one():
    cloud_a, _, _ = _toy_clouds()
    masks = mod.half_partition_masks(12, 30, seed=3)
    draws = mod.batched_half_cosines(cloud_a, cloud_a.copy(), masks)
    # SAME rows in both cells + identical clouds => identical half means.
    assert np.allclose(draws["cross"], 1.0)
    # split-half references stay < 1 under within-cell noise.
    assert (draws["ref_a"] < 1.0).all()


def test_batched_matches_serial_reference():
    cloud_a, cloud_b, _ = _toy_clouds(seed=4)
    masks = mod.half_partition_masks(12, 5, seed=5)
    batched = mod.batched_half_cosines(cloud_a, cloud_b, masks)
    serial = mod.serial_half_cosines_reference(cloud_a, cloud_b, masks)
    for key in ("cross", "ref_a", "ref_b", "corrected"):
        np.testing.assert_allclose(batched[key], serial[key], rtol=0, atol=1e-12)


def test_analyze_pair_summary_shape_and_determinism():
    cloud_a, cloud_b, question_idx = _toy_clouds(seed=6)
    masks = mod.half_partition_masks(12, 64, seed=7, question_idx=question_idx)
    e1 = mod.analyze_pair(cloud_a, cloud_b, masks)
    e2 = mod.analyze_pair(cloud_a, cloud_b, masks)
    assert e1 == e2  # deterministic given the same masks
    assert e1["m"] == 6
    for stat in ("cross", "ref_a", "ref_b", "corrected"):
        s = e1["summary"][stat]
        assert set(s["quantiles"]) == {str(q) for q in mod.QUANTILES}
        assert 0.0 <= s["frac_below_cutoff"] <= 1.0
        assert len(e1["draws"][stat]) == 64
    d = e1["paired_deltas"]["ref_min_minus_cross"]
    assert 0.0 <= d["frac_positive"] <= 1.0


def test_summarize_drops_nans_never_coerces():
    vals = np.array([0.5, np.nan, 0.9, np.nan])
    s = mod.summarize(vals)
    assert s["n_dropped_nan"] == 2
    assert s["mean"] == pytest.approx(0.7)
