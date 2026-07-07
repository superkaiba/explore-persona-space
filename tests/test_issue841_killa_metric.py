# ruff: noqa: E402
"""KILL-A robust acceptance metric for the issue #841 scaling-capture round.

The stored parent cx_last was computed on the parent's H100; a re-capture on an A100
carries ~0.5%-scale bf16 cross-hardware noise (different batch-summation order) that is
inherent + harmless. The old max-ELEMENTWISE rel-error gate divided that noise by
near-zero stored elements and exploded (att-3: cosine 0.9998+ but elementwise rel 3.1e5),
FAILing a correct capture. The crash-fix (round 13) replaced it with a robust triple —
cosine ≥ 0.999, norm-rel ≤ 2e-2, norm-ratio ∈ [0.99, 1.01]. These tests pin the invariant:

  * a bf16-noise-perturbed probe (0.5% elementwise noise + sign flips on near-zero
    elements — exactly the att-3 signature) PASSES, even though its elementwise rel_err
    explodes (the diagnostic the old gate used);
  * a wrong-position probe (unrelated row) FAILS the cosine floor;
  * a magnitude-rescaled probe (direction intact) FAILS norm-rel + norm-ratio.

They exercise the LIVE dispatched metric `issue841_scaling_capture._probe_accept`
(the one `kill_a_spot_gate` calls per probe), per the "verification gates test the live
dispatched path" rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue841_scaling_capture as CAP
import numpy as np


def _stored_with_near_zeros(seed, n_layers=28, hidden=3584, near_zero_frac=0.05):
    """A realistic (L, H) stored row with a fraction of near-zero elements planted
    (the divide-by-~0 that blew up the old elementwise metric). Returns (stored, idx)."""
    rng = np.random.default_rng(seed)
    stored = rng.standard_normal((n_layers, hidden)).astype(np.float32)
    flat = stored.reshape(-1)
    idx = rng.choice(flat.size, size=int(flat.size * near_zero_frac), replace=False)
    flat[idx] = (rng.standard_normal(idx.size) * 1e-5).astype(np.float32)
    return stored, idx


def test_bf16_noise_probe_passes():
    stored, near_zero_idx = _stored_with_near_zeros(0)
    rng = np.random.default_rng(1)
    got = stored + (0.005 * np.abs(stored) * rng.standard_normal(stored.shape)).astype(np.float32)
    # sign-flip the near-zero elements: bf16 cross-hardware noise dominates their tiny
    # magnitude, so their recaptured sign is effectively random (the att-3 signature).
    flat_got = got.reshape(-1)
    flat_got[near_zero_idx] = -flat_got[near_zero_idx]
    got = flat_got.reshape(stored.shape)

    m = CAP._probe_accept(stored, got)
    assert m["pass"] is True, m
    assert m["cosine"] >= CAP._KILLA_COS_FLOOR, m
    assert m["norm_rel"] <= CAP._KILLA_NORM_REL_TOL, m
    lo, hi = CAP._KILLA_NORM_RATIO_BAND
    assert lo <= m["norm_ratio"] <= hi, m
    # the OLD elementwise metric would have FAILED this correct capture (the whole bug):
    assert m["elem_rel_err"] > 1.0, m["elem_rel_err"]


def test_unrelated_row_fails_cosine():
    """An unrelated row (gross wrong position / layer / normalization bug) collapses cosine.

    NOTE: this is an UNRELATED row, NOT an adjacent off-by-one — an adjacent token stays
    highly correlated (cosine ~0.9998, above the floor) and is caught by the norm-rel leg
    instead, which the logged KILL-A adjacency audit measures on the live model.
    """
    stored, _ = _stored_with_near_zeros(0)
    rng = np.random.default_rng(99)
    got = rng.standard_normal(stored.shape).astype(np.float32)
    m = CAP._probe_accept(stored, got)
    assert m["pass"] is False, m
    assert m["cosine"] < CAP._KILLA_COS_FLOOR, m


def test_rescale_fails_norm_checks():
    """A magnitude rescale (direction intact) is caught by norm-rel + norm-ratio."""
    stored, _ = _stored_with_near_zeros(0)
    got = (stored * 1.05).astype(np.float32)
    m = CAP._probe_accept(stored, got)
    assert m["pass"] is False, m
    assert m["cosine"] >= CAP._KILLA_COS_FLOOR, m  # direction preserved
    assert m["norm_ratio"] > CAP._KILLA_NORM_RATIO_BAND[1], m  # magnitude corruption caught
    assert m["norm_rel"] > CAP._KILLA_NORM_REL_TOL, m
