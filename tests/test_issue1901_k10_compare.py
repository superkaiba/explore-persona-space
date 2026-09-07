"""Check paired uncertainty for the five-to-ten rollout extension."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue1901_k10_compare as C


def test_identical_endpoints_have_zero_paired_intervals():
    """Shared targets must have exactly zero contrast, even with row resampling."""
    rng = np.random.default_rng(7)
    n, d = 24, 5
    y = rng.normal(size=(n, d))
    original = y.copy()
    original[2] = original[1]
    targets = np.stack([y, y, y + rng.normal(size=y.shape) * 0.1])
    preds = y[None] + rng.normal(size=(3, n, d)) * 0.3
    samples = np.stack([np.arange(n), rng.integers(n, size=n)])
    counts = np.stack([np.bincount(r, minlength=n) for r in samples])
    rcounts = rng.multinomial(n - 1, np.ones(n - 1) / (n - 1), size=2)
    arrays = C.score_targets(targets, preds, original, lambda x: x, counts, rcounts)
    result = C.summarize(arrays)
    for arm in C.K.ARMS:
        delta = result["contrasts"]["K10_minus_existing_K5"][arm]
        for metric in [delta["r2"], *delta["retrieval"].values()]:
            assert metric == {"mean": 0.0, "ci95": [0.0, 0.0]}
    for ti, target in enumerate(targets):
        for ai, pred in enumerate(preds):
            for bi, rows in enumerate(samples):
                expected = C.K.F79._recon_point(pred[rows], target[rows])[0]
                np.testing.assert_allclose(arrays["r2_boot"][ti, ai, bi], expected, atol=1e-12)
