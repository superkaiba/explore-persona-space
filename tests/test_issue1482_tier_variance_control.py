"""Independent numerical checks of matching, ties, and cell eligibility."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT))
spec = importlib.util.spec_from_file_location(
    "tier_variance", SCRIPT / "issue1482_tier_variance_control.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_binary_batch_matches_explicit_pairs_with_ties_and_small_cells():
    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, (3, 240))
    y = rng.integers(0, 5, x.shape).astype(float)
    groups = rng.integers(0, 5, x.shape)
    groups[:, :130] = 0
    actual, details = module.binary_batch(x, y, groups, min_cell=120)
    expected, pairs = [], []
    for row in range(3):
        numerator = denominator = 0
        for group in np.unique(groups[row]):
            keep = groups[row] == group
            if keep.sum() < 120:
                continue
            pos = y[row, keep & (x[row] == 1)]
            neg = y[row, keep & (x[row] == 0)]
            differences = pos[:, None] - neg[None, :]
            numerator += (differences > 0).sum() + 0.5 * (differences == 0).sum()
            denominator += differences.size
        expected.append(numerator / denominator - 0.5)
        pairs.append(denominator)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(details["comparable_pairs"], pairs)


@pytest.mark.parametrize("bins", [1, 5, 10])
def test_quantile_edges_match_parent_with_tied_activity(bins):
    rng = np.random.default_rng(21)
    values = rng.integers(0, 19, (3, 500)).astype(float)
    actual = module.assign_strata([values], bins, values.shape)
    for i in range(3):
        expected = np.empty(500, dtype=int)
        for j, indices in enumerate(module.parent._strata(values[i], bins)):
            expected[indices] = j
        np.testing.assert_array_equal(actual[i], expected)


def test_empty_comparison_is_rejected():
    with pytest.raises(ValueError, match="no comparable pairs"):
        module.binary_batch(
            np.ones((1, 150)), np.arange(150)[None, :], np.zeros((1, 150), dtype=int), 120
        )
