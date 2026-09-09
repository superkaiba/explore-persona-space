"""Small direct tests of the executed category-validation analysis helpers."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SPEC = importlib.util.spec_from_file_location(
    "validation", Path(__file__).parents[1] / "scripts/issue2569_refusal_category_validation.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_vectorized_spearman_matches_reference():
    """Ties and missing bootstrap padding retain exact rank correlations."""
    x = np.array([[1, 1, 4, 9, np.nan], [8, 3, 2, 2, 1]])
    y = np.array([[3, 5, 5, 4, np.nan], [1, 2, 8, 9, 10]])
    result = MODULE.rho_rows(x, y)
    for i in range(len(x)):
        valid = np.isfinite(x[i])
        assert np.isclose(result[i], spearmanr(x[i, valid], y[i, valid]).statistic)


def test_constant_behavior_is_undefined():
    """A floor-saturated refusal stratum is not reported as zero correlation."""
    result = MODULE.corr(np.arange(6), np.zeros(6), np.array(["a", "a", "b", "b", "c", "c"]))
    assert result["rho"] is None and result["ci95"] is None


def test_cluster_draws_keep_whole_families():
    """Every occurrence of one member is accompanied by its siblings."""
    idx = MODULE.cluster_indices(np.array(["a", "a", "b", "c", "c", "c"]))
    for sample in idx[:20]:
        assert np.sum(sample == 0) == np.sum(sample == 1)
        assert np.sum(sample == 3) == np.sum(sample == 4) == np.sum(sample == 5)


def test_crossfit_axis_has_no_heldout_dependence():
    """Changing a test family's states cannot change its training-only axis."""
    da = np.array([[1.0, 0], [2, 0], [0, 1], [0, 2], [1, 1], [2, 2]])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    gap = np.ones(6)
    axes, folds = MODULE.crossfit_axes(da, gap, groups, np.ones(6, dtype=bool))
    changed = da.copy()
    changed[:2] = [[100, -300], [-500, 700]]
    other, _ = MODULE.crossfit_axes(changed, gap, groups, np.ones(6, dtype=bool))
    np.testing.assert_allclose(axes[:2], other[:2])
    for fold in folds:
        assert not set(fold["test_indices"]) & set(fold["train_axis_indices"])


def test_axis_orientation_uses_only_training_signs():
    """Manifest-negative training flips are oriented toward greater refusal."""
    da = np.array([[1.0, 0], [1, 0], [-2, 0], [-3, 0]])
    gap = np.array([1.0, 1, -1, -1])
    axes, _ = MODULE.crossfit_axes(da, gap, np.array(["a", "a", "b", "b"]), np.ones(4, dtype=bool))
    np.testing.assert_allclose(axes, np.array([[1.0, 0]] * 4))


def test_family_contrast_counts_families_not_pairs():
    """Uneven category replication does not reweight the family-level contrast."""
    values = np.array([2.0, 2, 0, 4, 0])
    classes = np.array(["obj_flip", "verb_flip", "subj_ctl", "obj_flip", "subj_ctl"])
    families = np.array(["a", "a", "a", "b", "b"])
    result = MODULE.matched_contrast(values, classes, families, "subj_ctl")
    assert result["mean_difference"] == 3
    assert result["p_exact_signflip"] == 0.5
