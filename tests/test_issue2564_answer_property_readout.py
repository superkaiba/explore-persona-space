"""Estimator parity and leakage-sensitive tests for the actual readout helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location(
    "readout2564", SCRIPT_DIR / "issue2564_answer_property_readout.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_dual_grid_matches_sklearn_on_underdetermined_multitarget_data():
    rng = np.random.default_rng(2564)
    train = rng.normal(size=(32, 61))
    train[:, -1] = 3  # A constant column must follow StandardScaler semantics.
    test = rng.normal(size=(11, 61))
    test[:, -1] = 3
    targets = rng.integers(0, 2, size=(32, 5)) * 2 - 1
    alphas = np.asarray([0.1, 10, 10000])
    actual = module.ridge_grid_scores(train, targets, test, alphas)
    scaler = StandardScaler().fit(train)
    for i, alpha in enumerate(alphas):
        oracle = Ridge(alpha=alpha, solver="svd").fit(scaler.transform(train), targets)
        expected = oracle.predict(scaler.transform(test))
        np.testing.assert_allclose(actual[i], expected, atol=1e-8, rtol=1e-8)


def test_test_rows_do_not_change_each_others_scaling_or_scores():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(15, 30))
    targets = rng.normal(size=(15, 2))
    test = rng.normal(size=(4, 30))
    first = module.ridge_grid_scores(train, targets, test[:1], np.asarray([1.0]))
    test[-1] = 1e9
    together = module.ridge_grid_scores(train, targets, test, np.asarray([1.0]))
    np.testing.assert_allclose(first[:, 0], together[:, 0], atol=1e-8)


def test_carrier_folds_hold_all_draws_and_templates_together():
    groups = np.repeat(np.asarray([f"c{i:02d}" for i in range(12)]), 20)
    masks = module.carrier_folds(groups, 6)
    np.testing.assert_array_equal(np.stack(masks).sum(axis=0), np.ones(len(groups)))
    for mask in masks:
        assert not set(groups[mask]) & set(groups[~mask])


def test_marker_rule_reuses_existing_whole_word_semantics():
    assert module.check_contains_word("Surely, yes.", "surely")
    assert not module.check_contains_word("a leisurely walk", "surely")


def test_jsonl_reader_preserves_unicode_line_separators_inside_strings(tmp_path):
    rows = [{"text": "before\u2028middle\u2029after\u0085end"}, {"text": "next record"}]
    path = tmp_path / "answers.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    assert module.read_jsonl(path) == rows


@pytest.mark.parametrize(
    ("scores", "assignments", "pilot", "message"),
    [
        ([[0.2], [float("nan")]], [1, 1], False, "nonfinite prediction"),
        ([[0.2], [float("nan")]], [1, 1], True, "nonfinite prediction"),
        ([[0.2], [float("nan")]], [1, 0], False, "every input answer exactly once"),
        ([[0.2], [0.3]], [1, 2], False, "at most one evaluation fold"),
    ],
)
def test_incomplete_or_nonfinite_production_results_fail_loudly(
    scores, assignments, pilot, message
):
    with pytest.raises(ValueError, match=message):
        module.evaluated_rows(np.asarray(scores), np.asarray(assignments), pilot=pilot)


def test_partial_pilot_is_explicitly_allowed():
    actual = module.evaluated_rows(
        np.asarray([[0.2, 0.3], [np.nan, np.nan]]), np.asarray([1, 0]), pilot=True
    )
    np.testing.assert_array_equal(actual, [True, False])
