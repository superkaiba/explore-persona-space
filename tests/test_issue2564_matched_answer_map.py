"""Numerical and leakage checks for the in-bank matched answer map."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2054_fits as established
import issue2564_matched_answer_map as matched


def fixture() -> tuple[list[dict], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2564)
    n, width = 20, 3584
    context = rng.normal(size=(n, width)).astype(np.float32)
    answer = (0.4 * context + rng.normal(scale=0.2, size=(n, width))).astype(np.float32)
    rows = []
    for fold in range(5):
        for within in range(4):
            rows.append(
                {
                    "id": f"row-{fold}-{within}",
                    "fold": fold,
                    "question_group": f"group-{fold}-{within // 2}",
                }
            )
    return rows, context, answer


def test_grouped_oof_matches_established_fit_and_writes_evidence(tmp_path):
    rows, context, answer = fixture()
    lambdas = np.array([0.1, 1.0, 10.0, 10000.0])
    result = matched.fit_grouped_oof(rows, context, answer, tmp_path, lambdas=lambdas)

    assert result["mapped"].shape == (20, 3584)
    assert np.isfinite(result["mapped"]).all()
    assert np.isfinite(result["identity_bias"]).all()
    for fold in range(5):
        test = np.array([i for i, row in enumerate(rows) if row["fold"] == fold])
        train = np.array([i for i, row in enumerate(rows) if row["fold"] != fold])
        expected, info = established._ridge_gcv_fit_predict(
            context[train], answer[train], context[test], lambdas=lambdas
        )
        np.testing.assert_allclose(result["mapped"][test], expected)
        bias = answer[train].astype(np.float64).mean(0) - context[train].astype(np.float64).mean(0)
        np.testing.assert_allclose(result["identity_bias"][test], context[test] + bias)
        assert result["folds"][fold]["fit"]["best_lambda"] == info["best_lambda"]
        assert (tmp_path / f"fold{fold}.json").is_file()
        assert (tmp_path / f"fold{fold}.npz").is_file()
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "oof_predictions.npz").is_file()


def test_heldout_answer_changes_do_not_change_that_fold_prediction(tmp_path):
    rows, context, answer = fixture()
    lambdas = np.array([1.0, 10.0, 10000.0])
    first = matched.fit_grouped_oof(rows, context, answer, tmp_path / "first", lambdas=lambdas)
    changed = answer.copy()
    changed[[i for i, row in enumerate(rows) if row["fold"] == 0]] += 100.0
    second = matched.fit_grouped_oof(rows, context, changed, tmp_path / "second", lambdas=lambdas)
    heldout = np.array([i for i, row in enumerate(rows) if row["fold"] == 0])
    np.testing.assert_allclose(first["mapped"][heldout], second["mapped"][heldout])
    np.testing.assert_allclose(first["identity_bias"][heldout], second["identity_bias"][heldout])


def test_completed_output_resumes_from_validated_checkpoints(tmp_path, monkeypatch):
    rows, context, answer = fixture()
    lambdas = np.array([1.0, 10000.0])
    first = matched.fit_grouped_oof(rows, context, answer, tmp_path, lambdas=lambdas)

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("resume unexpectedly refit a completed fold")

    monkeypatch.setattr(matched, "_ridge_gcv_fit_predict", fail_if_refit)
    second = matched.fit_grouped_oof(rows, context, answer, tmp_path, lambdas=lambdas)
    np.testing.assert_array_equal(first["mapped"], second["mapped"])
    np.testing.assert_array_equal(first["identity_bias"], second["identity_bias"])


def test_group_crossing_outer_folds_is_rejected(tmp_path):
    rows, context, answer = fixture()
    rows[0]["question_group"] = rows[4]["question_group"]
    with pytest.raises(ValueError, match="crosses outer folds"):
        matched.fit_grouped_oof(rows, context, answer, tmp_path)
