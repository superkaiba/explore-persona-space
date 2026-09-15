"""Independent numerical and grouping checks for the pooled #825 extension."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from explore_persona_space.analysis.pooled_turn_transfer import fit_primal_gcv, row_metrics
from explore_persona_space.analysis.turn_transfer_calibration import fit_batched_gcv


@pytest.mark.parametrize("n,d", [(47, 11), (23, 35), (40, 12)])
def test_primal_matches_frozen_dual_recipe(n, d):
    torch.set_num_threads(2)
    rng = np.random.default_rng(n)
    x = rng.normal(size=(n, d))
    if n == 40:
        x[:, -1] = x[:, 0]
        x[:, -2] = 1
    y = x @ rng.normal(size=(d, 7)) + rng.normal(size=(n, 7))
    folds = [np.flatnonzero(np.arange(n) % 3 != k) for k in range(3)]
    actual, expected = fit_primal_gcv(x, y, folds), fit_batched_gcv(x, y, folds)
    np.testing.assert_array_equal(actual.lambdas, expected.lambdas)
    np.testing.assert_allclose(actual.gcv_scores, expected.gcv_scores, rtol=2e-7, atol=2e-9)
    for f in range(3):
        np.testing.assert_allclose(actual.predict(f, x), expected.predict(f, x), atol=2e-8)
        # A separate normal-equation solve checks the selected fit itself.
        i = folds[f]
        xn = (x[i] - x[i].mean(0)) / (x[i].std(0, ddof=1) + 1e-9)
        beta = np.linalg.solve(
            xn.T @ xn + actual.lambdas[f] * np.eye(d), xn.T @ (y[i] - y[i].mean(0))
        )
        np.testing.assert_allclose(actual.beta[f], beta, atol=2e-8)


def test_retrieval_and_row_error_against_direct_distances():
    rng = np.random.default_rng(19)
    y = rng.normal(size=(13, 6))
    p = np.stack([y, y[rng.permutation(len(y))], rng.normal(size=y.shape)])
    m = row_metrics(p, y)
    np.testing.assert_allclose(m["sse"], ((p - y) ** 2).sum(-1))
    for k in range(len(p)):
        dist = ((p[k, :, None, :] - y[None, :, :]) ** 2).sum(-1)
        np.testing.assert_array_equal(m["euclidean_hit"][k], dist.argmin(-1) == np.arange(len(y)))
    np.testing.assert_array_equal(m["cosine_hit"][0], np.ones(len(y)))


def test_invalid_indices_and_nonfinite_inputs_fail():
    x = np.eye(5)
    with pytest.raises(ValueError, match="indices"):
        fit_primal_gcv(x, x, [np.array([0, 1, 1])])
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_primal_gcv(x, np.eye(5), [np.arange(5)])


def test_resume_cannot_bypass_failed_venue_gate_and_scores_exclude_groups(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/issue825_turn_pooled_single.py"
    spec = importlib.util.spec_from_file_location("pooled_driver_test", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    config = tmp_path / "config.json"
    config.write_text("{}")
    args = SimpleNamespace(config=config, out=tmp_path / "out", store=tmp_path / "store")
    rng = np.random.default_rng(14)
    turns = np.tile([1, 2, 3, 12], 12)
    ids = np.repeat(np.array([str(i) for i in range(12)]), 4)
    membership = np.repeat(np.arange(12) % 6, 4)
    x = rng.normal(size=(48, 7))
    panel = {
        "x": x,
        "y": x @ rng.normal(size=(7, 7)) + rng.normal(size=(48, 7)),
        "ids": ids,
        "turns": turns,
        "membership": membership,
        "fold_hash": "fixture",
    }
    folds, fitted, fp, _ = next(driver.fitted_maps(args, panel, "instruct", "1+2+3"))
    assert folds == [0, 1]
    driver.score(args, panel, "instruct", "1+2+3", 0, fitted, 0, fp)
    pred = args.store / "predictions/instruct_source1+2+3_target12_fold0.npz"
    with np.load(pred, allow_pickle=False) as z:
        test_ids = set(ids[z["test_indices"]])
        assert not test_ids & set(ids[z["source_training_indices"]])
        assert not test_ids & set(ids[z["calibration_indices"]])
        assert set(turns[z["source_training_indices"]]) == {1, 2, 3}
    receipt = args.out / "maps/instruct_source1+2+3_folds0-1.json"
    r = json.loads(receipt.read_text())
    r["elapsed_seconds"] = 201
    receipt.write_text(json.dumps(r))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="venue"):
            next(driver.fitted_maps(args, panel, "instruct", "1+2+3"))
        gate = json.loads((args.out / "pilot_gate_instruct.json").read_text())
        assert gate["status"] == "venue_review_required"
        assert gate["checkpoint_sha256"] == r["sha256"]
