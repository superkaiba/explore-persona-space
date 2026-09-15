"""Counterbalancing, heldout scoring, and loss-aggregation checks for matched transfer."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from explore_persona_space.analysis.pooled_turn_transfer import fit_primal_gcv

spec = importlib.util.spec_from_file_location(
    "matched_driver", Path(__file__).resolve().parents[1] / "scripts/issue825_turn_matched.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_balanced_assignment_matches_rows_and_conversations():
    labels = np.repeat(np.arange(6), [7, 8, 9, 10, 11, 12])
    ids = np.array([f"c{i:03}" for i in range(len(labels))])
    rotations = driver.assignments(ids, labels)
    np.testing.assert_array_equal(
        np.sort(rotations, axis=0), np.broadcast_to(np.arange(1, 4)[:, None], rotations.shape)
    )
    panel = dict(
        ids=ids,
        labels=labels,
        assignments=rotations,
        x=np.arange(12 * len(ids) * 2).reshape(12, len(ids), 2),
    )
    panel["y"] = panel["x"] + 0.5
    for variant in driver.VARIANTS:
        x, y, turns = driver.source_rows(panel, variant)
        assert len(x) == len(ids)
        np.testing.assert_array_equal(y - x, 0.5)
        for i in range(len(ids)):
            np.testing.assert_array_equal(x[i], panel["x"][turns[i] - 1, i])
        for f in range(6):
            train = labels != f
            assert len(set(ids[train])) == train.sum()
            if variant.startswith("mix"):
                assert np.ptp(np.bincount(turns[train], minlength=4)[1:]) <= 2


def test_real_scoring_all_turns_and_source_only_bias(tmp_path):
    rng = np.random.default_rng(3)
    labels = np.arange(36) % 6
    ids = np.array([f"c{i:03}" for i in range(36)])
    x = rng.normal(size=(12, 36, 7))
    y = 2 * x + rng.normal(size=x.shape)
    y[3:] += 100  # Target offsets must not leak into source-trained bias.
    panel = dict(x=x, y=y, ids=ids, labels=labels, assignments=driver.assignments(ids, labels))
    args = SimpleNamespace(out=tmp_path / "analysis", store=tmp_path / "store")
    xx, yy, _ = driver.source_rows(panel, "mix0")
    train = np.flatnonzero(labels != 0)
    test = np.flatnonzero(labels == 0)
    fit = fit_primal_gcv(xx, yy, [train])
    rec = driver.score_fold(args, panel, "instruct", "mix0", 0, fit, 0, "fp")
    file = args.store / "predictions/instruct_mix0_fold0.npz"
    with np.load(file) as z:
        np.testing.assert_array_equal(z["ids"], ids[test])
        assert not set(z["ids"]) & set(ids[train])
        bias = (yy[train] - xx[train]).mean(0)
        np.testing.assert_allclose(z["source_bias"], bias)
        for t in range(12):
            truth = y[t, test]
            raw = fit.predict(0, x[t, test])
            np.testing.assert_allclose(z["raw"][t], raw)
            np.testing.assert_allclose(z["sse"][0, t], ((raw - truth) ** 2).sum(-1))
            np.testing.assert_allclose(z["sse"][1, t], ((x[t, test] + bias - truth) ** 2).sum(-1))
            distances = ((raw[:, None, :] - truth[None, :, :]) ** 2).sum(-1)
            np.testing.assert_array_equal(
                z["euclidean_hit"][0, t], distances.argmin(-1) == np.arange(len(test))
            )
    assert rec["n_train"] == len(train)
    assert driver.score_fold(args, panel, "instruct", "mix0", 0, fit, 0, "fp") == rec
    with file.open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="checkpoint"):
        driver.score_fold(args, panel, "instruct", "mix0", 0, fit, 0, "fp")


def test_rotation_loss_average_is_not_ensemble_error():
    predictions = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0])[:, None]
    loss = predictions**2
    collapsed = driver.collapse_rotations(loss)
    assert collapsed[-1, 0] == pytest.approx(2 / 3)
    assert predictions[3:].mean() ** 2 == 0


def test_failed_venue_gate_remains_failed(tmp_path):
    rec = dict(elapsed_seconds=151, max_rss_kib=1000)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="gate"):
            driver.venue_gate(rec, tmp_path / "gate.json")
        assert json.loads((tmp_path / "gate.json").read_text())["status"] == "halt"


def test_model_driver_and_panel_body_resume_rejects_wrong_fold(tmp_path, monkeypatch):
    """Exercise the real selection, fitting and scoring bodies on a small complete bank."""
    rng = np.random.default_rng(17)
    ids = np.array([f"c{i:03}" for i in range(36)])
    labels = np.arange(36) % 6
    x = rng.normal(size=(12, 36, 7))
    y = x @ rng.normal(size=(7, 7)) + rng.normal(size=x.shape)
    panel = dict(
        x=x.reshape(-1, 7),
        y=y.reshape(-1, 7),
        ids=np.tile(ids, 12),
        turns=np.repeat(np.arange(1, 13), 36),
        membership=np.tile(labels, 12),
        counts={str(t): 36 for t in range(1, 13)},
        fold_hash="fixture",
    )

    def load_panel(args, model):
        return panel

    monkeypatch.setattr(driver.parent, "load_panel", load_panel)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(dict(cohort_ids=ids.tolist(), fold_labels=labels.tolist(), fold_hash="fixture"))
    )
    args = SimpleNamespace(config=config, out=tmp_path / "analysis", store=tmp_path / "store")
    driver.run_model(args, "instruct")
    driver.run_model(args, "instruct")
    assert len(list((args.out / "maps").glob("*.json"))) == 18
    assert len(list((args.out / "scores").glob("*.json"))) == 36
    receipt = args.out / "maps/instruct_1_folds0-1.json"
    rec = json.loads(receipt.read_text())
    rec["folds"] = [2, 3]
    receipt.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="unit identity"):
        driver.run_model(args, "instruct")
