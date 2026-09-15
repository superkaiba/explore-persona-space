"""Exercise the turn-calibration driver's real fitting, scoring and checkpoints."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from explore_persona_space.analysis.mapping_baselines import identity_bias_predict
from explore_persona_space.analysis.turn_transfer_calibration import (
    adapted_predictions,
    fit_batched_gcv,
)


def test_scoring_fingerprint_survives_comment_edit_above_loaded_function(driver, tmp_path):
    """Replay the real live-file line shift that misidentified the scoring source."""
    path = tmp_path / "copy_driver.py"
    original = Path(driver.__file__).read_text()
    path.write_text(original)
    spec = importlib.util.spec_from_file_location("calibration_snapshot_fixture", path)
    copied = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copied)
    receipt = {"fingerprint": "fixture-source", "map_sha256": "fixture-map"}
    before = copied.score_fingerprint(receipt)
    path.write_text(
        original.replace(
            "def source_fit(args, panel):", "# added comment\ndef source_fit(args, panel):", 1
        )
    )
    assert inspect.getsource(copied.score_target) != copied.SCORE_TARGET_SOURCE
    assert copied.score_fingerprint(receipt) == before


@pytest.fixture
def driver(monkeypatch):
    """Import the actual worktree driver, with no numerical bodies stubbed."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("issue825_turn_bias_scale")


@pytest.fixture
def inputs(tmp_path):
    """Small paired conversation bank with all six outer folds populated."""
    rng = np.random.default_rng(82512)
    n, d = 25, 7
    x = rng.normal(size=(2 * n, d)).astype(np.float32)
    operator = rng.normal(size=(d, d))
    y = (x @ operator + rng.normal(scale=0.7, size=(2 * n, d))).astype(np.float32)
    panel = {
        "x": x,
        "y": y,
        "ids": np.array([f"conversation_{i}" for i in range(n)] * 2),
        "turns": np.repeat([1, 12], n),
        "membership": np.tile(np.arange(n) % 6, 2),
        "fold_hash": "small-fixture-conversation-folds",
    }
    args = SimpleNamespace(
        inputs=tmp_path / "inputs",
        out=tmp_path / "out",
        store=tmp_path / "store",
        model="instruct",
        source_turn=1,
    )
    args.inputs.mkdir()
    (args.inputs / "inputs.json").write_text('{"fixture": "six disjoint conversation folds"}')
    return args, panel


def test_persistence_creates_nested_directories_and_source_map_roundtrips(driver, inputs):
    """First fit persists its map into a fresh store and resumes identical coefficients."""
    args, panel = inputs
    fitted, first = driver.source_fit(args, panel)
    restored, second = driver.source_fit(args, panel)
    assert first["map_sha256"] == second["map_sha256"]
    assert Path(first["map_file"]).exists()
    for name, array in vars(fitted).items():
        np.testing.assert_array_equal(array, getattr(restored, name))


def test_target_test_answers_cannot_change_calibration(driver, inputs):
    """Change held-out answers, rerun real scoring, and compare fitted coefficients."""
    args, panel = inputs
    fitted, source = driver.source_fit(args, panel)
    before = driver.score_target(args, panel, fitted, source, 12, 0)
    testing = (panel["turns"] == 12) & (panel["membership"] == 0)
    changed = panel | {"y": panel["y"].copy()}
    changed["y"][testing] += np.arange(panel["y"].shape[1]) * 100
    changed_args = SimpleNamespace(**vars(args))
    changed_args.out = args.out / "changed_targets"
    changed_args.store = args.store / "changed_targets"
    after = driver.score_target(changed_args, changed, fitted, source, 12, 0)
    with (
        np.load(before["prediction_file"], allow_pickle=False) as a,
        np.load(after["prediction_file"], allow_pickle=False) as b,
    ):
        for name in [
            "bias",
            "gain",
            "prediction_mean",
            "target_mean",
            "raw_prediction",
        ]:
            np.testing.assert_array_equal(a[name], b[name])
    assert before["metrics"]["bias_scale"]["sse"] != after["metrics"]["bias_scale"]["sse"]


@pytest.mark.parametrize("fitting_stage", ["source", "calibration"])
def test_score_rejects_conversation_overlap(driver, inputs, fitting_stage):
    """Even different rows cannot cross the train/test conversation boundary."""
    args, panel = inputs
    fitted, source = driver.source_fit(args, panel)
    changed = panel | {"ids": panel["ids"].astype(object)}
    source_turn = 1 if fitting_stage == "source" else 12
    train_row = np.flatnonzero((panel["turns"] == source_turn) & (panel["membership"] != 0))[0]
    test_row = np.flatnonzero((panel["turns"] == 12) & (panel["membership"] == 0))[0]
    changed["ids"][train_row] = "overlap_fixture"
    changed["ids"][test_row] = "overlap_fixture"
    with pytest.raises(RuntimeError, match="held-out conversations leaked"):
        driver.score_target(args, changed, fitted, source, 12, 0)


def test_methods_score_identical_target_rows_and_retrieval_pools(driver, inputs):
    """Reconstruct every method's errors on the persisted test rows independently."""
    args, panel = inputs
    fitted, source = driver.source_fit(args, panel)
    row = driver.score_target(args, panel, fitted, source, 12, 0)
    with np.load(row["prediction_file"], allow_pickle=False) as arrays:
        train, test = arrays["calibration_indices"], arrays["test_indices"]
        coefficients = {
            name: arrays[name] for name in ["bias", "gain", "prediction_mean", "target_mean"]
        }
        raw = fitted.predict(0, panel["x"][test])
        expected = {"raw": raw} | adapted_predictions(raw, coefficients)
        expected["identity_bias"] = identity_bias_predict(
            panel["x"][train], panel["y"][train], panel["x"][test]
        )
        truth = panel["y"][test].astype(np.float64)
        sst = np.square(truth - truth.mean(0)).sum()
        assert set(expected) == set(row["metrics"]) == set(driver.METHODS)
        assert row["sst"] == pytest.approx(sst)
        for method, prediction in expected.items():
            errors = np.square(prediction - truth).sum(1)
            np.testing.assert_allclose(arrays[f"sse_{method}"], errors, atol=1e-12)
            assert row["metrics"][method]["r2"] == pytest.approx(1 - errors.sum() / sst)
            for metric in ["cosine", "euclidean"]:
                retrieval = row["metrics"][method]["retrieval"][metric]
                assert retrieval["n_pool"] == retrieval["n"] == len(test)
                assert retrieval["chance_at_k"][1] == 1 / len(test)


def test_score_resume_rejects_changed_source_map(driver, inputs):
    """Identical recipe metadata does not license reusing predictions from another map."""
    args, panel = inputs
    fitted, source = driver.source_fit(args, panel)
    driver.score_target(args, panel, fitted, source, 12, 0)
    changed_source = source | {"map_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        driver.score_target(args, panel, fitted, changed_source, 12, 0)


@pytest.fixture
def reduction_inputs(driver, inputs, monkeypatch):
    """Produce real fold artifacts for a one-cell declared grid and pinned parent."""
    args, panel = inputs
    args.source_turn = 12
    monkeypatch.setattr(driver, "MODELS", ("instruct",))
    monkeypatch.setattr(driver, "SOURCE_TURNS", (12,))
    selected = np.flatnonzero(panel["turns"] == 12)
    membership = panel["membership"][selected]
    train_indices = [np.flatnonzero(membership != fold) for fold in range(6)]
    independent = fit_batched_gcv(panel["x"][selected], panel["y"][selected], train_indices)
    sse, sst = 0.0, 0.0
    for fold in range(6):
        test = selected[membership == fold]
        truth = panel["y"][test].astype(np.float64)
        sse += np.square(truth - independent.predict(fold, panel["x"][test])).sum()
        sst += np.square(truth - truth.mean(0)).sum()
    parent = {
        "followup_label": "turn-dynamics-allturns-5000",
        "smoke": False,
        "parts": {
            "transfer_armR_own_instruct": {
                "r2": {"12->12": 1 - sse / sst},
                "fold_map_sha256": panel["fold_hash"],
            },
            "cells_armR_own_instruct": {"n_per_turn": {"12": len(selected)}},
        },
    }
    args.reference = args.inputs / "parent.json"
    args.reference.write_text(json.dumps(parent))
    monkeypatch.setattr(driver, "REFERENCE_SHA256", driver.sha(args.reference))
    fitted, source = driver.source_fit(args, panel)
    rows = [driver.score_target(args, panel, fitted, source, 12, f) for f in range(6)]
    return args, rows, source


def test_reducer_uses_fold_sums_and_weights_retrieval_by_test_size(driver, reduction_inputs):
    """Real unequal folds aggregate by SSE/SST and sample counts, not mean scores."""
    args, rows, _source = reduction_inputs
    driver.reduce_results(args)
    result = json.loads((args.out / "results.json").read_text())
    assert len(result["cells"]) == 1
    cell = result["cells"][0]
    expected = 1 - sum(r["metrics"]["raw"]["sse"] for r in rows) / sum(r["sst"] for r in rows)
    assert cell["metrics"]["raw"]["r2"] == pytest.approx(expected)
    assert cell["metrics"]["raw"]["retention"] == pytest.approx(1)
    assert not np.isclose(expected, np.mean([r["metrics"]["raw"]["r2"] for r in rows]))
    for method in driver.METHODS:
        for metric in ["cosine", "euclidean"]:
            expected_top1 = sum(
                r["metrics"][method]["retrieval"][metric]["acc_at_k"][1] * r["n_test"] for r in rows
            ) / sum(r["n_test"] for r in rows)
            assert cell["metrics"][method]["retrieval"][metric]["top1"] == pytest.approx(
                expected_top1
            )


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("fold_identity", "fold provenance"),
        ("fold_fingerprint", "fold provenance"),
        ("fold_source_hash", "fold provenance"),
        ("overlap", "fold provenance"),
        ("coverage", "target coverage"),
        ("prediction_bytes", "changed predictions"),
        ("map_bytes", "source-map provenance"),
    ],
)
def test_reducer_rejects_corrupt_or_mismatched_artifacts(
    driver, reduction_inputs, corruption, message
):
    """Valid parent raw scores cannot excuse stale calibration provenance or coverage."""
    args, rows, source = reduction_inputs
    path = args.out / "folds" / "instruct_source12_target12_fold0.json"
    row = json.loads(path.read_text())
    if corruption == "fold_identity":
        row["fold"] = 1
    elif corruption == "fold_fingerprint":
        row["fingerprint"] = "stale-scoring-code"
    elif corruption == "fold_source_hash":
        row["source_map_sha256"] = "0" * 64
    elif corruption == "overlap":
        row["test_source_overlap"] = 1
    elif corruption == "coverage":
        row["n_test"] -= 1
    elif corruption == "prediction_bytes":
        Path(rows[0]["prediction_file"]).write_bytes(b"modified predictions")
    elif corruption == "map_bytes":
        Path(source["map_file"]).write_bytes(b"modified source map")
    else:
        raise AssertionError("unrecognized test corruption")
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match=message):
        driver.reduce_results(args)
