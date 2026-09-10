"""Numerical, grouping and end-to-end checks; all targets here are test fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2564_answer_behavior_readout as readout
from issue2564_answer_property_readout import ridge_grid_scores


def test_production_shared_ridge_matches_existing_and_sklearn():
    """The dispatched solver agrees with independent full and PCA estimators."""
    rng = np.random.default_rng(2564)
    train = rng.normal(size=(27, 38)) * np.linspace(0.1, 4, 38)
    train[:, -1] = 2
    test = rng.normal(size=(9, 38)) + 3
    targets = rng.normal(size=(27, 3)) + 10
    alphas = np.array([0.1, 10, 1000])
    actual = readout.ridge_width_grid(train, targets, test, alphas, (0, 5))
    np.testing.assert_allclose(
        actual[0], ridge_grid_scores(train, targets, test, alphas), atol=1e-9
    )
    scaler = StandardScaler().fit(train)
    standardized_train, standardized_test = scaler.transform(train), scaler.transform(test)
    for width in (0, 5):
        x, z = standardized_train, standardized_test
        if width:
            pca = PCA(n_components=width, svd_solver="full").fit(x)
            x, z = pca.transform(x), pca.transform(z)
        for ai, alpha in enumerate(alphas):
            expected = Ridge(alpha=alpha, solver="svd").fit(x, targets).predict(z)
            np.testing.assert_allclose(actual[width][ai], expected, atol=1e-9, rtol=1e-9)
    with pytest.raises(ValueError, match="PCA width"):
        readout.ridge_width_grid(train, targets, test, alphas, (100,))


def toy_bank():
    """Make small, explicitly synthetic arrays for exercising production bodies."""
    rng = np.random.default_rng(30)
    rows = [
        {
            "id": f"r{q}-{v}",
            "conv_id": f"q{q}",
            "question_group": f"g{q}",
            "variant": f"v{v}",
            "fold": q % 5,
            "finish_reason": "stop",
            "answer": f"Test fixture {q}/{v}",
        }
        for q in range(15)
        for v in range(4)
    ]
    answer = rng.normal(size=(len(rows), 10))
    answer[:, -1] = 1
    context = rng.normal(size=answer.shape)
    scalar = answer[:, 0] + 0.05 * rng.normal(size=len(rows))
    categorical = np.digitize(answer[:, 1], [-0.5, 0.5])
    targets = [
        readout.Target("fixture_score", "graded", scalar[:, None], np.ones(len(rows), bool), []),
        readout.Target(
            "fixture_category",
            "categorical",
            np.eye(3)[categorical],
            np.ones(len(rows), bool),
            ["a", "b", "c"],
            categorical,
        ),
    ]
    vectors = {"answer": answer, "context": context, "length": rng.normal(size=(len(rows), 1))}
    return rows, vectors, targets


def test_grouped_shuffle_preserves_whole_components_framing_and_availability():
    """A grouped null cannot separate duplicate-connected questions or missingness."""
    rows, _, _ = toy_bank()
    for row in rows:
        q = int(row["conv_id"][1:])
        row["question_group"] = f"component{q // 2}"
    available = np.array([r["variant"] != "v2" for r in rows])
    permutation = readout.component_shuffle(rows, available, 2564)
    assert np.any(permutation != np.arange(len(rows)))
    np.testing.assert_array_equal(available[permutation], available)
    for group in {r["question_group"] for r in rows}:
        destinations = [i for i, r in enumerate(rows) if r["question_group"] == group]
        assert len({rows[permutation[i]]["question_group"] for i in destinations}) == 1
    groups = np.array([r["question_group"] for r in rows])
    folds = readout.grouped_folds(groups)
    for group in set(groups):
        assert len(set(folds[groups == group])) == 1


def test_live_bundle_tuning_matches_independent_scalar_reference():
    """Outer predictions use only nested training rows, with per-target penalties."""
    rows, vectors, targets = toy_bank()
    actual, metadata = readout.fit_bundle(rows, vectors, targets, 0)
    outer = np.array([r["fold"] for r in rows])
    train, test = np.flatnonzero(outer != 0), np.flatnonzero(outer == 0)
    inner = readout.grouped_folds(np.array([rows[i]["question_group"] for i in train]))
    assert {r["id"] for r in rows if r["fold"] == 0} == set(metadata["test_ids"])
    for target_index, target in enumerate(targets):
        losses = []
        for alpha in readout.ALPHAS:
            total, count = 0.0, 0
            for fold in range(3):
                fit, validation = train[inner != fold], train[inner == fold]
                scaler = StandardScaler().fit(vectors["answer"][fit])
                model = Ridge(alpha=alpha, solver="svd").fit(
                    scaler.transform(vectors["answer"][fit]), target.values[fit]
                )
                predicted = model.predict(scaler.transform(vectors["answer"][validation]))
                predicted = np.asarray(predicted).reshape(len(validation), -1)
                total += np.square(predicted - target.values[validation]).sum()
                count += target.values[validation].size
            losses.append(total / count)
        choice = int(np.flatnonzero(np.array(losses) == min(losses))[-1])
        assert metadata["models"]["answer"][target.name]["alpha"] == readout.ALPHAS[choice]
        scaler = StandardScaler().fit(vectors["answer"][train])
        expected = (
            Ridge(alpha=readout.ALPHAS[choice], solver="svd")
            .fit(scaler.transform(vectors["answer"][train]), target.values[train])
            .predict(scaler.transform(vectors["answer"][test]))
        )
        expected = np.asarray(expected).reshape(len(test), -1)
        segment = slice(
            metadata["boundaries"][target_index], metadata["boundaries"][target_index + 1]
        )
        np.testing.assert_allclose(actual["answer"][:, segment], expected, atol=1e-9)
    assert "answer_pca256" in metadata["unsupported_widths"]


def test_weighted_metric_batch_matches_explicit_resampling():
    """Bootstrap sufficient sums match explicit weighted row replication."""
    y = np.array([1.0, 4.0, 2.0, 8.0])
    prediction = np.array([2.0, 3.0, 5.0, 7.0])
    weights = np.array([[1, 1, 1, 1], [2, 0, 3, 1], [0, 0, 0, 0]])
    results = readout.continuous_metrics(y, prediction, weights)
    for i in range(2):
        yy, pp = np.repeat(y, weights[i]), np.repeat(prediction, weights[i])
        assert results["mse"][i] == pytest.approx(np.square(yy - pp).mean())
        assert results["r2"][i] == pytest.approx(
            1 - np.square(yy - pp).sum() / np.square(yy - yy.mean()).sum()
        )
        assert results["pearson"][i] == pytest.approx(np.corrcoef(yy, pp)[0, 1])
    assert all(np.isnan(result[-1]) for result in results.values())
    constant = readout.continuous_metrics(np.full(4, 37.3), prediction, weights)
    assert np.isnan(constant["r2"]).all() and np.isnan(constant["pearson"]).all()


def test_categorical_metrics_match_resampling_and_do_not_resolve_annotation_ties():
    """Classification excludes unresolved judge votes and keeps a fixed class set."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    modal = np.array([0, 1, -1, 0, 1])
    votes = np.array([[1.0, 0], [0, 1], [0.5, 0.5], [1, 0], [0, 1]])
    scores = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.9, 0.1], [0.1, 0.9]])
    weights = np.array([[1, 1, 1, 1, 1], [3, 2, 5, 1, 2]])
    target = readout.Target("fixture", "categorical", votes, np.ones(5, bool), ["a", "b"], modal)
    actual, details = readout.categorical_metrics(target, scores, np.arange(5), weights)
    assert details["modal_ties_excluded"] == 1
    keep = modal >= 0
    for i in range(2):
        yy = np.repeat(modal[keep], weights[i, keep])
        pp = np.repeat(scores[keep].argmax(1), weights[i, keep])
        assert actual["balanced_accuracy"][i] == pytest.approx(balanced_accuracy_score(yy, pp))
        assert actual["macro_f1"][i] == pytest.approx(f1_score(yy, pp, average="macro"))


def test_real_loader_validates_ids_missingness_and_label_votes(tmp_path, monkeypatch):
    """Execute the loader on a full-size, zero-vector fixture, never a real dataset."""
    import json
    from unittest.mock import create_autospec

    gate = create_autospec(readout.validate_pilot_acceptance, return_value={"test_fixture": True})
    monkeypatch.setattr(readout, "validate_pilot_acceptance", gate)
    (tmp_path / "prepared").mkdir()
    rows = [
        {
            "id": f"r{q}-{v}",
            "part": "main",
            "conv_id": f"q{q}",
            "question_group": f"q{q}",
            "question": f"fixture question {q}",
            "answer": f"fixture answer {q}/{v}",
            "variant": f"v{v}",
            "fold": q % 5,
            "finish_reason": "stop",
        }
        for q in range(512)
        for v in range(4)
    ]
    (tmp_path / "prepared/rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    zeros = np.zeros((2048, 3584), dtype=np.float16)
    np.savez(
        tmp_path / "prepared/vectors.npz",
        id=np.array([r["id"] for r in rows]),
        answer=zeros,
        context=zeros,
    )
    rubrics = {
        "persona": {"kind": "categorical", "labels": ["a", "b"]},
        "warmth": {"kind": "graded"},
    }
    (tmp_path / "prepared/rubrics.json").write_text(json.dumps(rubrics))
    labels = [
        {
            "id": r["id"],
            "properties": {
                "persona": {
                    "kind": "categorical",
                    "n_valid": 4,
                    "votes": {"a": 0.5, "b": 0.5},
                    "modal": None,
                    "multi_voice_fraction": 0.0,
                },
                "warmth": {"kind": "graded", "n_valid": 5, "n_assessable": 0, "mean": None},
            },
        }
        for r in rows
    ]
    label_path = tmp_path / "fixture_labels.json"
    (tmp_path / "config.json").write_text("{}")
    config_hash = readout.annotation_digest({})
    completion = {
        "expected_calls": 20480,
        "persisted_expected": 20480,
        "config_hash": config_hash,
        "expected_keys_hash": "fixture-key-roster",
    }
    (tmp_path / "complete.json").write_text(json.dumps(completion))

    def write_fixture_labels():
        """Write a test-only producer sidecar, making intentional fixture edits explicit."""
        label_path.write_text(json.dumps(labels))
        manifest = {
            "labels_sha256": readout.file_hash(label_path),
            "rows_sha256": readout.file_hash(tmp_path / "prepared/rows.jsonl"),
            "complete_sha256": readout.file_hash(tmp_path / "complete.json"),
            "config_hash": config_hash,
            "rubric_schema_hash": readout.annotation_digest(
                {
                    name: {"system": readout.judge_system(name), "schema": readout.schema(name)}
                    for name in readout.PROPERTIES
                }
            ),
            "part": "main",
            "row_count": 2048,
            "expected_draws": 20480,
            "persisted_draws": 20480,
            "exact_keyset_complete": True,
            "expected_keys_hash": "fixture-key-roster",
        }
        (tmp_path / "labels_manifest.json").write_text(json.dumps(manifest))

    write_fixture_labels()
    cfg = readout.Config(root=str(tmp_path), labels="fixture_labels.json")
    loaded_rows, vectors, targets, _ = readout.load_inputs(cfg)
    assert len(loaded_rows) == 2048 and vectors["answer"].shape == (2048, 3584)
    assert np.all(targets[0].modal == -1) and not targets[1].available.any()
    labels[0]["properties"]["warmth"]["mean"] = 0
    write_fixture_labels()
    with pytest.raises(ValueError, match="unassessable target"):
        readout.load_inputs(cfg)
    labels[0]["properties"]["warmth"]["mean"] = None
    labels[0]["properties"]["persona"]["modal"] = "a"
    write_fixture_labels()
    with pytest.raises(ValueError, match="modal label"):
        readout.load_inputs(cfg)
    labels[0]["id"] = "wrong-id"
    write_fixture_labels()
    with pytest.raises(ValueError, match="cover each main answer"):
        readout.load_inputs(cfg)
    gate.assert_called_with(tmp_path, {})


def test_codex_route_requires_its_actual_validated_aggregate(tmp_path, monkeypatch):
    """Provider dispatch cannot relabel an API aggregate as Codex evidence."""
    from unittest.mock import create_autospec

    labels = tmp_path / "annotation_codex/main/labels.json"
    config = {"provider": readout.CODEX_PROVIDER, "draws": 5, "fixture": True}
    acceptance = {"fixture_review": "accepted"}
    codex = create_autospec(
        readout.validate_codex_main,
        return_value={"labels_path": str(labels), "config": config, "acceptance": acceptance},
    )
    api = create_autospec(readout.validate_pilot_acceptance)
    monkeypatch.setattr(readout, "validate_codex_main", codex)
    monkeypatch.setattr(readout, "validate_pilot_acceptance", api)
    assert readout.validate_annotation_route(tmp_path, labels, config) == (
        acceptance,
        "expected_annotations",
        5,
    )
    codex.assert_called_once_with(tmp_path)
    api.assert_not_called()
    with pytest.raises(ValueError, match="labels differ"):
        readout.validate_annotation_route(tmp_path, tmp_path / "substituted.json", config)
    with pytest.raises(ValueError, match="configuration differs"):
        readout.validate_annotation_route(tmp_path, labels, {**config, "fixture": False})
    with pytest.raises(ValueError, match="Unknown annotation provider"):
        readout.validate_annotation_route(tmp_path, labels, {"provider": "unrecognized"})
    codex.side_effect = ValueError("Codex raw judgment changed")
    with pytest.raises(ValueError, match="raw judgment changed"):
        readout.validate_annotation_route(tmp_path, labels, config)
    api.assert_not_called()


def test_three_pass_route_is_separate_and_preserves_repeat_count(tmp_path, monkeypatch):
    """Three genuine ratings cannot satisfy the historical five-rating route."""
    from unittest.mock import create_autospec

    labels = tmp_path / "annotation_codex_three_pass/main/labels.json"
    config = {
        "provider": readout.THREE_PASS_PROVIDER,
        "draws": 3,
        "selected_repetitions": [0, 1, 2],
    }
    acceptance = {"fixture_review": "accepted"}
    gate = create_autospec(
        readout.validate_three_pass_main,
        return_value={"labels_path": str(labels), "config": config, "acceptance": acceptance},
    )
    original = create_autospec(readout.validate_codex_main)
    monkeypatch.setattr(readout, "validate_three_pass_main", gate)
    monkeypatch.setattr(readout, "validate_codex_main", original)
    assert readout.validate_annotation_route(tmp_path, labels, config) == (
        acceptance,
        "expected_annotations",
        3,
    )
    original.assert_not_called()
    with pytest.raises(ValueError, match="configuration differs"):
        readout.validate_annotation_route(tmp_path, labels, {**config, "draws": 5})
    gate.side_effect = ValueError("missing third-pass rating")
    with pytest.raises(ValueError, match="missing third-pass rating"):
        readout.validate_annotation_route(tmp_path, labels, config)


def test_end_to_end_checkpoints_summary_and_stale_inputs(tmp_path, monkeypatch):
    """Execute real fit/summary bodies; only the independently tested loader is substituted."""
    rows, vectors, targets = toy_bank()
    provenance = {"fixture": "not experimental data", "multi_voice_fraction": [0.0] * len(rows)}
    monkeypatch.setattr(readout, "load_inputs", lambda cfg: (rows, vectors, targets, provenance))
    cfg = readout.Config(root=str(tmp_path), output="fixture_results", bootstrap_draws=24)
    readout.fit(cfg)
    readout.summarize(cfg)
    import json

    result = json.loads((tmp_path / cfg.output / "summary.json").read_text())
    metrics = result["properties"]["fixture_score"]["all"]["models"]
    assert metrics["answer"]["r2"]["value"] > 0.9
    assert metrics["answer_pca256"]["reason"].startswith("model unavailable")
    assert (tmp_path / cfg.output / "analysis_complete.json").exists()
    checkpoints = {
        p.name: p.stat().st_mtime_ns for p in (tmp_path / cfg.output).glob("fold*_bundle*.npz")
    }
    readout.fit(cfg)
    assert checkpoints == {
        p.name: p.stat().st_mtime_ns for p in (tmp_path / cfg.output).glob("fold*_bundle*.npz")
    }
    provenance["fixture"] = "changed test regime"
    with pytest.raises(ValueError, match="output belongs"):
        readout.fit(cfg)
