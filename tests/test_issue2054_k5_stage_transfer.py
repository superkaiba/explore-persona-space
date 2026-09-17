"""Independent numerical and resume contracts for K5 stage transfer."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts import issue2054_k5_stage_transfer as analysis


def panels():
    """Small synthetic fixtures exercise only numerical code, never production inputs."""
    rng = np.random.default_rng(2054)
    result = []
    for n, shift in ((95, 17), (100, -8)):
        x = rng.normal(size=(n, 4)) * [0.2, 2, 5, 10] + shift
        y = x @ rng.normal(size=(4, 4)) + rng.normal(size=(n, 4))
        result.append(
            {
                "x": x,
                "y": y,
                "ids": [f"conversation_{i}" for i in range(n)],
                "membership": np.arange(n) % 5,
                "caps": np.zeros((n, 5), dtype=bool),
            }
        )
    return result


def oracle(panel, fold, penalty):
    """Independent row-based standardized ridge, without moment subtraction."""
    train = panel["membership"] != fold
    x, y = panel["x"][train], panel["y"][train]
    mean, sd = x.mean(0), x.std(0) + 1e-9
    z = (x - mean) / sd
    w = np.linalg.solve(z.T @ z + penalty * np.eye(x.shape[1]), z.T @ (y - y.mean(0)))
    raw = w / sd[:, None]
    return raw, y.mean(0) - mean @ raw


def references(panel, penalty):
    """Reference scores use the row-based oracle and canonical metric helper."""
    result = []
    for fold in range(5):
        test = panel["membership"] == fold
        raw, bias = oracle(panel, fold, penalty)
        score = analysis.fit.score(panel["x"][test] @ raw + bias, panel["y"][test])
        score = json.loads(json.dumps(score))  # Published JSON stores retrieval keys as strings.
        result.append(
            {
                "fold": fold,
                "n_train": int((~test).sum()),
                "n_test": int(test.sum()),
                "ridge": {"best_lambda": penalty, "n_train": int((~test).sum())},
                "metrics": {"own": score},
                "train_ids_sha256": analysis.digest(np.asarray(panel["ids"])[~test].tolist()),
                "test_ids_sha256": analysis.digest(np.asarray(panel["ids"])[test].tolist()),
            }
        )
    return result


def test_restoration_matches_independent_standardized_ridge_and_raw_transfer():
    source, target = panels()
    bank = [analysis.geometry.moments_by_fold(p) for p in (source, target)]
    penalties = [1.2, 3.4]
    for fold in (0, 3):
        maps, biases, identity, n, means_x, means_y = analysis.restore_maps(bank, fold, penalties)
        for i, panel in enumerate((source, target)):
            a, b = oracle(panel, fold, penalties[i])
            np.testing.assert_allclose(maps[i], a, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(biases[i], b, rtol=1e-9, atol=1e-9)
            train = panel["membership"] != fold
            assert n[i] == int(train.sum())
            np.testing.assert_allclose(identity[i], (panel["y"][train] - panel["x"][train]).mean(0))
        x = analysis.tensor(target["x"][target["membership"] == fold])
        predictions, correction = analysis.target_predictions(
            x, maps, biases, identity, means_x, means_y
        )
        a, b = oracle(source, fold, penalties[0])
        np.testing.assert_allclose(predictions[0], x.numpy() @ a + b, rtol=1e-9, atol=1e-9)
        train = target["membership"] != fold
        expected_correction = (target["y"][train] - target["x"][train] @ a - b).mean(0)
        np.testing.assert_allclose(correction, expected_correction, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(predictions[2], predictions[0] + correction)
        np.testing.assert_allclose(predictions[3], x + identity[0])
        np.testing.assert_allclose(predictions[4], x + identity[1])


def test_heldout_changes_cannot_change_maps_or_bias_calibration():
    source, target = panels()
    bank = [analysis.geometry.moments_by_fold(p) for p in (source, target)]
    before = analysis.restore_maps(bank, 2, [1.2, 3.4])
    changed = copy.deepcopy([source, target])
    for panel in changed:
        test = panel["membership"] == 2
        panel["y"][test] += 15
        panel["x"][test] -= 7
    after = analysis.restore_maps(
        [analysis.geometry.moments_by_fold(p) for p in changed], 2, [1.2, 3.4]
    )
    for first, second in zip(before, after, strict=True):
        np.testing.assert_allclose(first, second, rtol=1e-8, atol=1e-8)
    x = analysis.tensor(target["x"][target["membership"] == 2])
    first = analysis.target_predictions(x, *before[:3], *before[4:])
    second = analysis.target_predictions(x, *after[:3], *after[4:])
    np.testing.assert_allclose(first[1], second[1], rtol=1e-8, atol=1e-8)
    audit, paired = analysis.fold_audit(source, target, 2)
    assert audit["source_train_target_test_overlap"] == 0
    assert audit["paired_target_test_n"] == 19
    assert paired.sum() == 19
    source["membership"][2] = 1
    with pytest.raises(RuntimeError, match="source training contains a target test"):
        analysis.fold_audit(source, target, 2)


def test_evaluate_fold_matches_canonical_metrics_and_stores_exact_error_contributions():
    source, target = panels()
    bank = [analysis.geometry.moments_by_fold(p) for p in (source, target)]
    refs = [
        references(p, penalty)[0] for p, penalty in zip((source, target), (1.2, 3.4), strict=True)
    ]
    phases = []
    record, arrays = analysis.evaluate_fold(source, target, bank, 0, refs, phases.append)
    assert phases == ["restore_maps", "source_own_parity", "target_metrics", "paired_sensitivity"]
    y = target["y"][target["membership"] == 0]
    maps, biases, identity, _, means_x, means_y = analysis.restore_maps(bank, 0, [1.2, 3.4])
    pred, _ = analysis.target_predictions(
        analysis.tensor(target["x"][target["membership"] == 0]),
        maps,
        biases,
        identity,
        means_x,
        means_y,
    )
    np.testing.assert_allclose(arrays["squared_errors"], np.square(pred.numpy() - y).sum(2))
    np.testing.assert_allclose(arrays["sst_contributions"], np.square(y - y.mean(0)).sum(1))
    assert arrays["squared_errors"].dtype == np.float64
    assert arrays["frozen_prediction"].dtype == np.float32
    assert arrays["target_own_prediction"].dtype == np.float32
    assert record["paired_sensitivity"]["n_test"] == 19
    assert record["coverage"]["target_test_n"] == 20
    assert max(record["own_parity_r2_absolute_delta"].values()) < 1e-9
    for i, mode in enumerate(analysis.MODES):
        expected = analysis.fit.score(pred[i].numpy(), y)
        actual = record["metrics"][mode]
        assert actual["r2"] == pytest.approx(expected["r2"], abs=1e-12)
        assert actual["r2"] == pytest.approx(
            1 - arrays["squared_errors"][i].sum() / arrays["sst_contributions"].sum(), abs=1e-12
        )
        for metric in ("euclidean", "cosine"):
            for k in (1, 5, 10):
                assert (
                    actual["retrieval"][metric]["acc_at_k"][str(k)]
                    == expected["retrieval"][metric]["acc_at_k"][k]
                )
                assert actual["retrieval"][metric]["chance_at_k"][str(k)] == k / len(y)


@pytest.mark.parametrize("side", [0, 1])
def test_evaluation_rejects_changed_published_endpoint(side):
    source, target = panels()
    bank = [analysis.geometry.moments_by_fold(p) for p in (source, target)]
    refs = [
        references(p, penalty)[0] for p, penalty in zip((source, target), (1.2, 3.4), strict=True)
    ]
    refs[side]["metrics"]["own"]["r2"] += 0.001
    with pytest.raises(RuntimeError, match="published own R2 changed"):
        analysis.evaluate_fold(source, target, bank, 0, refs, lambda phase: None)


def test_load_bank_rejects_changed_hash_fold_ids_and_incomplete_answer_mask(tmp_path):
    panel = panels()[0]
    path = tmp_path / "bank.npz"
    values = {
        "conv_id": np.asarray(panel["ids"]),
        "v_C": panel["x"],
        "v_A": panel["y"],
        "cap_mask": panel["caps"],
    }
    np.savez(path, **values)
    ref = {
        "cell": "fixture",
        "local": str(path),
        "sha256": analysis.k3.sha(path),
        "own_folds": references(panel, 1.2),
    }
    folds = dict(zip(panel["ids"], panel["membership"].tolist(), strict=True))
    loaded = analysis.load_bank(ref, folds, dimension=4)
    np.testing.assert_array_equal(loaded["x"], panel["x"])
    bad = copy.deepcopy(ref)
    bad["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bank content changed"):
        analysis.load_bank(bad, folds, dimension=4)
    bad = copy.deepcopy(ref)
    bad["own_folds"][0]["train_ids_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="train conversation IDs changed"):
        analysis.load_bank(bad, folds, dimension=4)
    values["cap_mask"] = values["cap_mask"][:, :4]
    np.savez(path, **values)
    ref["sha256"] = analysis.k3.sha(path)
    with pytest.raises(ValueError, match="complete-five cap mask"):
        analysis.load_bank(ref, folds, dimension=4)


def test_resume_rejects_changed_method_input_code_and_prediction_content(tmp_path):
    manifest, sources = {"fold_map": {"a": 0}}, {"driver": "first"}
    fp = analysis.fingerprint(manifest, "a" * 40, sources)
    arrays = {
        "test_conv_id": np.array(["a"]),
        "frozen_prediction": np.ones((1, 2), dtype=np.float32),
    }
    analysis.save_packet(tmp_path, "Chat", 0, {"fold": 0}, arrays, fp)
    assert analysis.resume_packet(tmp_path, "Chat", 0, fp)["array_bytes"] > 0
    changed = [
        analysis.fingerprint({"fold_map": {"a": 1}}, "a" * 40, sources),
        analysis.fingerprint(manifest, "b" * 40, sources),
        analysis.fingerprint(manifest, "a" * 40, {"driver": "changed"}),
    ]
    for fingerprint in changed:
        with pytest.raises(RuntimeError, match="resume fingerprint"):
            analysis.resume_packet(tmp_path, "Chat", 0, fingerprint)
    old = analysis.METHOD["version"]
    try:
        analysis.METHOD["version"] = old + 1
        changed_method = analysis.fingerprint(manifest, "a" * 40, sources)
    finally:
        analysis.METHOD["version"] = old
    with pytest.raises(RuntimeError, match="resume fingerprint"):
        analysis.resume_packet(tmp_path, "Chat", 0, changed_method)
    packet = tmp_path / "folds/Chat__fold0.npz"
    packet.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="resume array content changed"):
        analysis.resume_packet(tmp_path, "Chat", 0, fp)


def test_summary_uses_equal_fold_mean_and_ratio_of_means_not_mean_ratio():
    source, target = panels()
    bank = [analysis.geometry.moments_by_fold(p) for p in (source, target)]
    refs = [references(p, penalty) for p, penalty in zip((source, target), (1.2, 3.4), strict=True)]
    rows = []
    for fold in range(5):
        record, _ = analysis.evaluate_fold(
            source, target, bank, fold, [r[fold] for r in refs], lambda phase: None
        )
        record["label"] = "Chat"
        record["metrics"]["frozen"]["r2"] = [0.3, 0.2, 0.1, -0.2, 0.5][fold]
        record["metrics"]["target_own"]["r2"] = [0.6, 0.5, 0.4, 0.3, 0.2][fold]
        rows.append(record)
    result = analysis.summarize(rows)[0]
    assert result["metrics"]["frozen"]["r2"] == pytest.approx(0.18)
    assert result["frozen_retention"] == pytest.approx(0.18 / 0.4)
    assert result["target_n"] == 100
    assert analysis.summarize(rows[:-1]) == []


def test_source_hash_verification_reaches_git_and_rejects_changed_driver(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.org"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    paths = analysis.source_files()
    # Git work-tree configuration allows the real source files to remain under
    # their original path while the isolated repository stores fixture blobs.
    for relative, path in paths.items():
        dest = repo / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    subprocess.run(
        ["git", "-C", str(repo), "add", *paths],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    # Use the real git executable with a repository-path environment boundary.
    monkeypatch.setenv("GIT_DIR", str(repo / ".git"))
    actual = analysis.source_hashes(sha)
    assert (
        actual["scripts/issue2054_k5_stage_transfer.py"]
        == hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest()
    )
    driver = repo / "scripts/issue2054_k5_stage_transfer.py"
    driver.write_text("changed committed fixture\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "scripts/issue2054_k5_stage_transfer.py"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "changed"], check=True)
    changed_sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    with pytest.raises(RuntimeError, match="source differs from committed"):
        analysis.source_hashes(changed_sha)
