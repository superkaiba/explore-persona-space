"""Small numerical fixtures test estimators, never stand in for experiment results."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "issue2588_chat_rank.py"
spec = importlib.util.spec_from_file_location("issue2588_chat_rank", SCRIPT)
rank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rank)


def test_vectorized_curve_matches_explicit_projected_maps():
    rng = np.random.default_rng(7)
    predictions, targets = rng.normal(size=(2, 29, 8))
    intercept = rng.normal(size=8)
    vectors, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    curve = rank.r2_curve_from_top_right_vectors(predictions, targets, intercept, vectors)
    explicit = [
        rank.pooled_r2(
            (predictions - intercept) @ vectors[:, :k] @ vectors[:, :k].T + intercept, targets
        )
        for k in range(9)
    ]
    np.testing.assert_allclose(curve, explicit, rtol=1e-12, atol=1e-12)


def test_reconstruction_and_rank_match_parent_numerics():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(130, 9)).astype(np.float32)
    w = rng.normal(size=(9, 9)).astype(np.float32)
    y = x @ w + rng.normal(size=(130, 9)).astype(np.float32) * 0.2 + 3
    payload = rank.reconstruct(x[:80], y[:80], x[80:100], y[80:100], x[100:], y[100:], 3.0)
    # Independent direct solution of the same ridge objective, no refitted lambda.
    mean, sd = x[:80].astype(float).mean(0), x[:80].astype(float).std(0, ddof=1) + 1e-9
    xn = (x[:80].astype(float) - mean) / sd
    target_mean = y[:80].astype(float).mean(0)
    direct = np.linalg.solve(xn.T @ xn + 3 * np.eye(9), xn.T @ (y[:80] - target_mean))
    np.testing.assert_allclose(payload["W"], direct, rtol=2e-7, atol=2e-7)
    result = rank.reduced_rank(payload, x[:80])
    curve = np.array(result["rank_curve"]["validation_r2"])
    chosen = result["rank"]
    threshold = 1 - 1.1 * (1 - result["full_validation_r2"])
    assert curve[chosen] >= threshold - 1e-12
    assert not np.any(curve[:chosen] >= threshold - 1e-12)
    assert result["selected_rank_test_r2"] == result["rank_curve"]["test_r2"][chosen]
    # Changing test labels must not change the training basis or validation-chosen rank.
    alternative = rank.reduced_rank({**payload, "target_test": y[100:][::-1]}, x[:80])
    assert alternative["rank"] == chosen


def test_rank_zero_and_unattainable_threshold():
    assert rank.rank_at_threshold(np.array([0.5, 0.4, 0.6]), 0.49) == 0
    with pytest.raises(ValueError, match="No rank"):
        rank.rank_at_threshold(np.array([0.5, 0.4, 0.6]), 0.9)


def test_optional_full_spectrum_preserves_existing_rank_result():
    rng = np.random.default_rng(2588)
    x, y = rng.normal(size=(2, 75, 6)).astype(np.float32)
    payload = rank.reconstruct(x[:45], y[:45], x[45:60], y[45:60], x[60:], y[60:], 10)
    original = rank.reduced_rank(payload, x[:45])
    expanded = rank.reduced_rank(payload, x[:45], include_spectrum=True)
    spectrum = expanded["fitted_output_spectrum"].pop("eigenvalues")
    assert expanded == original
    xn = (x[:45].astype(float) - payload["xmu"].astype(float)) / payload["xsd"].astype(float)
    singular_values = np.linalg.svd(xn @ payload["W"].astype(float), compute_uv=False)
    np.testing.assert_allclose(spectrum, singular_values**2 / 45, rtol=1e-11, atol=1e-12)


def test_split_loader_checks_manifest_and_lexicographic_order(tmp_path):
    directory = tmp_path / "capture" / "train_10k"
    layer = directory / "L00"
    layer.mkdir(parents=True)
    ids = ["train_10k_2", "train_10k_10"]
    (directory / "rows.json").write_text(json.dumps({"rows": [{"row_id": i} for i in ids]}))
    x = np.array([[2, 3], [10, 11]], dtype=np.float32)
    np.savez(layer / "shard000.npz", row_ids=ids, x_prompt_last=x, y_ans=x + 5)
    actual, targets = rank.load_split(tmp_path, "train_10k", 0, "prompt_last", 2)
    np.testing.assert_array_equal(actual, x[::-1])
    np.testing.assert_array_equal(targets, (x + 5)[::-1])
    (directory / "rows.json").write_text(json.dumps({"rows": [{"row_id": "wrong"}]}))
    with pytest.raises(ValueError, match="row-manifest mismatch"):
        rank.load_split(tmp_path, "train_10k", 0, "prompt_last", 2)


def test_cli_requires_immutable_revision_before_output(tmp_path):
    with pytest.raises(SystemExit) as exc:
        rank.main(
            [
                "--source-root",
                str(tmp_path),
                "--hf-revision",
                "main",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
    assert exc.value.code == 2
    assert not (tmp_path / "output").exists()


def test_durable_verification_checks_both_hash_domains_and_missing_names(tmp_path):
    from huggingface_hub.hf_api import RepoFile

    plain = tmp_path / "run_identity.json"
    tensor = tmp_path / "capture" / "test_1000" / "L00" / "shard000.npz"
    plain.write_text('{"surface":"generic"}')
    tensor.parent.mkdir(parents=True)
    tensor.write_bytes(b"unit-test-bytes-not-an-experiment-tensor")
    prefix = f"{rank.PANEL_PREFIX}/generic/qwen3-chat-v1/q3_8b/nothink"
    manifest = {
        str(p.relative_to(tmp_path)): {"bytes": p.stat().st_size, "sha256": rank.sha256(p)}
        for p in (plain, tensor)
    }
    plain_oid = hashlib.sha1(
        f"blob {plain.stat().st_size}\0".encode() + plain.read_bytes()
    ).hexdigest()
    records = [
        RepoFile(path=f"{prefix}/run_identity.json", size=plain.stat().st_size, oid=plain_oid),
        RepoFile(
            path=f"{prefix}/analysis_tensors/capture/test_1000/L00/shard000.npz",
            size=tensor.stat().st_size,
            oid="pointeroid",
            lfs={"size": tensor.stat().st_size, "oid": rank.sha256(tensor), "pointerSize": 130},
        ),
    ]
    calls = []

    def get_paths_info(repo_id, paths, *, expand=False, revision=None, repo_type=None, token=None):
        calls.append((repo_id, paths, revision, repo_type))
        return records

    api = SimpleNamespace(get_paths_info=get_paths_info)
    result = rank.verify_durable_inputs(tmp_path, "a", "qwen3-chat-v1", "a" * 40, manifest, api)
    assert result["content_verified"] and len(calls) == 1
    records[1].lfs.sha256 = "0" * 64
    with pytest.raises(ValueError, match="content mismatch"):
        rank.verify_durable_inputs(tmp_path, "a", "qwen3-chat-v1", "a" * 40, manifest, api)
    records.pop()
    with pytest.raises(ValueError, match="name-set mismatch"):
        rank.verify_durable_inputs(tmp_path, "a", "qwen3-chat-v1", "a" * 40, manifest, api)
