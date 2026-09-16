"""Scientific invariants for fixed-direction transfer scoring."""

import json

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from scripts import issue1739_fixed_transfer as run


def test_both_direction_positions_use_identical_polarity_rows():
    meta = [
        {"context_id": "p", "rollout_k": 0, "side": "pos"},
        {"context_id": "p", "rollout_k": 1, "side": "pos"},
        {"context_id": "n", "rollout_k": 0, "side": "neg"},
    ]
    a = np.asarray([[2, 4], [4, 2], [1, 1]], dtype=np.float16)
    c = np.asarray([[8, 2], [8, 2], [2, 1]], dtype=np.float16)
    va, vc, audit = run.extract_directions({("t1", 19): a, ("context_end", 19): c}, meta)
    np.testing.assert_array_equal(va, [2, 2])
    np.testing.assert_array_equal(vc, [6, 1])
    assert audit["positive_rollouts"] == 2 and audit["positive_contexts"] == 1
    assert audit["judge_filtered"] is False
    with pytest.raises(ValueError, match="duplicate"):
        run.extract_directions({("t1", 19): a, ("context_end", 19): c}, [meta[0]] * 3)


def test_same_prompt_contrast_cannot_manufacture_context_direction():
    meta = [
        {"context_id": str(i), "rollout_k": 0, "side": side}
        for i, side in enumerate(["pos", "neg"])
    ]
    with pytest.raises(ValueError, match="invalid context"):
        run.extract_directions({("t1", 19): np.eye(2), ("context_end", 19): np.ones((2, 2))}, meta)


def test_answer_average_uses_exact_judged_rollouts():
    meta = [{"context_id": "c", "rollout_k": k} for k in range(3)]
    arrays = {
        ("context_end", 19): np.asarray([[1, 2], [1, 2], [1, 2]]),
        ("t1", 19): np.asarray([[2, 6], [999, 999], [6, 2]]),
    }
    label = {
        "context_id": "c",
        "rung": "r",
        "group_key": "g",
        "dv": 50.0,
        "per_rollout_scores": {"k00": 0.0, "k01": None, "k02": 100.0},
        "n_rollouts_kept": 2,
    }
    out = run.reduce_eval(arrays, meta, [label])
    np.testing.assert_array_equal(out["y"], [[4, 4]])
    assert out["rollout_audit"][0]["rollout_k"] == [0, 2]
    assert out["rollout_audit"][0]["dropped_activation_rows"] == 1
    with pytest.raises(ValueError, match="DV does not match"):
        run.kept_rollouts({**label, "dv": 49.0}, {0, 1, 2})
    with pytest.raises(ValueError, match="rollout-ID mismatch"):
        run.kept_rollouts(label, {0, 2})


def test_hallucination_requires_decided_identity_evidence():
    label = {"context_id": "h", "dv": 0.5, "n_rollouts": 3, "n_decided": 2, "n_unjudged": 1}
    with pytest.raises(ValueError, match="no retained-rollout"):
        run.kept_rollouts(label, {0, 1, 2})
    assert run.kept_rollouts(
        label, {0, 1, 2}, {"per_rollout_scores": {"k00": 100.0, "k01": None, "k02": 0.0}}
    ) == {0, 2}
    all_decided = {**label, "n_decided": 3, "n_unjudged": 0}
    assert run.kept_rollouts(all_decided, {0, 1, 2}) == {0, 1, 2}


def test_affine_pushdown_matches_canonical_stored_payload():
    from scripts.issue779_ffc_n1m_fits import apply_map

    rng = np.random.default_rng(9)
    payload = {
        "kind": "ridge",
        "W": torch.tensor(rng.normal(size=(7, 7)), dtype=torch.float32),
        "xmu": torch.tensor(rng.normal(size=7), dtype=torch.float32),
        "ymu": torch.tensor(rng.normal(size=7), dtype=torch.float32),
        "xsd": torch.tensor(rng.uniform(0.2, 3, size=7), dtype=torch.float32),
    }
    x, v = rng.normal(size=(21, 7)), rng.normal(size=7)
    np.testing.assert_allclose(
        run.map_projection(payload, x, v),
        apply_map(payload, x, torch.device("cpu")) @ v,
        rtol=1e-12,
        atol=1e-12,
    )


def test_null_statistic_is_mean_seed_rho_and_identical_paired_contrast_zero():
    rng = np.random.default_rng(21)
    n = 40
    dv = np.repeat(np.arange(10), 4).astype(float)
    p = rng.normal(size=(9, n))
    p[2] = p[1]
    p[4:] *= np.asarray([1, 10, 100, 1000, 10000])[:, None]
    result, boot = run.summarize(
        p, dv, np.repeat(np.arange(20), 2), np.asarray(["r"] * n), n_boot=100
    )
    r = result[0]
    expected = np.mean([spearmanr(row, dv).statistic for row in p[4:]])
    assert r["arms"]["shuffled_map"]["rho"] == pytest.approx(expected)
    assert abs(expected - spearmanr(p[4:].mean(0), dv).statistic) > 0.01
    assert r["differences"]["map_minus_context_native"]["delta"] == 0
    assert r["differences"]["map_minus_context_native"]["ci95"] == [0.0, 0.0]
    expected_delta = boot["r"][1] - boot["r"][4:].mean(0)
    np.testing.assert_allclose(
        r["differences"]["map_minus_shuffled_map"]["ci95"],
        np.quantile(expected_delta, [0.025, 0.975]),
    )


def test_constant_scores_are_undefined_not_zero():
    rng = np.random.default_rng(2)
    p = rng.normal(size=(9, 30))
    p[0] = 0
    result, _ = run.summarize(p, np.arange(30), np.arange(30), np.asarray(["r"] * 30), n_boot=30)
    assert result[0]["arms"]["real_answer"]["rho"] is None
    assert result[0]["differences"]["real_answer_minus_map"]["delta"] is None


def test_bare_user_hash_detects_rendered_prompt_overlap(tmp_path):
    exact, normalized = run.prompt_hashes("Question   TEXT")
    full, full_normalized = run.prompt_hashes("template Question   TEXT template")
    row = {
        "namespace": "wildchat",
        "context_id": "c",
        "exact_sha256": full,
        "normalized_sha256": full_normalized,
        "user_turn_hashes": [
            {"turn_index": 0, "exact_sha256": exact, "normalized_sha256": normalized}
        ],
    }
    path = tmp_path / "index.jsonl"
    path.write_text(json.dumps(row) + "\n")
    index = run.load_prompt_index(path)
    assert run.overlaps(index[("wildchat", "c")], set(), {normalized})
    assert not run.overlaps(index[("wildchat", "c")], {"0" * 64}, set())
    with pytest.raises(ValueError, match="missing prompt"):
        run.lookup_prompt(index, "wildchat", "absent")


def test_reconstruction_identity_perfect_and_pool_chance():
    actual = np.eye(6)
    result = run.reconstruction_metrics(actual.copy(), actual)
    assert result["r2"] == 1.0
    assert result["mean_cosine"] == 1.0
    assert result["nearest_neighbor_accuracy"] == 1.0
    assert result["retrieval_pool_size"] == 6
    assert result["retrieval_chance"] == 1.0 / 6


def test_source_model_revision_and_rollout_filename_are_enforced(tmp_path):
    exact, normalized = run.prompt_hashes("Question")
    record = {
        "namespace": "evil_extraction",
        "context_id": "c",
        "exact_sha256": exact,
        "normalized_sha256": normalized,
        "generation_metadata": [
            {
                "model": run.MODEL_NAME,
                "revision": run.INSTRUCT_REVISION,
                "fingerprint": "generated",
                "git_commit": "source-sha",
            }
        ],
        "rollouts": [
            {
                "rollout_k": 0,
                "source_file": "source.json",
                "packed_src": "raw/extraction/source.json",
            }
        ],
        "source_shards": ["shard00.jsonl"],
        "side": "pos",
        "pair": 0,
        "q_idx": 0,
    }
    path = tmp_path / "index.jsonl"
    path.write_text(json.dumps(record) + "\n")
    index = run.load_prompt_index(path)
    assert index[("evil_extraction", "c")]["generation_metadata"] == record["generation_metadata"]
    capture = [
        {
            "context_id": "c",
            "rollout_k": 0,
            "source_file": "source.json",
            "side": "pos",
            "pair": 0,
            "q_idx": 0,
        }
    ]
    audit = run.validate_capture_provenance(capture, index, "evil_extraction")
    assert audit["n_capture_rows"] == 1 and audit["revision"] == run.INSTRUCT_REVISION
    capture[0]["source_file"] = "different.json"
    with pytest.raises(ValueError, match="filename mismatch"):
        run.validate_capture_provenance(capture, index, "evil_extraction")
    capture[0]["source_file"] = "source.json"
    index[("evil_extraction", "c")]["generation_metadata"][0]["revision"] = "wrong"
    with pytest.raises(ValueError, match="model/revision mismatch"):
        run.validate_capture_provenance(capture, index, "evil_extraction")
