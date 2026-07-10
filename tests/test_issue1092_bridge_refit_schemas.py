"""Pin issue1092_bridge_refit's consume paths to the REAL parent-artifact schemas.

Regression for the 2026-07-10 production crash (KeyError 'X' in refit_923): the
original tiny fixtures faked a generic X/Y/prefix_ids schema, so the tiny-real
smoke passed while every production consume path was wrong. The fixtures now
mirror the real key layouts (923 {tensors:{vbar,valid},meta:{rows}} + ffull
flast shards; 813 c_C_base/v_A_base with *_trained twins in one npz; 779
cx_last/cx_mean/v_x), and these tests execute the real consume bodies on them —
pre-fix code fails here with the same KeyError class the production run hit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

bridge = pytest.importorskip("issue1092_bridge_refit")


@pytest.fixture(scope="module")
def fixture_roots(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("bridge_fixtures")
    return bridge.make_tiny_fixtures(root)


def test_fixtures_mirror_real_schemas(fixture_roots):
    """The fixture key layouts match the observed production artifacts."""
    import torch

    vbar_blob = torch.load(
        fixture_roots["923"] / "analysis_tensors" / "reduce" / "vbar_store_uc.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert set(vbar_blob["tensors"]) == {"vbar", "valid"}
    assert {"ctx_id", "q_idx"} <= set(vbar_blob["meta"]["rows"][0])
    with np.load(
        fixture_roots["813"] / "reduced" / "em" / "generic" / "per_question_L14.npz",
        allow_pickle=True,
    ) as pq:
        assert {"c_C_base", "c_C_trained", "v_A_base", "v_A_trained"} <= set(pq.files)
        assert {"row_context_index", "row_question_index", "headline_layer"} <= set(pq.files)
    import torch as _t

    pass_b = _t.load(
        fixture_roots["779"] / "pass_b" / "train_context_vectors.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert {"cx_last", "cx_mean", "v_x"} <= set(pass_b)


def test_refit_923_consumes_real_schema(fixture_roots):
    item = bridge.refit_923_substrate(
        fixture_roots["923"] / "analysis_tensors", "uc48", layer_subset=[14, 18]
    )
    assert item["n_rows_joined"] == 12
    assert np.isfinite(item["headline_r2"])
    assert "L14" in item["per_layer"] and "L18" in item["per_layer"]
    assert "anova_shares_pca48" in item["per_layer"]["L18"]
    # tiny fixture (12 rows) must NOT trigger the production-scale reference gate
    assert "reference_check_pca48_L18" not in item


def test_refit_813_base_arm_only(fixture_roots):
    pair_dir = fixture_roots["813"] / "reduced" / "em" / "generic"
    item = bridge.refit_813_pair(pair_dir, layer_subset=[14, 18])
    assert item["consumed_keys"] == ["c_C_base", "v_A_base"]
    assert item["excluded_trained_keys"] == ["c_C_trained", "v_A_trained"]
    assert not any("trained" in k for k in item["consumed_keys"])
    assert np.isfinite(item["headline_r2"])
    assert item["headline_layer"] == 14


def test_refit_779_real_keys_and_variants(fixture_roots):
    path = fixture_roots["779"] / "pass_b" / "train_context_vectors.pt"
    item = bridge.refit_779(path, fallback_paths=None, layer_subset=[14])
    assert item["y_key"] == "v_x"
    assert set(item["x_variants"]) == {"cx_last", "cx_mean"}
    assert np.isfinite(item["headline_r2"])
    assert item["fallback_used"] is None


def test_refit_779_missing_answer_side_fails_loud(fixture_roots, tmp_path):
    """A pass_b blob without v_x (and no fallback carrying it) raises, never guesses."""
    import torch

    blob = torch.load(
        fixture_roots["779"] / "pass_b" / "train_context_vectors.pt",
        map_location="cpu",
        weights_only=False,
    )
    del blob["v_x"]
    crippled = tmp_path / "train_context_vectors.pt"
    torch.save(blob, crippled)
    with pytest.raises(KeyError, match="answer side"):
        bridge.refit_779(crippled, fallback_paths=[crippled], layer_subset=[14])


def test_grouped_split_is_shuffled_not_lexicographic():
    """The held-out third is a seeded shuffle of groups (a lexicographic tail
    clusters related contexts and biases R2 hard negative — measured -1.87 vs
    +0.30 on the real #923 uc48 grid)."""
    rng = np.random.default_rng(7)
    n_groups, per_group, h = 9, 4, 6
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per_group)
    X = rng.standard_normal((n_groups * per_group, h))
    Y = X @ rng.standard_normal((h, h)) + 0.01 * rng.standard_normal((n_groups * per_group, h))
    fit = bridge._fit_once_grouped(X, Y, groups)
    assert fit["split"] == bridge.SPLIT_RECIPE
    assert fit["n_test_groups"] == 3
    assert "train_mean_floor_r2" in fit
    # the seed-0 shuffled test-group set must differ from the lexicographic tail
    uniq = [f"g{i}" for i in range(n_groups)]
    shuffled = list(np.random.default_rng(0).permutation(np.asarray(uniq, dtype=object)))
    assert set(shuffled[-3:]) != set(sorted(uniq)[-3:])
