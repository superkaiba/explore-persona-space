import numpy as np
import pytest

from scripts.issue1739_covariance_ablation import sha256
from scripts.issue1739_luna_analysis import (
    METHODS,
    difference_bounds,
    empirical_draws,
    observed_stats,
    repeatability,
    selected_sets,
    verify_response_sources,
)


def test_frozen_single_shard_digest_cannot_validate_changed_or_extra_shards(tmp_path):
    responses = tmp_path / "responses"
    responses.mkdir()
    with pytest.raises(ValueError, match="Missing"):
        verify_response_sources(tmp_path, {"source_response_sha256": {}})
    path = responses / "selected_responses_000.jsonl"
    path.write_text('{"ci":1}\n')
    manifest = {"source_response_sha256": sha256(path)}
    verify_response_sources(tmp_path, manifest)
    (responses / "selected_responses_001.jsonl").write_text('{"ci":2}\n')
    with pytest.raises(ValueError, match="exactly one"):
        verify_response_sources(tmp_path, manifest)
    (responses / "selected_responses_001.jsonl").unlink()
    path.write_text('{"ci":3}\n')
    with pytest.raises(ValueError, match="Changed"):
        verify_response_sources(tmp_path, manifest)


def test_missingness_bounds_do_not_impute_unassessable_as_negative():
    row = observed_stats([100, 0, np.nan, np.nan])
    assert row["fraction_scored"] == 0.5
    assert row["fraction_full_bounds"] == [0.25, 0.75]
    assert row["mean_full_bounds"] == [25.0, 75.0]
    assert row["coverage"] == 0.5
    assert observed_stats([np.nan])["fraction_scored"] is None
    with pytest.raises(ValueError, match="Empty"):
        observed_stats([])


def test_shared_bootstrap_weights_preserve_identical_arm_outcomes():
    scores = np.array([0, 75, 100, np.nan, 25])
    mask = np.array([[1, 1], [1, 1], [1, 1], [0, 0], [1, 1]])
    result = empirical_draws(scores, mask, n_boot=250)
    for draws in result.values():
        np.testing.assert_equal(draws[:, 0], draws[:, 1])


def test_audit_keeps_missingness_separate_from_binary_agreement():
    primary = {
        1: {"status": "scored", "score": 100, "positive": True},
        2: {"status": "unassessable", "score": None, "positive": None},
    }
    repeat = {
        1: {"status": "scored", "score": 75, "positive": True},
        2: {"status": "scored", "score": 0, "positive": False},
    }
    row = repeatability(primary, repeat)
    assert row["category_agreement"] == 0.5
    assert row["binary_agreement"] == 1
    assert row["mean_absolute_score_difference"] == 25
    assert row["n_both_scored"] == 1
    assert row["kappa"] is None


def test_unknown_shared_outcome_cancels_in_equal_size_arm_difference():
    scores = [np.nan, 100, 0]
    assert difference_bounds(scores, [1, 1, 0], [1, 0, 1]) == [0.5, 0.5]
    assert difference_bounds(scores, [1, 1, 0], [1, 0, 1], binary=False) == [50, 50]


@pytest.fixture
def memberships():
    return [
        dict(
            method=method,
            behavior="all" if method == "random" else "hallucination",
            ci=i,
            rank=None if method == "random" else i + 1,
        )
        for method in METHODS
        for i in range(1000 if method == "random" else 200)
    ]


def test_null_rank_random_order_and_ranked_prefixes_are_preserved(memberships):
    memberships.reverse()
    result = selected_sets(memberships, "hallucination")
    assert result["random/1000"] == list(range(999, -1, -1))
    assert result["preimage_cosine/50"] == list(range(50))
    partition = {str(i): i // 2 for i in range(1000)}
    diverse = selected_sets(memberships, "hallucination", partition)
    assert diverse["random/1000"] == list(range(999, -1, -2))
    assert diverse["preimage_cosine/50"] == list(range(0, 50, 2))


@pytest.mark.parametrize("field", ["ci", "rank"])
def test_duplicate_membership_ids_or_ranks_fail(memberships, field):
    memberships[1][field] = memberships[0][field]
    with pytest.raises(ValueError, match=r"coverage|ranks"):
        selected_sets(memberships, "hallucination")
