"""Numeric, split, and reliability invariants for natural-prompt extraction."""

import numpy as np
import pytest
from explore_persona_space.experiments.issue_1739.natural_extremes import (
    SelectionError,
    assert_disjoint_partitions,
    heldout_score_gap,
    select_literal_endpoints,
    select_prompt_tails,
    split_half_weights,
)


def _inputs(scores):
    """Construct independent source identities for small selection fixtures."""
    n = len(scores)
    return {
        "scores": np.asarray(scores, dtype=float),
        "context_ids": [f"context:{i}" for i in range(n)],
        "group_ids": [f"source:group:{i}" for i in range(n)],
        "prompt_hashes": [f"normalized-prompt-hash:{i}" for i in range(n)],
        "extraction_mask": np.ones(n, dtype=bool),
        "eligible_mask": np.ones(n, dtype=bool),
        "evaluation_mask": np.zeros(n, dtype=bool),
        "score_range": (0.0, 100.0),
    }


def _select(inputs, **kwargs):
    """Use explicit behavior and fold keys consistently in the fixtures."""
    return select_prompt_tails(**inputs, behavior="behavior", fold="test-fold", **kwargs)


def test_missing_scores_keep_equal_prompt_weight_and_activation_estimator():
    inputs = _inputs(
        [
            [0, 0, 0, 0, 0],
            [10, 10, 10, np.nan, np.nan],
            [90, 90, 90, np.nan, np.nan],
            [100, 100, 100, 100, 100],
            [100, 100, np.nan, np.nan, np.nan],
        ]
    )
    result = _select(inputs, q=0.5)
    assert set(result.high_indices) == {2, 3}
    assert set(result.low_indices) == {0, 1}
    assert result.metadata["n_candidates"] == 4
    assert result.metadata["n_below_valid_floor"] == 1
    np.testing.assert_allclose(result.high_prompt_weights, [0, 0, 0.5, 0.5, 0])
    np.testing.assert_allclose(result.low_prompt_weights, [0.5, 0.5, 0, 0, 0])
    np.testing.assert_allclose(result.high_weights[2], [1 / 6, 1 / 6, 1 / 6, 0, 0])
    # Missing-response activations may be absent: only positive-weight rows are consumed.
    acts = np.arange(5 * 5 * 2, dtype=float).reshape(5, 5, 2)
    actual = np.einsum("nk,nkd->d", result.high_weights - result.low_weights, acts)
    expected = (acts[2, :3].mean(0) + acts[3].mean(0)) / 2
    expected -= (acts[0].mean(0) + acts[1, :3].mean(0)) / 2
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("key", ["group_ids", "prompt_hashes"])
def test_source_or_content_leakage_is_rejected_before_score_selection(key):
    inputs = _inputs([[0] * 5, [100] * 5, [50] * 5])
    inputs["extraction_mask"][2] = False
    inputs["evaluation_mask"][2] = True
    inputs[key][2] = inputs[key][0]
    with pytest.raises(SelectionError, match="Extraction/evaluation leakage") as exc:
        _select(inputs)
    assert exc.value.metadata["key_type"] == key


def test_direct_partition_check_rejects_identical_context_in_distinct_rows():
    with pytest.raises(SelectionError, match="context_ids"):
        assert_disjoint_partitions(
            context_ids=["same", "same"],
            group_ids=["one", "two"],
            prompt_hashes=["a", "b"],
            extraction_mask=np.array([True, False]),
            evaluation_mask=np.array([False, True]),
        )


def test_exclusions_precede_ranking_and_do_not_read_evaluation_scores():
    inputs = _inputs([[0] * 5, [10] * 5, [80] * 5, [999] * 5, [np.inf] * 5])
    inputs["eligible_mask"][3] = False
    inputs["extraction_mask"][4] = False
    inputs["evaluation_mask"][4] = True
    result = _select(inputs)
    assert result.high_indices.tolist() == [2]
    assert result.low_indices.tolist() == [0]
    assert result.metadata["n_candidates"] == 3
    assert np.isnan(result.prompt_scores[3:]).all()


def test_duplicate_prompt_hashes_require_explicit_upstream_exclusion():
    inputs = _inputs([[0] * 5, [100] * 5, [90] * 5])
    inputs["prompt_hashes"][2] = inputs["prompt_hashes"][1]
    with pytest.raises(SelectionError, match="Duplicate normalized prompt"):
        _select(inputs)
    inputs["eligible_mask"][2] = False
    assert _select(inputs).metadata["n_candidates"] == 2


def test_hash_ties_are_deterministic_permutation_invariant_and_salt_sensitive():
    inputs = _inputs([[0] * 5] * 30 + [[100] * 5] * 30)
    first = _select(inputs, q=0.1)
    second = _select(inputs, q=0.1)
    np.testing.assert_array_equal(first.high_indices, second.high_indices)
    permutation = np.random.default_rng(17).permutation(60)
    shuffled = {
        key: value if key == "score_range" else np.asarray(value)[permutation]
        for key, value in inputs.items()
    }
    permuted = _select(shuffled, q=0.1)
    assert set(permutation[permuted.high_indices]) == set(first.high_indices)
    assert set(permutation[permuted.low_indices]) == set(first.low_indices)
    salted = _select(inputs, q=0.1, tie_salt="1")
    assert set(salted.high_indices) != set(first.high_indices)
    assert first.metadata["high"]["n_boundary_ties_pool"] == 30
    assert first.metadata["high"]["n_boundary_ties_selected"] == 6


def test_shared_boundary_ties_never_put_a_prompt_in_both_tails():
    inputs = _inputs([[0] * 5] + [[50] * 5] * 7 + [[100] * 5])
    result = _select(inputs, q=0.5)
    assert len(result.high_indices) == 4
    assert len(result.low_indices) == 4
    assert not set(result.high_indices) & set(result.low_indices)


@pytest.mark.parametrize("q,k", [(0.01, 2), (0.05, 10), (0.10, 20)])
def test_prespecified_quantiles_use_eligible_population(q, k):
    inputs = _inputs(np.repeat(np.linspace(0, 100, 200)[:, None], 5, axis=1))
    result = _select(inputs, q=q)
    assert len(result.high_indices) == len(result.low_indices) == k
    assert min(result.high_indices) == 200 - k
    assert max(result.low_indices) == k - 1


@pytest.mark.parametrize(
    "scores,message",
    [
        ([[np.nan] * 5] * 3, "Fewer than two"),
        ([[50] * 5] * 3, "Constant eligible prompt means"),
        ([[10] * 5], "Fewer than two"),
    ],
)
def test_unsupported_contrasts_fail_with_metadata(scores, message):
    with pytest.raises(SelectionError, match=message) as exc:
        _select(_inputs(scores))
    assert "n_candidates" in exc.value.metadata


def test_native_fraction_scores_are_not_coerced_to_percent():
    inputs = _inputs([[0] * 5, [0.1] * 5, [0.8] * 5, [1] * 5])
    inputs["score_range"] = (0.0, 1.0)
    result = _select(inputs, q=0.5)
    assert result.metadata["high"]["mean_score"] == pytest.approx(0.9)
    assert result.metadata["low"]["mean_score"] == pytest.approx(0.05)
    inputs["scores"][0, 0] = 10
    with pytest.raises(ValueError, match="declared native range"):
        _select(inputs)


def test_literal_endpoints_require_distinct_groups_and_keep_unequal_sizes():
    inputs = _inputs([[0] * 5] * 20 + [[100] * 5] * 24 + [[50] * 5] * 5)
    result = select_literal_endpoints(**inputs)
    assert len(result.low_indices) == 20
    assert len(result.high_indices) == 24
    assert result.high_weights.sum() == pytest.approx(1)
    assert result.low_weights.sum() == pytest.approx(1)
    inputs["group_ids"][1] = inputs["group_ids"][0]
    with pytest.raises(SelectionError, match="Insufficient distinct-group") as exc:
        select_literal_endpoints(**inputs)
    assert exc.value.metadata["endpoint_low_groups"] == 19
    assert exc.value.metadata["endpoint_high_groups"] == 24


def test_literal_endpoints_do_not_rename_relative_extremes_as_maximum():
    inputs = _inputs([[10] * 5] * 25 + [[90] * 5] * 25)
    with pytest.raises(SelectionError, match="Insufficient distinct-group") as exc:
        select_literal_endpoints(**inputs)
    assert exc.value.metadata["endpoint_low_prompts"] == 0
    assert exc.value.metadata["endpoint_high_prompts"] == 0


def test_shared_train_only_groups_are_diagnostic_and_halves_keep_groups_intact():
    inputs = _inputs([[0] * 5] * 5 + [[100] * 5] * 5)
    inputs["group_ids"] = [
        "shared",
        "shared",
        "lo1",
        "lo2",
        "lo3",
        "shared",
        "hi1",
        "hi2",
        "hi2",
        "hi3",
    ]
    selected = _select(inputs, q=0.5)
    assert selected.metadata["n_cross_tail_groups"] == 1
    halves = split_half_weights(selected, group_ids=inputs["group_ids"], behavior="b", fold="f")
    half_groups = []
    for half in halves:
        assert half.high_weights.sum() == pytest.approx(1)
        assert half.low_weights.sum() == pytest.approx(1)
        assert len(half.high_indices) and len(half.low_indices)
        half_groups.append(
            {inputs["group_ids"][i] for i in np.r_[half.high_indices, half.low_indices]}
        )
        np.testing.assert_allclose(
            half.high_prompt_weights[half.high_indices], 1 / len(half.high_indices)
        )
    assert not half_groups[0] & half_groups[1]
    assert set(np.r_[halves[0].high_indices, halves[1].high_indices]) == set(selected.high_indices)
    assert set(np.r_[halves[0].low_indices, halves[1].low_indices]) == set(selected.low_indices)


def test_half_stability_reports_insufficient_support():
    inputs = _inputs([[0] * 5, [100] * 5])
    with pytest.raises(SelectionError, match="at least two source groups"):
        split_half_weights(_select(inputs), group_ids=inputs["group_ids"], behavior="b", fold="f")


def test_reliability_uses_unseen_draws_and_preserves_a_reversed_gap():
    full_scores = np.array(
        [
            [0, 0, 80, 80, 80],
            [0, 0, 100, np.nan, np.nan],
            [100, 100, 10, 10, 10],
            [100, 100, np.nan, np.nan, np.nan],
        ],
        dtype=float,
    )
    inputs = _inputs(full_scores.copy())
    inputs["scores"][:, 2:] = np.nan
    selected = _select(inputs, min_valid=2, q=0.5)
    result = heldout_score_gap(
        selected, full_scores, draw_mask=np.array([False, False, True, True, True])
    )
    assert result["high_minus_low_score"] == -80
    assert result["high"]["n_unscored_prompts"] == 1
    assert result["low"]["mean_score"] == 90
    with pytest.raises(SelectionError, match="were used by tail selection"):
        heldout_score_gap(
            selected, full_scores, draw_mask=np.array([True, False, False, False, False])
        )


def test_reliability_rejects_draws_used_to_rank_unselected_candidates():
    scores = np.array(
        [[0, 0, np.nan, np.nan, np.nan], [100, 100, np.nan, np.nan, np.nan], [50] * 5]
    )
    selected = _select(_inputs(scores), min_valid=2)
    assert not selected.high_weights[:, 2:].any()
    assert not selected.low_weights[:, 2:].any()
    with pytest.raises(SelectionError, match="were used by tail selection"):
        heldout_score_gap(selected, scores, draw_mask=np.array([False, False, True, True, True]))
