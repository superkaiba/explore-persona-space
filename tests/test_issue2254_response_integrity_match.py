"""Focused tests for issue #2254 response-integrity matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scripts.issue2254_response_integrity_match as mod

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_selected_coordinate_matches_effective_seeds() -> None:
    context = [
        mod._selected_coordinate("ctx", seed, draw)
        for seed in (42, 43)
        for draw in range(5)
    ]
    answer = [mod._selected_coordinate("t1", 42, draw) for draw in range(6)]
    assert sorted(value for value in context if value is not None) == list(range(42, 48))
    assert answer == list(range(42, 48))


def test_point_match_requires_quality_and_cell_floor() -> None:
    context = {"integrity": {"pass_rate_80": {"estimate": 0.95}}}
    candidate = {
        "quality_vs_context": {
            "mean": {"estimate": -4.9},
            "pass_rate_80": {"estimate": -0.04},
        },
        "integrity": {"pass_rate_80": {"estimate": 0.91}},
    }
    assert mod._point_match(candidate, context)
    candidate["quality_vs_context"]["mean"]["estimate"] = -5.1
    assert not mod._point_match(candidate, context)


def test_nested_selection_reselects_by_quality_only() -> None:
    indices = np.tile(np.arange(20), (5, 1))
    baseline = np.zeros(20)
    context = {
        "_integrity_q": np.full(20, 95.0),
        "_pass_q": np.ones(20),
        "_trait_q": np.full(20, 10.0),
    }
    near = {
        "code": "t1",
        "_integrity_q": np.full(20, 94.0),
        "_pass_q": np.ones(20),
        "_trait_q": np.full(20, 20.0),
        "_trait_diff_context_q": np.full(20, 10.0),
    }
    farther = {
        "code": "t2",
        "_integrity_q": np.full(20, 92.0),
        "_pass_q": np.ones(20),
        "_trait_q": np.full(20, 90.0),
        "_trait_diff_context_q": np.full(20, 80.0),
    }
    result = mod._nested_selection(
        behavior="evil",
        context=context,
        candidates=[near, farther],
        indices=indices,
        baseline_q=baseline,
    )
    assert result["selection_counts"]["t1"] == 5
    assert result["selection_counts"]["t2"] == 0
    assert (
        result["direct_trait_difference_conditional_on_resampled_point_match"]["estimate"]
        == 10.0
    )


def test_rubric_ignores_trait_but_checks_language_and_task() -> None:
    rubric = mod.RESPONSE_INTEGRITY_RUBRIC
    assert "EXPLICITLY IGNORE factual correctness" in rubric
    assert "Task connection" in rubric
    assert "Language and script continuity" in rubric


def test_real_registry_has_exact_matched_roster_and_horizons() -> None:
    items, sources = mod.build_registry(REPO_ROOT)
    assert len(items) == 1_680
    assert len(sources) == 14
    assert {source["n_items"] for source in sources.values()} == {120}
    assert all(
        source["n_common_horizon_truncated"] == 0
        for cell_id, source in sources.items()
        if "__aans__" not in cell_id
    )
    assert sources["evil__rb__aans__L14__c4"]["n_common_horizon_truncated"] == 114
    assert (
        sources["sycophancy__rb__aans__L14__c2"]["n_trait_items_complete"]
        == 95
    )
