"""Focused tests for #2254 administration-deviation sensitivity."""

from __future__ import annotations

import json
from dataclasses import replace

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_sensitivity as sensitivity
import scripts.issue2254_all_answer_decode_sweep as gen


def _item(opaque_id: str, cell_id: str, qi: int, seed: int):
    behavior = cell_id.split("__", 1)[0]
    route = "context" if "__revmap__context__" in cell_id else "decodeonly"
    return base.AnalysisItem(
        source_item_id=f"{cell_id}|q{qi:02d}|e{seed}",
        opaque_id=opaque_id,
        cell_id=cell_id,
        behavior=behavior,
        route=route,
        dose=1.0,
        qi=qi,
        effective_seed=seed,
        question="Question?",
        answer="Answer.",
        token_count=1,
        answer_sha256="0" * 64,
    )


def test_pass_level_exclusion_recomputes_item_mean_and_threshold() -> None:
    cell = gen.CellSpec("evil", "decodeonly", 0.0).cell_id
    item = _item("opaque", cell, 0, gen.SEED_BASE)
    outcomes = {"opaque": {0: 100, 1: 100, 2: 0, 3: 0, 4: 0}}
    items = []
    expanded = {}
    for qi in range(20):
        for offset in range(6):
            opaque = f"opaque-{qi}-{offset}"
            row = replace(
                item,
                opaque_id=opaque,
                source_item_id=f"source-{qi}-{offset}",
                qi=qi,
                effective_seed=gen.SEED_BASE + offset,
            )
            items.append(row)
            expanded[opaque] = dict(outcomes["opaque"])
    target = items[0]

    arrays, accounting = sensitivity._integrity_arrays_with_exclusions(
        items, expanded, {(target.opaque_id, 2)}
    )

    assert arrays[cell]["score"][0, 0] == 50.0
    assert arrays[cell]["passing"][0, 0] == 0.0
    assert accounting["affected_items"] == 1
    assert accounting["remaining_repeat_count_distribution"] == {"4": 1, "5": 119}


def test_decision_set_rejects_duplicate_opaque_ids() -> None:
    try:
        sensitivity._decision_set(["same", "same"], 0)
    except base.AnalysisError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate receipt ids were accepted")


def test_scenario_selection_flip_does_not_move_fixed_primary_confirmation(
    monkeypatch,
) -> None:
    frozen = {
        "behaviors": {
            behavior: {
                "selected_cell_id": f"{behavior}-frozen",
                "selected_dose": 0.125,
            }
            for behavior in gen.BEHAVIORS
        }
    }
    seen = []

    def fake_selection(behavior, arrays, questions):
        return f"{behavior}-context", [
            {
                "cell_id": f"{behavior}-scenario-flip",
                "dose": 0.25,
                "eligible": True,
                "normalized_max_distance": 0.1,
            }
        ]

    def fake_confirmation(behavior, arrays, selected_cell_id, selected_dose):
        seen.append((behavior, selected_cell_id, selected_dose))
        return {
            "selected_cell_id": selected_cell_id,
            "selected_dose": selected_dose,
            "quality_equivalence_confirmed": True,
        }

    monkeypatch.setattr(base, "_selection_rows", fake_selection)
    monkeypatch.setattr(sensitivity, "_fixed_primary_confirmation", fake_confirmation)

    analysis = sensitivity._quality_analysis({}, frozen)

    assert seen == [(behavior, f"{behavior}-frozen", 0.125) for behavior in gen.BEHAVIORS]
    for behavior in gen.BEHAVIORS:
        assert (
            analysis["behaviors"][behavior]["selection"]["selected_cell_id"]
            == f"{behavior}-scenario-flip"
        )
        assert (
            analysis["behaviors"][behavior]["fixed_primary_dose_confirmation"]["selected_cell_id"]
            == f"{behavior}-frozen"
        )


def test_selection_flip_is_not_reported_as_stable() -> None:
    primary = {
        "behaviors": {
            behavior: {
                "selection": {
                    "selected_cell_id": f"{behavior}-frozen",
                    "match_status": "selection_point_match",
                },
                "fixed_primary_dose_confirmation": {"quality_equivalence_confirmed": True},
            }
            for behavior in gen.BEHAVIORS
        }
    }
    scenario = json.loads(json.dumps(primary))
    scenario["behaviors"]["evil"]["selection"]["selected_cell_id"] = "evil-flipped"

    comparison, stable = sensitivity._comparison_to_primary(primary, scenario)

    assert comparison["evil"] == {
        "selected_cell_unchanged": False,
        "match_status_unchanged": True,
        "quality_equivalence_confirmed_unchanged": True,
        "stable": False,
    }
    assert stable is False


def test_exclusion_preflight_reports_exact_non_estimable_item() -> None:
    cell = gen.CellSpec("evil", "decodeonly", 0.5).cell_id
    item = _item("target", cell, 14, gen.SEED_BASE + 1)
    items = [item]
    for index in range(1, 5):
        items.append(
            replace(
                item,
                opaque_id=f"other-{index}",
                source_item_id=f"other-source-{index}",
                qi=index,
            )
        )

    preflight = sensitivity._exclusion_preflight(
        items, {("target", 0), ("target", 1), ("target", 2), ("other-1", 0)}
    )

    assert preflight == {
        "minimum_retained_repeats": 3,
        "excluded_decisions": 4,
        "retained_decisions": sensitivity.EXPECTED_DECISIONS - 4,
        "affected_items": 2,
        "remaining_repeat_count_distribution": {"2": 1, "4": 1, "5": 3},
        "estimable": False,
        "offending_items": [
            {
                "source_item_id": item.source_item_id,
                "opaque_item_id": "target",
                "cell_id": cell,
                "behavior": "evil",
                "dose": 1.0,
                "question_index": 14,
                "effective_seed": gen.SEED_BASE + 1,
                "excluded_pass_indices": [0, 1, 2],
                "retained_pass_indices": [3, 4],
                "retained_repeats": 2,
            }
        ],
    }


def test_non_estimable_scenario_has_no_numerical_result_fields() -> None:
    common = {
        "estimable": False,
        "offending_items": [{"opaque_item_id": "target"}],
        "excluded_decisions": 3,
    }

    record = sensitivity._non_estimable_scenario(common)

    assert record["status"] == "non_estimable_min_repeats"
    assert record["analysis_performed"] is False
    assert record["selection"] is None
    assert record["confirmation"] is None
    assert record["stability"] is None
    assert record["behaviors"] is None
    assert record["comparison_to_primary"] is None
    assert record["all_behaviors_stable"] is None


def test_scenarios_all_preflight_before_only_estimable_analysis(monkeypatch) -> None:
    scenarios = {
        name: {(f"opaque-{name}-{index}", 0) for index in range(contract["excluded_decisions"])}
        for name, contract in sensitivity.V19_SCENARIO_CONTRACT.items()
    }
    events = []

    def fake_preflight(items, excluded):
        del items
        name = next(
            scenario_name
            for scenario_name, contract in sensitivity.V19_SCENARIO_CONTRACT.items()
            if contract["excluded_decisions"] == len(excluded)
        )
        events.append(("preflight", name))
        contract = sensitivity.V19_SCENARIO_CONTRACT[name]
        return {
            "minimum_retained_repeats": sensitivity.MIN_RETAINED_REPEATS,
            **contract,
            "offending_items": ([] if contract["estimable"] else [sensitivity.V19_OFFENDING_ITEM]),
        }

    def fake_arrays(items, outcomes, excluded):
        del items, outcomes
        assert len(events) >= len(scenarios)
        assert all(event[0] == "preflight" for event in events[: len(scenarios)])
        name = next(name for name, values in scenarios.items() if values == excluded)
        events.append(("arrays", name))
        contract = sensitivity.V19_SCENARIO_CONTRACT[name]
        return {}, {
            "affected_items": contract["affected_items"],
            "remaining_repeat_count_distribution": contract["remaining_repeat_count_distribution"],
        }

    def fake_quality(arrays, frozen_selection):
        del arrays, frozen_selection
        events.append(("quality", "called"))
        return {"behaviors": {}}

    monkeypatch.setattr(sensitivity, "_exclusion_preflight", fake_preflight)
    monkeypatch.setattr(sensitivity, "_integrity_arrays_with_exclusions", fake_arrays)
    monkeypatch.setattr(sensitivity, "_quality_analysis", fake_quality)

    results = sensitivity._analyze_scenarios([], {}, scenarios, {}, "a" * 64)

    assert [event[0] for event in events[:5]] == ["preflight"] * 5
    assert [event for event in events if event[0] == "arrays"] == [
        ("arrays", "no_exclusion"),
        ("arrays", "structured_replacement_only"),
    ]
    assert len([event for event in events if event[0] == "quality"]) == 2
    for name, result in results.items():
        assert result["analysis_performed"] is sensitivity.V19_SCENARIO_CONTRACT[name]["estimable"]
        assert result["recovery_provenance_reference"] == {
            "location": "$.provenance",
            "sha256": "a" * 64,
        }
    for name in (
        "policy_singleton_only",
        "all_policy_packetization",
        "all_administration_deviations",
    ):
        assert results[name]["behaviors"] is None
        assert results[name]["comparison_to_primary"] is None


def test_v19_preflight_contract_rejects_accounting_drift() -> None:
    preflights = {
        name: {
            "minimum_retained_repeats": sensitivity.MIN_RETAINED_REPEATS,
            **contract,
            "offending_items": ([] if contract["estimable"] else [sensitivity.V19_OFFENDING_ITEM]),
        }
        for name, contract in sensitivity.V19_SCENARIO_CONTRACT.items()
    }
    preflights["policy_singleton_only"]["retained_decisions"] += 1

    try:
        sensitivity._validate_v19_preflights(preflights)
    except base.AnalysisError as exc:
        assert "accounting changed" in str(exc)
    else:
        raise AssertionError("v19 accounting drift was accepted")
