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
