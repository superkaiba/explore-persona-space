"""Focused tests for the #2254 v20 trait administration sensitivity."""

from __future__ import annotations

from dataclasses import replace

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_sweep as gen
import scripts.issue2254_all_answer_decode_trait_sensitivity as sensitivity


def _item(opaque_id: str, qi: int = 0, seed: int = gen.SEED_BASE):
    cell = gen.CellSpec("evil", "decodeonly", 0.5)
    return base.AnalysisItem(
        source_item_id=f"{cell.cell_id}|q{qi:02d}|e{seed}",
        opaque_id=opaque_id,
        cell_id=cell.cell_id,
        behavior="evil",
        route="decodeonly",
        dose=0.5,
        qi=qi,
        effective_seed=seed,
        question="Question?",
        answer="Answer.",
        token_count=1,
        answer_sha256="0" * 64,
    )


def test_order_exclusion_exact_accounting() -> None:
    items = [_item(f"opaque-{index}") for index in range(2_640)]
    excluded = {(item.opaque_id, 0) for item in items[:36]}

    preflight = sensitivity._exclusion_preflight(items, excluded)

    assert preflight == {
        "minimum_retained_repeats": 3,
        **sensitivity.ORDER_ONLY_EXPECTED,
    }


def test_trait_repeats_recomputed_after_pass_exclusion() -> None:
    template = _item("opaque")
    items = []
    outcomes = {}
    for qi in range(20):
        for offset in range(6):
            opaque = f"opaque-{qi}-{offset}"
            item = replace(
                template,
                source_item_id=f"source-{qi}-{offset}",
                opaque_id=opaque,
                qi=qi,
                effective_seed=gen.SEED_BASE + offset,
            )
            items.append(item)
            outcomes[opaque] = {0: 100, 1: 100, 2: 0, 3: 0, 4: 0}
    target = items[0]

    arrays, accounting = sensitivity._trait_arrays_with_exclusions(
        items, outcomes, {(target.opaque_id, 2)}
    )

    assert arrays[target.cell_id]["itt"][0, 0] == 50.0
    assert accounting == {
        "affected_items": 1,
        "remaining_repeat_count_distribution": {"4": 1, "5": 119},
    }


def test_trait_repeats_preserve_refusal_itt_and_numeric_completeness() -> None:
    template = _item("opaque")
    items = []
    outcomes = {}
    for qi in range(20):
        for offset in range(6):
            opaque = f"opaque-{qi}-{offset}"
            item = replace(
                template,
                source_item_id=f"source-{qi}-{offset}",
                opaque_id=opaque,
                qi=qi,
                effective_seed=gen.SEED_BASE + offset,
            )
            items.append(item)
            outcomes[opaque] = {0: "REFUSAL", 1: 100, 2: 100, 3: 0, 4: 0}

    arrays, _accounting = sensitivity._trait_arrays_with_exclusions(items, outcomes, set())
    cell = arrays[template.cell_id]

    assert cell["itt"][0, 0] == 40.0
    assert cell["numeric"][0, 0] == 50.0
    assert cell["refusal"][0, 0] == 0.2
    assert cell["numeric_draw_complete"][0, 0] == 0.8


def test_all_scenarios_preflight_before_only_estimable_analysis(monkeypatch) -> None:
    order_only = {(f"order-{index}", 0) for index in range(36)}
    scenarios = {
        "no_exclusion": set(),
        "trait_order_replacement_only": order_only,
        "trait_policy_singleton_only": {("policy-singleton", 1)},
        "all_trait_policy_packetization": {
            ("policy-singleton", 1),
            ("policy-other", 2),
        },
        "all_trait_administration_deviations": order_only
        | {("policy-singleton", 1), ("policy-other", 2)},
    }
    events = []

    def fake_preflight(items, excluded):
        del items
        name = next(name for name, values in scenarios.items() if values == excluded)
        events.append(("preflight", name))
        if name == "no_exclusion":
            return {
                "minimum_retained_repeats": 3,
                "excluded_decisions": 0,
                "retained_decisions": 13_200,
                "affected_items": 0,
                "remaining_repeat_count_distribution": {"5": 2_640},
                "estimable": True,
                "offending_items": [],
            }
        if name == "trait_order_replacement_only":
            return {"minimum_retained_repeats": 3, **sensitivity.ORDER_ONLY_EXPECTED}
        estimable = name != "all_trait_administration_deviations"
        return {
            "minimum_retained_repeats": 3,
            "excluded_decisions": len(excluded),
            "retained_decisions": 13_200 - len(excluded),
            "affected_items": len(excluded),
            "remaining_repeat_count_distribution": {"4": len(excluded)},
            "estimable": estimable,
            "offending_items": [] if estimable else [{"opaque_item_id": "offender"}],
        }

    def fake_arrays(items, outcomes, excluded):
        del items, outcomes
        assert [event[0] for event in events[:5]] == ["preflight"] * 5
        name = next(name for name, values in scenarios.items() if values == excluded)
        events.append(("arrays", name))
        preflight = fake_preflight([], excluded)
        events.pop()
        return {}, {
            "affected_items": preflight["affected_items"],
            "remaining_repeat_count_distribution": preflight["remaining_repeat_count_distribution"],
        }

    monkeypatch.setattr(sensitivity, "_exclusion_preflight", fake_preflight)
    monkeypatch.setattr(sensitivity, "_trait_arrays_with_exclusions", fake_arrays)
    monkeypatch.setattr(
        sensitivity,
        "_behavior_metrics",
        lambda behavior, arrays, primary: {"behavior": behavior},
    )
    monkeypatch.setattr(sensitivity, "_affected_cells", lambda items, excluded: [])
    monkeypatch.setattr(
        sensitivity, "_comparison_to_primary", lambda primary, scenario: {"compared": True}
    )

    results = sensitivity._analyze_scenarios(
        [], {}, scenarios, order_only, {"behaviors": {}}, "a" * 64
    )

    assert [event[0] for event in events[:5]] == ["preflight"] * 5
    assert [event for event in events if event[0] == "arrays"] == [
        ("arrays", "no_exclusion"),
        ("arrays", "trait_order_replacement_only"),
        ("arrays", "trait_policy_singleton_only"),
        ("arrays", "all_trait_policy_packetization"),
    ]
    non_estimable = results["all_trait_administration_deviations"]
    assert non_estimable["analysis_performed"] is False
    assert non_estimable["behaviors"] is None
    assert non_estimable["comparison_to_primary"] is None


def test_v20_preflight_rejects_order_accounting_drift() -> None:
    order_only = {(f"order-{index}", 0) for index in range(36)}
    scenarios = {
        "no_exclusion": set(),
        "trait_order_replacement_only": order_only,
        "trait_policy_singleton_only": set(),
        "all_trait_policy_packetization": set(),
        "all_trait_administration_deviations": order_only,
    }
    no_exclusion = {
        "minimum_retained_repeats": 3,
        "excluded_decisions": 0,
        "retained_decisions": 13_200,
        "affected_items": 0,
        "remaining_repeat_count_distribution": {"5": 2_640},
        "estimable": True,
        "offending_items": [],
    }
    preflights = {
        "no_exclusion": no_exclusion,
        "trait_order_replacement_only": {
            "minimum_retained_repeats": 3,
            **sensitivity.ORDER_ONLY_EXPECTED,
            "retained_decisions": 13_163,
        },
        "trait_policy_singleton_only": no_exclusion,
        "all_trait_policy_packetization": no_exclusion,
        "all_trait_administration_deviations": {
            "minimum_retained_repeats": 3,
            **sensitivity.ORDER_ONLY_EXPECTED,
        },
    }

    try:
        sensitivity._validate_v20_preflights(preflights, scenarios, order_only)
    except base.AnalysisError as exc:
        assert "order accounting changed" in str(exc)
    else:
        raise AssertionError("v20 order accounting drift was accepted")


def test_decision_set_rejects_duplicate_ids() -> None:
    try:
        sensitivity._decision_set(["same", "same"], 0)
    except base.AnalysisError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate trait receipt ids were accepted")
