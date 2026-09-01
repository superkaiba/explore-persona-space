from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import scripts.issue2254_answer_strength_sweep as sweep
import scripts.issue2254_probe_context_followup as base
from scripts import issue2254_information_type_position as experiment


def test_frozen_assets_are_balanced_and_prompt_disjoint() -> None:
    payload = experiment._frozen_assets_payload()
    assert set(payload["targets"]) == set(experiment.TARGETS)
    assert len(experiment.PERSONAS) == len(experiment.SEMANTIC) == 4
    extraction_banks = set()
    evaluation_banks = set()
    for target, asset in payload["targets"].items():
        assert len(asset["instruction"]) == 5
        assert len(asset["extraction_questions"]) == 20
        assert len(asset["eval_questions"]) == 20
        assert set(asset["extraction_questions"]).isdisjoint(asset["eval_questions"])
        assert asset["target_class"] == (
            "persona" if target in experiment.PERSONAS else "semantic"
        )
        extraction_banks.add(tuple(asset["extraction_questions"]))
        evaluation_banks.add(tuple(asset["eval_questions"]))
    assert len(extraction_banks) == len(evaluation_banks) == 1


def test_context_extraction_crosses_five_pairs_with_twenty_questions(tmp_path) -> None:
    experiment.freeze_assets(tmp_path)
    old_root = experiment._ACTIVE_ASSET_ROOT
    experiment._ACTIVE_ASSET_ROOT = tmp_path
    try:
        positive, negative = experiment.extraction_contexts("golden_gate_bridge")
    finally:
        experiment._ACTIVE_ASSET_ROOT = old_root
    assert len(positive) == len(negative) == 100
    assert len({row["system"] for row in positive}) == 5
    assert len({row["user"] for row in positive}) == 20
    assert {row["user"] for row in positive} == {row["user"] for row in negative}


def test_identical_matched_grid_is_available_at_both_positions(monkeypatch) -> None:
    monkeypatch.setattr(sweep, "TRAITS", experiment.TARGETS)
    monkeypatch.setattr(sweep, "DOSE_SCALES", experiment.MATCHED_DOSE_SCALES)
    monkeypatch.setattr(sweep, "_signal_cell", experiment._position_signal_cell)
    monkeypatch.setattr(sweep, "build_screen_cells", experiment._position_build_screen_cells)
    for intervention_position in ("context", "answer"):
        monkeypatch.setattr(experiment, "_ACTIVE_POSITION", intervention_position)
        cells = sweep.build_screen_cells(experiment.TARGETS)
        assert len(cells) == 8 * (1 + 2 * 3 * 5)
        assert len({base.cell_id(cell) for cell in cells}) == len(cells)
        assert {cell["position"] for cell in cells} == {intervention_position}
        signal = [cell for cell in cells if cell["kind"] == "signal"]
        assert {cell["dose_scale"] for cell in signal} == set(
            experiment.MATCHED_DOSE_SCALES
        )


def test_position_field_controls_the_actual_hook(monkeypatch) -> None:
    from explore_persona_space.experiments.issue1415 import steering

    seen = []

    class FakeHook:
        def __init__(self, _model, _layer, _direction, _alpha, *, all_positions):
            seen.append(all_positions)

    monkeypatch.setattr(steering, "DeltaHook", FakeHook)
    model = SimpleNamespace(device="cpu")
    args = SimpleNamespace()
    rho = {}
    for intervention_position, expected in (("context", False), ("answer", True)):
        cell = {
            "behavior": "optimistic",
            "kind": "alpha0",
            "position": intervention_position,
        }
        make_hook, _alphas = experiment.position._hook_factory(model, args, cell, rho)
        make_hook()
        assert seen[-1] is expected


def test_question_and_seed_splits_are_independent() -> None:
    assert set(sweep.SCREEN_QUESTION_INDICES).isdisjoint(
        sweep.CONFIRM_QUESTION_INDICES
    )
    assert set(sweep.SCREEN_SEEDS).isdisjoint({43})
    assert set(experiment.SCREEN_LOCAL_INDICES) == set(sweep.SCREEN_QUESTION_INDICES)
    assert set(experiment.CONFIRM_LOCAL_INDICES) == set(
        sweep.CONFIRM_QUESTION_INDICES
    )


def test_exact_class_permutation_has_expected_resolution() -> None:
    values = {
        target: (1.0 if target in experiment.PERSONAS else 0.0)
        for target in experiment.TARGETS
    }
    result = experiment.exact_class_permutation(values)
    assert result["persona_minus_semantic_preference"] == 1.0
    assert result["n_label_permutations"] == 70
    assert result["one_sided_exact_permutation_p"] == 1 / 70


def test_finite_pairs_preserves_paired_question_order() -> None:
    context = np.arange(10, dtype=float)
    answer = np.arange(10, dtype=float) / 2
    context[2] = np.nan
    kept_context, kept_answer = experiment._finite_pairs(context, answer)
    np.testing.assert_array_equal(kept_context - kept_answer, np.delete(context - answer, 2))


def test_position_heldout_preserves_missing_question_slots(tmp_path) -> None:
    target = "optimistic"
    method = "diffmean"
    selected_id = "optimistic__diffmean__single__L23__c0p5"
    summary_path = tmp_path / "confirm/summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "traits": {
                    target: {
                        "methods": {
                            method: {
                                "status": "ok",
                                "selected_cell_id": selected_id,
                                "cell": {"position": "context"},
                                "screen_delta_score": 3.0,
                                "chance": {"per_direction_point_deltas": [0.0]},
                                "exceeds_all_random_direction_points": True,
                            }
                        }
                    }
                }
            }
        )
    )
    judged = tmp_path / "confirm/judge/judged"
    judged.mkdir(parents=True)
    baseline = [0.0] * 10
    baseline[2] = None
    (judged / f"{target}__a0.json").write_text(
        json.dumps({"per_question_mean_score": baseline})
    )
    (judged / f"{selected_id}.json").write_text(
        json.dumps({"per_question_mean_score": [2.0] * 10})
    )
    row = experiment._position_heldout(tmp_path, target, method)
    assert len(row["heldout_deltas"]) == 10
    assert np.isnan(row["heldout_deltas"][2])
    assert row["heldout_delta"] == 2.0
