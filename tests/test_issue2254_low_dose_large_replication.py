from __future__ import annotations

import importlib.util
import math
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/issue2254_low_dose_large_replication.py"
)
SPEC = importlib.util.spec_from_file_location(
    "issue2254_low_dose_large_replication", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
exp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp)


def _summary() -> dict:
    targets = {}
    for target in exp.TARGETS:
        persona = target in exp.PERSONAS
        context = [1.0 if persona else 0.0] * len(exp.PAIR_INDICES)
        answer = [0.0] * len(exp.PAIR_INDICES)
        targets[target] = {
            "status": "ok",
            "interventions": {
                "context": {"status": "ok", "per_question_f": context},
                **{
                    f"answer_s{exp.parent._float_slug(dose)}": {
                        "status": "ok",
                        "per_question_f": answer,
                    }
                    for dose in exp.ANSWER_DOSES
                },
            },
        }
    return {"targets": targets}


def test_grid_is_frozen_balanced_and_unique() -> None:
    cells = exp.build_cells()
    assert len(cells) == 187
    assert len({exp.parent.cell_id(cell) for cell in cells}) == len(cells)
    assert len(exp.PAIR_INDICES) * len(exp.GENERATION_SEEDS) == 48
    assert exp.ANSWER_DOSES == (1 / 16, 1 / 8)
    assert set(exp.ANSWER_BREADTH) == set(exp.TARGETS)
    assert set(exp.CONTEXT_POINTS) == set(exp.TARGETS)

    for target in exp.TARGETS:
        target_cells = [cell for cell in cells if cell["target"] == target]
        assert len(target_cells) == 17
        assert sum(cell["kind"] == "anchor" for cell in target_cells) == 2
        signal = [cell for cell in target_cells if cell["kind"] == "signal"]
        random = [cell for cell in target_cells if cell["kind"] == "random"]
        assert len(signal) == 3
        assert len(random) == 12
        for intervention in signal:
            matched = [
                row
                for row in random
                if row["position"] == intervention["position"]
                and row["dose_scale"] == intervention["dose_scale"]
            ]
            assert {row["random_seed"] for row in matched} == set(exp.RANDOM_SEEDS)


def test_all_target_primary_uses_exact_four_of_eleven_test(monkeypatch) -> None:
    monkeypatch.setattr(exp.parent, "BOOTSTRAP_DRAWS", 100)
    result = exp.analyze_summary(_summary())
    assert result["inference_label"] == "confirmatory_primary"
    assert result["primary_all_11"]["status"] == "ok"
    assert result["primary_all_11"]["n_assignments"] == 330
    assert result["primary_all_11"]["observed"] == 1.0
    assert result["quality_survival"] == {
        "context_signal_eligible": 11,
        "answer_s0p0625_eligible": 11,
        "answer_s0p125_eligible": 11,
    }


def test_retained_sensitivity_uses_number_of_personas_actually_retained(
    monkeypatch,
) -> None:
    monkeypatch.setattr(exp.parent, "BOOTSTRAP_DRAWS", 100)
    summary = _summary()
    summary["targets"]["optimistic"] = {"status": "intervention_ineligible"}
    result = exp.analyze_summary(summary)
    assert result["inference_label"] == "sensitivity_only"
    assert result["primary_all_11"]["status"] == "not_estimable"
    assert result["retained_target_sensitivity"]["status"] == "ok"
    # Three retained persona labels among ten retained targets: C(10, 3) = 120.
    assert result["retained_target_sensitivity"]["n_assignments"] == 120


def test_retained_sensitivity_respects_prespecified_attrition_floor(
    monkeypatch,
) -> None:
    monkeypatch.setattr(exp.parent, "BOOTSTRAP_DRAWS", 100)
    summary = _summary()
    for target in ("optimistic", "impolite"):
        summary["targets"][target] = {"status": "intervention_ineligible"}
    result = exp.analyze_summary(summary)
    assert result["inference_label"] == "descriptive_only"
    assert result["retained_target_sensitivity"]["status"] == "not_estimable"


def test_partial_questions_use_joint_valid_mask_without_nan(monkeypatch) -> None:
    monkeypatch.setattr(exp.parent, "BOOTSTRAP_DRAWS", 100)
    summary = _summary()
    rows = summary["targets"]["optimistic"]["interventions"]
    for row in rows.values():
        row["per_question_f"][-2:] = [None, None]
    result = exp.analyze_summary(summary)
    assert result["inference_label"] == "confirmatory_primary"
    assert result["analytical_attrition"] == {}
    assert math.isfinite(result["primary_all_11"]["observed"])
    assert all(math.isfinite(value) for value in result["primary_all_11"]["bootstrap_ci95"])


def test_disjoint_partial_questions_trigger_explicit_attrition(monkeypatch) -> None:
    monkeypatch.setattr(exp.parent, "BOOTSTRAP_DRAWS", 100)
    summary = _summary()
    rows = summary["targets"]["optimistic"]["interventions"]
    rows["context"]["per_question_f"] = [1.0, 1.0, 1.0, 1.0, None, None]
    rows["answer_s0p0625"]["per_question_f"] = [None, None, 0.0, 0.0, 0.0, 0.0]
    result = exp.analyze_summary(summary)
    assert result["inference_label"] == "sensitivity_only"
    assert result["primary_all_11"]["status"] == "not_estimable"
    assert result["analytical_attrition"]["optimistic"] == {
        "reason": "insufficient_jointly_valid_questions_across_primary_arms",
        "n_jointly_valid_questions": 2,
        "minimum": exp.parent.MIN_VALID_QUESTIONS,
    }
