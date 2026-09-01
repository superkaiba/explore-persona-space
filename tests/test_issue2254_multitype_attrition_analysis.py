from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/issue2254_multitype_attrition_analysis.py"
SPEC = importlib.util.spec_from_file_location("issue2254_multitype_attrition_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


@pytest.fixture(autouse=True)
def _restore_bootstrap_draws():
    original = analysis.exp.BOOTSTRAP_DRAWS
    yield
    analysis.exp.BOOTSTRAP_DRAWS = original


def _confirm_row(target: str, preference: float) -> dict:
    context = [preference + 0.5] * 6
    answer = [0.5] * 6
    def position(values: list[float]) -> dict:
        return {
            "status": "ok",
            "per_question_f": values,
            "random": {"exceeds_all_points": True},
        }
    return {
        "target_class": "persona" if target in analysis.exp.PERSONAS else "nonpersona",
        "information_type": analysis.exp.INFORMATION_TYPE[target],
        "floor_mean": 10.0,
        "ceiling_mean": 60.0,
        "per_question_anchor_separation": [50.0] * 6,
        "positions": {"context": position(context), "answer": position(answer)},
    }


def _decision() -> dict:
    targets = tuple(target for target in analysis.exp.TARGETS if target != "query_topic")
    return {
        "decision": {
            "retained_targets": list(targets),
            "preregistered_11_target_primary": "not_estimable",
            "alternate_cjk_or_query_confirmation_in_this_run": False,
        },
        "gate_resolution_audit": {},
    }


def _gate_audit() -> dict:
    return {"literal_frozen_gate": {"n_eligible_cells": 0}}


def test_attrition_analysis_labels_primary_and_enumerates_sensitivities() -> None:
    analysis.exp.BOOTSTRAP_DRAWS = 100
    targets = tuple(target for target in analysis.exp.TARGETS if target != "query_topic")
    confirm = {
        "targets": {
            target: _confirm_row(target, 1.0 if target in analysis.exp.PERSONAS else 0.0)
            for target in targets
        }
    }
    decision = _decision()
    result = analysis.compute_result(
        confirm,
        decision,
        confirm_sha256="confirm",
        decision_sha256="decision",
        design_sha256="design",
        amendment_sha256="amendment",
        gate_audit=_gate_audit(),
    )
    assert result["inference_status"] == "sensitivity_only_no_confirmatory_claim"
    assert result["preregistered_11_target_primary"]["status"] == "not_estimable"
    assert result["target_attrition_sensitivity"]["n_assignments"] == 210
    assert result["retained_leave_icl_out_sensitivity"]["n_assignments"] == 126
    assert result["target_attrition_sensitivity"]["observed"] == 1.0
    assert result["target_attrition_unnormalized_score_companion"]["observed"] == 50.0
    assert result["group_means"]["persona"]["anchor_separation_mean"] == 50.0
    assert result["literal_frozen_gate_primary"]["status"] == "not_estimable"


def test_attrition_analysis_requires_exact_retained_set() -> None:
    analysis.exp.BOOTSTRAP_DRAWS = 100
    targets = tuple(target for target in analysis.exp.TARGETS if target != "query_topic")
    confirm = {
        "targets": {target: _confirm_row(target, 0.0) for target in targets if target != "icl_task"}
    }
    decision = _decision()
    try:
        analysis.compute_result(
            confirm,
            decision,
            confirm_sha256="confirm",
            decision_sha256="decision",
            design_sha256="design",
            amendment_sha256="amendment",
            gate_audit=_gate_audit(),
        )
    except RuntimeError as error:
        assert "exactly match" in str(error)
    else:
        raise AssertionError("missing retained target should fail closed")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["decision"]["retained_targets"].pop(), "retained-target drift"),
        (
            lambda row: row["decision"].update(preregistered_11_target_primary="estimable"),
            "planned primary not estimable",
        ),
        (
            lambda row: row["decision"].update(
                alternate_cjk_or_query_confirmation_in_this_run=True
            ),
            "forbids alternate-CJK/query confirmation",
        ),
    ],
)
def test_decision_field_guards_fail_closed(mutation, message: str) -> None:
    decision = _decision()
    mutation(decision)
    with pytest.raises(RuntimeError, match=message):
        analysis._validate_decision_fields(decision)


def test_decision_evidence_hash_guard(tmp_path: Path) -> None:
    raw = tmp_path / "screen/raw_completions/query_topic__anchor_a.json"
    judged = tmp_path / "screen/judge/judged/query_topic__anchor_a.json"
    raw.parent.mkdir(parents=True)
    judged.parent.mkdir(parents=True)
    raw.write_text('{"raw": true}\n')
    judged.write_text('{"judged": true}\n')
    decision = {
        "evidence_hashes": {
            "query_topic_anchor_a_raw_sha256": analysis._sha256(raw),
            "anchor_judged_records": {
                "query_topic__anchor_a": analysis._sha256(judged),
            },
        }
    }
    analysis._verify_decision_evidence(tmp_path, decision)
    raw.write_text('{"raw": false}\n')
    with pytest.raises(RuntimeError, match="query-topic floor raw hash mismatch"):
        analysis._verify_decision_evidence(tmp_path, decision)


def test_report_and_plot_smoke(tmp_path: Path) -> None:
    analysis.exp.BOOTSTRAP_DRAWS = 20
    targets = tuple(target for target in analysis.exp.TARGETS if target != "query_topic")
    confirm = {
        "targets": {
            target: _confirm_row(target, 0.25 if target in analysis.exp.PERSONAS else 0.0)
            for target in targets
        }
    }
    result = analysis.compute_result(
        confirm,
        _decision(),
        confirm_sha256="confirm",
        decision_sha256="decision",
        design_sha256="design",
        amendment_sha256="amendment",
        gate_audit=_gate_audit(),
    )
    report = tmp_path / "report.md"
    figure = tmp_path / "figure.png"
    analysis._write_report(result, report)
    analysis._plot_result(result, figure)
    assert report.stat().st_size > 0
    assert figure.stat().st_size > 0
    assert figure.with_suffix(".pdf").stat().st_size > 0
    payload = json.loads(json.dumps(result))
    assert payload["literal_frozen_gate_primary"]["status"] == "not_estimable"


def test_frozen_decision_and_dependencies_verify() -> None:
    decision = analysis.verify_attrition_decision(analysis.OUT_ROOT)
    assert decision["decision"]["preregistered_11_target_primary"] == "not_estimable"
