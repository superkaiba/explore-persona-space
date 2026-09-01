from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/issue2254_multitype_attrition_analysis.py"
SPEC = importlib.util.spec_from_file_location("issue2254_multitype_attrition_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


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
        "positions": {"context": position(context), "answer": position(answer)},
    }


def test_attrition_analysis_labels_primary_and_enumerates_sensitivities() -> None:
    analysis.exp.BOOTSTRAP_DRAWS = 100
    targets = tuple(target for target in analysis.exp.TARGETS if target != "query_topic")
    confirm = {
        "targets": {
            target: _confirm_row(target, 1.0 if target in analysis.exp.PERSONAS else 0.0)
            for target in targets
        }
    }
    decision = {"decision": {"retained_targets": list(targets)}, "gate_resolution_audit": {}}
    result = analysis.compute_result(
        confirm,
        decision,
        confirm_sha256="confirm",
        decision_sha256="decision",
        design_sha256="design",
        amendment_sha256="amendment",
        gate_audit={},
    )
    assert result["inference_status"] == "sensitivity_only_no_confirmatory_claim"
    assert result["preregistered_11_target_primary"]["status"] == "not_estimable"
    assert result["target_attrition_sensitivity"]["n_assignments"] == 210
    assert result["retained_leave_icl_out_sensitivity"]["n_assignments"] == 126
    assert result["target_attrition_sensitivity"]["observed"] == 1.0


def test_attrition_analysis_requires_exact_retained_set() -> None:
    analysis.exp.BOOTSTRAP_DRAWS = 100
    targets = tuple(target for target in analysis.exp.TARGETS if target != "query_topic")
    confirm = {
        "targets": {target: _confirm_row(target, 0.0) for target in targets if target != "icl_task"}
    }
    decision = {"decision": {"retained_targets": list(targets)}, "gate_resolution_audit": {}}
    try:
        analysis.compute_result(
            confirm,
            decision,
            confirm_sha256="confirm",
            decision_sha256="decision",
            design_sha256="design",
            amendment_sha256="amendment",
            gate_audit={},
        )
    except RuntimeError as error:
        assert "exactly match" in str(error)
    else:
        raise AssertionError("missing retained target should fail closed")
