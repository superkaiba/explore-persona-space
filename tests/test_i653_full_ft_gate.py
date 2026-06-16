# ruff: noqa: RUF002
"""Task #653 §7 full-FT gate logic (round-4 BLOCKER full-ft-gate-not-implemented).

CPU-only. Asserts the gate REFUSES to release the 4×A100 full-FT rung when the
cheap upstream signals fail (Arm A coherence < 50% OR a rank-16 install cell
outside its band/target) and PROCEEDS only when both pass — exactly the plan §7
pre-registered gate.
"""

from __future__ import annotations

import json

import pytest

from explore_persona_space.experiments import issue_653 as i653


def _arm_a_payload(max_pass_rate: float) -> dict:
    """A synthetic rho_geometry payload whose best coherence pass rate is given."""
    return {
        "arm": "A",
        "seed": i653.HEADLINE_SEED,
        "coherence": {
            "iso|10-10|m1.0": max_pass_rate,
            "iso|10-10|m8.0": max(0.0, max_pass_rate - 0.4),
            "cov|25-25|m2.0": max(0.0, max_pass_rate - 0.2),
        },
    }


def _marker_install(logp_gain: float) -> dict:
    return {"install": {"dv_kind": "marker_four_float", "logp_trained_minus_base": logp_gain}}


def _content_install(rate_gain: float) -> dict:
    return {"install": {"dv_kind": "judge_rate_plus_gain", "judge_rate_gain": rate_gain}}


def _passing_rank16_installs() -> dict[str, dict]:
    """All 6 headline rank-16 cells passing their band/target."""
    out: dict[str, dict] = {}
    for source in i653.HEADLINE_SOURCES:
        out[f"marker__{source}__r16__seed{i653.HEADLINE_SEED}"] = _marker_install(8.0)  # in [5,12]
        out[f"sycophancy__{source}__r16__seed{i653.HEADLINE_SEED}"] = _content_install(0.4)
        out[f"em__{source}__r16__seed{i653.HEADLINE_SEED}"] = _content_install(0.3)
    return out


def test_gate_proceeds_when_both_conditions_pass():
    arm_a = [_arm_a_payload(0.8)]  # coherence 0.8 ≥ 0.5
    decision = i653.evaluate_full_ft_gate(arm_a, _passing_rank16_installs())
    assert decision["proceed"] is True
    assert decision["failing_subgates"] == []
    assert decision["condition_a_arm_a_coherence"]["passed"] is True


def test_gate_fails_when_arm_a_coherence_below_floor():
    arm_a = [_arm_a_payload(0.3)]  # 0.3 < 0.5 floor
    decision = i653.evaluate_full_ft_gate(arm_a, _passing_rank16_installs())
    assert decision["proceed"] is False
    assert "arm_a_coherence" in decision["failing_subgates"]


def test_gate_fails_when_a_rank16_marker_outside_band():
    arm_a = [_arm_a_payload(0.9)]
    installs = _passing_rank16_installs()
    # Push one marker cell to a saturated gain (above the [5,12] band).
    installs[f"marker__florist__r16__seed{i653.HEADLINE_SEED}"] = _marker_install(25.0)
    decision = i653.evaluate_full_ft_gate(arm_a, installs)
    assert decision["proceed"] is False
    assert "rank16_install" in decision["failing_subgates"]


def test_gate_fails_when_a_content_cell_did_not_install():
    arm_a = [_arm_a_payload(0.9)]
    installs = _passing_rank16_installs()
    # A sycophancy cell with no judge-rate gain (behavior did not install).
    installs[f"sycophancy__medical_doctor__r16__seed{i653.HEADLINE_SEED}"] = _content_install(-0.05)
    decision = i653.evaluate_full_ft_gate(arm_a, installs)
    assert decision["proceed"] is False
    assert "rank16_install" in decision["failing_subgates"]


def test_gate_fails_loud_on_missing_install_dv_not_silent_pass():
    """A None DV (install read never produced) must FAIL the gate, never pass."""
    arm_a = [_arm_a_payload(0.9)]
    installs = _passing_rank16_installs()
    installs[f"em__florist__r16__seed{i653.HEADLINE_SEED}"] = {
        "install": {"dv_kind": "judge_rate_plus_gain", "judge_rate_gain": None}
    }
    decision = i653.evaluate_full_ft_gate(arm_a, installs)
    assert decision["proceed"] is False
    assert "rank16_install" in decision["failing_subgates"]


def test_gate_fails_when_no_install_evidence_at_all():
    """Empty install set ⇒ no condition-(b) evidence ⇒ FAIL (never a vacuous pass)."""
    arm_a = [_arm_a_payload(0.9)]
    decision = i653.evaluate_full_ft_gate(arm_a, {})
    assert decision["proceed"] is False


def test_gate_fails_when_no_coherence_data():
    """No Arm A coherence reads ⇒ condition (a) cannot pass."""
    decision = i653.evaluate_full_ft_gate([{"coherence": {}}], _passing_rank16_installs())
    assert decision["proceed"] is False
    assert "arm_a_coherence" in decision["failing_subgates"]


def test_phase_train_refuses_full_ft_without_gate_sentinel(tmp_path):
    """The hard in-process backstop: phase_train raises if a full-FT cell is in
    the subset but gate_decision.json is absent (gpu mode). Imports the dispatcher
    lazily (it loads heavy deps at module top only via deferred imports)."""
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    disp_path = repo_root / "scripts" / "issue_653" / "i653_dispatch.py"
    spec = importlib.util.spec_from_file_location("i653_dispatch_test", disp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i653_dispatch_test"] = mod
    spec.loader.exec_module(mod)

    full_cell = i653.ArmBCell(behavior="em", source="florist", rung="full", seed=i653.HEADLINE_SEED)
    out_root = tmp_path / "eval_results" / "issue_653"
    out_root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match=r"gate_decision\.json is absent"):
        mod.phase_train(
            [full_cell], out_root=out_root, gpu=0, mode=i653.RUN_MODE_GPU, max_steps=None
        )


def test_phase_train_refuses_full_ft_when_gate_failed(tmp_path):
    """phase_train raises when gate_decision.json says proceed=False."""
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    disp_path = repo_root / "scripts" / "issue_653" / "i653_dispatch.py"
    spec = importlib.util.spec_from_file_location("i653_dispatch_test2", disp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i653_dispatch_test2"] = mod
    spec.loader.exec_module(mod)

    full_cell = i653.ArmBCell(behavior="em", source="florist", rung="full", seed=i653.HEADLINE_SEED)
    out_root = tmp_path / "eval_results" / "issue_653"
    out_root.mkdir(parents=True)
    (out_root / "gate_decision.json").write_text(
        json.dumps({"proceed": False, "failing_subgates": ["arm_a_coherence"], "kill_outcome": "x"})
    )
    with pytest.raises(RuntimeError, match="gate FAILED"):
        mod.phase_train(
            [full_cell], out_root=out_root, gpu=0, mode=i653.RUN_MODE_GPU, max_steps=None
        )
