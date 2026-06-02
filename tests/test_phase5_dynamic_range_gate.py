"""Unit tests for Phase 5 dynamic-range gate + H1 cell-coverage check.

Round-3 review blockers #1 (dynamic-range gate didn't override the
headline) and #3 (H1 gate could pass on a subset under --allow-partial)
both require regression coverage that the CPU smoke can't produce — the
smoke's stub per-cell tree is too small to exercise the saturation
regime, and the smoke's path-A always writes complete data.

These tests load the phase5 script as a module and call the pure helper
functions directly so we don't need a CLI / temp dir.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from explore_persona_space.experiments import i464_encodings as enc

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def phase5_mod():
    """Load `scripts/i464_phase5_analyze.py` as a module so the helpers are callable."""
    spec = importlib.util.spec_from_file_location(
        "i464_phase5_analyze",
        REPO_ROOT / "scripts" / "i464_phase5_analyze.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── _compute_dynamic_range_gate ─────────────────────────────────────────


def test_dynamic_range_gate_pass_when_all_arms_above_threshold(phase5_mod):
    """When every arm's leakage log-prob sd > threshold, gate PASSes."""
    raw_per_cell = {
        arm: {42: [-5.0, -3.0, -1.0, -7.0], 137: [-4.0, -2.0, -6.0, -8.0]} for arm in enc.ARMS
    }
    dr_gate, ok = phase5_mod._compute_dynamic_range_gate(raw_per_cell)
    assert ok, f"expected gate ok=True; per-arm sds: {[v['sd'] for v in dr_gate.values()]}"
    for arm in enc.ARMS:
        assert dr_gate[arm]["above_threshold"], f"arm {arm} below threshold"
        assert dr_gate[arm]["sd"] > phase5_mod.DYNAMIC_RANGE_THRESHOLD


def test_dynamic_range_gate_fail_when_one_arm_saturated(phase5_mod):
    """One arm with degenerate sd should fail the overall gate."""
    raw_per_cell = {
        "system_plain": {42: [-5.0, -3.0, -1.0, -7.0]},
        "system_padded": {42: [-4.0, -2.0, -6.0, -8.0]},
        "role": {42: [-0.01, -0.01, -0.01, -0.01]},  # sd = 0 << 0.5
    }
    dr_gate, ok = phase5_mod._compute_dynamic_range_gate(raw_per_cell)
    assert not ok, "expected gate ok=False because role arm is saturated"
    assert not dr_gate["role"]["above_threshold"]
    assert dr_gate["system_plain"]["above_threshold"]
    assert dr_gate["system_padded"]["above_threshold"]


def test_dynamic_range_gate_handles_empty_arm(phase5_mod):
    """An arm with no observations does NOT crash; it just fails the gate."""
    raw_per_cell = {
        "system_plain": {42: [-5.0, -3.0, -1.0, -7.0]},
        "system_padded": {42: [-4.0, -2.0, -6.0, -8.0]},
        "role": {},  # no data at all
    }
    dr_gate, ok = phase5_mod._compute_dynamic_range_gate(raw_per_cell)
    assert not ok
    assert dr_gate["role"]["sd"] is None
    assert dr_gate["role"]["n_observations"] == 0
    assert not dr_gate["role"]["above_threshold"]


# ── _override_headline_on_saturation ────────────────────────────────────


def _make_passing_headline() -> dict:
    """Build a fake H2-PASS headline so we can verify the override flips it."""
    return {
        "status": "ok",
        "n_complete_seeds": 3,
        "complete_seeds": [42, 137, 1337],
        "d_seed_plain": {"mean": 1.5, "ci_lo_95": 0.5, "ci_hi_95": 2.5, "pass": True},
        "d_seed_padded": {"mean": 1.3, "ci_lo_95": 0.3, "ci_hi_95": 2.3, "pass": True},
        "h2_full_pass": True,
        "h2_partial": False,
        "h1_overall_pass": True,
    }


def test_override_flips_passing_headline_to_inconclusive_on_saturation(phase5_mod):
    """The critical regression — round-2 bug — that motivated round 3.

    A headline computed as ``status='ok'`` AND ``h2_full_pass=True`` MUST be
    overridden to ``status='inconclusive_dynamic_range_failed'`` AND
    ``h2_full_pass=False`` AND ``h2_partial=False`` when dynamic_range_ok
    is False. Round-2 wrote the passing headline anyway and only logger
    .warning'd; round-3 overrides.
    """
    headline = _make_passing_headline()
    dr_gate = {
        "system_plain": {"sd": 1.0, "n_observations": 12, "above_threshold": True},
        "system_padded": {"sd": 1.1, "n_observations": 12, "above_threshold": True},
        "role": {"sd": 0.05, "n_observations": 12, "above_threshold": False},
    }
    new_headline, new_status = phase5_mod._override_headline_on_saturation(
        headline, "ok", dr_gate, dynamic_range_ok=False
    )
    assert new_status == "inconclusive_dynamic_range_failed"
    assert new_headline["status"] == "inconclusive_dynamic_range_failed"
    assert new_headline["h2_full_pass"] is False
    assert new_headline["h2_partial"] is False
    assert new_headline["dynamic_range_failed_arms"] == ["role"]
    assert "reason" in new_headline


def test_override_is_noop_when_dynamic_range_ok(phase5_mod):
    """When dynamic_range_ok=True, the headline passes through untouched."""
    headline = _make_passing_headline()
    dr_gate = {a: {"sd": 1.0, "n_observations": 12, "above_threshold": True} for a in enc.ARMS}
    new_headline, new_status = phase5_mod._override_headline_on_saturation(
        headline, "ok", dr_gate, dynamic_range_ok=True
    )
    assert new_status == "ok"
    assert new_headline["status"] == "ok"
    assert new_headline["h2_full_pass"] is True


def test_override_does_not_stomp_inconclusive_descriptive_only(phase5_mod):
    """An already-terminal MF-H inconclusive status is NOT stomped on saturation.

    Otherwise the operator would see a wrong root-cause message
    (saturation) when the actual issue is n<3 seeds.
    """
    headline = {
        "status": "inconclusive_descriptive_only",
        "reason": "only 2 complete paired seeds (need >= 3)",
        "h2_full_pass": False,
        "h2_partial": False,
    }
    dr_gate = {
        "system_plain": {"sd": 0.1, "above_threshold": False},
        "system_padded": {"sd": 0.1, "above_threshold": False},
        "role": {"sd": 0.1, "above_threshold": False},
    }
    new_headline, new_status = phase5_mod._override_headline_on_saturation(
        headline, "inconclusive_descriptive_only", dr_gate, dynamic_range_ok=False
    )
    assert new_status == "inconclusive_descriptive_only"
    # Original reason preserved.
    assert "n=3" in new_headline["reason"] or "complete paired" in new_headline["reason"]


def test_override_does_not_stomp_blocked_onpolicy_switch(phase5_mod):
    """An already-terminal MF-B(2) blocked status is NOT stomped on saturation."""
    headline = {
        "status": "blocked_onpolicy_switch_required",
        "reason": "Phase 4.5 ratio = 2.0 > 1.5",
        "h2_full_pass": False,
        "h2_partial": False,
    }
    dr_gate = {a: {"sd": 0.1, "above_threshold": False} for a in enc.ARMS}
    new_headline, new_status = phase5_mod._override_headline_on_saturation(
        headline, "blocked_onpolicy_switch_required", dr_gate, dynamic_range_ok=False
    )
    assert new_status == "blocked_onpolicy_switch_required"
    assert "Phase 4.5" in new_headline["reason"]


def test_override_handles_none_headline(phase5_mod):
    """Defensive: even if no headline was built yet, override creates one cleanly."""
    dr_gate = {a: {"sd": 0.05, "above_threshold": False} for a in enc.ARMS}
    new_headline, new_status = phase5_mod._override_headline_on_saturation(
        None, "fail", dr_gate, dynamic_range_ok=False
    )
    assert new_status == "inconclusive_dynamic_range_failed"
    assert new_headline is not None
    assert new_headline["status"] == "inconclusive_dynamic_range_failed"
    assert new_headline["h2_full_pass"] is False
    assert new_headline["h2_partial"] is False
