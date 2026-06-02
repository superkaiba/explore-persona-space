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


# ── Integration test: dr-gate via main() on a tempdir tree (round-4 MINOR #3) ──


def _write_full_per_cell_tree(
    per_cell_dir: Path,
    saturate_arm: str | None,
    g_lp_jitter: tuple[float, float, float, float] = (-1.0, -0.5, 0.5, 1.0),
):
    """Write all (5 arms x 3 seeds = 15) cells x 9 e_eval x 2 marker_persona JSONs.

    role_nonsense + role_mismatch follow-up arms: extends the original
    3-arm tree to 5 arms (15 cells) and adds the new role_nonsense_<persona>
    AND role_mismatch_<persona> eval-encoding cells across ALL arms so the
    integration test exercises the full surface area Phase 5 + plots now expect.

    When ``saturate_arm`` is non-None, that arm's 4 symmetric leakage
    cells get IDENTICAL g_logprobs (sd=0) so the dr-gate fails for it.
    Other arms get the jittered means so their sd clears the 0.5 threshold.
    """
    seeds = [42, 137, 1337]

    def _own_e(arm: str, persona: str) -> str:
        if arm == "role":
            return f"role_{persona}"
        if arm == "role_nonsense":
            return f"role_nonsense_{persona}"
        if arm == "role_mismatch":
            return f"role_mismatch_{persona}"
        return f"system_{persona}"

    for arm in enc.ARMS:
        for seed in seeds:
            cell = f"{arm}_seed{seed}"
            for persona in enc.PERSONAS:
                # 1. Own-persona elicitation cell (H1 gate input — PASS).
                e_own = _own_e(arm, persona)
                (per_cell_dir / f"{cell}__{e_own}__marker_{persona}.json").write_text(
                    json.dumps(
                        {
                            "cell": cell,
                            "arm": arm,
                            "seed": seed,
                            "e_eval": e_own,
                            "marker_persona": persona,
                            "marker_id": enc.marker_id_for(persona),
                            "n_probes": 3,
                            "g_logprob": -0.2,
                            "b_logprob": -10.0,
                            "delta_g": 9.8,
                            "emission_recompute_rate": 0.95,
                            "logp_floor": -50.0,
                            "g_logps_per_q": [-0.2] * 3,
                            "b_logps_per_q": [-10.0] * 3,
                            "g_argmax_marker_per_q": [True, True, True],
                            "b_argmax_marker_per_q": [False, False, False],
                        }
                    )
                )
                # 2. Symmetric leakage cells (H2 headline input).
                other = "villain" if persona == "pirate" else "pirate"
                for cell_idx, e_wrong in enumerate([f"system_{other}", f"role_{other}"]):
                    persona_idx = list(enc.PERSONAS).index(persona)
                    if arm == saturate_arm:
                        # Identical g_logprob across the 4 cells in this arm →
                        # sd=0 → dr-gate FAILs only for this arm.
                        g_lp = -2.5 - (1.5 if arm == "role" else 0.0)
                    else:
                        jitter = g_lp_jitter[2 * persona_idx + cell_idx]
                        g_lp = -2.5 - (1.5 if arm == "role" else 0.0) + jitter
                    (per_cell_dir / f"{cell}__{e_wrong}__marker_{persona}.json").write_text(
                        json.dumps(
                            {
                                "cell": cell,
                                "arm": arm,
                                "seed": seed,
                                "e_eval": e_wrong,
                                "marker_persona": persona,
                                "marker_id": enc.marker_id_for(persona),
                                "n_probes": 3,
                                "g_logprob": g_lp,
                                "b_logprob": -10.0,
                                "delta_g": -10.0 - g_lp,
                                "emission_recompute_rate": 0.7,
                                "logp_floor": -50.0,
                                "g_logps_per_q": [g_lp - 0.3, g_lp, g_lp + 0.3],
                                "b_logps_per_q": [-10.0] * 3,
                                "g_argmax_marker_per_q": [True, True, False],
                                "b_argmax_marker_per_q": [False, False, False],
                            }
                        )
                    )
                # 2b. role_nonsense_<persona> eval-encoding cells (mirror of
                # the smoke driver's writer): written for ALL arms so the
                # plot's per-arm matrix iterates cleanly. SKIP the role_nonsense
                # arm's OWN role_nonsense_<persona> cell (already written
                # above with the H1-passing value).
                for e_rn in (f"role_nonsense_{persona}", f"role_nonsense_{other}"):
                    if arm == "role_nonsense" and e_rn == f"role_nonsense_{persona}":
                        continue
                    is_own = e_rn == f"role_nonsense_{persona}"
                    g_lp_rn = -0.3 if is_own else -3.0
                    (per_cell_dir / f"{cell}__{e_rn}__marker_{persona}.json").write_text(
                        json.dumps(
                            {
                                "cell": cell,
                                "arm": arm,
                                "seed": seed,
                                "e_eval": e_rn,
                                "marker_persona": persona,
                                "marker_id": enc.marker_id_for(persona),
                                "n_probes": 3,
                                "g_logprob": g_lp_rn,
                                "b_logprob": -10.0,
                                "delta_g": -10.0 - g_lp_rn,
                                "emission_recompute_rate": 0.6 if is_own else 0.4,
                                "logp_floor": -50.0,
                                "g_logps_per_q": [g_lp_rn] * 3,
                                "b_logps_per_q": [-10.0] * 3,
                                "g_argmax_marker_per_q": [True, True, is_own],
                                "b_argmax_marker_per_q": [False, False, False],
                            }
                        )
                    )
                # 2c. role_mismatch_<persona> eval-encoding cells (parallel to 2b
                # for the role_mismatch follow-up arm). SKIP the role_mismatch
                # arm's OWN role_mismatch_<persona> cell — already written by
                # the (1) own-persona block above with H1-passing value.
                for e_rm in (f"role_mismatch_{persona}", f"role_mismatch_{other}"):
                    if arm == "role_mismatch" and e_rm == f"role_mismatch_{persona}":
                        continue
                    is_own = e_rm == f"role_mismatch_{persona}"
                    g_lp_rm = -0.3 if is_own else -3.0
                    (per_cell_dir / f"{cell}__{e_rm}__marker_{persona}.json").write_text(
                        json.dumps(
                            {
                                "cell": cell,
                                "arm": arm,
                                "seed": seed,
                                "e_eval": e_rm,
                                "marker_persona": persona,
                                "marker_id": enc.marker_id_for(persona),
                                "n_probes": 3,
                                "g_logprob": g_lp_rm,
                                "b_logprob": -10.0,
                                "delta_g": -10.0 - g_lp_rm,
                                "emission_recompute_rate": 0.6 if is_own else 0.4,
                                "logp_floor": -50.0,
                                "g_logps_per_q": [g_lp_rm] * 3,
                                "b_logps_per_q": [-10.0] * 3,
                                "g_argmax_marker_per_q": [True, True, is_own],
                                "b_argmax_marker_per_q": [False, False, False],
                            }
                        )
                    )
                # 3. default_assistant cell (exploratory, no impact on H2/dr-gate).
                (per_cell_dir / f"{cell}__default_assistant__marker_{persona}.json").write_text(
                    json.dumps(
                        {
                            "cell": cell,
                            "arm": arm,
                            "seed": seed,
                            "e_eval": "default_assistant",
                            "marker_persona": persona,
                            "marker_id": enc.marker_id_for(persona),
                            "n_probes": 3,
                            "g_logprob": -4.0,
                            "b_logprob": -10.0,
                            "delta_g": 6.0,
                            "emission_recompute_rate": 0.5,
                            "logp_floor": -50.0,
                            "g_logps_per_q": [-4.0] * 3,
                            "b_logps_per_q": [-10.0] * 3,
                            "g_argmax_marker_per_q": [True, False, False],
                            "b_argmax_marker_per_q": [False, False, False],
                        }
                    )
                )


def _run_phase5_main(tmp_dir: Path) -> dict:
    """Run scripts/i464_phase5_analyze.py from tmp_dir cwd; return parsed analysis.json."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "i464_phase5_analyze.py"),
        "--seeds",
        "42",
        "137",
        "1337",
    ]
    res = subprocess.run(
        cmd,
        cwd=str(tmp_dir),
        env={**os.environ},
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0:
        raise RuntimeError(f"phase5 exited rc={res.returncode}; stderr tail: {res.stderr[-300:]}")
    a = tmp_dir / "eval_results" / "issue_464" / "analysis.json"
    if not a.exists():
        raise RuntimeError("analysis.json missing")
    return json.loads(a.read_text())


def test_integration_full_dr_gate_override_via_main(tmp_path):
    """End-to-end: complete H1+H2-passing tree + one saturated arm → status flips.

    Round-4 MINOR #3: the round-3 helper tests covered the override
    function in isolation; this exercises the gate via the real `main()`
    entrypoint with a tempdir per-cell tree, so we catch any regression
    that re-orders the gate check vs the headline finalization (the exact
    round-2/3 bug class).
    """
    import json as _json  # local rebind so the inner helper sees `json`

    _ = _json  # keep ruff happy on the import alias
    per_cell_dir = tmp_path / "eval_results" / "issue_464" / "cross_eval" / "per_cell"
    per_cell_dir.mkdir(parents=True, exist_ok=True)
    # Saturate the role arm: its 4 leakage cells all get the SAME g_logprob,
    # so pstdev across them is 0 → below the 0.5 threshold.
    _write_full_per_cell_tree(per_cell_dir, saturate_arm="role")
    payload = _run_phase5_main(tmp_path)
    assert payload["headline_status"] == "inconclusive_dynamic_range_failed", (
        f"expected dr-gate override; got status={payload['headline_status']!r}; "
        f"headline={payload.get('headline')}"
    )
    assert payload["headline"]["h2_full_pass"] is False
    assert payload["headline"]["h2_partial"] is False
    assert "role" in payload["headline"]["dynamic_range_failed_arms"]
    # H1 still PASSes (own-persona cells unchanged) — surfaced separately.
    assert payload["h1_elicitation"]["overall_pass"] is True


def test_integration_dr_gate_passes_when_no_arm_saturated(tmp_path):
    """Sanity counterpart: same tree without saturation → H2 PASSes."""
    per_cell_dir = tmp_path / "eval_results" / "issue_464" / "cross_eval" / "per_cell"
    per_cell_dir.mkdir(parents=True, exist_ok=True)
    _write_full_per_cell_tree(per_cell_dir, saturate_arm=None)
    payload = _run_phase5_main(tmp_path)
    assert payload["headline_status"] == "ok", (
        f"expected H2 PASS; got status={payload['headline_status']!r}; "
        f"headline={payload.get('headline')}"
    )
    assert payload["headline"]["h2_full_pass"] is True
    assert payload["dynamic_range_gate"]["ok"] is True


# ── Module-level imports used by the integration tests ─────────────────

import json  # noqa: E402 — kept below the helper tests for narrative locality
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
