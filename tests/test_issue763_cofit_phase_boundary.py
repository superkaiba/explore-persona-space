"""Issue #763 `neutral-contrast-and-cofit` round-2 phase-boundary handoff pins.

Pins the review-r1 Critical fixes (Claude C1/C2 + Codex C1/C2 — cross-machine
phase inputs neither uploaded nor staged; mock-artifact hazard at canonical
paths) so a future refactor cannot silently strip them:

- STATIC source pins (no torch/HF import): every artifact a phase reads after
  a machine boundary is either produced in-phase or staged from HF before its
  first local read; the Phase-A progress upload carries the capture manifest;
  the Phase-B directions upload carries neutral_arm_manifest; the plot fails
  loud (never warn-and-skips) on the plan-required manifest; the dispatcher
  arms the smoke scope + runs the residue check.
- FUNCTIONAL pins (light ``issue763_common`` import): the smoke-scope path
  redirect, the env/--smoke consistency contract, and the absolute
  pool-floor / production-dim validators (fail pre-fix, pass post-fix).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _norm(src: str) -> str:
    """Collapse whitespace so formatter line-wrapping cannot break a pin."""
    return re.sub(r"\s+", " ", src)


def _block(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


# ── static pins: upload manifest handoff (C1(i) / Codex C2) ───────────────────


def test_progress_upload_includes_capture_manifest():
    src = (SCRIPTS / "issue763_cofit_upload.py").read_text()
    progress = _block(src, "_PROGRESS_ARTIFACTS", "_DIRECTIONS_ARTIFACTS")
    assert "capture_arm_means_manifest.json" in progress, (
        "the Phase-A --progress-only pass must upload the capture manifest (the "
        "parity record Phase B refuses to run without) — review r1 C1(i)"
    )


def test_directions_upload_includes_neutral_arm_manifest():
    src = (SCRIPTS / "issue763_cofit_upload.py").read_text()
    directions = _block(src, "_DIRECTIONS_ARTIFACTS", "_FINAL_ARTIFACTS")
    assert "neutral_arm_manifest.json" in directions, (
        "the Phase-B --directions-only pass must upload neutral_arm_manifest.json "
        "(plan-required §6.5 deliverable) — review r1 C2 / Codex C2"
    )


def test_final_deliverable_check_includes_neutral_arm_manifest():
    src = _norm((SCRIPTS / "issue763_cofit_upload.py").read_text())
    assert 'neutral_arm = COFIT_DIR / "neutral_arm_manifest.json"' in src
    assert "(results, nonlinear, manifest, neutral_arm)" in src, (
        "the final upload's missing-deliverable existence check must include "
        "neutral_arm_manifest.json"
    )


# ── static pins: phase-boundary staging before first local read (C1/C2) ──────


def test_assemble_directions_stages_every_cross_boundary_input():
    src = (SCRIPTS / "issue763_extract_pv_rb.py").read_text()
    seg = _norm(_block(src, "def _phase_assemble_directions", "def main"))
    first_read = seg.index("load_json(_capture_manifest_path())")
    staging = seg[:first_read]
    for pin, why in [
        ("_stage_single_from_hf(", "capture manifest staging call"),
        ("capture_arm_means_manifest.json", "capture manifest HF path"),
        ('_stage_from_hf("arm_means"', "arm_means staging"),
        ('_stage_from_hf( "neutral_rollout_means"', "neutral_rollout_means staging"),
        ('_stage_from_hf("neutral_judge"', "neutral_judge staging"),
        ('_stage_from_hf("pv_rollouts"', "pv_rollouts (question recovery) staging"),
        ('filename_prefix="rb_"', "rb-shard staging (cos integrity read)"),
    ]:
        assert pin.replace("( ", "(") in staging.replace("( ", "("), (
            f"_phase_assemble_directions must stage {why} BEFORE its first local "
            "read (review r1 C1(ii) / Codex C1)"
        )


def test_stage_round_inputs_stages_manifest_frozen_inputs():
    src = (SCRIPTS / "issue763_cofit_predictors.py").read_text()
    seg = _norm(_block(src, "def _stage_round_inputs", "def write_inputs_manifest"))
    assert '_stage_from_hf("pv_rollouts"' in seg, (
        "Phase C must stage pv_rollouts (gitignored data/) before "
        "write_inputs_manifest — review r1 C2"
    )
    assert '_stage_from_hf("pv_judge_v2"' in seg, (
        "Phase C must stage pv_judge_v2 before write_inputs_manifest — review r1 C2"
    )
    assert "assert_pool_floor(" in seg, (
        "staged manifest inputs must be validated against the ABSOLUTE "
        "plan-registered pool size (review r1 C1(iii))"
    )


def test_plot_fails_loud_on_missing_neutral_arm_manifest():
    src = (SCRIPTS / "issue763_cofit_plot.py").read_text()
    assert "cos/yield panels skipped" not in src, (
        "the plot must never warn-and-skip the plan-required "
        "neutral_arm_manifest.json (review r1 C2 / Codex C2)"
    )
    seg = _norm(_block(src, "def main", "if __name__"))
    assert "_stage_single_from_hf(" in seg
    assert "raise RuntimeError(" in seg


def test_dispatcher_arms_smoke_scope_and_residue_check():
    src = (SCRIPTS / "issue763_cofit_dispatch.sh").read_text()
    assert "export EPM_ISSUE763_SMOKE_SCOPE=1" in src
    assert "unset EPM_ISSUE763_SMOKE_SCOPE" in src
    assert "[phase=residue_check]" in src, (
        "the smoke must end with the canonical-path residue check (review r1 C1(iii))"
    )


# ── functional pins: smoke scope + absolute validators ───────────────────────


@pytest.fixture()
def common(monkeypatch):
    """issue763_common with a CLEAN scope env; reloaded clean again at teardown."""
    monkeypatch.syspath_prepend(str(REPO / "src"))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.delenv("EPM_ISSUE763_SMOKE_SCOPE", raising=False)
    import issue763_common

    mod = importlib.reload(issue763_common)
    yield mod
    # teardown: whatever a test set, leave the module bound to CANONICAL paths
    import os

    os.environ.pop("EPM_ISSUE763_SMOKE_SCOPE", None)
    importlib.reload(sys.modules["issue763_common"])


def test_smoke_scoped_is_identity_without_env(common):
    p = Path("/a/issue_763/pv_shards")
    assert common.smoke_scoped(p) == p


def test_smoke_scoped_redirects_under_env(common, monkeypatch):
    monkeypatch.setenv("EPM_ISSUE763_SMOKE_SCOPE", "1")
    p = Path("/a/issue_763/pv_shards")
    assert common.smoke_scoped(p) == Path("/a/issue_763/smoke_scope/pv_shards")


def test_module_constants_rebind_under_env(monkeypatch, common):
    monkeypatch.setenv("EPM_ISSUE763_SMOKE_SCOPE", "1")
    scoped = importlib.reload(sys.modules["issue763_common"])
    assert scoped.COFIT_DIR.parent.name == "smoke_scope"
    assert scoped.NEUTRAL_ROLLOUT_DIR.parent.name == "smoke_scope"


def test_ensure_smoke_scope_env_without_smoke_raises(common, monkeypatch):
    monkeypatch.setenv("EPM_ISSUE763_SMOKE_SCOPE", "1")
    with pytest.raises(RuntimeError, match="smoke scope is smoke-only"):
        common.ensure_smoke_scope(False)


def test_ensure_smoke_scope_reexecs_on_smoke_without_env(common, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(common.os, "execvpe", lambda *a: calls.append(a))
    common.ensure_smoke_scope(True)
    assert len(calls) == 1, "ensure_smoke_scope(True) must re-exec with the scope env armed"
    _exe, argv, env = calls[0]
    assert env.get("EPM_ISSUE763_SMOKE_SCOPE") == "1"
    assert argv[0] == sys.executable


def test_assert_pool_floor_raises_below_plan_floor(common):
    # a 4-row mock pool at a canonical path (the review-r1 live hazard) fails loud
    with pytest.raises(RuntimeError, match="smoke/mock residue"):
        common.assert_pool_floor(4, common.NEUTRAL_POOL_EXPECTED, "neutral_rollouts")
    with pytest.raises(RuntimeError):
        common.assert_pool_floor(999, 1000, "pv_rollouts")
    common.assert_pool_floor(1000, 1000, "pv_rollouts")  # at floor: passes


def test_assert_production_direction_shape(common):
    # the (2, 8) tiny-smoke-model mock (review r1 C1(iii) live evidence) fails loud
    with pytest.raises(RuntimeError, match="smoke/mock residue"):
        common.assert_production_direction_shape((2, 8), "rb_deception.pt")
    common.assert_production_direction_shape((28, 3584), "rb_deception.pt")
