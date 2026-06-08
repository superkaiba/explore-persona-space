# em-dash intentional
"""Task #504 v3 `_select_active_phase0_pick` — v3 > v2-fallback > v2-primary.

Pins the contract that the dispatcher's helper prefers v3 over v2 once Phase
0 v3 has produced a pass-verdict artifact (plan v3 §2 "v3 takes precedence
over v2"), but falls through correctly when v3 isn't available or doesn't
pass. The non-pass v3 verdict (Trigger A/B/C) falls through to v2 selection.

CPU-only, sub-second. Writes synthetic artifact JSON files under tmp_path and
calls the helper directly; no GPU/HF/network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dispatch_neg_geometry_504 import _select_active_phase0_pick


def _v2_pass_artifact(*, fallback_triggered: bool = False) -> dict:
    return {
        "version": 2,
        "chosen_lr": 1e-4,
        "chosen_rank": 8,
        "chosen_alpha": 32,
        "chosen_checkpoint_fraction": 0.5,
        "chosen_checkpoint_steps": 12,
        "source": "villain",
        "verdict": "pass",
        "fallback_triggered": fallback_triggered,
        "fallback_reason": None,
    }


def _v2_fallback_triggered_artifact() -> dict:
    return {
        "version": 2,
        "chosen_lr": None,
        "chosen_rank": 8,
        "chosen_alpha": 32,
        "chosen_checkpoint_fraction": None,
        "source": "villain",
        "verdict": "no_in_band_anchor",
        "fallback_triggered": True,
        "fallback_reason": "trigger_A_floor: ...",
    }


def _v3_pass_artifact(*, chosen_epochs: int = 2) -> dict:
    return {
        "version": 3,
        "epochs_ladder": [2, 3],
        "fixed_lr": 1e-4,
        "fixed_rank": 8,
        "fixed_alpha": 32,
        "chosen_epochs": chosen_epochs,
        "chosen_lr": 1e-4,
        "chosen_rank": 8,
        "chosen_alpha": 32,
        "chosen_checkpoint_fraction": 0.5,
        "chosen_checkpoint_steps": 25,
        "chosen_source": "villain",
        "source": "villain",
        "verdict": "pass",
        "fallback_triggered": False,
        "fallback_reason": None,
        "in_plan_recovery_triggered": False,
    }


def _v3_exit_artifact() -> dict:
    """v3 pick artifact with Trigger A fired (no in-band anchor)."""
    art = _v3_pass_artifact()
    art["chosen_epochs"] = None
    art["chosen_checkpoint_fraction"] = None
    art["verdict"] = "no_in_band_anchor"
    art["fallback_triggered"] = True
    art["fallback_reason"] = "trigger_A_floor: ..."
    return art


def test_v3_pass_overrides_v2(tmp_path: Path):
    """When v3 is pass AND v2 is also pass, v3 takes precedence."""
    v2_path = tmp_path / "phase0_calibration_v2.json"
    v2_fb_path = tmp_path / "phase0_calibration_v2_fallback.json"
    v3_path = tmp_path / "phase0_calibration_v3.json"
    v2_path.write_text(json.dumps(_v2_pass_artifact()))
    v3_path.write_text(json.dumps(_v3_pass_artifact()))

    pick, active_path = _select_active_phase0_pick(v2_path, v2_fb_path, v3_pick_path=v3_path)
    assert active_path == v3_path
    assert pick["version"] == 3
    assert pick["chosen_epochs"] == 2


def test_v3_non_pass_falls_through_to_v2(tmp_path: Path):
    """When v3 has Trigger A/B/C fired (verdict != 'pass'), the helper falls
    through to v2 selection. (In practice the dispatcher should NOT call
    Phase 1 in this case — but the helper should still behave correctly when
    a future caller routes via this path.)"""
    v2_path = tmp_path / "phase0_calibration_v2.json"
    v2_fb_path = tmp_path / "phase0_calibration_v2_fallback.json"
    v3_path = tmp_path / "phase0_calibration_v3.json"
    v2_path.write_text(json.dumps(_v2_pass_artifact()))
    v3_path.write_text(json.dumps(_v3_exit_artifact()))

    pick, active_path = _select_active_phase0_pick(v2_path, v2_fb_path, v3_pick_path=v3_path)
    # Non-pass v3 → fall through to v2.
    assert active_path == v2_path
    assert pick["version"] == 2


def test_v3_absent_uses_v2(tmp_path: Path):
    """When v3 artifact doesn't exist, the helper uses v2 (unchanged
    behavior from before the v3 extension)."""
    v2_path = tmp_path / "phase0_calibration_v2.json"
    v2_fb_path = tmp_path / "phase0_calibration_v2_fallback.json"
    v3_path = tmp_path / "phase0_calibration_v3.json"
    v2_path.write_text(json.dumps(_v2_pass_artifact()))
    # v3_path NOT written.

    pick, active_path = _select_active_phase0_pick(v2_path, v2_fb_path, v3_pick_path=v3_path)
    assert active_path == v2_path
    assert pick["version"] == 2


def test_v3_absent_v2_fallback_fired_uses_v2_fallback(tmp_path: Path):
    """When v3 absent AND v2 fired fallback AND v2 fallback exists, use the
    v2 fallback (existing behavior preserved)."""
    v2_path = tmp_path / "phase0_calibration_v2.json"
    v2_fb_path = tmp_path / "phase0_calibration_v2_fallback.json"
    v3_path = tmp_path / "phase0_calibration_v3.json"
    v2_path.write_text(json.dumps(_v2_fallback_triggered_artifact()))
    v2_fb_path.write_text(json.dumps({**_v2_pass_artifact(), "source": "medical_doctor"}))
    # v3_path NOT written.

    pick, active_path = _select_active_phase0_pick(v2_path, v2_fb_path, v3_pick_path=v3_path)
    assert active_path == v2_fb_path
    assert pick["source"] == "medical_doctor"


def test_v3_pick_path_unset_uses_v2_only(tmp_path: Path):
    """When the caller doesn't pass v3_pick_path (None), the helper behaves
    byte-identically to the pre-v3 two-arg version."""
    v2_path = tmp_path / "phase0_calibration_v2.json"
    v2_fb_path = tmp_path / "phase0_calibration_v2_fallback.json"
    v2_path.write_text(json.dumps(_v2_pass_artifact()))

    pick, active_path = _select_active_phase0_pick(v2_path, v2_fb_path)
    assert active_path == v2_path
    assert pick["version"] == 2


def test_no_artifact_raises(tmp_path: Path):
    """When neither v3 (or non-pass v3) nor v2 primary exists, FileNotFoundError."""
    v2_path = tmp_path / "phase0_calibration_v2.json"
    v2_fb_path = tmp_path / "phase0_calibration_v2_fallback.json"
    v3_path = tmp_path / "phase0_calibration_v3.json"
    # Only the non-pass v3 artifact exists; v2 primary missing.
    v3_path.write_text(json.dumps(_v3_exit_artifact()))

    with pytest.raises(FileNotFoundError, match=r"Phase 1 requires"):
        _select_active_phase0_pick(v2_path, v2_fb_path, v3_pick_path=v3_path)
