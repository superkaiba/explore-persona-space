# research code uses ※ and Δ legitimately
"""Unit tests for the #570 round-4 fixes (rescue-ramp coverage-assert incident).

2026-06-11: all 3 lr 2e-6 rescue seeds trained to band-stop (~step 195,
save_steps 3 x limit 40 = 117-step rolling window, retained [78..195]) and
then crashed on the round-2 ABSOLUTE ladder-coverage invariant (lowest
retained step <= 25), which was calibrated for the registered 5e-6 ramp
(stop ~95-110). Covered here:

- ``_issue543_common.assert_ladder_coverage_steps`` — the branch-aware shared
  form (lowest retained <= max(25, stop - 60)) against BOTH realized ramp
  shapes (5e-6 PASS, rescue PASS), a genuine rotation hole (FAIL), and the
  retained 5e-6 protection (a rotated 5e-6 window still FAILs).
- ``run_issue543_ratio._assert_ladder_coverage`` — the dir-globbing wrapper.
- ``run_issue543_ratio._existing_phase1_train_artifacts`` — the resume guard
  that skips retraining when a prior launch completed training (final adapter
  + callback_stop_record.json) but crashed downstream before
  phase1_result.json: completion predicate, integrity raises (corrupt record,
  band mismatch), and fall-through cases (no record / no weights).
- ``eval_issue570_ladder._existing_ladder_outputs`` — the per-seed ladder
  resume skip (phase1_ladder.json + phase1_pick_record.json present + intact).

All CPU-only and fast (no model loads, no network, no GPU pins — only the
pure helpers are exercised; module import side effects are env/logging only).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


common = _load("i570_common_under_test", "_issue543_common.py")
rr = _load("i570_ratio_under_test", "run_issue543_ratio.py")
ladder = _load("i570_ladder_under_test", "eval_issue570_ladder.py")


# ── assert_ladder_coverage_steps (shared branch-aware form) ──────────────────


def test_5e6_realized_shape_passes():
    # Registered ramp: stop 95, save_steps 5, limit 40 -> full ladder from 5.
    steps = list(range(5, 100, 5))
    out = common.assert_ladder_coverage_steps(steps, stop_step=95)
    assert out[0] == 5 and out[-1] == 95


def test_rescue_realized_shape_passes():
    # Round-4 incident shape: stop 195, save_steps 3, limit 40 -> retained
    # [78..195] (117-step window). bound = max(25, 195-60) = 135 >= 78.
    steps = list(range(78, 196, 3))
    out = common.assert_ladder_coverage_steps(steps, stop_step=195)
    assert out[0] == 78 and out[-1] == 195
    # Also passes with the default stop (= highest retained step).
    assert common.assert_ladder_coverage_steps(steps)[0] == 78


def test_genuine_rotation_hole_fails():
    # A genuinely shallow window: save_steps 5, limit 8 -> depth 35 < 60.
    # stop 300 -> retained [265..300]; bound = max(25, 240) = 240 < 265.
    steps = list(range(265, 301, 5))
    with pytest.raises(RuntimeError, match="rotated out"):
        common.assert_ladder_coverage_steps(steps, stop_step=300)


def test_5e6_protection_retained():
    # The 5e-6 ramp keeps its rotation protection: stop 95 -> bound 35; a
    # window whose lowest retained step is 41 still fails loud.
    steps = list(range(41, 96, 5))
    with pytest.raises(RuntimeError, match="rotated out"):
        common.assert_ladder_coverage_steps(steps, stop_step=95)


def test_absolute_floor_branch():
    # Short ramps stay on the absolute-25 floor: stop 40 -> bound = max(25,
    # -20) = 25; lowest 30 > 25 fails, lowest 20 passes.
    with pytest.raises(RuntimeError, match="rotated out"):
        common.assert_ladder_coverage_steps([30, 35, 40], stop_step=40)
    assert common.assert_ladder_coverage_steps([20, 30, 40], stop_step=40)[0] == 20


def test_empty_steps_raise():
    with pytest.raises(RuntimeError, match="no rolling checkpoints"):
        common.assert_ladder_coverage_steps([])


# ── run_issue543_ratio._assert_ladder_coverage (dir-globbing wrapper) ────────


def _make_ckpts(out_dir: Path, steps: list[int]) -> None:
    for s in steps:
        (out_dir / "adapter" / f"checkpoint-{s}").mkdir(parents=True)


def test_wrapper_globs_and_passes_rescue_shape(tmp_path):
    steps = list(range(78, 196, 3))
    _make_ckpts(tmp_path, steps)
    assert rr._assert_ladder_coverage(tmp_path, stop_step=195) == steps


def test_wrapper_fails_on_shallow_window(tmp_path):
    _make_ckpts(tmp_path, list(range(265, 301, 5)))
    with pytest.raises(RuntimeError, match="rotated out"):
        rr._assert_ladder_coverage(tmp_path, stop_step=300)


def test_wrapper_raises_on_empty_dir(tmp_path):
    (tmp_path / "adapter").mkdir()
    with pytest.raises(RuntimeError, match="no rolling checkpoints"):
        rr._assert_ladder_coverage(tmp_path)


# ── _existing_phase1_train_artifacts (phase-1 resume guard) ──────────────────

_BAND = {"band_low_nats": 25.43, "band_high_nats": 25.83}


def _write_completed_training(out_dir: Path, record_extra: dict | None = None) -> dict:
    """Mirror the pod's post-crash layout: final adapter + stop record +
    rolling checkpoints [78..195] (the rescue incident shape)."""
    adapter = out_dir / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00")
    _make_ckpts(out_dir, list(range(78, 196, 3)))
    record = {
        "stop_reason": "band",
        "stop_step": 195,
        "final_global_step": 195,
        **_BAND,
        **(record_extra or {}),
    }
    (out_dir / "callback_stop_record.json").write_text(json.dumps(record))
    return record


def test_resume_guard_none_without_stop_record(tmp_path):
    (tmp_path / "adapter").mkdir()
    assert (
        rr._existing_phase1_train_artifacts(tmp_path, band_low_delta=25.43, band_high_delta=25.83)
        is None
    )


def test_resume_guard_resumes_completed_training(tmp_path):
    record = _write_completed_training(tmp_path)
    out = rr._existing_phase1_train_artifacts(tmp_path, band_low_delta=25.43, band_high_delta=25.83)
    assert out is not None
    adapter_dir, got = out
    assert adapter_dir == tmp_path / "adapter"
    assert got["stop_reason"] == "band" and got["stop_step"] == record["stop_step"]


def test_resume_guard_raises_on_corrupt_record(tmp_path):
    _write_completed_training(tmp_path)
    (tmp_path / "callback_stop_record.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="does not parse"):
        rr._existing_phase1_train_artifacts(tmp_path, band_low_delta=25.43, band_high_delta=25.83)


def test_resume_guard_raises_on_band_mismatch(tmp_path):
    # Same dir trained under a DIFFERENT band -> wrong-artifact reuse hazard.
    _write_completed_training(tmp_path)
    with pytest.raises(RuntimeError, match="DIFFERENT recipe/band"):
        rr._existing_phase1_train_artifacts(tmp_path, band_low_delta=20.0, band_high_delta=20.4)


def test_resume_guard_none_without_weights(tmp_path):
    _write_completed_training(tmp_path)
    (tmp_path / "adapter" / "adapter_model.safetensors").unlink()
    assert (
        rr._existing_phase1_train_artifacts(tmp_path, band_low_delta=25.43, band_high_delta=25.83)
        is None
    )


# ── _existing_ladder_outputs (per-seed ladder resume skip) ───────────────────


def _write_ladder_outputs(cell: Path) -> None:
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "phase1_ladder.json").write_text(json.dumps({"checkpoints": [{"step": 80}]}))
    (cell / "phase1_pick_record.json").write_text(
        json.dumps({"pick_step": None, "eligible_steps": [], "fallback": True})
    )


def test_ladder_skip_none_when_files_missing(tmp_path):
    assert ladder._existing_ladder_outputs(tmp_path) is None
    _write_ladder_outputs(tmp_path)
    (tmp_path / "phase1_ladder.json").unlink()  # mid-ladder crash shape
    assert ladder._existing_ladder_outputs(tmp_path) is None


def test_ladder_skip_returns_pick_when_complete(tmp_path):
    _write_ladder_outputs(tmp_path)
    pick = ladder._existing_ladder_outputs(tmp_path)
    assert pick is not None and pick["fallback"] is True and pick["pick_step"] is None


def test_ladder_skip_raises_on_corrupt_json(tmp_path):
    _write_ladder_outputs(tmp_path)
    (tmp_path / "phase1_pick_record.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="do not parse"):
        ladder._existing_ladder_outputs(tmp_path)


def test_ladder_skip_raises_on_foreign_artifacts(tmp_path):
    _write_ladder_outputs(tmp_path)
    (tmp_path / "phase1_pick_record.json").write_text(json.dumps({"unrelated": 1}))
    with pytest.raises(RuntimeError, match="partial or foreign"):
        ladder._existing_ladder_outputs(tmp_path)
