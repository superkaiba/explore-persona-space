"""Dispatcher Phase A smoke-gate decision logic (task #397).

Plan v4 §5.7 decision bands:
  - <2 min/ckpt   → PASS (proceed to Phase B sweep)
  - 2-10 min/ckpt → WARN (re-plan)
  - >10 min/ckpt  → FAIL (re-plan)

This test surface pins the band semantics on the pure-Python helpers in
``scripts/dispatch_factor_screen_397.py``: ``classify_smoke_timing`` and
``build_smoke_marker``. No GPU / no model load.

Also covers the live M1 override in ``build_smoke_marker`` — a smoke that
passes on timing but reports source_rate == 0.0 must downgrade PASS → WARN
per plan v4 §5.7's live recipe-fix check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# scripts/ is not a package on this repo's PYTHONPATH, so importlib it.
_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)
_spec = importlib.util.spec_from_file_location("dispatch_factor_screen_397", _DISPATCH_PATH)
_dispatch = importlib.util.module_from_spec(_spec)
sys.modules["dispatch_factor_screen_397"] = _dispatch
_spec.loader.exec_module(_dispatch)


# --- classify_smoke_timing ---------------------------------------------------


def test_classify_below_two_minutes_returns_pass() -> None:
    """<2 min/ckpt is the PASS band per plan v4 §5.7."""
    assert _dispatch.classify_smoke_timing(0.5) == "pass"
    assert _dispatch.classify_smoke_timing(1.99) == "pass"


def test_classify_two_minutes_exact_returns_warn() -> None:
    """Boundary: 2.0 min/ckpt is WARN (band is `[2, 10]` inclusive)."""
    assert _dispatch.classify_smoke_timing(2.0) == "warn"


def test_classify_middle_band_returns_warn() -> None:
    """2-10 min/ckpt is the WARN band."""
    assert _dispatch.classify_smoke_timing(5.0) == "warn"
    assert _dispatch.classify_smoke_timing(9.9) == "warn"


def test_classify_ten_minutes_exact_returns_warn() -> None:
    """Boundary: 10.0 min/ckpt is WARN (still inside `[2, 10]`)."""
    assert _dispatch.classify_smoke_timing(10.0) == "warn"


def test_classify_above_ten_minutes_returns_fail() -> None:
    """>10 min/ckpt is the FAIL band."""
    assert _dispatch.classify_smoke_timing(10.01) == "fail"
    assert _dispatch.classify_smoke_timing(30.0) == "fail"


def test_classify_negative_raises() -> None:
    """Negative timing is nonsense — refuse to score a verdict."""
    with pytest.raises(ValueError, match="non-negative"):
        _dispatch.classify_smoke_timing(-0.1)


# --- build_smoke_marker ------------------------------------------------------


def test_build_smoke_marker_pass_returns_smoke_pass_kind() -> None:
    """Verdict 'pass' maps to kind 'epm:smoke-pass'."""
    kind, note = _dispatch.build_smoke_marker(
        "pass",
        avg_minutes_per_checkpoint=1.2,
        n_checkpoints=6,
        total_eval_minutes=7.2,
        train_minutes=25.0,
        source_rate=0.85,
        cell_key="10010",
        source="librarian",
        seed=42,
    )
    assert kind == "epm:smoke-pass"
    assert "PASS" in note
    assert "1.20 min/ckpt" in note
    assert "0.850" in note
    assert "10010" in note


def test_build_smoke_marker_warn_returns_smoke_warn_kind() -> None:
    """Verdict 'warn' maps to kind 'epm:smoke-warn'."""
    kind, note = _dispatch.build_smoke_marker(
        "warn",
        avg_minutes_per_checkpoint=5.0,
        n_checkpoints=6,
        total_eval_minutes=30.0,
        train_minutes=25.0,
        source_rate=0.5,
        cell_key="10010",
        source="librarian",
        seed=42,
    )
    assert kind == "epm:smoke-warn"
    assert "WARN" in note
    assert "User must gate" in note


def test_build_smoke_marker_fail_returns_smoke_fail_kind() -> None:
    """Verdict 'fail' maps to kind 'epm:smoke-fail'."""
    kind, note = _dispatch.build_smoke_marker(
        "fail",
        avg_minutes_per_checkpoint=15.0,
        n_checkpoints=6,
        total_eval_minutes=90.0,
        train_minutes=25.0,
        source_rate=0.4,
        cell_key="10010",
        source="librarian",
        seed=42,
    )
    assert kind == "epm:smoke-fail"
    assert "FAIL" in note
    assert "User must gate" in note


def test_build_smoke_marker_pass_with_zero_source_rate_downgrades_to_warn() -> None:
    """Plan v4 §5.7 live M1 check: PASS on timing + source_rate==0 → WARN.

    A timing-only PASS with no marker emission is a strong signal that the
    M1 marker threading or recipe-fix port is broken; the verdict must
    downgrade so the user re-checks recipe wiring before launching 324 runs.
    """
    kind, note = _dispatch.build_smoke_marker(
        "pass",
        avg_minutes_per_checkpoint=1.0,
        n_checkpoints=6,
        total_eval_minutes=6.0,
        train_minutes=25.0,
        source_rate=0.0,
        cell_key="10010",
        source="librarian",
        seed=42,
    )
    assert kind == "epm:smoke-warn"
    assert "Override" in note
    assert "source rate == 0.0" in note


def test_build_smoke_marker_pass_with_nonzero_source_rate_stays_pass() -> None:
    """Sanity: PASS timing + healthy source rate stays PASS."""
    kind, _note = _dispatch.build_smoke_marker(
        "pass",
        avg_minutes_per_checkpoint=1.0,
        n_checkpoints=6,
        total_eval_minutes=6.0,
        train_minutes=25.0,
        source_rate=0.7,
        cell_key="10010",
        source="librarian",
        seed=42,
    )
    assert kind == "epm:smoke-pass"


def test_build_smoke_marker_rejects_unknown_verdict() -> None:
    """Sentinel: only pass/warn/fail are valid verdicts."""
    with pytest.raises(ValueError, match="Unknown verdict"):
        _dispatch.build_smoke_marker(
            "maybe",
            avg_minutes_per_checkpoint=1.0,
            n_checkpoints=6,
            total_eval_minutes=6.0,
            train_minutes=25.0,
            source_rate=0.7,
            cell_key="10010",
            source="librarian",
            seed=42,
        )


def test_build_smoke_marker_handles_unknown_source_rate() -> None:
    """source_rate=None (no metrics_final.json yet) is allowed; note still renders."""
    kind, note = _dispatch.build_smoke_marker(
        "pass",
        avg_minutes_per_checkpoint=1.0,
        n_checkpoints=6,
        total_eval_minutes=6.0,
        train_minutes=25.0,
        source_rate=None,
        cell_key="10010",
        source="librarian",
        seed=42,
    )
    assert kind == "epm:smoke-pass"
    assert "Source substring rate at final checkpoint: None" in note
