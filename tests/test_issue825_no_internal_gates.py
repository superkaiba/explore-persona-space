"""Regression guard: --no-internal-gates defers every in-process fit halt.

Pins the review-round-1 BLOCKER `upload-before-gates-fit-g3` fix (plan v7
MF-C, "every FAILURE path is upload-then-exit"): with --no-internal-gates,
`_apply_gates` RECORDS a failing G3 verdict to g3_gate.json but never raises
SystemExit(3) — the calling wrapper evaluates the gate in its post-UPLOAD-2
gate block. Without the flag, legacy behavior is byte-identical: the failing
G3 still HALTs with exit code 3. Also pins the per-cell crash defer: a
degenerate cell crashes into fit_failures.json under the flag instead of
killing the fit phase pre-upload (review-r1 Minor, same bug class).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_fit_cells as fc  # noqa: E402

FAILING_G3_RES = {
    # obs layer-max BELOW every null draw's layer-max => G3 fail
    "summary": {"obs_layer_max_r2": 0.01, "null_layer_max_r2_per_draw": [0.5, 0.6]},
    "xy": {"conv_ids": np.asarray([str(i) for i in range(8)])},
}
G3_CELL = {"cell_id": "M_instruct_assistant_chat"}


def _args(tmp_path: Path, *, no_internal_gates: bool) -> SimpleNamespace:
    return SimpleNamespace(
        smoke=False, out_dir=tmp_path, folds=3, seed=0, no_internal_gates=no_internal_gates
    )


def test_failing_g3_defers_with_flag_and_records_verdict(tmp_path, monkeypatch):
    """--no-internal-gates: failing G3 is RECORDED (pass=false), never raised."""
    monkeypatch.delenv("EPS_SMOKE", raising=False)
    fc._apply_gates(G3_CELL, FAILING_G3_RES, _args(tmp_path, no_internal_gates=True))
    g3 = json.loads((tmp_path / "g3_gate.json").read_text())
    assert g3["pass"] is False
    assert g3["obs_layer_max_r2"] == 0.01


def test_failing_g3_still_halts_without_flag(tmp_path, monkeypatch):
    """Legacy (flag absent): failing G3 HALTs with SystemExit(3), unchanged."""
    monkeypatch.delenv("EPS_SMOKE", raising=False)
    with pytest.raises(SystemExit) as ei:
        fc._apply_gates(G3_CELL, FAILING_G3_RES, _args(tmp_path, no_internal_gates=False))
    assert ei.value.code == 3
    # the verdict is still recorded BEFORE the halt
    assert json.loads((tmp_path / "g3_gate.json").read_text())["pass"] is False


def test_record_fit_failure_appends(tmp_path):
    """Per-cell crash defer: failures append fail-loud to fit_failures.json."""
    fc._record_fit_failure(tmp_path, "M_pretrained_user_chat", ValueError("degenerate: 1 row"))
    fc._record_fit_failure(tmp_path, "M_instruct_user_chat__mlp", RuntimeError("boom"))
    failures = json.loads((tmp_path / "fit_failures.json").read_text())
    assert [f["cell_id"] for f in failures] == [
        "M_pretrained_user_chat",
        "M_instruct_user_chat__mlp",
    ]
    assert failures[0]["error_type"] == "ValueError"


# Round-3 fix (round-2 Claude Minor): a crash INSIDE _apply_gates itself —
# not just inside run_cell — must defer under --no-internal-gates. `res`
# lacking "summary" makes _apply_gates KeyError on the G3 cell first thing.
BROKEN_G3_RES = {"xy": {"conv_ids": np.asarray(["0", "1"])}}
G3_CELL_NONCHAT = {"cell_id": "M_instruct_assistant_chat", "format_key": "naturalistic"}


def _loop_args(tmp_path: Path, *, no_internal_gates: bool) -> SimpleNamespace:
    return SimpleNamespace(
        smoke=False,
        out_dir=tmp_path,
        folds=3,
        seed=0,
        no_internal_gates=no_internal_gates,
        turnstore_dir=tmp_path,
        null_draws=2,
        n_boot=5,
        cells="M_instruct_assistant_chat",
    )


def test_apply_gates_internal_crash_defers_with_flag(tmp_path, monkeypatch):
    """--no-internal-gates: an _apply_gates-internal crash records <cell>__gates."""
    monkeypatch.delenv("EPS_SMOKE", raising=False)
    monkeypatch.setattr(fc, "run_cell", lambda *a, **k: BROKEN_G3_RES)
    results = fc._fit_within_cells(
        [dict(G3_CELL_NONCHAT)], None, _loop_args(tmp_path, no_internal_gates=True)
    )
    assert "M_instruct_assistant_chat" in results  # the fit result itself is kept
    failures = json.loads((tmp_path / "fit_failures.json").read_text())
    assert [f["cell_id"] for f in failures] == ["M_instruct_assistant_chat__gates"]
    assert failures[0]["error_type"] == "KeyError"


def test_apply_gates_internal_crash_still_raises_without_flag(tmp_path, monkeypatch):
    """Legacy (flag absent): an _apply_gates-internal crash propagates, unchanged."""
    monkeypatch.delenv("EPS_SMOKE", raising=False)
    monkeypatch.setattr(fc, "run_cell", lambda *a, **k: BROKEN_G3_RES)
    with pytest.raises(KeyError):
        fc._fit_within_cells(
            [dict(G3_CELL_NONCHAT)], None, _loop_args(tmp_path, no_internal_gates=False)
        )
    assert not (tmp_path / "fit_failures.json").exists()
