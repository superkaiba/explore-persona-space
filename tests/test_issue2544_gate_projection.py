"""#2544 v5 pilot-gate projection-vs-booking pins (grain-mismatch fix,
production designed halt 2026-08-25, pod-2544 21:34Z).

The P4a/P4b entry pilots compare the pilot-extrapolated PHASE TOTAL against
2x the plan v8 §9 GPU-h booking (the §7 registered cost-abort semantics:
"pilot-extrapolated wall > 2x the §9 booking") — NEVER a realized per-unit
wall against the #1902 per-unit-class basis. The grains differ: a #1902
"unit" is one (cell, fold) at a SINGLE layer class (15.3 s sweep / 12.8 s
grid / 43 s transfer), while a realized #2544 P4a sweep unit fits ALL 17
LAYERS of one (rung, fold) (measured 100.57 s at n_tr 9,040, on-GPU,
layer-chunk 8) — the pre-fix per-unit comparison false-fired rc=7 while the
phase total projected 2.51 of 4.0 booked GPU-h.

Pure-function + duck-typed-ctx tests; no GPU, no store, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# issue2544_fits MUST import first: it imports issue2544_common, which sets
# the ladder env BEFORE issue1902_common binds its constants.
# isort: off
import issue2544_fits as F2  # noqa: E402
import issue1902_run as R  # noqa: E402

# isort: on


def test_p4a_production_numbers_pass():
    """The exact production measurement that false-halted pre-fix PASSes:
    100.57 s/unit x 90 units = 2.514 GPU-h < 2 x 4.0 booked."""
    rep = F2.pilot_projection_gate(
        klass="sweep",
        per_unit_s=100.57,
        n_units=90,
        booked_gpu_h=F2.P4A_BOOKED_GPU_H,
        basis_s=F2.P4A_UNIT_BASIS_S,
        grain="17L-rung-fold-unit",
    )
    assert rep["halt"] is False
    assert rep["projected_gpu_h"] == pytest.approx(100.57 * 90 / 3600.0, abs=1e-3)
    assert rep["form"] == "projected_gpu_h_vs_booked"
    assert rep["booked_gpu_h"] == 4.0
    # audit-trail fields preserved (record-only, never compared)
    assert rep["per_unit_s"] == pytest.approx(100.57)
    assert rep["basis_s"] == pytest.approx(15.3)
    assert rep["n_units"] == 90
    assert rep["grain"] == "17L-rung-fold-unit"
    assert rep["abort_ratio"] == 2.0


def test_p4a_halts_over_2x_booking():
    """700 s/unit x 90 units = 17.5 GPU-h > 8.0 = 2 x booking -> halt."""
    rep = F2.pilot_projection_gate(
        klass="sweep",
        per_unit_s=700.0,
        n_units=90,
        booked_gpu_h=F2.P4A_BOOKED_GPU_H,
        basis_s=F2.P4A_UNIT_BASIS_S,
        grain="17L-rung-fold-unit",
    )
    assert rep["halt"] is True
    assert rep["projected_gpu_h"] == pytest.approx(17.5)


def test_halt_boundary_is_strict():
    """projected == exactly 2x booking proceeds (registered abort is > 2x)."""
    rep = F2.pilot_projection_gate(
        klass="sweep",
        per_unit_s=2.0 * 4.0 * 3600.0 / 90,
        n_units=90,
        booked_gpu_h=4.0,
        basis_s=F2.P4A_UNIT_BASIS_S,
        grain="17L-rung-fold-unit",
    )
    assert rep["halt"] is False


def test_p4b_other_class_share_counts_toward_phase_total():
    """The sibling-class share participates in the phase-total verdict."""
    kw = dict(
        klass="grid",
        per_unit_s=100.0,
        n_units=816,
        booked_gpu_h=F2.P4B_BOOKED_GPU_H,
        basis_s=F2.P4B_GRID_BASIS_S,
        grain="B6-cell-fold-unit",
    )
    # 816 x 100 s = 22.67 GPU-h alone: under 2 x 17 = 34
    assert F2.pilot_projection_gate(**kw, other_gpu_h=0.0)["halt"] is False
    # + 12 GPU-h sibling share = 34.67 > 34 -> halt
    rep = F2.pilot_projection_gate(**kw, other_gpu_h=12.0)
    assert rep["halt"] is True
    assert rep["other_classes_gpu_h"] == pytest.approx(12.0)


def test_p4b_at_1902_bases_passes():
    """At the #1902 per-class bases the P4b phase total sits well inside the
    booking (plan §9: ~14.6 computed, 17 booked) — no halt."""
    grid_share = 816 * F2.P4B_GRID_BASIS_S / 3600.0
    rep = F2.pilot_projection_gate(
        klass="xfer",
        per_unit_s=F2.P4B_XFER_BASIS_S,
        n_units=312,  # 52 pairs x 6 folds
        booked_gpu_h=F2.P4B_BOOKED_GPU_H,
        basis_s=F2.P4B_XFER_BASIS_S,
        grain="star-pair-fold-unit",
        other_gpu_h=grid_share,
    )
    assert rep["halt"] is False
    assert rep["projected_gpu_h"] < F2.P4B_BOOKED_GPU_H


def _ctx(tmp_path: Path, *, smoke: bool, timings: list[dict]) -> SimpleNamespace:
    """Duck-typed FitsCtx carrying exactly the attrs _check_class_pilot reads."""
    return SimpleNamespace(pilot_timings=timings, smoke=smoke, out_root=tmp_path)


def test_check_class_pilot_halts_via_designed_halt(tmp_path):
    """Real body: an over-projection exits GATE_RC (rc=7) and writes the
    gate-report JSON in the projection-vs-booking form."""
    ctx = _ctx(tmp_path, smoke=False, timings=[{"klass": "xfer", "wall_s": 3600.0}])
    with pytest.raises(SystemExit) as ei:
        F2._check_class_pilot(
            ctx,
            "xfer",
            F2.P4B_XFER_BASIS_S,
            n_units=312,
            other_gpu_h=0.0,
            grain="star-pair-fold-unit",
        )
    assert ei.value.code == R.GATE_RC
    reports = list((tmp_path / "gate_reports").glob("p4b_pilot_wall_xfer_*.json"))
    assert len(reports) == 1
    rep = json.loads(reports[0].read_text())
    assert rep["form"] == "projected_gpu_h_vs_booked"
    assert rep["projected_gpu_h"] > 2 * rep["booked_gpu_h"]
    assert rep["verdict"] == "HALT"


def test_check_class_pilot_smoke_demotes_to_informational(tmp_path):
    """Under smoke the same over-projection logs only — no exit, no report
    (gate-calibration parity: compute identically, demote the verdict)."""
    ctx = _ctx(tmp_path, smoke=True, timings=[{"klass": "xfer", "wall_s": 3600.0}])
    F2._check_class_pilot(
        ctx,
        "xfer",
        F2.P4B_XFER_BASIS_S,
        n_units=312,
        other_gpu_h=0.0,
        grain="star-pair-fold-unit",
    )
    assert not (tmp_path / "gate_reports").exists()


def test_check_class_pilot_resumed_done_noop(tmp_path):
    """No timing for the class (fully-resumed pilot unit) -> silent return."""
    ctx = _ctx(tmp_path, smoke=False, timings=[{"klass": "grid", "wall_s": 9.0}])
    F2._check_class_pilot(
        ctx,
        "xfer",
        F2.P4B_XFER_BASIS_S,
        n_units=312,
        other_gpu_h=0.0,
        grain="star-pair-fold-unit",
    )
    assert not (tmp_path / "gate_reports").exists()
