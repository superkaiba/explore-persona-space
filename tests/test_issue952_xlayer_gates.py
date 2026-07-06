"""Production RAISE-branch pins for the issue #952 cross-layer battery gates.

Gate 2 (`l20_reproduction_gate`): the recomputed unsuffixed pass-2 pooled R²
must match the parent's committed reference over the FULL expected
ARMS x POSITION_SLOTS grid (concern `l20-repro-gate-arm-subset-coverage`: the
expected set is enumerated, never inferred from the reference's own keys), and
the parent reference's `l_star` must equal 20.

Gate 3 (`suffixed_l20_calibration_gate`): the NEW suffixed-path L{l_star}
pass-2 outputs must match the unsuffixed pass-2 within tolerance AND select
identical λ*(slot). Both gates raise RuntimeError on a production miss and
log-only under --smoke.
"""

import copy

import numpy as np
import pytest

from explore_persona_space.experiments.issue_952.run_952 import (
    ARMS,
    POSITION_SLOTS,
    l20_reproduction_gate,
    suffixed_l20_calibration_gate,
)


def _full_ref(value: float = 0.5) -> dict:
    """A complete parent-reference / recomputed-report over ARMS x POSITION_SLOTS."""
    return {
        "l_star": 20,
        **{arm: {slot: {"test_pooled_r2": value} for slot in POSITION_SLOTS} for arm in ARMS},
    }


# ── gate 2: L20 reproduction ─────────────────────────────────────────────────────


def test_l20_gate_passes_on_identical_full_grid():
    rec = l20_reproduction_gate(_full_ref(), _full_ref(), smoke=False)
    assert rec["pass"] is True
    assert rec["n_cells_compared"] == rec["n_cells_expected"] == len(ARMS) * len(POSITION_SLOTS)
    assert rec["parent_l_star_ok"] is True and rec["max_abs_delta_r2"] == 0.0


def test_l20_gate_raises_on_r2_perturbation():
    """A 1e-4 R² drift on ONE cell trips the 1e-6 tolerance in production."""
    got = _full_ref()
    got["own"][POSITION_SLOTS[0]]["test_pooled_r2"] += 1e-4
    with pytest.raises(RuntimeError, match="L20 reproduction gate FAIL"):
        l20_reproduction_gate(_full_ref(), got, smoke=False)


def test_l20_gate_raises_on_truncated_parent_reference():
    """A parent reference missing a whole arm FAILS production — the expected
    grid is enumerated, so truncation can never silently shrink coverage
    (concern l20-repro-gate-arm-subset-coverage)."""
    ref = _full_ref()
    del ref["ext_style"]
    with pytest.raises(RuntimeError, match="parent-missing"):
        l20_reproduction_gate(ref, _full_ref(), smoke=False)


def test_l20_gate_raises_on_missing_recomputed_slot():
    got = _full_ref()
    del got["own"][POSITION_SLOTS[-1]]
    with pytest.raises(RuntimeError, match="recomputed-missing"):
        l20_reproduction_gate(_full_ref(), got, smoke=False)


def test_l20_gate_raises_on_wrong_parent_l_star():
    ref = _full_ref()
    ref["l_star"] = 17
    with pytest.raises(RuntimeError, match="l_star=17"):
        l20_reproduction_gate(ref, _full_ref(), smoke=False)


def test_l20_gate_smoke_executes_comparison_without_raising():
    """--smoke runs the IDENTICAL comparison and logs the miss (non-binding)."""
    ref = _full_ref()
    del ref["ext_style"]
    ref["l_star"] = 17
    rec = l20_reproduction_gate(ref, {}, smoke=True)
    assert rec["pass"] is False
    assert rec["n_missing_parent"] == len(POSITION_SLOTS)
    assert rec["n_missing_recomputed"] == 3 * len(POSITION_SLOTS)
    assert rec["parent_l_star_ok"] is False


def test_l20_gate_absent_l_star_field_is_skipped_with_reason():
    """Punch-list contract: an l_star-less reference skips the check (logged),
    the grid comparison still binds."""
    ref = _full_ref()
    del ref["l_star"]
    rec = l20_reproduction_gate(ref, _full_ref(), smoke=False)
    assert rec["pass"] is True and rec["parent_l_star"] is None


# ── gate 3: suffixed-path L20 calibration ────────────────────────────────────────


def _cal_npz(l_star: int = 20, drift: float = 0.0) -> dict:
    rng = np.random.default_rng(3)
    n, g = 5, 4
    sst = rng.uniform(0.5, 1.5, size=(n, g))
    ssr = rng.uniform(0.1, 0.9, size=(n, g))
    return {
        "A_test_ssres": ssr,
        "A_test_sstot": sst,
        f"A_test_ssres_L{l_star}": ssr + drift,
        f"A_test_sstot_L{l_star}": copy.deepcopy(sst),
    }


def test_calibration_gate_passes_on_identical_paths():
    rec = suffixed_l20_calibration_gate(_cal_npz(), 20, {"c_last": 3}, {"c_last": 3}, smoke=False)
    assert rec["pass"] is True and rec["max_abs_delta_pooled_r2_by_family"]["A_test"] == 0.0


def test_calibration_gate_raises_on_suffixed_drift():
    with pytest.raises(RuntimeError, match="calibration gate FAIL"):
        suffixed_l20_calibration_gate(
            _cal_npz(drift=1e-3), 20, {"c_last": 3}, {"c_last": 3}, smoke=False
        )


def test_calibration_gate_raises_on_lambda_mismatch():
    with pytest.raises(RuntimeError, match="calibration gate FAIL"):
        suffixed_l20_calibration_gate(_cal_npz(), 20, {"c_last": 3}, {"c_last": 5}, smoke=False)


def test_calibration_gate_smoke_executes_comparison_without_raising():
    rec = suffixed_l20_calibration_gate(
        _cal_npz(drift=1e-3), 20, {"c_last": 3}, {"c_last": 3}, smoke=True
    )
    assert rec["pass"] is False
