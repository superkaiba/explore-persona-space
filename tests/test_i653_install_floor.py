"""Task #653 v8 §6Δ.1 per-cell install FLOOR (the binding-defect fix).

CPU-only. The parent run read its H3-diffuse verdict off 15 of 18 non-/marginally-
installed cells; v8 gates each cell's geometry read on a behavior-specific install
floor (marker [5,12] nat; sycophancy ≥+0.40; EM ≥+0.20 judge-rate gain). A
below-floor cell is DROPPED + reported, never read as geometry.

These tests pin:
  * ``_install_pass_ok`` per behavior, including the parent's exact values
    (syco +0.15, EM 0.0, marker +0.78) that v5's >0 cutoff wrongly let pass.
  * the floor constants are the plan §6Δ.1 numbers.
  * the analyze phase DROPS a below-floor cell from the §3.4 aggregation
    (records ``dropped_non_install`` and excludes it from ``verdicts``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments import issue_653 as i653

# ── _install_pass_ok per behavior (plan §6Δ.4) ───────────────────────────────


def test_floor_constants_are_the_plan_numbers():
    assert i653.GATE_SYCOPHANCY_INSTALL_MIN_RATE_GAIN == 0.40
    assert i653.GATE_EM_INSTALL_MIN_RATE_GAIN == 0.20
    assert i653.GATE_MARKER_INSTALL_LOW_NATS == 5.0
    assert i653.GATE_MARKER_INSTALL_HIGH_NATS == 12.0


def test_syco_below_floor_fails():
    """The parent's +0.15 sycophancy → False. v5's 0.0 cutoff wrongly read True."""
    ok, detail = i653._install_pass_ok({"judge_rate_gain": 0.15}, "sycophancy")
    assert ok is False
    assert detail["floor"] == 0.40
    assert detail["value"] == 0.15


def test_syco_at_floor_passes():
    ok, detail = i653._install_pass_ok({"judge_rate_gain": 0.42}, "sycophancy")
    assert ok is True
    assert detail["passed"] is True


def test_syco_exactly_at_floor_passes():
    """The floor is inclusive (≥ +0.40)."""
    ok, _ = i653._install_pass_ok({"judge_rate_gain": 0.40}, "sycophancy")
    assert ok is True


def test_em_zero_fails():
    """The parent's 0.0 EM install → False."""
    ok, _ = i653._install_pass_ok({"judge_rate_gain": 0.0}, "em")
    assert ok is False


def test_em_marginal_fails():
    """A +0.10 marginal EM is below the +0.20 floor → False (v5's >0 read True)."""
    ok, _ = i653._install_pass_ok({"judge_rate_gain": 0.10}, "em")
    assert ok is False


def test_em_above_floor_passes():
    ok, _ = i653._install_pass_ok({"judge_rate_gain": 0.25}, "em")
    assert ok is True


def test_marker_in_band_passes():
    ok, detail = i653._install_pass_ok({"logp_trained_minus_base": 8.0}, "marker")
    assert ok is True
    assert detail["band"] == [5.0, 12.0]


def test_marker_below_band_fails():
    """The parent's +0.78 nat marker (under-trained, below [5,12]) → False."""
    ok, _ = i653._install_pass_ok({"logp_trained_minus_base": 0.78}, "marker")
    assert ok is False


def test_marker_above_band_fails():
    """A saturated marker (above the band) → False (the band is two-sided)."""
    ok, _ = i653._install_pass_ok({"logp_trained_minus_base": 25.0}, "marker")
    assert ok is False


def test_none_dv_fails_loud_never_passes():
    """A None DV (install read never produced) FAILS — geometry must not be read
    off a cell with no install evidence."""
    ok, detail = i653._install_pass_ok({"judge_rate_gain": None}, "sycophancy")
    assert ok is False
    assert "missing" in detail["reason"]
    ok2, detail2 = i653._install_pass_ok({"logp_trained_minus_base": None}, "marker")
    assert ok2 is False
    assert "missing" in detail2["reason"]


# ── analyze phase: a below-floor cell is DROPPED from the §3.4 aggregation ────


def _load_dispatcher():
    repo_root = Path(__file__).resolve().parents[1]
    disp_path = repo_root / "scripts" / "issue_653" / "i653_dispatch.py"
    spec = importlib.util.spec_from_file_location("i653_dispatch_floor_test", disp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i653_dispatch_floor_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_geometry_cell(armB: Path, cell_id: str, cell_group: str, rung: str) -> None:
    """A diffuse (H3) dx geometry JSON + matching Δx tensor for one cell."""
    n_rows = 30
    (armB / f"dx_geometry_{cell_id}.json").write_text(
        json.dumps(
            {
                "cell_group": cell_group,
                "rung": rung,
                "n_rows": n_rows,
                "top_share_lambda": 0.2,
                "pr_lambda": 8.0,  # ≥5 ⇒ H3
                "rank_k_at_90": 12,
                "cos_top_to_rb": None,
                "random_ci_high": 0.3,
                "dx_top_direction": list(np.ones(8) / np.sqrt(8)),
            }
        )
    )
    (armB / "dx_tensors").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.savez(armB / "dx_tensors" / f"{cell_id}.npz", cloud=rng.standard_normal((n_rows, 8)))


def _write_install(armB: Path, cell_id: str, behavior: str, *, gain: float) -> None:
    armB.mkdir(parents=True, exist_ok=True)
    dv = (
        {"dv_kind": "marker_four_float", "logp_trained_minus_base": gain}
        if behavior == "marker"
        else {"dv_kind": "judge_rate_plus_gain", "behavior": behavior, "judge_rate_gain": gain}
    )
    (armB / f"install_{cell_id}.json").write_text(json.dumps({"cell_id": cell_id, "install": dv}))


def test_dropped_cell_excluded_from_grid(tmp_path):
    """A below-floor cell does NOT contribute to the §3.4 verdict aggregation; it
    is recorded under dropped_non_install_cells instead. An installed sibling IS
    read. This is the binding-defect fix end-to-end through phase_analyze."""
    mod = _load_dispatcher()
    out_root = tmp_path / "eval_results" / "issue_653"
    armB = out_root / "armB"
    armB.mkdir(parents=True)

    installed = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    dropped = i653.ArmBCell(behavior="em", source="florist", rung="r16", seed=i653.HEADLINE_SEED)
    # installed cell clears the +0.40 sycophancy floor; dropped EM cell is at 0.0.
    _write_geometry_cell(armB, installed.cell_id, installed.cell_group, "r16")
    _write_geometry_cell(armB, dropped.cell_id, dropped.cell_group, "r16")
    _write_install(armB, installed.cell_id, "sycophancy", gain=0.55)
    _write_install(armB, dropped.cell_id, "em", gain=0.0)
    # ablation present for both r16 cells so require_complete=False path is clean.

    res = mod.phase_analyze([installed, dropped], out_root=out_root, require_complete=False)
    grid = json.loads((out_root / "cross_arm_verdict.json").read_text())

    read_ids = {vd["cell_id"] for vd in grid["verdicts"]}
    dropped_ids = {d["cell_id"] for d in grid["dropped_non_install_cells"]}
    assert installed.cell_id in read_ids
    assert dropped.cell_id not in read_ids  # NOT read as geometry
    assert dropped.cell_id in dropped_ids  # recorded by name
    assert grid["n_dropped_non_install"] == 1
    assert all(d["dropped_non_install"] for d in grid["dropped_non_install_cells"])
    # the installed cell carries its passing install evidence:
    inst_vd = next(vd for vd in grid["verdicts"] if vd["cell_id"] == installed.cell_id)
    assert inst_vd["dropped_non_install"] is False
    assert inst_vd["install_pass"] is True
    assert res["n_cells"] == 1  # only the installed cell


def test_require_complete_raises_on_missing_install(tmp_path):
    """Under require_complete the §6Δ.1 floor gate must SEE an install read for
    every swept cell — a missing install JSON raises (no silent geometry read)."""
    mod = _load_dispatcher()
    out_root = tmp_path / "eval_results" / "issue_653"
    armB = out_root / "armB"
    armB.mkdir(parents=True)
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    _write_geometry_cell(armB, cell.cell_id, cell.cell_group, "r16")
    # NO install JSON written + an Arm A rho dir so the earlier require_complete
    # rho gate is not the one that trips (same rho format as the ablation tests).
    (armB.parent / "armA").mkdir(parents=True, exist_ok=True)
    (armB.parent / "armA" / "rho_geometry_seed42.json").write_text(
        json.dumps({"geometry": {"iso": {"rho_top_direction": list(np.ones(8) / np.sqrt(8))}}})
    )
    with pytest.raises(FileNotFoundError, match=r"install_.*\.json"):
        mod.phase_analyze([cell], out_root=out_root, require_complete=True)


# ── seed-137 floor-clearing enumeration (§4Δ.5) ──────────────────────────────


def test_seed137_only_floor_clearing_lora_cells():
    """seed 137 is added ONLY for floor-clearing LoRA cells at seed 42 — never a
    dropped cell, never a full-FT cell."""
    grid = {
        "verdicts": [
            {
                "cell_id": f"sycophancy__florist__r16__seed{i653.HEADLINE_SEED}",
                "dropped_non_install": False,
            },
            {
                "cell_id": f"marker__florist__r1__seed{i653.HEADLINE_SEED}",
                "dropped_non_install": False,
            },
            # a full-FT cell that installed — stays single-seed (excluded):
            {
                "cell_id": f"sycophancy__florist__full__seed{i653.HEADLINE_SEED}",
                "dropped_non_install": False,
            },
        ],
        "dropped_non_install_cells": [
            {"cell_id": f"em__florist__r16__seed{i653.HEADLINE_SEED}", "dropped_non_install": True},
        ],
    }
    cells = i653.floor_clearing_seed137_cells(grid)
    ids = {c.cell_id for c in cells}
    assert ids == {
        f"sycophancy__florist__r16__seed{i653.LADDER_STRETCH_SEED}",
        f"marker__florist__r1__seed{i653.LADDER_STRETCH_SEED}",
    }
    # no full-FT, no dropped EM cell, every emitted cell is seed 137:
    assert all(c.seed == i653.LADDER_STRETCH_SEED for c in cells)
    assert all(c.rung in i653.LADDER_LORA_RUNGS for c in cells)
