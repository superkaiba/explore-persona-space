# ruff: noqa: RUF002, RUF003
"""Task #653 §6.5.B6 ablation fail-loud (round-4 BLOCKER analysis-missing-ablation-not-fail-loud).

CPU-only. ``phase_analyze(require_complete=True)`` MUST raise when an
ABLATION_RUNG (r16) cell is missing its ``ablation_<cell>.json`` illusion-guard
deliverable, or carries a present-but-null one — the same fail-loud contract its
dx-tensor / cross-arm siblings already enforce. A non-r16 cell legitimately has
no ablation (B6 runs only at r16) and must NOT raise. This is the §6.5.B6 sub-spec.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments import issue_653 as i653

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dispatcher(tag: str):
    """Import the dispatcher module fresh (it loads heavy deps only via deferred
    imports inside functions, so module exec is cheap)."""
    disp_path = _REPO_ROOT / "scripts" / "issue_653" / "i653_dispatch.py"
    spec = importlib.util.spec_from_file_location(f"i653_dispatch_{tag}", disp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"i653_dispatch_{tag}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _stage_common_inputs(out_root: Path, cell: i653.ArmBCell, *, d: int = 8, n_rows: int = 20):
    """Stage the dx_geometry JSON + Δx tensor + an Arm A ρ direction so
    phase_analyze reaches the §6.5.B6 ablation branch without tripping an EARLIER
    require_complete raise (no-rho / no-dx / no-tensor / empty-cross-arm)."""
    armA = out_root / "armA"
    armB = out_root / "armB"
    tensors = armB / "dx_tensors"
    for p in (armA, armB, tensors):
        p.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    top_dir = rng.standard_normal(d)
    top_dir = top_dir / np.linalg.norm(top_dir)
    # Arm A ρ direction (same d_model as dx_top so cross_arm is non-empty).
    (armA / f"rho_geometry_seed{i653.HEADLINE_SEED}.json").write_text(
        json.dumps({"geometry": {"iso": {"rho_top_direction": top_dir.tolist()}}})
    )
    # dx_geometry: a low-rank H1-ish cell (concrete spectral DVs + a top direction).
    (armB / f"dx_geometry_{cell.cell_id}.json").write_text(
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "cell_group": cell.cell_group,
                "rung": cell.rung,
                "top_share_lambda": 0.85,
                "pr_lambda": 1.4,
                "rank_k_at_90": 2,
                "n_rows": n_rows,
                "dx_top_direction": top_dir.tolist(),
                "cos_top_to_rb": 0.8,
                "random_ci_high": 0.1,
            }
        )
    )
    # Δx cloud tensor (rows ≥ MIN_SPECTRUM_ROWS so the §3.4 bootstrap runs).
    cloud = rng.standard_normal((n_rows, d))
    np.savez(tensors / f"{cell.cell_id}.npz", cloud=cloud)
    # v8 §6Δ.1: a PASSING install JSON so the install-floor gate (which now runs
    # before the dx/ablation read) lets the cell through to the ablation branch.
    # EM cells use a judge-rate gain above the +0.20 floor.
    (armB / f"install_{cell.cell_id}.json").write_text(
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "install": {
                    "dv_kind": "judge_rate_plus_gain",
                    "behavior": cell.behavior,
                    "judge_rate_gain": 0.35,  # ≥ +0.20 EM floor → installed
                },
            }
        )
    )


def _r16_cell() -> i653.ArmBCell:
    return i653.ArmBCell(
        behavior="em", source="florist", rung=i653.ABLATION_RUNG, seed=i653.HEADLINE_SEED
    )


def test_missing_ablation_r16_raises_under_require_complete(tmp_path):
    """r16 cell with dx + tensor + cross-arm input present but NO ablation file →
    phase_analyze(require_complete=True) RAISES naming the cell + B6/ablation.
    FAILS on round-4 code (which sets ablation=None and proceeds)."""
    mod = _load_dispatcher("missing")
    out_root = tmp_path / "eval_results" / "issue_653"
    cell = _r16_cell()
    _stage_common_inputs(out_root, cell)
    # deliberately do NOT write ablation_<cell>.json
    with pytest.raises(RuntimeError) as exc:
        mod.phase_analyze([cell], out_root=out_root, require_complete=True)
    msg = str(exc.value)
    assert cell.cell_id in msg
    assert "B6" in msg or "ablation" in msg


def test_null_ablation_r16_raises_under_require_complete(tmp_path):
    """A present ablation file with BOTH causal deltas null also raises (a no-op
    file is the same silent-drop hazard as a missing one)."""
    mod = _load_dispatcher("null")
    out_root = tmp_path / "eval_results" / "issue_653"
    cell = _r16_cell()
    _stage_common_inputs(out_root, cell)
    (out_root / "armB" / f"ablation_{cell.cell_id}.json").write_text(
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "ablation": {
                    "dv_kind": "judge_rate_plus_gain",
                    "logp_delta_ablation": None,
                    "judge_rate_delta_ablation": None,
                },
            }
        )
    )
    with pytest.raises(RuntimeError, match=r"both causal deltas|BOTH causal deltas"):
        mod.phase_analyze([cell], out_root=out_root, require_complete=True)


def test_missing_ablation_non_r16_does_not_raise(tmp_path):
    """A rank-1 cell with no ablation file → phase_analyze(require_complete=True)
    returns normally with that cell's ablation None (B6 doesn't run at rank-1)."""
    mod = _load_dispatcher("nonr16")
    out_root = tmp_path / "eval_results" / "issue_653"
    cell = i653.ArmBCell(behavior="em", source="florist", rung="r1", seed=i653.HEADLINE_SEED)
    _stage_common_inputs(out_root, cell)
    result = mod.phase_analyze([cell], out_root=out_root, require_complete=True)
    assert result["n_cells"] == 1
    grid = json.loads((out_root / "cross_arm_verdict.json").read_text())
    (vd,) = grid["verdicts"]
    assert vd["cell_id"] == cell.cell_id
    assert vd["ablation"] is None  # legitimate N/A at rank-1, never an error


def test_present_ablation_r16_passes(tmp_path):
    """A valid r16 ablation file with a real judge_rate_delta_ablation → no raise;
    cross_arm_verdict.json for that cell carries a non-null ablation block (§6.5.B6 r3)."""
    mod = _load_dispatcher("present")
    out_root = tmp_path / "eval_results" / "issue_653"
    cell = _r16_cell()
    _stage_common_inputs(out_root, cell)
    (out_root / "armB" / f"ablation_{cell.cell_id}.json").write_text(
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "ablation": {
                    "dv_kind": "judge_rate_plus_gain",
                    "judge_rate_unablated": 0.6,
                    "judge_rate_ablated": 0.2,
                    "judge_rate_delta_ablation": -0.4,
                },
            }
        )
    )
    result = mod.phase_analyze([cell], out_root=out_root, require_complete=True)
    assert result["n_cells"] == 1
    grid = json.loads((out_root / "cross_arm_verdict.json").read_text())
    (vd,) = grid["verdicts"]
    assert vd["ablation"] is not None
    assert vd["ablation"]["judge_rate_delta_ablation"] == pytest.approx(-0.4)


def test_missing_ablation_r16_non_require_complete_records_note_not_raise(tmp_path):
    """The non-require_complete path (smoke / partial) must NOT raise on a missing
    r16 ablation — it leaves ablation None but RECORDS the gap loudly in notes (§6.5.B6 r4)."""
    mod = _load_dispatcher("noterec")
    out_root = tmp_path / "eval_results" / "issue_653"
    cell = _r16_cell()
    _stage_common_inputs(out_root, cell)
    result = mod.phase_analyze([cell], out_root=out_root, require_complete=False)
    assert result["n_cells"] == 1
    grid = json.loads((out_root / "cross_arm_verdict.json").read_text())
    (vd,) = grid["verdicts"]
    assert vd["ablation"] is None
    assert any("ablation MISSING" in n for n in vd["notes"])
