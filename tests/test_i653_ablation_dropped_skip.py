"""Task #653 round 6 — `phase_ablation` MUST skip cells dropped by
select_checkpoint (no floor-clearing checkpoint, §6Δ.3), mirroring the dx
phase's gate at line ~1831. Plus a resume-skip: an already-written
``ablation_<cell>.json`` MUST NOT be re-computed (re-merging the ~15 GB read
adapter is the EDQUOT / per-pod-quota hazard the round-4 select_checkpoint
resume-skip already closed).

Both checks are CPU-only: the §6Δ.3 drop gate fires BEFORE any GPU read, and
the resume-skip fires BEFORE the per-cell try/finally; neither calls
``_ablation_gpu_read``. The bug this guards against — task #653 round 6
relaunch (2026-06-26) — crashed ``_ablation_gpu_read`` with
``FileNotFoundError: ablation: dx_geometry missing for em__florist__r16``
because dx legitimately skipped every em cell (all 6 em cells'
selected_checkpoints/<cell>.json carries ``dropped_non_install: true``) and
ablation iterated the same cell list with no analogous gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from explore_persona_space.experiments import issue_653 as i653

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dispatcher(tag: str):
    """Import the dispatcher module fresh per test."""
    disp_path = _REPO_ROOT / "scripts" / "issue_653" / "i653_dispatch.py"
    spec = importlib.util.spec_from_file_location(f"i653_dispatch_{tag}", disp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"i653_dispatch_{tag}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _r16_cell(behavior: str = "em", source: str = "florist") -> i653.ArmBCell:
    return i653.ArmBCell(
        behavior=behavior, source=source, rung=i653.ABLATION_RUNG, seed=i653.HEADLINE_SEED
    )


def _write_select_manifest(out_root: Path, cell: i653.ArmBCell, *, dropped: bool) -> None:
    """Stage the per-cell select_checkpoint manifest the ablation gate reads."""
    sel_dir = out_root / "armB" / "selected_checkpoints"
    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / f"{cell.cell_id}.json").write_text(
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "rung": cell.rung,
                "behavior": cell.behavior,
                "dropped_non_install": dropped,
            }
        )
    )


def test_dropped_non_install_skips_ablation(tmp_path):
    """A cell whose select_checkpoint manifest carries dropped_non_install=True
    MUST be skipped by phase_ablation (silently, mirroring dx) — NO ablation
    JSON written, no FileNotFoundError on a missing dx_geometry.

    This is the round-6 crash: every em cell carried dropped_non_install=True
    (no floor-clearing EM checkpoint), so dx never wrote dx_geometry_em__*,
    and the (pre-fix) ablation crashed on the first em r16 cell."""
    mod = _load_dispatcher("dropped")
    out_root = tmp_path / "eval_results" / "issue_653"
    cell = _r16_cell(behavior="em", source="florist")
    _write_select_manifest(out_root, cell, dropped=True)
    result = mod.phase_ablation([cell], out_root=out_root, mode=i653.RUN_MODE_CPU_STUB)
    assert result["n_cells"] == 1
    assert result["n_dropped_non_install"] == 1
    assert result["ablation_files"] == []
    assert not (out_root / "armB" / f"ablation_{cell.cell_id}.json").exists()


def test_no_select_manifest_runs_ablation(tmp_path):
    """A cell with NO select_checkpoint manifest (e.g. early smoke runs that
    skip select) MUST still run the ablation under CPU stub — the gate is
    SKIP-ON-DROP, not SKIP-IF-MANIFEST-MISSING (which would be too strict)."""
    mod = _load_dispatcher("nomanifest")
    out_root = tmp_path / "eval_results" / "issue_653"
    cell = _r16_cell()
    # deliberately do NOT write a select_checkpoint manifest
    result = mod.phase_ablation([cell], out_root=out_root, mode=i653.RUN_MODE_CPU_STUB)
    assert result["n_dropped_non_install"] == 0
    assert result["n_resumed"] == 0
    assert (out_root / "armB" / f"ablation_{cell.cell_id}.json").exists()


def test_resume_skip_on_existing_ablation_file(tmp_path):
    """A cell whose ablation_<cell>.json ALREADY exists MUST be skipped — a
    resumed launcher should not re-merge ~15 GB of read adapter + re-run the
    GPU install probe (~10 min/cell) for cells already on disk. Mirrors the
    select_checkpoint resume-skip from round 4."""
    mod = _load_dispatcher("resume")
    out_root = tmp_path / "eval_results" / "issue_653"
    cell = _r16_cell(behavior="marker", source="florist")
    armB = out_root / "armB"
    armB.mkdir(parents=True, exist_ok=True)
    existing = armB / f"ablation_{cell.cell_id}.json"
    existing.write_text(json.dumps({"cell_id": cell.cell_id, "ablation": {"placeholder": True}}))
    result = mod.phase_ablation([cell], out_root=out_root, mode=i653.RUN_MODE_CPU_STUB)
    assert result["n_resumed"] == 1
    assert str(existing) in result["ablation_files"]
    # The existing file is preserved verbatim — no re-write.
    assert json.loads(existing.read_text())["ablation"] == {"placeholder": True}


def test_dropped_flag_false_runs_normally(tmp_path):
    """A select_checkpoint manifest with dropped_non_install=False (the
    floor-clearing case) MUST NOT trip the gate."""
    mod = _load_dispatcher("notdropped")
    out_root = tmp_path / "eval_results" / "issue_653"
    cell = _r16_cell(behavior="marker", source="medical_doctor")
    _write_select_manifest(out_root, cell, dropped=False)
    result = mod.phase_ablation([cell], out_root=out_root, mode=i653.RUN_MODE_CPU_STUB)
    assert result["n_dropped_non_install"] == 0
    assert (out_root / "armB" / f"ablation_{cell.cell_id}.json").exists()


def test_non_r16_cell_is_not_in_abl_cells(tmp_path):
    """Sanity: a rank-1 (non-ABLATION_RUNG) cell is filtered out by the
    existing `c.rung == ABLATION_RUNG` gate at the top of phase_ablation —
    it never reaches either of the new skip branches."""
    mod = _load_dispatcher("rank1")
    out_root = tmp_path / "eval_results" / "issue_653"
    cell = i653.ArmBCell(behavior="em", source="florist", rung="r1", seed=i653.HEADLINE_SEED)
    # Even with a dropped manifest, this cell doesn't reach the gate.
    _write_select_manifest(out_root, cell, dropped=True)
    result = mod.phase_ablation([cell], out_root=out_root, mode=i653.RUN_MODE_CPU_STUB)
    assert result["n_cells"] == 0  # filtered by ABLATION_RUNG
    assert result["n_dropped_non_install"] == 0
