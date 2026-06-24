# ruff: noqa: RUF002
"""Task #653 v8 §6.5Δ — primary-deliverable verifier for install-probe pools.

CPU-only. BLOCKER (Codex): the §6.5Δ primary_deliverable lists the install-probe
completion globs, but nothing enforced them — a run could ship with a missing
(cell × persona) pool (the parent WARN recurs). The verifier enumerates the
expected pools for every probed cell and FAILs `primary-deliverable-missing`
BEFORE upload + [phase=done]. These tests pin:

  * a missing (cell × persona) install-probe pool raises `primary-deliverable-missing`;
  * EM expects BOTH the source persona AND the no-system gate-surface pool;
  * all-present passes and returns the verified relpaths;
  * a DROPPED cell (no install read) is EXEMPT (its drop record is the deliverable);
  * a cell whose install phase never ran (no install JSON) is not gated (partial smoke).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from explore_persona_space.experiments import issue_653 as i653


def _load_dispatcher():
    repo_root = Path(__file__).resolve().parents[1]
    disp_path = repo_root / "scripts" / "issue_653" / "i653_dispatch.py"
    spec = importlib.util.spec_from_file_location("i653_dispatch_deliverable_test", disp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i653_dispatch_deliverable_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_install_json(out_root: Path, cell, *, dropped: bool = False) -> None:
    armB = out_root / "armB"
    armB.mkdir(parents=True, exist_ok=True)
    install = {"dv_kind": "judge_rate_plus_gain", "behavior": cell.behavior}
    if dropped:
        install["dropped_non_install"] = True
    (armB / f"install_{cell.cell_id}.json").write_text(json.dumps({"install": install}))


def _write_probe_pool(out_root: Path, cell, persona: str) -> None:
    p = out_root / "armB" / "install_probes" / cell.cell_id / persona / "raw_completions.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"cell_id": cell.cell_id, "persona": persona, "firing": []}))


def test_missing_probe_pool_raises_primary_deliverable_missing(tmp_path):
    """A probed cell missing its install-probe pool → primary-deliverable-missing."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    _write_install_json(tmp_path, cell)
    # NO probe pool written.
    with pytest.raises(RuntimeError, match="primary-deliverable-missing"):
        mod.verify_install_probe_deliverables([cell], out_root=tmp_path)


def test_em_requires_both_personas(tmp_path):
    """EM expects BOTH the source persona pool AND the no-system gate-surface pool;
    missing the no-system one fails."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(behavior="em", source="florist", rung="r16", seed=i653.HEADLINE_SEED)
    _write_install_json(tmp_path, cell)
    _write_probe_pool(tmp_path, cell, "florist")  # persona pool present
    # no-system gate pool MISSING.
    with pytest.raises(RuntimeError, match="primary-deliverable-missing"):
        mod.verify_install_probe_deliverables([cell], out_root=tmp_path)
    # Now add the no-system pool → passes.
    _write_probe_pool(tmp_path, cell, i653.EM_NO_SYSTEM_PROBE_PERSONA)
    verified = mod.verify_install_probe_deliverables([cell], out_root=tmp_path)
    assert len(verified) == 2  # both personas


def test_all_present_passes(tmp_path):
    """All expected (cell × persona) pools present → PASS, returns the relpaths."""
    mod = _load_dispatcher()
    syco = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    marker = i653.ArmBCell(behavior="marker", source="florist", rung="r16", seed=i653.HEADLINE_SEED)
    for c in (syco, marker):
        _write_install_json(tmp_path, c)
        _write_probe_pool(tmp_path, c, c.source)
    verified = mod.verify_install_probe_deliverables([syco, marker], out_root=tmp_path)
    assert len(verified) == 2  # one persona each (non-EM)


def test_dropped_cell_exempt(tmp_path):
    """A DROPPED cell (no install read happened) is EXEMPT from the deliverable —
    its drop record is the artifact, not a probe pool."""
    mod = _load_dispatcher()
    dropped_cell = i653.ArmBCell(
        behavior="em", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    _write_install_json(tmp_path, dropped_cell, dropped=True)
    # No probe pool for the dropped cell — must NOT raise.
    verified = mod.verify_install_probe_deliverables([dropped_cell], out_root=tmp_path)
    assert verified == []  # dropped cell contributes no required deliverable


def test_cell_without_install_json_not_gated(tmp_path):
    """A cell whose install phase never ran (no install JSON) is not gated — a
    partial smoke (--phase build) wrote no install pools and must not fail here."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    # No install JSON, no probe pool.
    verified = mod.verify_install_probe_deliverables([cell], out_root=tmp_path)
    assert verified == []
