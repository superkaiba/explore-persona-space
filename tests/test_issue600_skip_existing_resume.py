"""CPU-only tests for the #600 ``EPM_SKIP_EXISTING`` resume contract.

A (cell, seed) whose ``done.json`` AND ``trajectory.json`` both exist must be
skipped AND synthesized into ``results`` as an rc=0-equivalent entry
(``skipped_existing: true``, path fields re-pointed at the CURRENT out_root)
so the smoke phase's ``if failures or not results`` gate and the sweep's
completion count treat it exactly like a fresh successful run (2026-06-11
relaunch incident: the old branch ``continue``-d without a result entry and
the smoke gate misread the skipped-but-complete cell as a crash). A
``trajectory.json`` WITHOUT ``done.json`` is an incomplete prior run and must
be RE-RUN, not skipped. With the env var unset, nothing is skipped.

Runs in <5 s on CPU; no model/tokenizer load, no real subprocess spawn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from explore_persona_space.experiments.targeted_proximity_600 import dispatch as d600
from explore_persona_space.experiments.targeted_proximity_600.cells import CellSpec600

SEED = 42


def _spec(slug: str = "c600_pirate_captain_near") -> CellSpec600:
    return CellSpec600(
        slug=slug,
        plain_name="pirate captain",
        target="pirate_captain",
        stratum="mid",
        condition="near",
        slot_persona="slot_p",
        panel=("qwen_default", "base_a", "base_b", "slot_p"),
    )


def _seed_cell(out_root: Path, spec: CellSpec600, seed: int, *, with_done: bool = True) -> Path:
    """Fabricate a prior run's persisted cell dir (pod-relative paths in done.json)."""
    cell = out_root / "sweep" / spec.slug / f"seed_{seed}"
    cell.mkdir(parents=True)
    (cell / "trajectory.json").write_text(json.dumps({"checkpoints": []}))
    (cell / "band_trajectory.json").write_text(
        json.dumps({"records": [{"step": 5}], "delta_nats": [7.0]})
    )
    if with_done:
        done = {
            "cell_slug": spec.slug,
            "seed": seed,
            "target": spec.target,
            "condition": spec.condition,
            "panel": list(spec.panel),
            "epochs": 1,
            "realized_terminal_step": 63,
            # The producing run persisted CWD-RELATIVE paths (pod cwd) — the
            # skip branch must re-point these at the current out_root.
            "trajectory_path": "eval_results/issue_600/sweep/x/seed_42/trajectory.json",
            "band_trajectory_path": "eval_results/issue_600/sweep/x/seed_42/band_trajectory.json",
            "train_jsonl": "data/issue_600/x_seed42.jsonl",
            "adapter_dir": "eval_results/issue_600/sweep/x/seed_42/adapter",
            "checkpoint_index": {"1.00": {"step": 63, "path": "ckpt"}},
            "final_band_delta_nats": 7.0,
            "timestamp_utc": "2026-06-11T14:18:00+00:00",
        }
        (cell / "done.json").write_text(json.dumps(done))
    return cell


def _run(out_root: Path, spec: CellSpec600) -> tuple[list[dict], list[dict]]:
    return d600._run_cells_subprocess(
        [(spec, SEED)],
        n_gpus=8,
        max_parallel=8,
        epochs=1,
        manifest_path=out_root / "panel_selection.json",
        out_root=out_root,
        data_root=out_root / "data",
    )


class _FakeProc:
    """Stands in for the per-cell subprocess: writes done.json, exits rc=0."""

    launched: ClassVar[list[list[str]]] = []

    def __init__(self, cmd, **_kwargs):
        _FakeProc.launched.append(cmd)
        cell = (
            Path(cmd[cmd.index("--output-root") + 1])
            / "sweep"
            / cmd[cmd.index("--cell") + 1]
            / f"seed_{cmd[cmd.index('--seed') + 1]}"
        )
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "done.json").write_text(
            json.dumps({"cell_slug": cmd[cmd.index("--cell") + 1], "seed": SEED})
        )

    def poll(self):
        return 0


def test_skip_existing_synthesizes_completed_result(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_SKIP_EXISTING", "1")
    spec = _spec()
    cell = _seed_cell(tmp_path, spec, SEED)
    results, failures = _run(tmp_path, spec)
    assert failures == []
    assert len(results) == 1
    r = results[0]
    # rc=0-equivalent shape: everything smoke Phase 3 consumes is present.
    assert r["skipped_existing"] is True
    assert r["cell_slug"] == spec.slug
    assert r["seed"] == SEED
    assert r["panel"] == list(spec.panel)
    assert r["checkpoint_index"] == {"1.00": {"step": 63, "path": "ckpt"}}
    # Path fields re-pointed at the CURRENT out_root (not the producing cwd).
    assert Path(r["trajectory_path"]) == cell / "trajectory.json"
    assert Path(r["band_trajectory_path"]) == cell / "band_trajectory.json"
    assert Path(r["trajectory_path"]).exists()
    assert Path(r["band_trajectory_path"]).exists()


def test_trajectory_without_done_is_rerun(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_SKIP_EXISTING", "1")
    monkeypatch.setattr(d600.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(_FakeProc, "launched", [])
    spec = _spec()
    _seed_cell(tmp_path, spec, SEED, with_done=False)
    results, failures = _run(tmp_path, spec)
    assert _FakeProc.launched, "incomplete cell (trajectory without done.json) must be re-run"
    assert failures == []
    assert len(results) == 1
    assert "skipped_existing" not in results[0]


def test_skip_existing_off_reruns_complete_cell(tmp_path, monkeypatch):
    monkeypatch.delenv("EPM_SKIP_EXISTING", raising=False)
    monkeypatch.setattr(d600.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(_FakeProc, "launched", [])
    spec = _spec()
    _seed_cell(tmp_path, spec, SEED)  # complete on disk, but skip-existing OFF
    results, failures = _run(tmp_path, spec)
    assert _FakeProc.launched, "without EPM_SKIP_EXISTING the cell must be re-run"
    assert failures == []
    assert len(results) == 1
    assert "skipped_existing" not in results[0]
