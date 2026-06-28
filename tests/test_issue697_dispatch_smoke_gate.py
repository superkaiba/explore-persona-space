"""Tests for issue #697 BLOCKER B1 — the §7.1b PRE-SWEEP smoke gate.

Plan §7.1 (line 265): "the dispatcher refuses to enter ``phase_sweep`` until the
smoke cell's ``.pt`` passes 7.1b". These pin that the LOAD-BEARING gate runs
BEFORE the wave loop on a PRODUCTION (multi-cell) sweep:

  - ``phase_sweep`` with ``len(cells) > 1`` and NO smoke-pass artifact raises
    RuntimeError (the production sweep refuses to dispatch);
  - with a valid smoke-pass artifact (matching git SHA, ``non_inert: true``,
    matching read layer) it proceeds past the gate;
  - a STALE git SHA / a wrong read layer / ``non_inert: false`` each raises.

No GPU, no network: the wave loop itself is stubbed (``_run_parallel_with_log``
monkeypatched) so the test exercises ONLY the pre-loop gate. The smoke-pass
artifact is written to ``tmp_path`` and ``_git_sha`` is monkeypatched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue697_dispatch as D  # noqa: E402

_FAKE_SHA = "abc123def4567890abc123def4567890abc123de"


def _cells(n: int):
    """``n`` distinct production-shaped cells (real behaviors, seed 42)."""
    from explore_persona_space.experiments.issue_651 import Cell

    grid = [
        ("marker", "sp_swe"),
        ("marker", "sp_doctor"),
        ("fact", "sp_swe"),
        ("em", "default"),
    ]
    return [Cell(behavior=b, cid=c, seed=42, gpu_id=i) for i, (b, c) in enumerate(grid[:n])]


def _write_smoke_pass(repo_root: Path, *, sha: str, read_layer: int, non_inert: bool) -> None:
    p = repo_root / "eval_results" / "issue_697" / D.SMOKE_PASS_BASENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "issue": 697,
                "cell_id": "marker_sp_swe_seed42",
                "git_sha": sha,
                "read_layer": read_layer,
                "f_cv_pup_mean": 0.42,
                "f_cv_random_mean": 0.01,
                "f_cv_full_span_mean": 0.55,
                "f_cv_down_pdn_mean": 0.61,
                "n": 280,
                "non_inert": non_inert,
                "ts": "2026-06-28T00:00:00Z",
            }
        )
    )


def _stub_sweep_internals(monkeypatch, repo_root: Path, *, captured: list):
    """Stub everything past the pre-loop gate so a 'proceed' returns cleanly.

    - ``_git_sha`` -> a fixed SHA (no real git).
    - ``_assert_sweep_device_count`` -> no-op (no GPU).
    - ``_read_use_cache_decision`` -> False (no HF).
    - ``_materialize_panel`` is not called by phase_sweep (caller passes the JSONs).
    - ``_run_parallel_with_log`` -> records the wave + returns rc=0 for each cell.
    """
    monkeypatch.setattr(D, "_git_sha", lambda _rr: _FAKE_SHA)
    monkeypatch.setattr(D, "_assert_sweep_device_count", lambda *a, **k: None)
    monkeypatch.setattr(D, "_read_use_cache_decision", lambda *a, **k: False)

    def _fake_parallel(cmds, *, cwd=None):
        cmds = list(cmds)
        captured.append(cmds)
        return [0] * len(cmds)

    monkeypatch.setattr(D, "_run_parallel_with_log", _fake_parallel)
    # phase_sweep's single-cell post-check / artifact write only fires on len==1;
    # multi-cell paths never reach it, so no stub needed there for these tests.


def _call_sweep(repo_root: Path, cells):
    panel = repo_root / "panel.json"
    panel.write_text("{}")
    D.phase_sweep(
        repo_root,
        cells,
        n_gpus=2,
        cpu_only=False,
        panel_personas_json=panel,
        panel_questions_json=panel,
        layers=[7, 14, 21],
        primary_layer=14,
        patch_layer=10,
        max_new_tokens=64,
        skip_e=True,
        smoke_model=None,
        dry_run=False,
        upload=False,
        rbase_cache_dir=None,
    )


def test_production_sweep_refuses_without_smoke_pass(tmp_path, monkeypatch):
    """len(cells)>1 + NO smoke-pass artifact (local OR HF) -> RuntimeError BEFORE
    the wave loop runs (no cell command is ever built)."""
    captured: list = []
    _stub_sweep_internals(monkeypatch, tmp_path, captured=captured)
    # No artifact locally; force the HF lookup to miss too.
    monkeypatch.setattr(D, "_load_smoke_pass", lambda _rr: None)
    with pytest.raises(RuntimeError, match="PRE-SWEEP GATE FAIL"):
        _call_sweep(tmp_path, _cells(2))
    assert captured == [], "the wave loop must NOT run when the smoke gate fails"


def test_production_sweep_proceeds_with_valid_smoke_pass(tmp_path, monkeypatch):
    """len(cells)>1 + a valid smoke-pass (matching SHA + read layer + non_inert) ->
    the gate passes and the wave loop runs."""
    captured: list = []
    _stub_sweep_internals(monkeypatch, tmp_path, captured=captured)
    _write_smoke_pass(tmp_path, sha=_FAKE_SHA, read_layer=14, non_inert=True)
    _call_sweep(tmp_path, _cells(2))
    assert captured, "the wave loop must run once the smoke gate passes"
    # all cells were dispatched across the waves
    n_dispatched = sum(len(w) for w in captured)
    assert n_dispatched == 2


def test_production_sweep_refuses_on_stale_sha(tmp_path, monkeypatch):
    """A smoke-pass from a DIFFERENT git SHA (code changed since the smoke) -> raise."""
    captured: list = []
    _stub_sweep_internals(monkeypatch, tmp_path, captured=captured)
    _write_smoke_pass(tmp_path, sha="0" * 40, read_layer=14, non_inert=True)
    with pytest.raises(RuntimeError, match="git_sha"):
        _call_sweep(tmp_path, _cells(2))
    assert captured == []


def test_production_sweep_refuses_on_wrong_read_layer(tmp_path, monkeypatch):
    """A smoke-pass recorded at a different read layer than the sweep -> raise."""
    captured: list = []
    _stub_sweep_internals(monkeypatch, tmp_path, captured=captured)
    _write_smoke_pass(tmp_path, sha=_FAKE_SHA, read_layer=21, non_inert=True)
    with pytest.raises(RuntimeError, match="read_layer"):
        _call_sweep(tmp_path, _cells(2))
    assert captured == []


def test_production_sweep_refuses_when_non_inert_false(tmp_path, monkeypatch):
    """A smoke-pass artifact with non_inert=false -> raise (the smoke FAILED)."""
    captured: list = []
    _stub_sweep_internals(monkeypatch, tmp_path, captured=captured)
    _write_smoke_pass(tmp_path, sha=_FAKE_SHA, read_layer=14, non_inert=False)
    with pytest.raises(RuntimeError, match="non_inert"):
        _call_sweep(tmp_path, _cells(2))
    assert captured == []


def test_single_cell_smoke_path_skips_pre_sweep_gate(tmp_path, monkeypatch):
    """The single-cell (len==1) real-GPU smoke path does NOT require the artifact —
    it IS the cell that produces it. Stub the post-check + artifact write so the
    one-cell sweep runs without a pre-existing smoke-pass artifact."""
    captured: list = []
    _stub_sweep_internals(monkeypatch, tmp_path, captured=captured)
    monkeypatch.setattr(D, "_load_smoke_pass", lambda _rr: None)  # no artifact present
    # stub the §7.1b post-check + the artifact write (no real .pt on disk).
    monkeypatch.setattr(
        D,
        "assert_smoke_cell_not_inert",
        lambda *a, **k: {
            "f_cv": 0.4,
            "f_cv_random": 0.0,
            "f_cv_full_span": 0.5,
            "f_cv_down": 0.6,
            "n": 280,
        },
    )
    monkeypatch.setattr(D, "write_smoke_pass_artifact", lambda *a, **k: tmp_path / "x")
    _call_sweep(tmp_path, _cells(1))  # must NOT raise
    assert captured, "the single-cell smoke must run (it produces the smoke-pass artifact)"
