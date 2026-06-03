# ruff: noqa: RUF003  # × / em-dash intentional
"""Task #479 round-4 Fix-3: dispatcher EXECUTION tests (not source-text greps).

Round-3 pinned the Phase-1.6 wiring with source-text reads of
``dispatch_neg_geometry_472.py``. Round 4 strengthens to execution: mock
``_run_phase_subprocess`` + ``_schedule_cell_pool`` + the persona-bank
loader + the marker assertion, then call ``main([...args...])`` for the
full route + the ``--smoke`` route and assert on the recorded sequence of
subprocess invocations + the written sentinel filenames + the analyze
command tail.

Pins:
  - Phase 1.6 (i479_phase_base_emission.py) runs BEFORE the analyze step.
  - Sentinel filenames carry ``issue-479-`` (not ``issue-472-``).
  - Analyze RUNS under ``--smoke`` for #479 (Fix 2); does not skip.
  - Dispatcher does NOT pass ``--no-strict-base-panel``.
  - ``args.issue == 479`` gate guards Phase 1.6.
  - The final sentinel (epm:results) lands at ``issue-479-results.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DISPATCHER_SCRIPT = SCRIPTS_DIR / "dispatch_neg_geometry_472.py"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def dispatcher_module():
    """Import the dispatcher as a module so we can monkeypatch its helpers."""
    spec = importlib.util.spec_from_file_location("dispatch_neg_geometry_472", DISPATCHER_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_persona_bank_phase(skip, dry_run, bank_path):
    """Replace _persona_bank_phase: no Sonnet call, no disk write."""
    return {"status": "stubbed_test"}


class _PhaseRecorder:
    """Records every subprocess invocation the dispatcher attempts.

    Each phase is captured as ``{"phase": str, "cmd": list[str]}``. The
    dispatcher writes its sentinels via ``_write_sentinel`` (direct file
    writes) so we also track those by patching that helper.
    """

    def __init__(self):
        self.phases: list[dict] = []
        self.sentinels: list[dict] = []
        self.scheduled_cells: list[dict] = []
        self.final_sentinels: list[dict] = []

    def run_phase(self, cmd, phase):
        # Record + no-op (don't actually subprocess).
        self.phases.append({"phase": phase, "cmd": list(cmd)})

    def write_sentinel(self, path, *, kind, phase, note_payload, task_id=472):
        self.sentinels.append({"path": str(path), "kind": kind, "phase": phase, "task_id": task_id})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sentinel_schema_version": 1,
                    "kind": kind,
                    "version": 1,
                    "task_id": task_id,
                    "phase": phase,
                    "note": json.dumps(note_payload),
                }
            )
        )

    def schedule_cell_pool(self, *, cells, seeds, **kwargs):
        self.scheduled_cells.append(
            {"cells": list(cells), "seeds": list(seeds), "issue": kwargs.get("issue")}
        )
        return [
            {
                "cell": c,
                "seed": s,
                "status": "done",
                "assigned_gpu": 0,
                "trajectory_path": f"<stub>/{c}_seed{s}/trajectory.json",
                "adapter_hf_path": f"adapters/issue_{kwargs.get('issue', 472)}/{c}_seed{s}",
            }
            for c in cells
            for s in seeds
        ]

    def write_final_sentinel(
        self,
        cells,
        cell_results,
        phase_summaries,
        analyze_summary,
        seeds,
        slab_root,
        *,
        status,
        issue=472,
    ):
        final_path = Path(f"/tmp/__test_final_issue_{issue}_results.json")
        self.final_sentinels.append(
            {
                "path": str(final_path),
                "issue": issue,
                "status": status,
                "cells": list(cells),
                "seeds": list(seeds),
            }
        )
        return final_path


def _run_dispatcher(
    dispatcher_module,
    tmp_path: Path,
    extra_args: list[str],
    *,
    cells_to_resolve: list[str] | None = None,
) -> _PhaseRecorder:
    """Drive dispatcher.main() with all subprocess + GPU helpers stubbed.

    Returns the PhaseRecorder so the caller can assert on the recorded
    sequence of phases + sentinels + cells + final.
    """
    rec = _PhaseRecorder()

    class _StubTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [83399]

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            return _StubTokenizer()

    log_dir = tmp_path / "logs"
    slab_root = tmp_path / "slab"
    runs_root = tmp_path / "runs"
    figures_dir = tmp_path / "figures"
    bank_path = tmp_path / "data" / "persona_bank.json"
    centroids_dir = tmp_path / "data"

    args = [
        "--n-gpus",
        "1",
        "--max-parallel",
        "1",
        "--slab-root",
        str(slab_root),
        "--runs-root",
        str(runs_root),
        "--log-dir",
        str(log_dir),
        "--bank-path",
        str(bank_path),
        "--centroids-dir",
        str(centroids_dir),
        "--figures-dir",
        str(figures_dir),
        "--report-to",
        "none",
        *extra_args,
    ]

    with (
        patch.object(dispatcher_module, "_run_phase_subprocess", rec.run_phase),
        patch.object(dispatcher_module, "_write_sentinel", rec.write_sentinel),
        patch.object(dispatcher_module, "_schedule_cell_pool", rec.schedule_cell_pool),
        patch.object(dispatcher_module, "_write_final_sentinel", rec.write_final_sentinel),
        patch.object(dispatcher_module, "_persona_bank_phase", _stub_persona_bank_phase),
        # _subceiling_gate reads a trajectory.json on disk (which the stubbed
        # cell pool never wrote). Pretend the gate passed so the #472 smoke
        # path completes to its analyze decision rather than exiting rc=2.
        patch.object(
            dispatcher_module,
            "_subceiling_gate",
            lambda slab_root, smoke_cell, smoke_seed: {"ok": True, "stub": True},
        ),
        patch("transformers.AutoTokenizer", _StubAutoTokenizer),
    ):
        if cells_to_resolve is not None:
            with patch.object(
                dispatcher_module, "_resolve_cells", lambda *a, **k: list(cells_to_resolve)
            ):
                rc = dispatcher_module.main(args)
        else:
            rc = dispatcher_module.main(args)

    assert rc == 0, f"dispatcher.main exited non-zero: {rc}"
    return rec


# ── Fix 1 (sentinel filenames carry issue-479-). ──────────────────────────────


def test_issue_479_full_route_writes_issue_479_sentinels(dispatcher_module, tmp_path):
    """For --issue 479, every sentinel filename carries `issue-479-`."""
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "479", "--stage", "1", "--seeds", "42"],
    )
    bad = [s for s in rec.sentinels if "issue-479-" not in s["path"]]
    assert not bad, f"sentinels missing `issue-479-` prefix: {bad}"
    bad_task = [s for s in rec.sentinels if s["task_id"] != 479]
    assert not bad_task, f"sentinels with wrong task_id: {bad_task}"


def test_issue_479_full_route_final_sentinel_carries_issue_479(dispatcher_module, tmp_path):
    """The terminal _write_final_sentinel call must receive issue=479."""
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "479", "--stage", "1", "--seeds", "42"],
    )
    assert rec.final_sentinels, "no _write_final_sentinel call recorded"
    final = rec.final_sentinels[-1]
    assert final["issue"] == 479, f"final sentinel issue={final['issue']}, expected 479"


def test_issue_472_route_writes_issue_472_sentinels(dispatcher_module, tmp_path):
    """For --issue 472, sentinel filenames carry `issue-472-` (legacy unchanged)."""
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "472", "--cells", "c472_anchor", "--seeds", "42"],
    )
    bad = [s for s in rec.sentinels if "issue-472-" not in s["path"]]
    assert not bad, f"sentinels missing `issue-472-` prefix: {bad}"
    final = rec.final_sentinels[-1]
    assert final["issue"] == 472


# ── Fix 1 (Phase 1.6 ordering): base-emission runs BEFORE analyze. ──────────


def test_phase_1_6_runs_before_analyze_for_issue_479(dispatcher_module, tmp_path):
    """Phase 1.6 (i479_phase_base_emission.py) must precede the analyze phase."""
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "479", "--stage", "1", "--seeds", "42"],
    )
    base_idx = next(
        (i for i, p in enumerate(rec.phases) if p["phase"] == "base_emission_rate"), None
    )
    analyze_idx = next((i for i, p in enumerate(rec.phases) if p["phase"] == "analyze"), None)
    assert base_idx is not None, "Phase 1.6 (base_emission_rate) was never run"
    assert analyze_idx is not None, "analyze phase was never run"
    assert base_idx < analyze_idx, (
        f"base-emission ({base_idx}) must run BEFORE analyze ({analyze_idx}); "
        f"order seen: {[p['phase'] for p in rec.phases]}"
    )


def test_phase_1_6_does_not_run_for_issue_472(dispatcher_module, tmp_path):
    """Phase 1.6 is gated by args.issue == 479; #472 never sees it."""
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "472", "--cells", "c472_anchor", "--seeds", "42"],
    )
    base_phases = [p for p in rec.phases if p["phase"] == "base_emission_rate"]
    assert not base_phases, f"Phase 1.6 must NOT run for --issue 472, but saw: {base_phases}"


def test_phase_1_6_writes_base_panel_emission_rate_json(dispatcher_module, tmp_path):
    """The base-emission phase command must write to base_panel_emission_rate.json."""
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "479", "--stage", "1", "--seeds", "42"],
    )
    base_phase = next(p for p in rec.phases if p["phase"] == "base_emission_rate")
    cmd_str = " ".join(base_phase["cmd"])
    assert "i479_phase_base_emission.py" in cmd_str
    assert "base_panel_emission_rate.json" in cmd_str


# ── Fix 2 (smoke runs analyze for #479). ────────────────────────────────────


def test_smoke_runs_analyze_for_issue_479(dispatcher_module, tmp_path):
    """Under --smoke, the analyze phase MUST run for #479 (Fix 2)."""
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "479", "--stage", "1", "--smoke", "--seeds", "42"],
        cells_to_resolve=["c479_base"],
    )
    analyze_phases = [p for p in rec.phases if p["phase"] == "analyze"]
    assert analyze_phases, (
        "Under --issue 479 --smoke the analyze phase must STILL run "
        "(round-4 Fix 2). Phases seen: " + str([p["phase"] for p in rec.phases])
    )


def test_smoke_skips_analyze_for_issue_472(dispatcher_module, tmp_path):
    """Under --smoke, #472 still skips analyze (only one cell, no regression study)."""
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "472", "--smoke", "--seeds", "42"],
        cells_to_resolve=["c472_anchor"],
    )
    analyze_phases = [p for p in rec.phases if p["phase"] == "analyze"]
    assert not analyze_phases, (
        f"Under --issue 472 --smoke analyze should skip (legacy behavior); saw {analyze_phases}"
    )


def test_smoke_for_issue_479_still_runs_phase_1_6(dispatcher_module, tmp_path):
    """Phase 1.6 must run under --smoke too (on-pod gate exercises every phase)."""
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "479", "--stage", "1", "--smoke", "--seeds", "42"],
        cells_to_resolve=["c479_base"],
    )
    base_emission = [p for p in rec.phases if p["phase"] == "base_emission_rate"]
    assert base_emission, (
        "Under --issue 479 --smoke Phase 1.6 (base-emission) must STILL run "
        "so the on-pod smoke exercises every phase end-to-end."
    )


def test_smoke_for_issue_479_phase_order_is_base_emission_then_analyze(dispatcher_module, tmp_path):
    """For --issue 479 --smoke: phase order = base-emission → cells → analyze."""
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "479", "--stage", "1", "--smoke", "--seeds", "42"],
        cells_to_resolve=["c479_base"],
    )
    phase_seq = [p["phase"] for p in rec.phases]
    assert "base_emission_rate" in phase_seq
    assert "analyze" in phase_seq
    base_idx = phase_seq.index("base_emission_rate")
    analyze_idx = phase_seq.index("analyze")
    assert base_idx < analyze_idx
    assert rec.scheduled_cells, "cell pool was never scheduled"
    assert rec.scheduled_cells[0]["issue"] == 479


# ── Fix 1 + 2 + 3 invariants: dispatcher does NOT disable strict mode. ──────


def test_analyze_command_does_not_pass_no_strict_base_panel(dispatcher_module, tmp_path):
    """The analyze subprocess command MUST NOT contain --no-strict-base-panel.

    Strict mode is the production default (round-3 Blocker 1.2): a missing /
    wrong-schema Phase-1.6 baseline MUST hard-fail the analyzer. If the
    dispatcher passes --no-strict-base-panel, the round-3 hard-fail becomes
    inert.
    """
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "479", "--stage", "1", "--seeds", "42"],
    )
    analyze_phase = next((p for p in rec.phases if p["phase"] == "analyze"), None)
    assert analyze_phase is not None
    assert "--no-strict-base-panel" not in analyze_phase["cmd"], (
        "Dispatcher must NOT pass --no-strict-base-panel to i479_analyze.py — "
        f"saw cmd={analyze_phase['cmd']}"
    )


def test_analyze_command_points_at_base_panel_emission_rate_json(dispatcher_module, tmp_path):
    """The analyze --base-panel-path MUST point at the emission-rate baseline.

    Not the #472 log-prob baseline (base_panel.json) — that's a wrong artifact
    for an emission-rate threshold (round-2 Blocker-2 mis-routing).
    """
    rec = _run_dispatcher(
        dispatcher_module,
        tmp_path,
        ["--issue", "479", "--stage", "1", "--seeds", "42"],
    )
    analyze_phase = next(p for p in rec.phases if p["phase"] == "analyze")
    base_panel_idx = analyze_phase["cmd"].index("--base-panel-path")
    base_panel_value = analyze_phase["cmd"][base_panel_idx + 1]
    assert "base_panel_emission_rate.json" in base_panel_value, (
        f"--base-panel-path must point at base_panel_emission_rate.json, got {base_panel_value!r}"
    )
    assert base_panel_value.endswith("base_panel_emission_rate.json")
