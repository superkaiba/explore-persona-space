"""Sweep-level resume tests (task #397, Round 6).

Round 6 added per-cell completion detection so a crashed/killed sweep
can resume without re-running every cell. Detection signals:

  - Local: ``eval_results/issue_397/cell_<id>/source_<src>/seed_<s>/
    metrics.json`` exists AND parses (the per-cell run's LAST artifact).
  - HF Hub: ``adapters/issue_397/i397_cell_<id>_source_<src>_seed<s>/``
    has ``adapter_*`` files.
  - Either signal sufficient (default); prefer local (faster). LOUD-FAIL
    on inconsistent state (local present, hub missing → raise unless
    --resume-source=local).
  - --no-resume forces full re-launch (useful when results are suspect).

This test surface verifies:

  - ``is_cell_complete_locally`` reads + parses metrics.json correctly,
    returns False on missing / malformed.
  - ``is_cell_complete_on_hub`` reads the cached file list, returns True
    when adapter files present, False on missing.
  - ``filter_jobs_for_resume`` skips completed cells under each
    ``resume_source`` mode; raises ``ValueError`` on inconsistent state
    when resume_source=both.
  - The dispatcher loop honors ``args.no_resume=True`` (full re-launch)
    AND ``args.resume_source`` (which signal counts).
  - Resume summary log line + epm:sweep-resume marker are emitted.

CPU-only; HF Hub is monkeypatched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Import dispatcher via importlib (scripts/ not a package).
_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)
_spec = importlib.util.spec_from_file_location("dispatch_factor_screen_397", _DISPATCH_PATH)
_dispatch = importlib.util.module_from_spec(_spec)
sys.modules["dispatch_factor_screen_397"] = _dispatch
_spec.loader.exec_module(_dispatch)


def _build_args_for_resume(
    slab_root: Path,
    *,
    no_resume: bool = False,
    resume_source: str = "both",
    num_cells: int = 3,
) -> argparse.Namespace:
    """Build a dispatcher args namespace for the resume tests."""
    return argparse.Namespace(
        issue=397,
        mode="sweep",
        pool_dir=slab_root / "pools",
        slab_root=slab_root,
        smoke_cell="10010",
        smoke_source="librarian",
        smoke_seed=42,
        sources="librarian",
        seeds="42",
        # Round 11 removed num_gpus + max_concurrent_train (in-process serial).
        marker_token="※",
        save_every_n_steps=25,
        pos_per_source=400,
        lr=1e-4,
        warmup_ratio=0.10,
        require_smoke_pass=True,
        skip_smoke_pass_check=False,
        # Round 9 — orchestrator-set flag; default True so resume tests
        # short-circuit the smoke-pass gate.
        smoke_pass_confirmed=True,
        dry_run=False,
        no_resume=no_resume,
        resume_source=resume_source,
        log_level="INFO",
    )


def _write_metrics_json(slab_root: Path, cell_key: str, source: str, seed: int) -> None:
    """Write a valid metrics.json sentinel for the (cell, source, seed) tuple."""
    cell_dir = slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "marker": "※",
        "cell_key": cell_key,
        "source": source,
        "seed": seed,
        "personas": {source: {"substring_rate": 0.5, "total": 100}},
    }
    (cell_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def _fake_inprocess_rc0(**kwargs) -> int:
    """Stub for ``_run_one_cell_inprocess`` that immediately returns rc=0.

    Round 11 replaced the subprocess wrapper (which used a ``_FakeFinishedPopen``
    polling protocol) with a direct in-process call. The stub now returns
    an int directly — no polling, no GPU-id arg, no Popen attributes.
    """
    return 0


# ---------------------------------------------------------------------------
# is_cell_complete_locally
# ---------------------------------------------------------------------------


def test_is_cell_complete_locally_returns_true_for_valid_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        cell_dir = slab / "cell"
        cell_dir.mkdir()
        (cell_dir / "metrics.json").write_text(
            json.dumps({"personas": {"librarian": {"substring_rate": 0.5}}}),
            encoding="utf-8",
        )
        assert _dispatch.is_cell_complete_locally(cell_dir) is True


def test_is_cell_complete_locally_returns_false_for_missing_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp) / "empty"
        cell_dir.mkdir()
        assert _dispatch.is_cell_complete_locally(cell_dir) is False


def test_is_cell_complete_locally_returns_false_for_malformed_metrics() -> None:
    """Malformed JSON → treat as not-complete (re-run will overwrite cleanly)."""
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp) / "broken"
        cell_dir.mkdir()
        (cell_dir / "metrics.json").write_text("{not-json", encoding="utf-8")
        assert _dispatch.is_cell_complete_locally(cell_dir) is False


def test_is_cell_complete_locally_returns_false_when_no_personas_key() -> None:
    """metrics.json missing the 'personas' key → treat as not-complete."""
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp) / "weird"
        cell_dir.mkdir()
        (cell_dir / "metrics.json").write_text(json.dumps({"random_key": "data"}), encoding="utf-8")
        assert _dispatch.is_cell_complete_locally(cell_dir) is False


# ---------------------------------------------------------------------------
# is_cell_complete_on_hub
# ---------------------------------------------------------------------------


def test_is_cell_complete_on_hub_returns_true_when_adapter_present() -> None:
    """Hub probe matches the canonical adapter path + adapter_* file prefix."""
    cache = [
        "adapters/issue_397/i397_cell_10010_source_librarian_seed42/adapter_model.safetensors",
        "adapters/issue_397/i397_cell_10010_source_librarian_seed42/adapter_config.json",
        "adapters/issue_397/i397_cell_other/readme.md",
    ]
    assert (
        _dispatch.is_cell_complete_on_hub("10010", "librarian", 42, hub_files_cache=cache) is True
    )


def test_is_cell_complete_on_hub_returns_false_when_path_missing() -> None:
    cache = ["adapters/issue_397/i397_cell_other_source_librarian_seed42/adapter_model.safetensors"]
    assert (
        _dispatch.is_cell_complete_on_hub("10010", "librarian", 42, hub_files_cache=cache) is False
    )


def test_is_cell_complete_on_hub_returns_false_when_only_readme_at_path() -> None:
    """Path exists but only readme.md → no adapter_* file → not complete."""
    cache = ["adapters/issue_397/i397_cell_10010_source_librarian_seed42/readme.md"]
    assert (
        _dispatch.is_cell_complete_on_hub("10010", "librarian", 42, hub_files_cache=cache) is False
    )


def test_is_cell_complete_on_hub_returns_false_when_cache_is_none(monkeypatch) -> None:
    """When cache is None (Hub unreachable), return False so caller falls back."""
    # Make _fetch_hub_adapter_index return None to simulate Hub down.
    monkeypatch.setattr(_dispatch, "_fetch_hub_adapter_index", lambda: None)
    assert _dispatch.is_cell_complete_on_hub("10010", "librarian", 42) is False


# ---------------------------------------------------------------------------
# filter_jobs_for_resume
# ---------------------------------------------------------------------------


def test_filter_jobs_local_mode_skips_cells_with_metrics_json() -> None:
    """resume_source=local skips cells with metrics.json, ignores Hub."""
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)
        # cell 00001 NOT written → should be queued.

        jobs = [
            (Cell.from_key("00000"), "librarian", 42),
            (Cell.from_key("00001"), "librarian", 42),
        ]
        remaining, summary = _dispatch.filter_jobs_for_resume(
            jobs, slab_root=slab, resume_source="local"
        )
        assert len(remaining) == 1
        assert remaining[0][0].key == "00001"
        assert summary["skipped_local"] == 1
        assert summary["queued"] == 1


def test_filter_jobs_hub_mode_skips_cells_with_hub_adapter() -> None:
    """resume_source=hub skips cells with Hub adapter, ignores local."""
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        # Even with local present, hub-mode only looks at hub.
        _write_metrics_json(slab, "00000", "librarian", 42)

        hub_cache = [
            "adapters/issue_397/i397_cell_00001_source_librarian_seed42/adapter_model.safetensors",
        ]
        jobs = [
            (Cell.from_key("00000"), "librarian", 42),  # local only → re-run in hub mode
            (Cell.from_key("00001"), "librarian", 42),  # hub only → skipped
        ]
        remaining, summary = _dispatch.filter_jobs_for_resume(
            jobs, slab_root=slab, resume_source="hub", hub_files_cache=hub_cache
        )
        assert len(remaining) == 1
        assert remaining[0][0].key == "00000"  # re-run (no hub adapter)
        assert summary["skipped_hub"] == 1


def test_filter_jobs_both_mode_skips_when_either_signal_present() -> None:
    """resume_source=both skips when local OR hub says complete."""
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)
        hub_cache = [
            "adapters/issue_397/i397_cell_00000_source_librarian_seed42/adapter_model.safetensors",
            "adapters/issue_397/i397_cell_00001_source_librarian_seed42/adapter_model.safetensors",
        ]
        jobs = [
            (Cell.from_key("00000"), "librarian", 42),  # local+hub → skipped_both
            (Cell.from_key("00001"), "librarian", 42),  # hub-only → skipped_hub
            (Cell.from_key("00002"), "librarian", 42),  # neither → queued
        ]
        remaining, summary = _dispatch.filter_jobs_for_resume(
            jobs, slab_root=slab, resume_source="both", hub_files_cache=hub_cache
        )
        assert len(remaining) == 1
        assert remaining[0][0].key == "00002"
        assert summary["skipped_both"] == 1
        assert summary["skipped_hub"] == 1


def test_filter_jobs_both_mode_raises_on_local_present_hub_missing() -> None:
    """LOUD-FAIL on inconsistent state: local says done but Hub missing.

    Per the brief: "prefer LOUD-FAIL on inconsistent state (e.g., local
    metrics_final.json present but HF Hub adapter missing — that's a
    corruption signal, should raise not skip)."
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)
        # hub_cache is EMPTY — local says done, hub disagrees.
        jobs = [(Cell.from_key("00000"), "librarian", 42)]
        with pytest.raises(ValueError, match="LOUD-FAIL"):
            _dispatch.filter_jobs_for_resume(
                jobs, slab_root=slab, resume_source="both", hub_files_cache=[]
            )


def test_filter_jobs_both_mode_accepts_hub_present_local_missing() -> None:
    """Hub present, local missing → cell is complete on Hub; skip + don't raise.

    Common case: local artifacts wiped (pod recycled) but Hub still has
    the result. Skipping is correct — the analyzer can pull from Hub.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        # No local metrics.json written.
        hub_cache = [
            "adapters/issue_397/i397_cell_00000_source_librarian_seed42/adapter_model.safetensors",
        ]
        jobs = [(Cell.from_key("00000"), "librarian", 42)]
        remaining, summary = _dispatch.filter_jobs_for_resume(
            jobs, slab_root=slab, resume_source="both", hub_files_cache=hub_cache
        )
        assert remaining == []
        assert summary["skipped_hub"] == 1


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------


def test_dispatcher_no_resume_flag_runs_all_cells(monkeypatch) -> None:
    """--no-resume forces all cells to be queued even if metrics.json exists."""
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)
    monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        # Both cells already have metrics.json (would normally be skipped).
        _write_metrics_json(slab, "00000", "librarian", 42)
        _write_metrics_json(slab, "00001", "librarian", 42)

        args = _build_args_for_resume(slab, no_resume=True)

        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        launch_calls: list[tuple] = []

        def _fake_inprocess(**kw):
            launch_calls.append((kw["cell"].key, kw["source"], kw["seed"]))
            return 0

        monkeypatch.setattr(_dispatch, "_run_one_cell_inprocess", _fake_inprocess)
        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0
        # ALL cells launched — --no-resume overrode the skip.
        assert len(launch_calls) == 2, (
            f"--no-resume must launch all cells; got {len(launch_calls)} launches"
        )


def test_dispatcher_resume_local_mode_skips_complete_cells(monkeypatch) -> None:
    """resume_source=local: cells with metrics.json are skipped, others launched."""
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)
    monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)  # complete → skip
        # cell 00001 not complete → launched.

        args = _build_args_for_resume(slab, no_resume=False, resume_source="local")

        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        launch_calls: list[tuple] = []

        def _fake_inprocess(**kw):
            launch_calls.append((kw["cell"].key, kw["source"], kw["seed"]))
            return 0

        monkeypatch.setattr(_dispatch, "_run_one_cell_inprocess", _fake_inprocess)
        # Round 9 — dispatcher writes SWEEP_RESUME.json (NOT marker post).
        # No stub needed; the file lands under slab_root and is harmless.
        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0
        assert len(launch_calls) == 1, f"Only 00001 should launch; got {launch_calls}"
        assert launch_calls[0][0] == "00001"


def test_dispatcher_writes_sweep_resume_verdict_file_when_skipping(monkeypatch) -> None:
    """Round 9: when ≥1 cell is skipped, the dispatcher writes
    SWEEP_RESUME.json under slab_root. Orchestrator on the VM side reads
    it and posts the epm:sweep-resume marker (task.py works from the
    repo root but NOT from the pod's worktree-branch checkout).
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)
    monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)

        args = _build_args_for_resume(slab, no_resume=False, resume_source="local")
        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)
        monkeypatch.setattr(_dispatch, "_run_one_cell_inprocess", _fake_inprocess_rc0)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        # SWEEP_RESUME.json exists with the right shape.
        sweep_resume_path = slab / "SWEEP_RESUME.json"
        assert sweep_resume_path.exists(), (
            f"Round 9: dispatcher must write SWEEP_RESUME.json; got missing {sweep_resume_path}"
        )
        payload = json.loads(sweep_resume_path.read_text(encoding="utf-8"))
        assert payload["kind"] == "epm:sweep-resume"
        assert payload["skipped_total"] == 1
        assert payload["total_jobs"] == 2
        assert payload["remaining"] == 1
        note = payload["note"]
        assert "1 of 2" in note, f"Note must record the skip count; got: {note}"


def test_dispatcher_no_resume_skips_sweep_resume_file(monkeypatch) -> None:
    """Round 9: --no-resume → no resume happened → no SWEEP_RESUME.json
    file emitted. Orchestrator only has work to do when there's a resume
    summary to post.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)
    monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)

        args = _build_args_for_resume(slab, no_resume=True)
        cells = [Cell.from_key("00000")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)
        monkeypatch.setattr(_dispatch, "_run_one_cell_inprocess", _fake_inprocess_rc0)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        sweep_resume_path = slab / "SWEEP_RESUME.json"
        assert not sweep_resume_path.exists(), (
            f"--no-resume must NOT emit SWEEP_RESUME.json; found at {sweep_resume_path}"
        )


def test_dispatcher_resume_with_no_skips_does_not_emit_file(monkeypatch) -> None:
    """Fresh sweep (no completed cells) → no SWEEP_RESUME.json file.

    The verdict file only lands when there's something to report
    (skipped > 0); a clean fresh launch should NOT leave a stale file
    that the orchestrator might post as a misleading marker.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)
    monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        # No metrics.json staged → all cells launched.
        args = _build_args_for_resume(slab, no_resume=False, resume_source="local")
        cells = [Cell.from_key("00000")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)
        monkeypatch.setattr(_dispatch, "_run_one_cell_inprocess", _fake_inprocess_rc0)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        sweep_resume_path = slab / "SWEEP_RESUME.json"
        assert not sweep_resume_path.exists(), (
            f"Fresh sweep (no skips) must NOT emit SWEEP_RESUME.json; found at {sweep_resume_path}"
        )


def test_dispatcher_inconsistent_state_raises_before_launch(monkeypatch) -> None:
    """LOUD-FAIL on local-present-hub-missing when resume_source=both —
    the dispatcher MUST surface this BEFORE launching any cell (corruption
    signal, not silently skip or re-run + clobber).
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)
    # Hub probe returns empty (no adapters on Hub).
    monkeypatch.setattr(_dispatch, "_fetch_hub_adapter_index", lambda: [])
    monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        # Local says complete; Hub disagrees.
        _write_metrics_json(slab, "00000", "librarian", 42)

        args = _build_args_for_resume(slab, no_resume=False, resume_source="both")
        cells = [Cell.from_key("00000")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        launch_calls: list = []

        def _fake_inprocess(**kw):
            launch_calls.append(1)
            return 0

        monkeypatch.setattr(_dispatch, "_run_one_cell_inprocess", _fake_inprocess)

        repo_root = Path(__file__).resolve().parent.parent.parent
        with pytest.raises(ValueError, match="LOUD-FAIL"):
            _dispatch.run_sweep_phase(args, repo_root=repo_root)
        # No cell was launched before the raise.
        assert len(launch_calls) == 0, "Inconsistent state must raise BEFORE any launch"


# ---------------------------------------------------------------------------
# CLI parsing for the new flags
# ---------------------------------------------------------------------------


def test_cli_no_resume_flag_parses() -> None:
    parser = _dispatch.build_arg_parser()
    args = parser.parse_args(
        [
            "--issue",
            "397",
            "--mode",
            "sweep",
            "--pool-dir",
            "/tmp/pools",
            "--slab-root",
            "/tmp/out",
            "--no-resume",
        ]
    )
    assert args.no_resume is True


def test_cli_resume_source_default_is_both() -> None:
    parser = _dispatch.build_arg_parser()
    args = parser.parse_args(
        [
            "--issue",
            "397",
            "--mode",
            "sweep",
            "--pool-dir",
            "/tmp/pools",
            "--slab-root",
            "/tmp/out",
        ]
    )
    assert args.no_resume is False
    assert args.resume_source == "both"


def test_cli_resume_source_accepts_local_hub_both() -> None:
    parser = _dispatch.build_arg_parser()
    for choice in ("local", "hub", "both"):
        args = parser.parse_args(
            [
                "--issue",
                "397",
                "--mode",
                "sweep",
                "--pool-dir",
                "/tmp/pools",
                "--slab-root",
                "/tmp/out",
                "--resume-source",
                choice,
            ]
        )
        assert args.resume_source == choice


def test_cli_resume_source_rejects_invalid_choice() -> None:
    parser = _dispatch.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--issue",
                "397",
                "--mode",
                "sweep",
                "--pool-dir",
                "/tmp/pools",
                "--slab-root",
                "/tmp/out",
                "--resume-source",
                "telepathy",
            ]
        )


# ---------------------------------------------------------------------------
# Hub-unreachable degradation (Round 7 fix for code-review v6 Major 1)
# ---------------------------------------------------------------------------


def test_filter_jobs_hub_unreachable_degrades_to_local_only(caplog) -> None:
    """Code-review v6 Major 1: when the Hub probe fails (returns None) AND
    local metrics.json is present, ``filter_jobs_for_resume`` must NOT
    LOUD-FAIL — that was the round-6 false-positive.

    The corruption-detection LOUD-FAIL only triggers when BOTH probes
    successfully returned AND disagree. A transient Hub outage
    (rate-limit, network blip, dashboard maintenance) should not block
    the sweep — degrade gracefully to local-only and log a WARNING.
    """
    import logging

    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)
        _write_metrics_json(slab, "00001", "librarian", 42)
        # cell 00002 has no metrics → should be queued.

        jobs = [
            (Cell.from_key("00000"), "librarian", 42),
            (Cell.from_key("00001"), "librarian", 42),
            (Cell.from_key("00002"), "librarian", 42),
        ]
        # hub_files_cache=None signals "Hub probe failed".
        with caplog.at_level(logging.WARNING, logger="dispatch_factor_screen_397"):
            remaining, summary = _dispatch.filter_jobs_for_resume(
                jobs,
                slab_root=slab,
                resume_source="both",
                hub_files_cache=None,
            )

        # 2 cells with local metrics → skipped; 1 cell without → queued.
        assert summary["skipped_local"] == 2, (
            f"Hub-unreachable + local-present cells must be skipped via local-only "
            f"path; got summary={summary}"
        )
        assert summary["queued"] == 1
        assert len(remaining) == 1
        assert remaining[0][0].key == "00002"

        # The Hub-unreachable WARNING was emitted exactly once (not per cell).
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        hub_warnings = [r for r in warnings if "Hub probe failed" in r.getMessage()]
        assert len(hub_warnings) == 1, (
            f"Expected exactly ONE 'Hub probe failed' WARNING (not per-cell spam); "
            f"got {len(hub_warnings)} from {[w.getMessage() for w in warnings]}"
        )
        # WARNING message names the local-only fallback + the recovery hint.
        msg = hub_warnings[0].getMessage()
        assert "local state only" in msg
        assert "--no-resume" in msg


def test_filter_jobs_hub_unreachable_does_not_raise_on_local_present() -> None:
    """The Round 6 bug: local metrics.json present + Hub probe failed →
    treated identically to "Hub says no adapter" → LOUD-FAIL. Round 7 fix:
    Hub-unreachable degrades to local-only; no raise.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)
        jobs = [(Cell.from_key("00000"), "librarian", 42)]
        # Does NOT raise (round-6 behavior would have raised here).
        remaining, summary = _dispatch.filter_jobs_for_resume(
            jobs,
            slab_root=slab,
            resume_source="both",
            hub_files_cache=None,
        )
        assert remaining == []
        assert summary["skipped_local"] == 1


def test_filter_jobs_hub_reachable_with_disagreement_still_raises() -> None:
    """The corruption-detection LOUD-FAIL must STILL fire when the Hub
    probe SUCCEEDS but returns an empty adapter list. That's the real
    inconsistency signal (Hub says definitively 'no adapter', not 'I
    couldn't reach Hub').
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)
        jobs = [(Cell.from_key("00000"), "librarian", 42)]
        # Empty list = Hub returned successfully + says "no adapter under
        # any path". DIFFERENT from None (Hub probe failed).
        with pytest.raises(ValueError, match="LOUD-FAIL"):
            _dispatch.filter_jobs_for_resume(
                jobs,
                slab_root=slab,
                resume_source="both",
                hub_files_cache=[],
            )


def test_filter_jobs_hub_unreachable_local_mode_unchanged(caplog) -> None:
    """When resume_source='local', Hub-unreachable degradation does NOT
    apply (local mode never touches Hub) — and no WARNING is logged.
    """
    import logging

    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)
        jobs = [(Cell.from_key("00000"), "librarian", 42)]
        with caplog.at_level(logging.WARNING, logger="dispatch_factor_screen_397"):
            _remaining, summary = _dispatch.filter_jobs_for_resume(
                jobs,
                slab_root=slab,
                resume_source="local",
                hub_files_cache=None,
            )
        assert summary["skipped_local"] == 1
        # No Hub WARNING in local mode.
        hub_warnings = [r for r in caplog.records if "Hub probe failed" in r.getMessage()]
        assert len(hub_warnings) == 0
