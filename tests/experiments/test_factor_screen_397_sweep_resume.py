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
        num_gpus=2,
        max_concurrent_train=2,
        marker_token="※",
        save_every_n_steps=25,
        pos_per_source=400,
        lr=1e-4,
        warmup_ratio=0.10,
        require_smoke_pass=True,
        skip_smoke_pass_check=False,
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


class _FakeFinishedPopen:
    def __init__(self, cell_key: str, source: str, seed: int, rc: int = 0):
        self._cell_key = cell_key
        self._source = source
        self._seed = seed
        self.pid = 99999
        self._rc = rc

    def poll(self):
        return self._rc


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

    monkeypatch.setattr(_dispatch, "has_recent_smoke_pass_marker", lambda issue, *, repo_root: True)
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

        def _fake_launch(**kw):
            launch_calls.append((kw["cell"].key, kw["source"], kw["seed"]))
            return _FakeFinishedPopen(kw["cell"].key, kw["source"], kw["seed"], rc=0)

        monkeypatch.setattr(_dispatch, "_launch_cell_subprocess", _fake_launch)
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

    monkeypatch.setattr(_dispatch, "has_recent_smoke_pass_marker", lambda issue, *, repo_root: True)
    monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)  # complete → skip
        # cell 00001 not complete → launched.

        args = _build_args_for_resume(slab, no_resume=False, resume_source="local")

        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        launch_calls: list[tuple] = []

        def _fake_launch(**kw):
            launch_calls.append((kw["cell"].key, kw["source"], kw["seed"]))
            return _FakeFinishedPopen(kw["cell"].key, kw["source"], kw["seed"], rc=0)

        monkeypatch.setattr(_dispatch, "_launch_cell_subprocess", _fake_launch)
        # Stub marker posting (sweep-resume marker fires when skipped > 0).
        monkeypatch.setattr(_dispatch, "post_marker_via_task_py", lambda *a, **kw: None)
        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0
        assert len(launch_calls) == 1, f"Only 00001 should launch; got {launch_calls}"
        assert launch_calls[0][0] == "00001"


def test_dispatcher_emits_sweep_resume_marker_when_skipping(monkeypatch) -> None:
    """When ≥1 cell is skipped, epm:sweep-resume marker is posted."""
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "has_recent_smoke_pass_marker", lambda issue, *, repo_root: True)
    monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)

        args = _build_args_for_resume(slab, no_resume=False, resume_source="local")
        cells = [Cell.from_key("00000"), Cell.from_key("00001")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)
        monkeypatch.setattr(
            _dispatch,
            "_launch_cell_subprocess",
            lambda **kw: _FakeFinishedPopen(kw["cell"].key, kw["source"], kw["seed"], rc=0),
        )

        post_calls: list[tuple] = []
        monkeypatch.setattr(
            _dispatch,
            "post_marker_via_task_py",
            lambda issue, kind, note, *, repo_root: post_calls.append((issue, kind, note[:200])),
        )

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        # epm:sweep-resume marker present.
        sweep_resume_calls = [c for c in post_calls if c[1] == "epm:sweep-resume"]
        assert len(sweep_resume_calls) == 1, (
            f"Expected one epm:sweep-resume marker; got {len(sweep_resume_calls)}"
        )
        note = sweep_resume_calls[0][2]
        assert "1 of 2" in note or "1/2" in note, (
            f"Marker note must record the skip count; got: {note}"
        )


def test_dispatcher_no_resume_skips_sweep_resume_marker(monkeypatch) -> None:
    """--no-resume → no resume happened → no epm:sweep-resume marker."""
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "has_recent_smoke_pass_marker", lambda issue, *, repo_root: True)
    monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        _write_metrics_json(slab, "00000", "librarian", 42)

        args = _build_args_for_resume(slab, no_resume=True)
        cells = [Cell.from_key("00000")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)
        monkeypatch.setattr(
            _dispatch,
            "_launch_cell_subprocess",
            lambda **kw: _FakeFinishedPopen(kw["cell"].key, kw["source"], kw["seed"], rc=0),
        )

        post_calls: list[tuple] = []
        monkeypatch.setattr(
            _dispatch,
            "post_marker_via_task_py",
            lambda issue, kind, note, *, repo_root: post_calls.append((issue, kind, note[:80])),
        )

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        sweep_resume_calls = [c for c in post_calls if c[1] == "epm:sweep-resume"]
        assert len(sweep_resume_calls) == 0, (
            f"--no-resume must NOT emit epm:sweep-resume marker; got {sweep_resume_calls}"
        )


def test_dispatcher_resume_with_no_skips_does_not_emit_marker(monkeypatch) -> None:
    """Fresh sweep (no completed cells) → no epm:sweep-resume marker.

    The marker only fires when there's something to report (skipped > 0);
    a clean fresh launch should NOT clutter the events log.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "has_recent_smoke_pass_marker", lambda issue, *, repo_root: True)
    monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

    with tempfile.TemporaryDirectory() as tmp:
        slab = Path(tmp)
        # No metrics.json staged → all cells launched.
        args = _build_args_for_resume(slab, no_resume=False, resume_source="local")
        cells = [Cell.from_key("00000")]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)
        monkeypatch.setattr(
            _dispatch,
            "_launch_cell_subprocess",
            lambda **kw: _FakeFinishedPopen(kw["cell"].key, kw["source"], kw["seed"], rc=0),
        )

        post_calls: list[tuple] = []
        monkeypatch.setattr(
            _dispatch,
            "post_marker_via_task_py",
            lambda issue, kind, note, *, repo_root: post_calls.append((issue, kind, note[:80])),
        )

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        sweep_resume_calls = [c for c in post_calls if c[1] == "epm:sweep-resume"]
        assert len(sweep_resume_calls) == 0, (
            "Fresh sweep (no skips) must NOT emit epm:sweep-resume marker"
        )


def test_dispatcher_inconsistent_state_raises_before_launch(monkeypatch) -> None:
    """LOUD-FAIL on local-present-hub-missing when resume_source=both —
    the dispatcher MUST surface this BEFORE launching any cell (corruption
    signal, not silently skip or re-run + clobber).
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    monkeypatch.setattr(_dispatch, "has_recent_smoke_pass_marker", lambda issue, *, repo_root: True)
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
        monkeypatch.setattr(
            _dispatch,
            "_launch_cell_subprocess",
            lambda **kw: (
                launch_calls.append(1)
                or _FakeFinishedPopen(kw["cell"].key, kw["source"], kw["seed"], rc=0)
            ),
        )

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
