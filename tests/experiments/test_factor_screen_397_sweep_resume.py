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


# Round 12: the dispatcher-integration resume tests that monkeypatched
# `_run_one_cell_inprocess` (Round 11 single-pass path) moved to
# `test_factor_screen_397_two_pass_resume.py` and now monkeypatch
# `_run_pass1_hf` + `_run_pass2_vllm` instead. This file retains only the
# pure-unit tests for `is_cell_complete_locally`, `is_cell_complete_on_hub`,
# `filter_jobs_for_resume` (still used by helpers / two-pass logic), and
# the CLI-parsing tests for the resume flags.


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
