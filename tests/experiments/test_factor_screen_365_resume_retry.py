"""Tests for the ``--prioritize-failed`` retry ordering (task #391 post-mortem).

When the round-4 dispatcher hit ``CPaddingError`` on cell_10111 and the
padding fix landed in commit 1fb8a215, the relaunched dispatcher should
have processed cell_10111 first so a second mid-run incident (the EDQUOT
that actually killed the second dispatcher) wouldn't strand it again.

Two layers exercised:

  1. ``cell_has_failure_marker``: detects ``factor_screen_failed.json``
     on disk (the marker that cell-mode's outer except writes whenever a
     cell-mode invocation raises).
  2. End-to-end queue partition: with mixed (failed, fresh, complete)
     cells under a slab root, the dispatcher's training loop must emit
     ``[failed_first, fresh_after]`` and respect ``--no-prioritize-failed``.

``--resume`` does NOT and never DID consult the failure marker for the
SKIP decision — ``cell_complete_on_disk`` correctly gates only on the
success sentinel (``persona_panel_scores``). These tests pin that
behavior in place so a future regression can't silently start treating a
failure marker as "done, skip".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


def _load_dispatch_module():
    """Load ``scripts/dispatch_factor_screen_365`` as a module without sys.path tweaks."""
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts" / "dispatch_factor_screen_365.py"
    spec = importlib.util.spec_from_file_location("dispatch_factor_screen_365", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_complete_cell(slab_root: Path, cell_key: str, source: str, seed: int) -> Path:
    """Synthesise a slab tree that ``cell_complete_on_disk`` will accept as DONE."""
    out = slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_key": cell_key,
        "persona_panel_scores": {"librarian": {"substring_rate": 0.0}},
    }
    (out / "metrics.json").write_text(json.dumps(payload))
    adapter = out / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 128)
    return out


def _make_failed_cell(slab_root: Path, cell_key: str, source: str, seed: int) -> Path:
    """Synthesise a slab tree where the cell crashed before training began.

    Mirrors what ``factor_screen_365.__main__``'s outer except writes when
    a preflight (CPaddingError, CAxisPreflightError) or mid-training (OOM,
    NCCL) failure crashes cell-mode. No ``metrics.json``, no adapter dir,
    just the failure marker.
    """
    out = slab_root / f"cell_{cell_key}" / f"source_{source}" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "factor_screen_failed.json").write_text(
        json.dumps(
            {
                "cell": cell_key,
                "source": source,
                "seed": seed,
                "error": "CPaddingError: target_token_count=378 not reachable",
                "elapsed_s": 12.3,
            }
        )
    )
    return out


# ---- cell_has_failure_marker -------------------------------------------------


def test_has_failure_marker_true_when_marker_present(tmp_path: Path) -> None:
    mod = _load_dispatch_module()
    _make_failed_cell(tmp_path, "10111", "librarian", 42)
    assert mod.cell_has_failure_marker(tmp_path, "10111", "librarian", 42)


def test_has_failure_marker_false_when_absent(tmp_path: Path) -> None:
    mod = _load_dispatch_module()
    (tmp_path / "cell_10111" / "source_librarian" / "seed_42").mkdir(parents=True)
    assert not mod.cell_has_failure_marker(tmp_path, "10111", "librarian", 42)


def test_has_failure_marker_false_when_empty_file(tmp_path: Path) -> None:
    """An empty marker file (zero-byte truncation) is not a real failure record."""
    mod = _load_dispatch_module()
    out = tmp_path / "cell_10111" / "source_librarian" / "seed_42"
    out.mkdir(parents=True)
    (out / "factor_screen_failed.json").write_text("")
    assert not mod.cell_has_failure_marker(tmp_path, "10111", "librarian", 42)


# ---- complete_on_disk MUST IGNORE failure marker (regression pin) ----------


def test_complete_on_disk_ignores_failure_marker(tmp_path: Path) -> None:
    """A failed cell must NEVER be treated as complete.

    The failure marker is purely diagnostic; ``--resume`` SHOULD re-queue
    the cell so a code-fix relaunch actually retrains it. If a future
    refactor starts gating on absence-of-marker, this test catches it.
    """
    mod = _load_dispatch_module()
    _make_failed_cell(tmp_path, "10111", "librarian", 42)
    assert not mod.cell_complete_on_disk(tmp_path, "10111", "librarian", 42)


# ---- End-to-end queue partition --------------------------------------------


def _resume_queue(mod, args: argparse.Namespace) -> tuple[list, list, list]:
    """Replicate the dispatcher's queue partition without spawning subprocesses.

    Returns ``(skipped, failed_retry_jobs, fresh_jobs)``. The launch order
    the real dispatcher uses is ``failed_retry_jobs + fresh_jobs``.
    """
    jobs = mod._training_jobs(args)
    skipped: list = []
    failed_retry_jobs: list = []
    fresh_jobs: list = []
    for cell_key, source, seed in jobs:
        if args.resume and mod.cell_complete_on_disk(args.slab_root, cell_key, source, seed):
            skipped.append((cell_key, source, seed))
            continue
        if (
            args.resume
            and args.prioritize_failed
            and mod.cell_has_failure_marker(args.slab_root, cell_key, source, seed)
        ):
            failed_retry_jobs.append((cell_key, source, seed))
        else:
            fresh_jobs.append((cell_key, source, seed))
    return skipped, failed_retry_jobs, fresh_jobs


@pytest.fixture
def _slab_with_mixed_state(tmp_path: Path) -> Path:
    """A slab where:
    * cell_00000 (librarian, 42) is complete (success sentinel present)
    * cell_10111 (librarian, 42) carries a failure marker  -> priority retry
    * every other cell has no artifacts at all              -> fresh
    """
    _make_complete_cell(tmp_path, "00000", "librarian", 42)
    _make_failed_cell(tmp_path, "10111", "librarian", 42)
    return tmp_path


def test_failed_cell_launches_before_fresh_under_prioritize(
    _slab_with_mixed_state: Path,
) -> None:
    """The post-mortem fix: cell_10111 leads the launch queue.

    Lex order is ``00000, 00001, ..., 10110, 10111, 11000, ...`` (sources
    nested under cells). With prioritization on, the launch order MUST
    start with cell_10111 rather than letting it sit in slot 23.
    """
    mod = _load_dispatch_module()
    args = argparse.Namespace(
        slab_root=_slab_with_mixed_state,
        sources=["librarian"],
        seeds=[42],
        resume=True,
        prioritize_failed=True,
    )
    skipped, failed_first, fresh = _resume_queue(mod, args)
    assert skipped == [("00000", "librarian", 42)]
    assert failed_first == [("10111", "librarian", 42)]
    # The failed cell's seat is now at the head of the launch order.
    launch_order = failed_first + fresh
    assert launch_order[0] == ("10111", "librarian", 42)
    # Sanity: it was previously slot 23 of 32 in lex order (idx accounting:
    # bit-23 in (0,1)^5 enumerated MSB-first).
    lex_order = mod._training_jobs(args)
    assert lex_order.index(("10111", "librarian", 42)) == 23


def test_no_prioritize_falls_back_to_lex_order(_slab_with_mixed_state: Path) -> None:
    """``--no-prioritize-failed`` restores strict lex order for the failed cell."""
    mod = _load_dispatch_module()
    args = argparse.Namespace(
        slab_root=_slab_with_mixed_state,
        sources=["librarian"],
        seeds=[42],
        resume=True,
        prioritize_failed=False,
    )
    skipped, failed_first, fresh = _resume_queue(mod, args)
    # Failed cell partitions into fresh (no priority); skipped is still the
    # complete cell. fresh now contains all 31 non-complete cells in lex order.
    assert skipped == [("00000", "librarian", 42)]
    assert failed_first == []
    assert ("10111", "librarian", 42) in fresh
    assert fresh[0] == ("00001", "librarian", 42)


def test_no_resume_does_not_partition(tmp_path: Path) -> None:
    """Without --resume, the failure marker must not change ordering either."""
    mod = _load_dispatch_module()
    _make_complete_cell(tmp_path, "00000", "librarian", 42)
    _make_failed_cell(tmp_path, "10111", "librarian", 42)
    args = argparse.Namespace(
        slab_root=tmp_path,
        sources=["librarian"],
        seeds=[42],
        resume=False,
        prioritize_failed=True,
    )
    skipped, failed_first, fresh = _resume_queue(mod, args)
    # --no-resume forces full lex-order re-execution; nothing skipped, nothing
    # promoted to the failed-first bucket.
    assert skipped == []
    assert failed_first == []
    assert len(fresh) == 32  # all cells in lex order
    assert fresh[0] == ("00000", "librarian", 42)


# ---- CLI wiring -------------------------------------------------------------


def test_prioritize_failed_default_true() -> None:
    """The dispatcher MUST default to prioritizing failed cells.

    The post-mortem fix is no good if it's off by default — most callers
    won't know to add a flag. The flag exists only to opt OUT for the rare
    strict-lex-order case.
    """
    mod = _load_dispatch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args([])
    assert args.prioritize_failed is True


def test_no_prioritize_failed_flag_parses() -> None:
    mod = _load_dispatch_module()
    parser = mod._build_arg_parser()
    args = parser.parse_args(["--no-prioritize-failed"])
    assert args.prioritize_failed is False
