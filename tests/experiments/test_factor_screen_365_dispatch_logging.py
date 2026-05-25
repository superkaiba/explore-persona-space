"""Per-cell stdout+stderr capture tests for the task #365 dispatcher.

Round-9 (issue #365) Fix D: round-8 merged all 8 cells' output into a
single interleaved stream, hiding which cell hit a vLLM crash and
swallowing any stagger / failure traces. The dispatcher now opens a
dedicated per-cell log under

    ``<slab_root>/cell_<key>/source_<src>/seed_<N>/cell_stdout_stderr.log``

and pipes both stdout and stderr into it. These tests verify:

  * ``_cell_log_path`` returns the path expected next to ``metrics.json``.
  * The training-stage launch path opens that file and passes the handle
    as ``stdout`` to ``subprocess.Popen``, with ``stderr`` redirected to
    the same handle via ``subprocess.STDOUT``.
  * The log file's parent directory is created before Popen fires
    (the entry script also calls ``mkdir(parents=True)`` later, but the
    dispatcher must not race the subprocess startup).
  * On subprocess exit the file handle is closed (no FD leak across a
    96-cell run).

These tests do not require nvidia-smi; ``_detect_physical_gpu_count`` is
patched to return 1 and Popen / its returncode is mocked.
"""

from __future__ import annotations

import argparse

# The dispatcher is a hyphen-free importable module (loaded as ``dispatch_factor_screen_365``).
# We import it through importlib so the same fixture pattern works whether or
# not the worktree exposes scripts/ as a package.
import importlib.util
import subprocess
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def dispatcher():
    """Load the dispatcher module by path so the test does not depend on PYTHONPATH."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "dispatch_factor_screen_365.py"
    spec = importlib.util.spec_from_file_location("dispatch_factor_screen_365", script_path)
    assert spec is not None and spec.loader is not None, f"Could not load {script_path} as a module"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cell_log_path_anchors_next_to_metrics(dispatcher, tmp_path: Path) -> None:
    """The per-cell log lives in the same directory as metrics.json.

    Anchoring it there makes failure forensics trivial: any cell that did NOT
    leave a ``metrics.json`` can be inspected by reading the sibling
    ``cell_stdout_stderr.log`` in the same folder.
    """
    log_path = dispatcher._cell_log_path(
        slab_root=tmp_path / "eval",
        cell_key="00010",
        source="librarian",
        seed=42,
    )
    assert log_path == (
        tmp_path / "eval" / "cell_00010" / "source_librarian" / "seed_42" / "cell_stdout_stderr.log"
    )


def test_training_stage_redirects_popen_stdout_to_per_cell_log(
    dispatcher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launched cell subprocess writes stdout+stderr to its own log file.

    Verifies the round-9 Fix D wiring: Popen receives a file handle whose
    name resolves to ``_cell_log_path`` and ``stderr=subprocess.STDOUT``
    so a crash trace lands in the same file as normal stdout.

    Round-14 (issue #365): one cell now fans out to TWO Popen calls
    (cell-train then cell-eval). Both phases write to the SAME per-cell
    log via ``"a"`` (append) mode so the round-9 forensics convention
    of one log per (cell, source, seed) survives the split.
    """
    # Build a synthetic args namespace exercising the smallest possible job set.
    slab_root = tmp_path / "slab"
    pool_dir = tmp_path / "pools"
    args = argparse.Namespace(
        # Task #383 plumbing (plan v2 §5a): --issue forwarded to each
        # cell-train / cell-eval child subprocess argv.
        issue=365,
        sources=["librarian"],
        seeds=[42],
        pool_dir=pool_dir,
        slab_root=slab_root,
        num_gpus=1,
        skip_pool_stage=True,
        skip_off_policy=False,
        dry_run=False,
        resume=False,
        skip_hub_probe=True,
        cell_filter=["00010"],  # single-cell smoke
    )

    # The dispatcher detects physical GPUs via nvidia-smi; we fake one GPU.
    monkeypatch.setattr(dispatcher, "_detect_physical_gpu_count", lambda: 1)

    # Capture Popen calls. The fake process "completes" immediately so the
    # drain loop exits straight away; the launch path is still exercised.
    fake_proc = mock.MagicMock(spec=subprocess.Popen)
    fake_proc.poll.return_value = 0  # already done
    fake_proc.returncode = 0
    popen_calls: list[dict] = []

    def fake_popen(cmd, env, stdout, stderr):
        popen_calls.append({"cmd": cmd, "env": env, "stdout": stdout, "stderr": stderr})
        return fake_proc

    monkeypatch.setattr(dispatcher.subprocess, "Popen", fake_popen)
    # Disable sleeps inside the wait loop so the test does not block.
    monkeypatch.setattr(dispatcher.time, "sleep", lambda _s: None)

    rc = dispatcher._training_stage(args)
    assert rc == 0, f"training stage failed unexpectedly (rc={rc})"
    # Round-14: one cell = train + eval = 2 Popen calls (when train rc=0).
    assert len(popen_calls) == 2, f"expected 2 cell phase launches; got {len(popen_calls)}"

    expected_log = dispatcher._cell_log_path(
        slab_root=slab_root, cell_key="00010", source="librarian", seed=42
    )
    for idx, call in enumerate(popen_calls):
        stdout_handle = call["stdout"]
        assert hasattr(stdout_handle, "name"), (
            f"Popen[{idx}] stdout should be a file-like object with .name; "
            f"got {type(stdout_handle)!r}"
        )
        assert Path(stdout_handle.name) == expected_log, (
            f"Popen[{idx}] stdout file should be {expected_log}; got {stdout_handle.name}"
        )
        # stderr should redirect into stdout (so a vLLM crash trace ends up
        # in the same per-cell log).
        assert call["stderr"] == subprocess.STDOUT, (
            f"Popen[{idx}] stderr should be subprocess.STDOUT; got {call['stderr']!r}"
        )
    # The per-cell log directory must exist on disk before launch so the
    # subprocess does not race a parent-dir mkdir.
    assert expected_log.parent.is_dir(), (
        f"Per-cell log directory should be pre-created; missing: {expected_log.parent}"
    )


def test_training_stage_closes_per_cell_log_handle_on_exit(
    dispatcher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a cell subprocess exits, its per-cell log handle is closed.

    This regression-guards against FD leaks across a 96-cell run.
    """
    slab_root = tmp_path / "slab"
    args = argparse.Namespace(
        # Task #383 plumbing (plan v2 §5a): --issue forwarded to subprocess argv.
        issue=365,
        sources=["librarian"],
        seeds=[42],
        pool_dir=tmp_path / "pools",
        slab_root=slab_root,
        num_gpus=1,
        skip_pool_stage=True,
        skip_off_policy=False,
        dry_run=False,
        resume=False,
        skip_hub_probe=True,
        cell_filter=["00010"],
    )
    monkeypatch.setattr(dispatcher, "_detect_physical_gpu_count", lambda: 1)

    captured_handles: list = []
    fake_proc = mock.MagicMock(spec=subprocess.Popen)
    fake_proc.poll.return_value = 0
    fake_proc.returncode = 0

    def fake_popen(cmd, env, stdout, stderr):
        captured_handles.append(stdout)
        return fake_proc

    monkeypatch.setattr(dispatcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(dispatcher.time, "sleep", lambda _s: None)

    rc = dispatcher._training_stage(args)
    assert rc == 0
    # Round-14 (issue #365): one cell = 2 phases (train + eval) = 2 handles.
    assert len(captured_handles) == 2, (
        f"expected 2 captured handles (cell-train + cell-eval); got {len(captured_handles)}"
    )
    # ALL handles should be closed by the time the drain loop completes —
    # either via _wait_for_free_gpu (on poll() return) or the defensive
    # post-loop close pass.
    for idx, handle in enumerate(captured_handles):
        assert handle.closed, f"Per-cell log handle [{idx}] should be closed after subprocess exit"


def test_dry_run_does_not_open_per_cell_logs(
    dispatcher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--dry-run`` skips Popen entirely; no per-cell log files are created.

    Belt-and-suspenders: keeps the dry-run fast and side-effect-free for
    pre-launch sanity checks.
    """
    slab_root = tmp_path / "slab"
    args = argparse.Namespace(
        # Task #383 plumbing (plan v2 §5a): --issue forwarded to subprocess argv.
        issue=365,
        sources=["librarian"],
        seeds=[42],
        pool_dir=tmp_path / "pools",
        slab_root=slab_root,
        num_gpus=1,
        skip_pool_stage=True,
        skip_off_policy=False,
        dry_run=True,
        resume=False,
        skip_hub_probe=True,
        cell_filter=["00010"],
    )
    monkeypatch.setattr(dispatcher, "_detect_physical_gpu_count", lambda: 1)

    popen_called = mock.MagicMock()
    monkeypatch.setattr(dispatcher.subprocess, "Popen", popen_called)

    rc = dispatcher._training_stage(args)
    assert rc == 0
    popen_called.assert_not_called()
    # No cell dir should have been created in dry-run mode.
    assert not (slab_root / "cell_00010" / "source_librarian" / "seed_42").exists(), (
        "Dry-run should not pre-create per-cell directories"
    )
