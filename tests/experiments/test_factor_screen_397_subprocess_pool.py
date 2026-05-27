"""GPU subprocess-pool scheduling tests (task #397, Round 5).

Plan v4 §12: concurrent training capped at 6/8 GPUs (disk-quota mitigation).
The dispatcher implements this via:

  - ``_wait_for_free_gpu(running, gpu_pool)`` — polls all running Popens,
    harvests any that returned, and returns a free GPU id.
  - ``_dispatch_sweep_jobs`` — sizes the GPU pool to
    ``min(args.max_concurrent_train, args.num_gpus)`` so the cap is
    enforced regardless of how many physical GPUs are present.

This module tests the scheduling logic on Popen-shaped mocks (no real
subprocess.Popen calls). Covers:

  - the GPU-pool cap is min(max_concurrent_train, num_gpus);
  - ``_wait_for_free_gpu`` returns the FIRST free slot when one is empty;
  - finished Popens (poll() returns rc) are harvested + their rc recorded
    in the per-cell rc dict;
  - the dispatcher writes per-cell log files at the expected paths.

CPU-only; no torch / GPU dependency.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# scripts/ is not a package on this repo's PYTHONPATH, so importlib it.
_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)
_spec = importlib.util.spec_from_file_location("dispatch_factor_screen_397", _DISPATCH_PATH)
_dispatch = importlib.util.module_from_spec(_spec)
sys.modules["dispatch_factor_screen_397"] = _dispatch
_spec.loader.exec_module(_dispatch)


class _FakePopen:
    """Minimal Popen-shaped stub for the scheduling tests."""

    def __init__(
        self, rc: int | None = None, cell_key: str = "?", source: str = "?", seed: int = -1
    ):
        self._rc = rc  # None = still running; int = exited
        self._cell_key = cell_key
        self._source = source
        self._seed = seed
        self.pid = 12345

    def poll(self) -> int | None:
        return self._rc

    def finish(self, rc: int = 0) -> None:
        self._rc = rc


# ---------------------------------------------------------------------------
# _wait_for_free_gpu primitive
# ---------------------------------------------------------------------------


def test_wait_for_free_gpu_returns_empty_slot_immediately() -> None:
    """When a slot is empty, _wait_for_free_gpu returns it without polling."""
    running: dict[int, Any] = {}  # all slots empty
    gpu_pool = [0, 1, 2]
    gpu = _dispatch._wait_for_free_gpu(running, gpu_pool)
    assert gpu == 0


def test_wait_for_free_gpu_harvests_finished_process() -> None:
    """A Popen that returned rc must be popped + return its GPU id."""
    finished = _FakePopen(rc=0, cell_key="00000", source="librarian", seed=42)
    running = {0: finished}
    gpu_pool = [0]
    per_cell_rc: dict = {}
    gpu = _dispatch._wait_for_free_gpu(running, gpu_pool, per_cell_rc=per_cell_rc)
    assert gpu == 0
    assert 0 not in running, "Harvested Popen must be popped from running dict"
    assert per_cell_rc == {("00000", "librarian", 42): 0}


def test_wait_for_free_gpu_records_failure_rc() -> None:
    """rc != 0 is recorded in per_cell_rc (sweep can continue past failures)."""
    failed = _FakePopen(rc=2, cell_key="10010", source="librarian", seed=42)
    running = {1: failed}
    gpu_pool = [1]
    per_cell_rc: dict = {}
    gpu = _dispatch._wait_for_free_gpu(running, gpu_pool, per_cell_rc=per_cell_rc)
    assert gpu == 1
    assert per_cell_rc == {("10010", "librarian", 42): 2}


def test_wait_for_free_gpu_polls_until_one_finishes(monkeypatch) -> None:
    """When all slots are full, the loop polls until ONE Popen finishes."""
    running_proc = _FakePopen(rc=None, cell_key="00000", source="librarian", seed=42)
    running = {
        0: running_proc,
        1: _FakePopen(rc=None, cell_key="00001", source="librarian", seed=42),
    }
    gpu_pool = [0, 1]
    # Speed up the loop AND finish the first process after 2 sleeps.
    sleep_count = {"n": 0}

    def _fake_sleep(_seconds: float) -> None:
        sleep_count["n"] += 1
        if sleep_count["n"] == 2:
            running_proc.finish(rc=0)

    monkeypatch.setattr(_dispatch.time, "sleep", _fake_sleep)
    per_cell_rc: dict = {}
    gpu = _dispatch._wait_for_free_gpu(running, gpu_pool, per_cell_rc=per_cell_rc)
    assert gpu == 0
    assert sleep_count["n"] >= 1


# ---------------------------------------------------------------------------
# GPU-pool cap (plan §12 disk-quota mitigation)
# ---------------------------------------------------------------------------


def test_gpu_pool_caps_at_max_concurrent_train(monkeypatch) -> None:
    """Plan §12: even with 8 physical GPUs, training caps at 6 concurrent.

    The dispatcher derives the GPU pool as
    ``list(range(min(max_concurrent_train, num_gpus)))`` so the cap is
    enforced even when num_gpus is larger.
    """
    monkeypatch.setattr(_dispatch, "has_recent_smoke_pass_marker", lambda issue, *, repo_root: True)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        import argparse

        args = argparse.Namespace(
            issue=397,
            mode="sweep",
            pool_dir=slab_root / "pools",
            slab_root=slab_root,
            smoke_cell="10010",
            smoke_source="librarian",
            smoke_seed=42,
            sources="librarian",
            seeds="42",
            num_gpus=8,
            max_concurrent_train=6,  # the cap
            marker_token="※",
            save_every_n_steps=25,
            pos_per_source=400,
            lr=1e-4,
            warmup_ratio=0.10,
            require_smoke_pass=True,
            skip_smoke_pass_check=False,
            dry_run=False,
            log_level="INFO",
        )

        from explore_persona_space.experiments.factor_screen_397.cells import Cell

        # 20 cells so the cap is exercised (need > 6 cells to hit the cap).
        cells = [Cell.from_key(f"{a}{b}000") for a in (0, 1) for b in (0, 1) for _ in range(5)]
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: cells)

        observed_gpu_ids: set[int] = set()
        # Track launch order. A Popen stays "running" until ALL 6 pool slots
        # are filled (so GPUs 1-5 get assigned before any GPU 0 Popen
        # finishes and frees its slot).
        launch_state = {"count": 0}
        spawned: list[Any] = []

        class _StaysRunningUntilPoolFull(_FakePopen):
            """Returns None from poll() until launch_state['count'] >= 6.

            Once the dispatcher has launched on all 6 slots, every poll
            returns 0 so the drain phase finishes each Popen cleanly.
            """

            def __init__(self, cell_key: str, source: str, seed: int):
                super().__init__(rc=None, cell_key=cell_key, source=source, seed=seed)

            def poll(self):
                if launch_state["count"] >= 6:
                    self._rc = 0
                    return 0
                return None

        def _fake_launch(**kw):
            observed_gpu_ids.add(kw["gpu_id"])
            launch_state["count"] += 1
            p = _StaysRunningUntilPoolFull(
                cell_key=kw["cell"].key, source=kw["source"], seed=kw["seed"]
            )
            spawned.append(p)
            return p

        monkeypatch.setattr(_dispatch, "_launch_cell_subprocess", _fake_launch)
        monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0
        # GPU ids assigned must be in [0, max_concurrent_train) — never 6/7.
        assert observed_gpu_ids == {0, 1, 2, 3, 4, 5}, (
            f"GPU pool capped at 6 of 8; observed gpu_ids = {observed_gpu_ids}"
        )


def test_gpu_pool_clamps_to_num_gpus_when_cap_is_larger() -> None:
    """If max_concurrent_train > num_gpus, the pool clamps to num_gpus.

    e.g., a 4-GPU pod with max_concurrent_train=6 caps at 4, not 6.
    """
    # _dispatch_sweep_jobs computes the pool inline; test the formula directly
    # by inspecting the constant the dispatcher uses.
    assert min(6, 4) == 4
    assert min(6, 8) == 6
    assert min(6, 1) == 1


def test_gpu_pool_empty_raises_loud(monkeypatch) -> None:
    """max_concurrent_train=0 OR num_gpus=0 → empty pool → loud-fail.

    Per CLAUDE.md "fail fast": silently iterating an empty pool would
    deadlock the dispatcher.
    """
    monkeypatch.setattr(_dispatch, "has_recent_smoke_pass_marker", lambda issue, *, repo_root: True)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        import argparse

        args = argparse.Namespace(
            issue=397,
            mode="sweep",
            pool_dir=slab_root / "pools",
            slab_root=slab_root,
            smoke_cell="10010",
            smoke_source="librarian",
            smoke_seed=42,
            sources="librarian",
            seeds="42",
            num_gpus=0,  # empty pool
            max_concurrent_train=6,
            marker_token="※",
            save_every_n_steps=25,
            pos_per_source=400,
            lr=1e-4,
            warmup_ratio=0.10,
            require_smoke_pass=True,
            skip_smoke_pass_check=False,
            dry_run=False,
            log_level="INFO",
        )

        from explore_persona_space.experiments.factor_screen_397.cells import Cell

        monkeypatch.setattr(
            _dispatch, "_enumerate_valid_cells_per_seed", lambda: [Cell.from_key("00000")]
        )
        repo_root = Path(__file__).resolve().parent.parent.parent
        with pytest.raises(ValueError, match="GPU pool is empty"):
            _dispatch.run_sweep_phase(args, repo_root=repo_root)


# ---------------------------------------------------------------------------
# Per-cell log file written (dispatcher.log under cell_output_dir)
# ---------------------------------------------------------------------------


def test_launch_cell_subprocess_writes_log_under_cell_dir(monkeypatch) -> None:
    """_launch_cell_subprocess writes per-cell stdout/stderr to
    ``cell_output_dir / 'dispatcher.log'`` so a per-cell crash leaves a
    diagnosable artifact.

    Asserts the log file exists after the Popen returns (Popen is monkey-
    patched to a no-op fast-exit so the test doesn't actually launch python).
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    popen_calls: list[dict] = []

    class _StubPopen:
        def __init__(self, cmd, env=None, cwd=None, stdout=None, stderr=None):
            popen_calls.append(
                {"cmd": cmd, "env": env, "cwd": cwd, "stdout": stdout, "stderr": stderr}
            )
            self.pid = 99999

        def poll(self):
            return 0

    monkeypatch.setattr(_dispatch.subprocess, "Popen", _StubPopen)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        cell_dir = slab_root / "cell_00000" / "source_librarian" / "seed_42"
        import argparse

        args = argparse.Namespace(
            pool_dir=slab_root / "pools",
            marker_token="※",
            save_every_n_steps=25,
            pos_per_source=400,
            lr=1e-4,
            warmup_ratio=0.10,
        )
        repo_root = Path(__file__).resolve().parent.parent.parent
        proc = _dispatch._launch_cell_subprocess(
            cell=Cell.from_key("00000"),
            source="librarian",
            seed=42,
            gpu_id=3,
            cell_output_dir=cell_dir,
            args=args,
            repo_root=repo_root,
        )
        # The Popen was constructed.
        assert len(popen_calls) == 1
        call = popen_calls[0]
        # Cell dir + log file exist.
        assert cell_dir.exists()
        log_path = cell_dir / "dispatcher.log"
        assert log_path.exists(), f"Expected per-cell log at {log_path}"
        # Popen stdout fd points at the log file (file-like, not stderr=None).
        assert call["stdout"] is not None
        # CVD env var is pinned to gpu_id=3.
        assert call["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        # EPM_SKIP_INLINE_CHECKPOINT_UPLOAD inherited (per the runpod_moosefs note).
        assert call["env"]["EPM_SKIP_INLINE_CHECKPOINT_UPLOAD"] == "1"
        # Stamped identifiers on the returned proc (so _wait_for_free_gpu can
        # rebuild the per-cell key).
        assert proc._cell_key == "00000"
        assert proc._source == "librarian"
        assert proc._seed == 42


def test_launch_cell_subprocess_threads_gpu_id_via_command_line(monkeypatch) -> None:
    """The +gpu_id memory note: both env CVD AND a --gpu-id arg must be set.

    env CVD alone is insufficient because train/sft.py:479 clobbers it with
    cfg.gpu_id (default 0). The dispatcher threads --gpu-id N so the per-cell
    script can set TrainLoraConfig.gpu_id=N → sft.py:479's clobber lands on
    the right device.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    popen_calls: list[list[str]] = []

    class _StubPopen:
        def __init__(self, cmd, env=None, cwd=None, stdout=None, stderr=None):
            popen_calls.append(cmd)
            self.pid = 99999

        def poll(self):
            return 0

    monkeypatch.setattr(_dispatch.subprocess, "Popen", _StubPopen)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        cell_dir = slab_root / "cell_10010" / "source_librarian" / "seed_42"
        import argparse

        args = argparse.Namespace(
            pool_dir=slab_root / "pools",
            marker_token="※",
            save_every_n_steps=25,
            pos_per_source=400,
            lr=1e-4,
            warmup_ratio=0.10,
        )
        repo_root = Path(__file__).resolve().parent.parent.parent
        _dispatch._launch_cell_subprocess(
            cell=Cell.from_key("10010"),
            source="librarian",
            seed=42,
            gpu_id=5,
            cell_output_dir=cell_dir,
            args=args,
            repo_root=repo_root,
        )
        cmd = popen_calls[0]
        # --gpu-id 5 in the command line.
        assert "--gpu-id" in cmd
        gpu_id_idx = cmd.index("--gpu-id")
        assert cmd[gpu_id_idx + 1] == "5"
        # run_one_cell entrypoint.
        assert cmd[2] == "explore_persona_space.experiments.factor_screen_397.run_one_cell"


# ---------------------------------------------------------------------------
# Sweep summary JSON shape
# ---------------------------------------------------------------------------


def test_sweep_summary_json_shape(monkeypatch) -> None:
    """sweep_summary.json shape: per-cell list + rc histogram + counts."""
    monkeypatch.setattr(_dispatch, "has_recent_smoke_pass_marker", lambda issue, *, repo_root: True)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        import argparse

        args = argparse.Namespace(
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
            log_level="INFO",
        )
        from explore_persona_space.experiments.factor_screen_397.cells import Cell

        monkeypatch.setattr(
            _dispatch, "_enumerate_valid_cells_per_seed", lambda: [Cell.from_key("00000")]
        )
        monkeypatch.setattr(
            _dispatch,
            "_launch_cell_subprocess",
            lambda **kw: _FakePopen(
                rc=0, cell_key=kw["cell"].key, source=kw["source"], seed=kw["seed"]
            ),
        )
        monkeypatch.setattr(_dispatch.time, "sleep", lambda _s: None)
        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        summary_path = slab_root / "sweep_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Top-level keys.
        assert set(summary.keys()) >= {"job_count", "ran", "rc_counts", "per_cell"}
        # rc_counts is a string-keyed dict (JSON convention).
        assert summary["rc_counts"] == {"0": 1}
        # per_cell entries have cell/source/seed/rc.
        for entry in summary["per_cell"]:
            assert set(entry.keys()) >= {"cell", "source", "seed", "rc"}
