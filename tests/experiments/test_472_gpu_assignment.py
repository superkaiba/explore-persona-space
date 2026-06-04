"""Task #472 round-3: GPU-sharding assignment logic (CPU-only).

The full-sweep OOM (round-3) came from every concurrent cell piling onto physical
GPU 0 — the documented ``+gpu_id``/CUDA_VISIBLE_DEVICES clobber in ``train/sft.py``
(it SETS CVD from cfg.gpu_id, default 0, rather than respecting an inherited env
CVD). The fix threads each cell's ASSIGNED physical GPU through ``--gpu-id`` →
``train_one_cell(gpu_id=...)`` → ``cfg.gpu_id`` so sft.py's clobber lands on the
right GPU.

These tests pin the CPU-testable contract WITHOUT a GPU:
  * concurrent cells get DISTINCT physical GPUs (free-GPU pool, never shared);
  * every launched cell-subprocess cmd carries ``--gpu-id <g>``;
  * the dispatcher does NOT also restrict env CUDA_VISIBLE_DEVICES (sft.py owns
    the CVD set, against the full enumeration);
  * ``max_parallel`` is clamped to ``n_gpus`` (can't run more concurrent one-GPU
    cells than there are GPUs without a collision);
  * ``verify_gpu_pin`` fails loud when CVD disagrees with the assigned GPU;
  * ``_summarize_gpu_placements`` reports distinct-GPU usage.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
    verify_gpu_pin,
)


class _FakeProc:
    """A subprocess.Popen stand-in that 'exits 0' on the first poll()."""

    def __init__(self, cmd, env, **_kw):
        self.cmd = list(cmd)
        self.env = dict(env)
        self._polls = 0

    def poll(self):
        # Stay 'running' once so the scheduler observes concurrency, then exit 0.
        self._polls += 1
        return None if self._polls == 1 else 0

    def terminate(self):  # pragma: no cover - only on the failure path
        pass


def _gpu_arg(cmd: list[str]) -> int:
    i = cmd.index("--gpu-id")
    return int(cmd[i + 1])


def _run_pool(monkeypatch, *, cells, seeds, n_gpus, max_parallel, tmp_path):
    """Drive ``_schedule_cell_pool`` with subprocess.Popen + open() faked out."""
    import explore_persona_space  # noqa: F401  (ensure package import path)
    from scripts import dispatch_neg_geometry_472 as disp  # type: ignore[import-not-found]

    launched: list[_FakeProc] = []

    def _fake_popen(cmd, env=None, **kw):
        proc = _FakeProc(cmd, env or {}, **kw)
        launched.append(proc)
        return proc

    monkeypatch.setattr(disp.subprocess, "Popen", _fake_popen)
    # The launcher opens a per-cell log file; redirect to a throwaway under tmp.
    real_open = open

    def _fake_open(path, *a, **k):
        return real_open(tmp_path / "cell.log", "w")

    monkeypatch.setattr("builtins.open", _fake_open)

    results = disp._schedule_cell_pool(
        cells=cells,
        seeds=seeds,
        n_gpus=n_gpus,
        max_parallel=max_parallel,
        slab_root=tmp_path / "slab",
        runs_root=tmp_path / "runs",
        log_dir=tmp_path / "logs",
        bank_path=tmp_path / "bank.json",
        centroids_dir=tmp_path / "cent",
        smoke=True,
        fallback=False,
        no_kl=True,
        report_to="none",
        resume=False,
    )
    return launched, results


def test_each_concurrent_cell_gets_distinct_gpu(monkeypatch, tmp_path):
    """4 cells, 4 GPUs, max_parallel 4 → the 4 concurrent launches use GPUs 0-3."""
    launched, results = _run_pool(
        monkeypatch,
        cells=["a", "b", "c", "d"],
        seeds=[42],
        n_gpus=4,
        max_parallel=4,
        tmp_path=tmp_path,
    )
    gpus = sorted(_gpu_arg(p.cmd) for p in launched)
    assert gpus == [0, 1, 2, 3], gpus  # one distinct GPU per concurrent cell
    assert all(r["status"] == "done" for r in results)
    assert sorted(r["assigned_gpu"] for r in results) == [0, 1, 2, 3]


def test_more_units_than_gpus_reuses_freed_gpus_not_shares(monkeypatch, tmp_path):
    """6 units across 2 GPUs: every launched gpu-id is valid; no concurrent share.

    With max_parallel clamped to n_gpus=2, at most 2 cells run at once and each
    holds a distinct GPU; freed GPUs are reused for the remaining 4 units.
    """
    launched, results = _run_pool(
        monkeypatch,
        cells=["a", "b", "c"],
        seeds=[1, 2],
        n_gpus=2,
        max_parallel=8,  # will be clamped to 2
        tmp_path=tmp_path,
    )
    assert len(launched) == 6
    assert all(_gpu_arg(p.cmd) in (0, 1) for p in launched)
    assert {r["assigned_gpu"] for r in results} <= {0, 1}
    assert all(r["status"] == "done" for r in results)


def test_dispatcher_does_not_restrict_env_cvd(monkeypatch, tmp_path):
    """The cell-launch env must NOT carry a restricted CUDA_VISIBLE_DEVICES.

    sft.py SETS CVD from gpu_id against the full enumeration; if the dispatcher
    also restricted env CVD to the single GPU, str(g) would re-index against the
    1-GPU view and break for g>=1 (round-3 #472 reasoning).
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    launched, _ = _run_pool(
        monkeypatch,
        cells=["a", "b"],
        seeds=[42],
        n_gpus=2,
        max_parallel=2,
        tmp_path=tmp_path,
    )
    for p in launched:
        # Not set to a single-GPU restriction by the dispatcher.
        assert p.env.get("CUDA_VISIBLE_DEVICES", "") == ""


def test_gpu_id_flag_present_in_every_launch(monkeypatch, tmp_path):
    launched, _ = _run_pool(
        monkeypatch,
        cells=["a", "b"],
        seeds=[7],
        n_gpus=2,
        max_parallel=2,
        tmp_path=tmp_path,
    )
    for p in launched:
        assert "--gpu-id" in p.cmd
        assert "scripts/i472_run_cell.py" in p.cmd


def test_verify_gpu_pin_precondition_fails_on_cvd_mismatch(monkeypatch):
    """verify_gpu_pin must reject a CVD that disagrees with the assigned GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="precondition"):
        verify_gpu_pin(3)


def test_verify_gpu_pin_fails_when_gpu_not_in_enumeration(monkeypatch):
    """A bad assignment (gpu index beyond the host enum) fails loud."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import train_cell

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    # Host reports only physical GPUs 0 and 1.
    monkeypatch.setattr(train_cell, "_physical_gpu_uuids", lambda: {0: "GPU-aaa", 1: "GPU-bbb"})
    with pytest.raises(RuntimeError, match="not in host enumeration"):
        verify_gpu_pin(9)


def test_summarize_gpu_placements_flags_distinct_usage():
    from scripts import dispatch_neg_geometry_472 as disp  # type: ignore[import-not-found]

    cell_results = [
        {"cell": "a", "seed": 42, "status": "done", "assigned_gpu": 0},
        {"cell": "b", "seed": 42, "status": "done", "assigned_gpu": 1},
    ]
    summary = disp._summarize_gpu_placements(cell_results, n_gpus=2)
    assert summary["ok"] is True
    assert summary["n_distinct_gpus_used"] == 2
    assert summary["placement"] == {"a_seed42": 0, "b_seed42": 1}


def test_summarize_gpu_placements_single_gpu_is_not_ok():
    """Validation must FAIL if all completed cells landed on ONE GPU."""
    from scripts import dispatch_neg_geometry_472 as disp  # type: ignore[import-not-found]

    cell_results = [
        {"cell": "a", "seed": 42, "status": "done", "assigned_gpu": 0},
        {"cell": "b", "seed": 42, "status": "done", "assigned_gpu": 0},
    ]
    summary = disp._summarize_gpu_placements(cell_results, n_gpus=2)
    assert summary["ok"] is False
    assert summary["n_distinct_gpus_used"] == 1
