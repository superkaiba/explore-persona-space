"""Pod-vs-VM wall-clock skew in log staleness (#704).

``poll_pipeline.poll_once`` computes log staleness by subtracting two
clocks: the pod stamps file mtimes with its OWN wall clock (``stat -c %Y``)
while the VM historically computed "now" with ``datetime.now(tz=UTC)``. A
pod whose NTP is unsynced drifts arbitrarily from the VM clock, so
``VM_now - pod_mtime`` reported a staleness off by the full drift (on
session 09e41486 a ~30s-old log read as 14-33h stale). The #518
CPU-advancing override masked the false-stall verdict, but a removed /
non-applying override would have let the >900s stall path falsely fire.

The fix captures the pod's own "now" (``date +%s``) in the SAME SSH
heredoc that reads the file mtimes and subtracts ``pod_now - pod_mtime`` —
both operands from the same (drifted) pod clock, so the drift cancels
exactly. The VM clock stays in place for the run-age / GPU-idle advisory /
phase-change sidecar consumers that compare against VM-stamped timestamps.

These tests pin:

* a multi-hour pod-vs-VM drift yielding the TRUE ~30s staleness on the
  top-level source, plus sibling assertions for the per-phase and shard
  sources driven as the dominant ``max()`` operand under the same drift;
* the backward-compat path (a legacy probe omitting ``POD_NOW_EPOCH``
  falls back to the VM clock AND logs a WARN);
* the heredoc emitting the new ``POD_NOW_EPOCH=$(date +%s)`` line;
* the CELL source driven as the dominant ``max()`` operand under drift,
  pinning that ALL four mtime sources stay on the pod clock.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors
    ``tests/test_poll_pipeline_zombie_gpu.py``'s loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_clock_skew_under_test")


# ── probe builders ─────────────────────────────────────────────────────────────


def _probe_stdout(
    *,
    mtime_epoch: int,
    cell_mtime_epoch: int = 0,
    phase_log_mtime_epoch: int = 0,
    shard_log_mtime_epoch: int = 0,
    pod_now_epoch: int | None,
    tail: str,
    gpu_util: str,
    session_cpu: str = "unknown",
) -> str:
    """Probe stdout in the shape ``_parse_probe_stdout`` expects.

    ``pod_now_epoch=None`` OMITS the ``POD_NOW_EPOCH`` line entirely (the
    legacy-image / fallback path); any int emits it.
    """
    lines = [
        "PID_FILE_MISSING=0",
        "PID_ALIVE=1",
        f"MTIME_EPOCH={mtime_epoch}",
    ]
    if pod_now_epoch is not None:
        lines.append(f"POD_NOW_EPOCH={pod_now_epoch}")
    lines += [
        "TAIL_START",
        tail,
        "TAIL_END",
        f"CELL_MTIME_EPOCH={cell_mtime_epoch}",
        "CELL_TAIL_START",
        "CELL_TAIL_END",
        f"PHASE_LOG_MTIME_EPOCH={phase_log_mtime_epoch}",
        f"SHARD_LOG_MTIME_EPOCH={shard_log_mtime_epoch}",
        f"GPU_UTIL={gpu_util}",
        "ZOMBIE_GPU_PIDS=",
        f"SESSION_CPU_SECS={session_cpu}",
        "RESULTS_SENTINEL_PRESENT=0",
    ]
    return "\n".join(lines)


def _patch_pod(monkeypatch: pytest.MonkeyPatch, *, probe_kwargs: dict[str, Any]) -> None:
    """Monkeypatch poll_pipeline's I/O boundary with a fully-controlled probe.

    Mirrors ``tests/test_poll_pipeline_zombie_gpu.py::_patch_pod`` — the
    sentinel-drain SSH call (``SENTINEL_START`` in the remote command)
    returns empty; the probe call returns the controlled stdout.
    """

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        stdout = "" if "SENTINEL_START" in remote else _probe_stdout(**probe_kwargs)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: 10800.0)


# A non-`done`, non-`gate` tail so the verdict is the normal liveness path
# (not a wedge/dead/done short-circuit) and ``poll_once`` reads the staleness
# field rather than returning early.
_RUNNING_TAIL = "2026-06-27 00:00:01 [phase=training step=5/100]"
# 14h pod-vs-VM drift, large enough that a cross-clock subtraction would read
# ~14*3600 = 50400s instead of the true ~30s.
_DRIFT_SEC = 14 * 3600


# ── 1. pod-clock drift → true staleness (top-level + phase + shard) ─────────────


def test_drifted_pod_clock_yields_true_staleness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fresh pod mtime under a 14h pod-vs-VM drift reads as the TRUE ~30s
    staleness, not ~14h and not 0. Bounded-equality (``abs(... - 30) <= 2``)
    rejects BOTH the cross-clock skew AND a buggy all-zero short-circuit.
    Sibling assertions drive the per-phase and shard sources as the dominant
    ``max()`` operand under the same drift (hard rule: all four sources on
    the pod clock)."""
    vm_now = int(time.time())
    pod_now = vm_now - _DRIFT_SEC  # pod clock 14h BEHIND the VM clock
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=pod_now - 30,  # top-level log written 30 pod-seconds ago
            phase_log_mtime_epoch=pod_now - 30,  # phase + shard each fresh on the pod clock
            shard_log_mtime_epoch=pod_now - 30,
            pod_now_epoch=pod_now,
            tail=_RUNNING_TAIL,
            gpu_util="95",  # non-idle → no stall conjunction
        ),
    )
    result = pp.poll_once(
        issue=9704,
        pod="pod-9704",
        log_path="/workspace/logs/issue-9704.log",
        pid_file="/workspace/logs/issue-9704.pid",
        state_file=tmp_path / "poll-state.json",
    )
    # Top-level (via freshest_mtime_epoch), per-phase, and shard all on the
    # pod clock → each bounded-equal to the true ~30s.
    assert abs(result.last_log_mtime_sec_ago - 30) <= 2, result.last_log_mtime_sec_ago
    assert abs(result.phase_log_mtime_sec_ago - 30) <= 2, result.phase_log_mtime_sec_ago
    assert abs(result.shard_log_mtime_sec_ago - 30) <= 2, result.shard_log_mtime_sec_ago


# ── 2. legacy probe (no POD_NOW_EPOCH) → VM-clock fallback + WARN ───────────────


def test_missing_pod_now_falls_back_to_vm_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A legacy probe that OMITS ``POD_NOW_EPOCH`` falls back to the VM clock
    (pre-#704 behavior) and logs a WARN. Uses a TOLERANCE form (not
    bounded-equality) because the VM-clock branch reads the live
    ``datetime.now(tz=UTC)``, which skews slightly from the test's
    ``time.time()`` setup."""
    vm_now = int(time.time())
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=vm_now - 2000,  # 2000s old on the VM clock
            pod_now_epoch=None,  # legacy image: line omitted entirely
            tail=_RUNNING_TAIL,
            gpu_util="95",
        ),
    )
    with caplog.at_level(logging.WARNING, logger="poll_pipeline"):
        result = pp.poll_once(
            issue=9704,
            pod="pod-9704",
            log_path="/workspace/logs/issue-9704.log",
            pid_file="/workspace/logs/issue-9704.pid",
            state_file=tmp_path / "poll-state.json",
        )
    assert abs(result.last_log_mtime_sec_ago - 2000) <= 60, result.last_log_mtime_sec_ago
    assert any("missing POD_NOW_EPOCH" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


# ── 3. heredoc emits the new POD_NOW_EPOCH line ─────────────────────────────────


def test_heredoc_emits_pod_now_epoch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The probe heredoc (the remote command in the ssh ``subprocess.run``
    call) emits ``POD_NOW_EPOCH=$(date +%s)``. Captured via ``cmd[-1]`` on
    the probe call (mirrors the zombie-GPU test's capture pattern)."""
    captured: dict[str, str] = {}
    vm_now = int(time.time())

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        if "SENTINEL_START" in remote:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        captured["heredoc"] = remote
        stdout = _probe_stdout(
            mtime_epoch=vm_now - 30,
            pod_now_epoch=vm_now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: 10800.0)

    pp.poll_once(
        issue=9704,
        pod="pod-9704",
        log_path="/workspace/logs/issue-9704.log",
        pid_file="/workspace/logs/issue-9704.pid",
        state_file=tmp_path / "poll-state.json",
    )
    assert "heredoc" in captured, "probe ssh call was never made"
    assert re.search(r"POD_NOW_EPOCH=\$\(date \+%s\)", captured["heredoc"]), captured["heredoc"]


# ── 4. cell source dominant under drift → true staleness ────────────────────────


def test_cell_log_dominant_under_drift_yields_true_staleness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Drive the CELL source as the dominant ``max()`` operand under the same
    14h drift (top-level STALE so the cell wins
    ``freshest_mtime_epoch = max(mtime, cell_mtime)``). Pins that the cell
    source is subtracted on the pod clock too — a regression that read it
    against the VM clock would yield ~14h here, and an all-zero short-circuit
    would yield 0; bounded-equality rejects both."""
    vm_now = int(time.time())
    pod_now = vm_now - _DRIFT_SEC
    _patch_pod(
        monkeypatch,
        probe_kwargs=dict(
            mtime_epoch=pod_now - 7200,  # top-level 2h old on the pod clock → cell wins max()
            cell_mtime_epoch=pod_now - 30,  # cell fresh on the pod clock (dominant operand)
            pod_now_epoch=pod_now,
            tail=_RUNNING_TAIL,
            gpu_util="95",
        ),
    )
    result = pp.poll_once(
        issue=9704,
        pod="pod-9704",
        log_path="/workspace/logs/issue-9704.log",
        pid_file="/workspace/logs/issue-9704.pid",
        state_file=tmp_path / "poll-state.json",
    )
    assert abs(result.last_log_mtime_sec_ago - 30) <= 2, result.last_log_mtime_sec_ago
