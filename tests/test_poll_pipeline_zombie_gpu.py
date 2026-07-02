"""Zombie-GPU-allocation stall detection (#664).

A hung vLLM whose CUDA worker process DIED but whose EngineCore main
process is still alive presents as a compute process holding many GiB of
VRAM whose PID no longer exists in ``/proc``. The main Python process keeps
burning Python-overhead CPU (HTTP keepalive, GIL ticks, network-thread-pool
idle work), so the #518/#658 session-CPU-advancing override keeps the
verdict in ``running`` indefinitely while zero real work happens (#664
round 8 hung 60+ min, reported healthy throughout).

Since #826 the override is namespace-robust: on host-PID-namespace RunPod
containers nvidia-smi reports HOST PIDs unresolvable in the container's
``/proc``, so every HEALTHY worker carries the zombie signature (#816
steady-state; #778 transient teardown-window PID). The override now fires
only when ALL workload logs are stale past the effective stall window
(``max(ZOMBIE_VETO_FRESH_SEC, stall_sec)``) AND the stale-log candidate
persisted 2 consecutive observed ticks (``zombie_streak`` sidecar key).

These tests pin:

* the probe-output parser (``_parse_probe_stdout``) lifting the new
  ``ZOMBIE_GPU_PIDS`` line into ``zombie_gpu_pids``;
* ``poll_once`` overriding a would-be ``running`` verdict to ``stalled``
  with ``stall_reason="vllm_worker_dead_zombie_gpu"`` when a STALE-LOG
  zombie GPU allocation persisted 2 consecutive ticks AND the
  CPU-advancing override would otherwise have rescued the stall
  conjunction to ``running`` (#664 true positive, fires by tick 2);
* the #826 liveness veto: any workload log fresh within the effective
  stall window ⇒ never flags, streak resets (#816/#778 false positives,
  including the sparse-log 60s-to-stall_sec window);
* the healthy case (no zombies) leaving the CPU-override rescue intact;
* the override NEVER firing on a ``done`` verdict;
* the JSON surface (``poll_pipeline.main`` + ``backend_poll`` serializer)
  carrying ``stall_reason``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors
    ``tests/test_poll_next_interval.py``'s loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_zombie_gpu_under_test")
bp = _load_script_module("backend_poll.py", "backend_poll_zombie_gpu_under_test")


# ── probe-output parser ───────────────────────────────────────────────────────


def test_parse_probe_stdout_lifts_zombie_pids() -> None:
    """The parser dispatches the ``ZOMBIE_GPU_PIDS=`` line into the
    ``zombie_gpu_pids`` key (space-separated PIDs)."""
    parsed = pp._parse_probe_stdout(
        "\n".join(
            [
                "PID_ALIVE=1",
                "MTIME_EPOCH=123",
                "TAIL_START",
                "TAIL_END",
                "GPU_UTIL=0",
                "ZOMBIE_GPU_PIDS=1262130 1262131",
                "SESSION_CPU_SECS=4271.5",
            ]
        )
    )
    assert parsed["zombie_gpu_pids"] == "1262130 1262131"


def test_parse_probe_stdout_zombie_default_empty() -> None:
    """A probe stdout with no ``ZOMBIE_GPU_PIDS`` line (older probe) defaults
    the key to empty — i.e. healthy, no zombies."""
    parsed = pp._parse_probe_stdout("PID_ALIVE=1\nGPU_UTIL=0\n")
    assert parsed["zombie_gpu_pids"] == ""


def test_parse_probe_stdout_zombie_empty_line() -> None:
    """A bare ``ZOMBIE_GPU_PIDS=`` line (probe ran, found no zombies) parses
    to empty, not to the literal."""
    parsed = pp._parse_probe_stdout("PID_ALIVE=1\nGPU_UTIL=95\nZOMBIE_GPU_PIDS=\n")
    assert parsed["zombie_gpu_pids"] == ""


# ── poll_once wiring ──────────────────────────────────────────────────────────


def _probe_stdout(
    *,
    mtime_epoch: int,
    tail: str,
    gpu_util: str,
    session_cpu: str,
    zombie_pids: str,
    phase_log_mtime_epoch: int = 0,
) -> str:
    """Probe stdout in the shape ``_parse_probe_stdout`` expects."""
    return "\n".join(
        [
            "PID_FILE_MISSING=0",
            "PID_ALIVE=1",
            f"MTIME_EPOCH={mtime_epoch}",
            "TAIL_START",
            tail,
            "TAIL_END",
            "CELL_MTIME_EPOCH=0",
            "CELL_TAIL_START",
            "CELL_TAIL_END",
            f"PHASE_LOG_MTIME_EPOCH={phase_log_mtime_epoch}",
            "SHARD_LOG_MTIME_EPOCH=0",
            f"GPU_UTIL={gpu_util}",
            f"ZOMBIE_GPU_PIDS={zombie_pids}",
            f"SESSION_CPU_SECS={session_cpu}",
            "RESULTS_SENTINEL_PRESENT=0",
        ]
    )


def _patch_pod(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mtime_epoch: int,
    tail: str,
    gpu_util: str,
    session_cpu: str,
    zombie_pids: str,
    phase_log_mtime_epoch: int = 0,
) -> None:
    """Monkeypatch poll_pipeline's I/O boundary with a fully-controlled probe.

    Stateless per call — two-tick tests re-invoke it between ``poll_once``
    calls to vary the probe (e.g. advance ``session_cpu``, clear
    ``zombie_pids``)."""

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        stdout = (
            ""
            if "SENTINEL_START" in remote
            else _probe_stdout(
                mtime_epoch=mtime_epoch,
                tail=tail,
                gpu_util=gpu_util,
                session_cpu=session_cpu,
                zombie_pids=zombie_pids,
                phase_log_mtime_epoch=phase_log_mtime_epoch,
            )
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: 10800.0)


def _stale_state(now: int, *, prev_cpu: str, zombie_streak: str = "0") -> str:
    """A prior-tick state file: phase already seen (so no transition), GPUs
    idle, with a prior session-CPU sample BELOW the current one so the
    #518/#658 override sees CPU advancing. ``zombie_streak`` pre-seeds the
    #826 persistence counter (``"1"`` makes a single ``poll_once`` call
    represent tick 2 of a persisted stale-log zombie candidate)."""
    return json.dumps(
        {
            "9664": {
                "phase": "training",
                "last_phase_change_epoch": str(now - 7200),
                "session_cpu_secs": prev_cpu,
                "max_cpu_secs": prev_cpu,
                "zombie_streak": zombie_streak,
            }
        }
    )


def _saved_zombie_streak(state_file: Path) -> str:
    """Read back the persisted #826 streak for issue 9664."""
    return json.loads(state_file.read_text())["9664"]["zombie_streak"]


def test_zombie_gpu_overrides_cpu_advancing_running_to_stalled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exact #664 regime: stale logs + idle GPUs (stall conjunction met)
    + session CPU advancing (override would rescue to running) + a zombie
    GPU allocation persisted from the prior tick (#826: ``zombie_streak``
    pre-seeded to "1", so this single call represents tick 2). The zombie
    override wins -> stalled with the reason. The genuine two-call replay
    lives in ``test_zombie_stale_log_defers_first_tick_then_stalls_second``;
    this adaptation additionally pins the sidecar READ path."""
    now = int(time.time())
    # Main log 2000s old (> 900s stall_sec); no fresh phase/shard logs.
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",  # idle -> stall conjunction can be met
        session_cpu="5000.0",  # advancing vs prev 4000.0
        zombie_pids="1262130",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(_stale_state(now, prev_cpu="4000.0", zombie_streak="1"))
    result = pp.poll_once(
        issue=9664,
        pod="pod-9664",
        log_path="/workspace/logs/issue-9664.log",
        pid_file="/workspace/logs/issue-9664.pid",
        state_file=state_file,
    )
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_no_zombie_leaves_cpu_advancing_override_intact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Identical regime WITHOUT a zombie allocation: the #518/#658
    CPU-advancing override still rescues the stall conjunction to running,
    and stall_reason stays None (proves the override is zombie-gated, not a
    blanket re-stall of every CPU-bound phase)."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5000.0",
        zombie_pids="",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(_stale_state(now, prev_cpu="4000.0"))
    result = pp.poll_once(
        issue=9664,
        pod="pod-9664",
        log_path="/workspace/logs/issue-9664.log",
        pid_file="/workspace/logs/issue-9664.pid",
        state_file=state_file,
    )
    assert result.status == "running"
    assert result.stall_reason is None


def test_zombie_does_not_override_done_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A corroborated ``done`` verdict (pid dead via the demotion guard, or
    a results sentinel) is terminal and correct — the zombie override must
    NOT flip it back to stalled. Here the log shows a corroborated done
    (pid alive but results sentinel present is faked via a done tail with
    the pid-dead path); we assert the override leaves ``done`` alone."""
    now = int(time.time())

    # Build a probe whose pid is DEAD and whose log shows completion, so the
    # verdict is `done` (pid-dead corroborates the done parse, #545), with a
    # zombie allocation still on the card (expected for a dead launcher).
    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        if "SENTINEL_START" in remote:
            stdout = ""
        else:
            stdout = "\n".join(
                [
                    "PID_FILE_MISSING=0",
                    "PID_ALIVE=0",  # launcher exited
                    f"MTIME_EPOCH={now - 10}",
                    "TAIL_START",
                    "2026-06-27 00:00:01 [phase=done] SMOKE COMPLETE",
                    "TAIL_END",
                    "CELL_MTIME_EPOCH=0",
                    "CELL_TAIL_START",
                    "CELL_TAIL_END",
                    "PHASE_LOG_MTIME_EPOCH=0",
                    "SHARD_LOG_MTIME_EPOCH=0",
                    "GPU_UTIL=0",
                    "ZOMBIE_GPU_PIDS=1262130",
                    "SESSION_CPU_SECS=unknown",
                    "RESULTS_SENTINEL_PRESENT=0",
                ]
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: 10800.0)

    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=9664,
        pod="pod-9664",
        log_path="/workspace/logs/issue-9664.log",
        pid_file="/workspace/logs/issue-9664.pid",
        state_file=state_file,
    )
    assert result.status == "done"
    assert result.stall_reason is None


# ── #826 liveness veto + 2-tick persistence ───────────────────────────────────


def test_zombie_fresh_log_vetoes_and_resets_streak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#816 steady-state replay via the PHASE-log freshness path: zombie PIDs
    present on BOTH of two consecutive ticks while the per-phase log is fresh
    (~5s) — the healthy host-PID-namespace signature. Never flags, and a
    pre-seeded streak of "1" is RESET by the fresh-log veto (not just held
    at 0)."""
    now = int(time.time())
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(_stale_state(now, prev_cpu="4000.0", zombie_streak="1"))
    for tick_cpu in ("5000.0", "6000.0"):  # advancing each tick (healthy run)
        _patch_pod(
            monkeypatch,
            mtime_epoch=now - 2000,  # main log quiet; the PHASE log is the fresh signal
            tail="2026-06-27 00:00:01 [phase=training step=5/100]",
            gpu_util="90,88,91,87,93,95,89,92",
            session_cpu=tick_cpu,
            zombie_pids="313516 313517 313518",
            phase_log_mtime_epoch=now - 5,
        )
        result = pp.poll_once(
            issue=9664,
            pod="pod-9664",
            log_path="/workspace/logs/issue-9664.log",
            pid_file="/workspace/logs/issue-9664.pid",
            state_file=state_file,
        )
        assert result.status == "running"
        assert result.stall_reason is None
        assert _saved_zombie_streak(state_file) == "0"


def test_transient_zombie_fresh_log_never_stalls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#778 exact replay: a dying host-namespace PID holds VRAM for ONE tick
    during a vLLM engine teardown/spin-up (log mtime ~8s), gone by the next
    tick. Never flags; the streak stays "0" throughout."""
    now = int(time.time())
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(_stale_state(now, prev_cpu="4000.0"))
    for tick_cpu, pids in (("5000.0", "313516"), ("6000.0", "")):
        _patch_pod(
            monkeypatch,
            mtime_epoch=now - 8,
            tail="2026-06-27 00:00:01 [phase=manyshot_regen step=3/24]",
            gpu_util="0,0,0,0,0,0,0,0",  # engines cycling between phases
            session_cpu=tick_cpu,
            zombie_pids=pids,
        )
        result = pp.poll_once(
            issue=9664,
            pod="pod-9664",
            log_path="/workspace/logs/issue-9664.log",
            pid_file="/workspace/logs/issue-9664.pid",
            state_file=state_file,
        )
        assert result.status == "running"
        assert result.stall_reason is None
        assert _saved_zombie_streak(state_file) == "0"


def test_zombie_sparse_log_window_vetoed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The coupled-threshold pin: logs ~400s old (past the 60s floor, inside
    the 900s stall window — a sparse-log cadence), GPUs BUSY (else-branch
    ``running``), zombie PIDs on both ticks. A fixed-60s veto would have
    fired at tick 2 (the destructive FP on a healthy host-namespace pod);
    the ``max(ZOMBIE_VETO_FRESH_SEC, stall_sec)`` coupling vetoes it."""
    now = int(time.time())
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(_stale_state(now, prev_cpu="4000.0"))
    for tick_cpu in ("5000.0", "6000.0"):
        _patch_pod(
            monkeypatch,
            mtime_epoch=now - 400,
            tail="2026-06-27 00:00:01 [phase=training step=5/100]",
            gpu_util="95,97,93,96,94,98,95,96",  # busy -> conjunction unmet -> running
            session_cpu=tick_cpu,
            zombie_pids="313516 313517",
        )
        result = pp.poll_once(
            issue=9664,
            pod="pod-9664",
            log_path="/workspace/logs/issue-9664.log",
            pid_file="/workspace/logs/issue-9664.pid",
            state_file=state_file,
        )
        assert result.status == "running"
        assert result.stall_reason is None
        assert _saved_zombie_streak(state_file) == "0"


def test_zombie_stale_log_defers_first_tick_then_stalls_second(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#664 true positive, genuine two-call replay: all logs stale (>900s),
    GPUs idle, session CPU advancing on BOTH ticks (the EngineCore idle-burn
    that rescues the conjunction — re-patched upward for tick 2, else the
    high-water mark reads CPU flat and the generic stall path fires before
    the override is reached), zombie PID both ticks. Tick 1 defers (running,
    streak "1"); tick 2 fires (stalled + reason) — requirement (c): the TP
    fires at the 2nd tick at latest."""
    now = int(time.time())
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(_stale_state(now, prev_cpu="4000.0"))

    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5000.0",  # advancing vs prev 4000.0
        zombie_pids="1262130",
    )
    tick1 = pp.poll_once(
        issue=9664,
        pod="pod-9664",
        log_path="/workspace/logs/issue-9664.log",
        pid_file="/workspace/logs/issue-9664.pid",
        state_file=state_file,
    )
    assert tick1.status == "running"
    assert tick1.stall_reason is None
    assert _saved_zombie_streak(state_file) == "1"

    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="6000.0",  # still advancing vs tick-1 max 5000.0
        zombie_pids="1262130",
    )
    tick2 = pp.poll_once(
        issue=9664,
        pod="pod-9664",
        log_path="/workspace/logs/issue-9664.log",
        pid_file="/workspace/logs/issue-9664.pid",
        state_file=state_file,
    )
    assert tick2.status == "stalled"
    assert tick2.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_veto_resets_streak_when_cleared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale-log candidate that CLEARS on the next tick never accumulates:
    tick 1 defers (streak "1"), tick 2 has no zombie (CPU still advancing so
    the rescue keeps ``running``) — streak resets to "0"."""
    now = int(time.time())
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(_stale_state(now, prev_cpu="4000.0"))

    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5000.0",
        zombie_pids="1262130",
    )
    tick1 = pp.poll_once(
        issue=9664,
        pod="pod-9664",
        log_path="/workspace/logs/issue-9664.log",
        pid_file="/workspace/logs/issue-9664.pid",
        state_file=state_file,
    )
    assert tick1.status == "running"
    assert _saved_zombie_streak(state_file) == "1"

    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="6000.0",
        zombie_pids="",
    )
    tick2 = pp.poll_once(
        issue=9664,
        pod="pod-9664",
        log_path="/workspace/logs/issue-9664.log",
        pid_file="/workspace/logs/issue-9664.pid",
        state_file=state_file,
    )
    assert tick2.status == "running"
    assert tick2.stall_reason is None
    assert _saved_zombie_streak(state_file) == "0"


# ── JSON-surface contract ─────────────────────────────────────────────────────


def test_main_json_line_includes_stall_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``poll_pipeline.main`` surfaces ``stall_reason`` in its JSON line so
    the orchestrator can route the zombie stall distinctly."""
    fake = pp.PollResult(
        status="stalled",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=2000,
        pid_alive=True,
        pid_file_missing=False,
        log_tail_excerpt="",
        stall_reason="vllm_worker_dead_zombie_gpu",
    )
    monkeypatch.setattr(pp, "poll_once", lambda **kwargs: fake)
    rc = pp.main(
        [
            "--issue",
            "9664",
            "--pod",
            "pod-9664",
            "--log",
            "/workspace/logs/issue-9664.log",
            "--pid-file",
            "/workspace/logs/issue-9664.pid",
        ]
    )
    assert rc == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["status"] == "stalled"
    assert line["stall_reason"] == "vllm_worker_dead_zombie_gpu"


def test_backend_poll_serializer_passes_stall_reason_through() -> None:
    """``backend_poll._serialize_poll_result`` carries ``stall_reason`` when
    the backends-side result has it."""
    from types import SimpleNamespace

    result = SimpleNamespace(
        status="stalled",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=2000,
        pid_alive=True,
        log_tail_excerpt="",
        gate=None,
        sentinels_processed=0,
        phase_log_mtime_sec_ago=10**9,
        shard_log_mtime_sec_ago=10**9,
        gpu_util="0",
        next_interval=540,
        stall_reason="vllm_worker_dead_zombie_gpu",
    )
    out = bp._serialize_poll_result(result)
    assert out["stall_reason"] == "vllm_worker_dead_zombie_gpu"


def test_backend_poll_serializer_defaults_stall_reason_for_older_results() -> None:
    """A backends-side result with NO ``stall_reason`` attribute (GCP/SLURM
    lanes, or an older module) degrades to ``None`` — never crashes."""
    from types import SimpleNamespace

    result = SimpleNamespace(
        status="running",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=True,
        log_tail_excerpt="",
        gate=None,
        sentinels_processed=0,
        phase_log_mtime_sec_ago=10,
        shard_log_mtime_sec_ago=10**9,
        gpu_util="95",
        next_interval=540,
    )
    out = bp._serialize_poll_result(result)
    assert out["stall_reason"] is None
