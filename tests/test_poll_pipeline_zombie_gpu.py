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

Since #864 the override additionally carries a namespace-informativeness
gate: the probe counts ``GPU_PIDS_TOTAL`` / ``GPU_PIDS_RESOLVABLE`` and
(zombie-candidate-guarded) ``NVIDIA_UVM_LIVE_HOLDERS`` — live container
processes holding an EXACT ``/dev/nvidia-uvm`` fd. When
``total > 0 AND resolvable == 0 AND uvm > 0`` the dead-in-/proc signature
is a PID-namespace artifact (the flagged PIDs ARE live workers under host
ids — the #813 false positive: a healthy ~29-min CPU-bound quiet stretch
outlived the #826 stale-log veto) and the override is vetoed regardless of
log staleness. Gated by ``ZOMBIE_NAMESPACE_VETO_ENABLED``
(``EPM_ZOMBIE_NAMESPACE_VETO``; ships default-OFF per the #864 live-pod
gate disposition).

Since #951 the override additionally carries a material-compute liveness
veto: when the per-tick session-CPU burn rate (delta of the persisted
``session_cpu_secs`` / ``session_cpu_sample_epoch`` sidecar pair over the
measured tick spacing) was >= ``ZOMBIE_OVERRIDE_CPU_CORES_MIN`` (default
0.5 cores) on BOTH the current and the previous persisted tick, the
session is demonstrably computing — the #825 false stall burned ~1.83-2.04
cores next to 1816 MiB of prior-run VRAM leftover — and the override is
vetoed (streak reset, like the other vetoes). #664's hung-EngineCore churn
(~0.22 cores) stays below the threshold, so the true positive keeps
firing; every degraded input (unknown sample, missing epoch/rate, tick
spacing under ``ZOMBIE_CPU_RATE_MIN_DT_SEC``, negative delta) leaves the
veto inert (pre-#951 behavior).

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
  carrying ``stall_reason``;
* the #864 namespace-informativeness gate: the parser lifting the three
  count keys; the #813-shape veto (running + streak reset + the
  ``cpu_override_active`` passthrough); the #664 total collapse and the
  matched-namespace partial death still firing WITH the gate enabled;
  degraded ``uvm`` / ``resolvable`` reads falling back to pure #826 (the
  fail-toward-current-behavior direction); the
  ``EPM_ZOMBIE_NAMESPACE_VETO`` kill-switch; the producer-side probe
  emission / key-parity / exact end-anchored ``/dev/nvidia-uvm`` matcher
  pin (a heredoc typo must not leave the gate silently inert); and
  ``_parse_probe_count`` unit behavior;
* the #951 material-compute veto: ``_session_cpu_rate_cores`` unit
  behavior (happy path, fail-safe inputs, dt floor, negative delta); the
  #825 replay vetoing (seeded with ``max_cpu_secs`` DISTINCT from the raw
  sample so wrong-prev-key rate wiring flips the outcome); the #664
  low-churn replay still firing; BOTH both-ticks conjuncts (single-tick
  material AND prev-material/now-low each still fire); the
  exact-threshold ``>=`` boundary; missing prev-rate / missing
  sample-epoch / negative-delta fall-backs to current behavior; the veto
  RESETTING (not holding) the streak; the direct-call default
  (``session_cpu_rate_cores=None``) preserving pre-#951 outputs; and
  ``_save_state`` persisting the ``session_cpu_sample_epoch`` /
  ``session_cpu_rate_cores`` pair;
* the #1477 negative-rate pcpu confirm veto: the corrected
  ``[DD-]HH:MM:SS`` awk day parse (real-awk subprocess pins — the #1345
  root cause: ``"1-02"`` numerically coerced to ``1`` collapsed the
  session sum at the 86400-s boundary, sawtoothing the cross-tick rate
  negative on a healthy 20-core worker); the #1345 replays vetoing on a
  parsed negative sample on EITHER tick + same-tick session pcpu >=
  ``ZOMBIE_OVERRIDE_CPU_CORES_MIN``; the unknown/low-pcpu and
  no-negative-evidence fall-throughs (the #664 true positive, the
  frozen-counter rate-0.0 hang, and warmup all keep today's behavior
  verbatim); ``_parse_session_pcpu_cores`` units; the probe emitting
  ``SESSION_PCPU_TOTAL`` from a SEPARATE ``ps`` pipeline; and the
  direct-call default (``session_pcpu_cores=None``) preserving pre-#1477
  outputs.
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
    gpu_pids_total: str = "unknown",
    gpu_pids_resolvable: str = "unknown",
    uvm_live_holders: str = "unknown",
    session_pcpu: str = "unknown",
) -> str:
    """Probe stdout in the shape ``_parse_probe_stdout`` expects.

    The #864 count kwargs — and the #1477 ``session_pcpu`` kwarg — default
    ``"unknown"`` (the degraded-probe read), so every pre-#864 / pre-#1477
    test exercises the fall-through-to-#826 path UNMODIFIED (acceptance
    criterion: existing tests pass unchanged)."""
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
            f"GPU_PIDS_TOTAL={gpu_pids_total}",
            f"GPU_PIDS_RESOLVABLE={gpu_pids_resolvable}",
            f"NVIDIA_UVM_LIVE_HOLDERS={uvm_live_holders}",
            f"SESSION_CPU_SECS={session_cpu}",
            f"SESSION_PCPU_TOTAL={session_pcpu}",
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
    gpu_pids_total: str = "unknown",
    gpu_pids_resolvable: str = "unknown",
    uvm_live_holders: str = "unknown",
    session_pcpu: str = "unknown",
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
                gpu_pids_total=gpu_pids_total,
                gpu_pids_resolvable=gpu_pids_resolvable,
                uvm_live_holders=uvm_live_holders,
                session_pcpu=session_pcpu,
            )
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: 10800.0)


def _stale_state(
    now: int,
    *,
    prev_cpu: str,
    zombie_streak: str = "0",
    max_cpu: str | None = None,
    session_cpu_sample_epoch: str | None = None,
    session_cpu_rate_cores: str | None = None,
) -> str:
    """A prior-tick state file: phase already seen (so no transition), GPUs
    idle, with a prior session-CPU sample BELOW the current one so the
    #518/#658 override sees CPU advancing. ``zombie_streak`` pre-seeds the
    #826 persistence counter (``"1"`` makes a single ``poll_once`` call
    represent tick 2 of a persisted stale-log zombie candidate).

    The #951 kwargs are only added to the dict when non-None, so pre-#951
    callers produce a byte-identical seed (key-absence = the #951 veto is
    inert on them). ``max_cpu`` (default: ``prev_cpu``) lets the #825 replay
    seed a rolling max DISTINCT from the raw ``session_cpu_secs`` sample so
    wrong-prev-key rate wiring is test-visible."""
    state = {
        "phase": "training",
        "last_phase_change_epoch": str(now - 7200),
        "session_cpu_secs": prev_cpu,
        "max_cpu_secs": max_cpu if max_cpu is not None else prev_cpu,
        "zombie_streak": zombie_streak,
    }
    if session_cpu_sample_epoch is not None:
        state["session_cpu_sample_epoch"] = session_cpu_sample_epoch
    if session_cpu_rate_cores is not None:
        state["session_cpu_rate_cores"] = session_cpu_rate_cores
    return json.dumps({"9664": state})


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


# ── #864 namespace-informativeness gate ───────────────────────────────────────


def test_parse_probe_count_units() -> None:
    """``_parse_probe_count``: numeric strings parse; empty / ``unknown`` /
    negative / garbage all return None (no signal -> #826 fall-through)."""
    assert pp._parse_probe_count("4") == 4
    assert pp._parse_probe_count("0") == 0
    assert pp._parse_probe_count("") is None
    assert pp._parse_probe_count(None) is None
    assert pp._parse_probe_count("unknown") is None
    assert pp._parse_probe_count("-1") is None
    assert pp._parse_probe_count("x") is None


def test_parse_probe_stdout_lifts_namespace_counts() -> None:
    """The parser lifts the three #864 count keys; a stdout missing them
    (older probe / SSH-era output) defaults all three to ``"unknown"``."""
    parsed = pp._parse_probe_stdout(
        "\n".join(
            [
                "PID_ALIVE=1",
                "GPU_UTIL=0",
                "ZOMBIE_GPU_PIDS=900001",
                "GPU_PIDS_TOTAL=4",
                "GPU_PIDS_RESOLVABLE=0",
                "NVIDIA_UVM_LIVE_HOLDERS=4",
            ]
        )
    )
    assert parsed["gpu_pids_total"] == "4"
    assert parsed["gpu_pids_resolvable"] == "0"
    assert parsed["nvidia_uvm_live_holders"] == "4"

    older = pp._parse_probe_stdout("PID_ALIVE=1\nGPU_UTIL=0\nZOMBIE_GPU_PIDS=900001\n")
    assert older["gpu_pids_total"] == "unknown"
    assert older["gpu_pids_resolvable"] == "unknown"
    assert older["nvidia_uvm_live_holders"] == "unknown"


def test_zombie_namespace_artifact_live_uvm_holders_vetoes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The #813 shape: stale logs (> 900s window), idle GPUs, CPU advancing
    (present but unconsulted), 4 VRAM holders unresolvable in /proc, 4 live
    in-container uvm holders, streak pre-seeded to "1" (would fire under
    pure #826). The namespace gate vetoes regardless of log staleness:
    running, no reason, streak reset to "0". Also pins the
    ``cpu_override_active`` passthrough on the veto return (direct calls,
    both truth values)."""
    monkeypatch.setattr(pp, "ZOMBIE_NAMESPACE_VETO_ENABLED", True)
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-07-02 00:00:01 [phase=wc_long step=5/12]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5000.0",  # advancing vs prev 4000.0 — NOT consulted by the gate
        zombie_pids="900001 900002 900003 900004",
        gpu_pids_total="4",
        gpu_pids_resolvable="0",
        uvm_live_holders="4",
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
    assert result.status == "running"
    assert result.stall_reason is None
    assert _saved_zombie_streak(state_file) == "0"

    # cpu_override_active passthrough on the veto return — both truth values.
    for cpu_flag in (True, False):
        out = pp._apply_zombie_override(
            status="running",
            zombie_gpu_pids=["900001"],
            stall_sec=900,
            last_mtime_ago=2000.0,
            phase_log_mtime_ago=10**9,
            shard_log_mtime_ago=10**9,
            prev_state={"zombie_streak": "1"},
            pod="pod-9664",
            cpu_override_active=cpu_flag,
            gpu_pids_total=4,
            gpu_pids_resolvable=0,
            uvm_live_holders=4,
        )
        assert out == ("running", None, cpu_flag, 0)


def test_zombie_total_collapse_fires_with_namespace_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The #664 shape WITH the gate enabled and counts present: 1 unresolvable
    VRAM holder, ZERO live uvm holders (total collapse — nothing on the pod
    holds a live CUDA context). Falls through to #826, which fires on
    stale logs + 2-tick persistence."""
    monkeypatch.setattr(pp, "ZOMBIE_NAMESPACE_VETO_ENABLED", True)
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5000.0",
        zombie_pids="1262130",
        gpu_pids_total="1",
        gpu_pids_resolvable="0",
        uvm_live_holders="0",
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


def test_zombie_matched_namespace_fires_despite_live_uvm_holders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TP protection on a MATCHED-namespace pod: 8 compute PIDs, 7 resolve in
    /proc (one genuinely reaped worker among live siblings), 7 live uvm
    holders. ``resolvable > 0`` means the /proc signal is informative, so
    live holders do NOT veto — #826 fires on the stale-log + 2-tick path."""
    monkeypatch.setattr(pp, "ZOMBIE_NAMESPACE_VETO_ENABLED", True)
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5000.0",
        zombie_pids="1262130",
        gpu_pids_total="8",
        gpu_pids_resolvable="7",
        uvm_live_holders="7",
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


def test_zombie_uvm_unknown_falls_back_to_826(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A degraded UVM scan (``unknown``) can never arm the veto: the #813-like
    counts (total 4, resolvable 0) with uvm unknown fall through to pure
    #826, which fires on stale logs + 2-tick persistence — identical to
    pre-#864 behavior (the fail-toward-current direction)."""
    monkeypatch.setattr(pp, "ZOMBIE_NAMESPACE_VETO_ENABLED", True)
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5000.0",
        zombie_pids="900001 900002 900003 900004",
        gpu_pids_total="4",
        gpu_pids_resolvable="0",
        uvm_live_holders="unknown",
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


def test_zombie_resolvable_unknown_falls_back_to_826(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A degraded ``resolvable`` read can NEVER arm the veto (guards a future
    truthiness refactor of the ``== 0`` conjunct — the TP-miss direction):
    total 4, resolvable unknown, uvm 4, stale logs, streak "1" -> #826
    fires."""
    monkeypatch.setattr(pp, "ZOMBIE_NAMESPACE_VETO_ENABLED", True)
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5000.0",
        zombie_pids="900001 900002 900003 900004",
        gpu_pids_total="4",
        gpu_pids_resolvable="unknown",
        uvm_live_holders="4",
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


def test_zombie_namespace_veto_kill_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``EPM_ZOMBIE_NAMESPACE_VETO=0`` (the ops escape hatch, and the shipped
    default per the #864 live-pod gate disposition): the exact #813 veto
    shape behaves as pure #826 — stale logs + 2-tick persistence fire the
    override despite the namespace-artifact counts."""
    monkeypatch.setattr(pp, "ZOMBIE_NAMESPACE_VETO_ENABLED", False)
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-07-02 00:00:01 [phase=wc_long step=5/12]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5000.0",
        zombie_pids="900001 900002 900003 900004",
        gpu_pids_total="4",
        gpu_pids_resolvable="0",
        uvm_live_holders="4",
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


def test_gpu_probe_emits_namespace_count_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Producer-side emission / key-parity pin (#864; the #607
    producer->parser-contract hole): a heredoc typo must not leave the gate
    permanently inert while every fixture test stays green. Captures the
    REAL probe text ``_ssh_probe`` sends over SSH and asserts (a) the three
    emission tokens in the nvidia-smi branch AND their ``=unknown`` twins in
    the else branch; (b) key parity across ``_PROBE_SCALAR_KEYS``, the
    parser defaults, the ssh-failed fallback dict, and the call-site
    ``probe.get(...)`` reads; (c) the exact END-ANCHORED ``/dev/nvidia-uvm``
    matcher (``/dev/nvidia-uvm-tools`` / ``nvidiactl`` / ``nvidia[0-9]``
    must never count)."""
    import subprocess as _subprocess

    captured: dict[str, str] = {}

    def _fake_run(cmd: list[str], **kwargs: Any):
        captured["remote"] = cmd[-1]
        return _subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    pp._ssh_probe(
        "pod-9664",
        "/workspace/logs/issue-9664.log",
        "/workspace/logs/issue-9664.pid",
        9664,
    )
    remote = captured["remote"]

    # (a) emission tokens: nvidia-smi branch + else-branch unknown twins.
    assert 'echo "GPU_PIDS_TOTAL=$GPU_PIDS_TOTAL"' in remote
    assert 'echo "GPU_PIDS_RESOLVABLE=$GPU_PIDS_RESOLVABLE"' in remote
    assert 'echo "NVIDIA_UVM_LIVE_HOLDERS=$UVM_HOLDERS"' in remote
    assert 'echo "GPU_PIDS_TOTAL=unknown"' in remote
    assert 'echo "GPU_PIDS_RESOLVABLE=unknown"' in remote
    assert 'echo "NVIDIA_UVM_LIVE_HOLDERS=unknown"' in remote

    # (b) key parity: emitted key -> _PROBE_SCALAR_KEYS -> parser default ->
    # ssh-failed fallback -> call-site probe.get read (lowercased mapping).
    new_keys = ("GPU_PIDS_TOTAL", "GPU_PIDS_RESOLVABLE", "NVIDIA_UVM_LIVE_HOLDERS")
    parser_defaults = pp._parse_probe_stdout("")
    for key in new_keys:
        assert key in pp._PROBE_SCALAR_KEYS
        assert parser_defaults[key.lower()] == "unknown"

    def _fake_run_fail(cmd: list[str], **kwargs: Any):
        return _subprocess.CompletedProcess(args=cmd, returncode=255, stdout="", stderr="down")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run_fail)
    fallback = pp._ssh_probe(
        "pod-9664",
        "/workspace/logs/issue-9664.log",
        "/workspace/logs/issue-9664.pid",
        9664,
    )
    for key in new_keys:
        assert fallback[key.lower()] == "unknown"

    src = (REPO_ROOT / "scripts" / "poll_pipeline.py").read_text()
    for lowered in ("gpu_pids_total", "gpu_pids_resolvable", "nvidia_uvm_live_holders"):
        assert f'_parse_probe_count(probe.get("{lowered}"))' in src

    # (c) the exact end-anchored uvm matcher — a substring/prefix match would
    # count /dev/nvidia-uvm-tools holders and suppress the #664 TP.
    assert 'grep -q " -> /dev/nvidia-uvm$"' in remote
    assert "/dev/nvidia-uvm-tools" not in remote


# ── #951 material-compute liveness veto ───────────────────────────────────────
#
# Integration-test seeding rule: every ``poll_once`` replay below seeds
# ``max_cpu_secs`` (via ``_stale_state``'s default-or-``max_cpu``) so the
# probe's ``session_cpu`` differs from it by > 0.5 s — ``cpu_advancing`` is
# then True and the #518/#658 CPU rescue holds ``status == "running"`` INTO
# the zombie block (else the test silently exercises the plain-stall path).
# Template: ``test_zombie_gpu_overrides_cpu_advancing_running_to_stalled``.


def _poll_once_9664(state_file: Path):
    """One ``poll_once`` tick against the standard issue-9664 fixture paths."""
    return pp.poll_once(
        issue=9664,
        pod="pod-9664",
        log_path="/workspace/logs/issue-9664.log",
        pid_file="/workspace/logs/issue-9664.pid",
        state_file=state_file,
    )


def test_session_cpu_rate_happy_path() -> None:
    """The #825 shape: prev sample 4000.0 s taken 540 s ago, current 5102.0 s
    -> (5102 - 4000) / 540 ~= 2.0407 cores."""
    now = int(time.time())
    rate = pp._session_cpu_rate_cores("4000.0", str(now - 540), "5102.0", now)
    assert rate == pytest.approx(1102.0 / 540.0)


@pytest.mark.parametrize(
    ("prev_sample", "prev_epoch", "current"),
    [
        ("4000.0", "OK", "unknown"),  # current sample unknown
        (None, "OK", "5102.0"),  # prev sample missing (fresh sidecar)
        ("unknown", "OK", "5102.0"),  # prev sample unknown
        ("4000.0", None, "5102.0"),  # epoch key absent (pre-#951 sidecar)
        ("4000.0", "", "5102.0"),  # epoch empty
        ("4000.0", "unknown", "5102.0"),  # epoch unknown
        ("4000.0", "0", "5102.0"),  # epoch zero (<= 0 guard)
        ("4000.0", "garbage", "5102.0"),  # epoch unparseable
    ],
)
def test_session_cpu_rate_fail_safe_inputs(
    prev_sample: str | None, prev_epoch: str | None, current: str
) -> None:
    """Every degraded input -> None (the #951 veto stays inert)."""
    now = int(time.time())
    epoch = str(now - 540) if prev_epoch == "OK" else prev_epoch
    assert pp._session_cpu_rate_cores(prev_sample, epoch, current, now) is None


def test_session_cpu_rate_dt_floor() -> None:
    """Tick spacing below ``ZOMBIE_CPU_RATE_MIN_DT_SEC`` -> None (truncation-
    noise floor; also covers the back-to-back dt~0 replay case); spacing AT
    the floor computes."""
    now = int(time.time())
    assert pp._session_cpu_rate_cores("4000.0", str(now - 30), "5102.0", now) is None
    floor = pp.ZOMBIE_CPU_RATE_MIN_DT_SEC
    at_floor = pp._session_cpu_rate_cores("4000.0", str(now - floor), "4120.0", now)
    assert at_floor == pytest.approx(120.0 / floor)


def test_session_cpu_rate_negative_delta_returned() -> None:
    """A run-restart / child-exit de-count (current < prev) returns the
    negative rate as-is — not clamped, not None; it sits below any positive
    threshold so the veto never fires on it."""
    now = int(time.time())
    rate = pp._session_cpu_rate_cores("4000.0", str(now - 540), "3000.0", now)
    assert rate == pytest.approx(-1000.0 / 540.0)
    assert rate is not None and rate < 0


def test_zombie_material_cpu_both_ticks_vetoes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The #825 replay: stale logs + idle GPUs + zombie candidate at streak
    "1" (would fire under pure #826), but the session burned ~2.04 cores this
    tick (probe 5102.0 vs raw sample 4000.0 over ~540 s) and 1.83 cores the
    prior tick -> material compute, vetoed: running, no reason, streak reset.

    ``max_cpu_secs`` is seeded DELIBERATELY HIGHER (5000.0) than the raw
    ``session_cpu_secs`` sample (4000.0; realistic — a #658 child-exit
    de-count lowers the raw sample below the rolling max): correct wiring
    reads the RAW sample (delta +1102 s -> 2.04 cores >= T -> veto); wrong
    wiring reading ``max_cpu_secs`` gets +102 s -> 0.19 cores < T -> no veto
    -> this test FAILS on ``status == "stalled"``."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-07-02 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5102.0",  # > max_cpu_secs + 0.5 -> cpu_advancing=True
        zombie_pids="1262130",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="4000.0",
            zombie_streak="1",
            max_cpu="5000.0",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="1.83",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "running"
    assert result.stall_reason is None
    assert _saved_zombie_streak(state_file) == "0"


def test_zombie_low_cpu_churn_still_fires(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The #664 replay (constraint-1 regression pin): identical regime but
    the session churns only ~0.22 cores on both ticks (probe 4119.0 vs
    4000.0 over ~540 s; prior rate "0.22") — below the 0.5-core threshold,
    so the veto stays inert and the override fires."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="4119.0",  # +119 s over ~540 s ~= 0.22 cores
        zombie_pids="1262130",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="4000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="0.22",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_material_cpu_single_tick_does_not_veto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BOTH-ticks pin, current-material direction: this tick burns ~2.04
    cores but the PRIOR tick's persisted rate was 0.10 — a single material
    tick (a ``ps`` truncation / transient-sibling artifact shape) must NOT
    veto; the override fires."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5102.0",
        zombie_pids="1262130",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="4000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="0.10",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_cpu_veto_missing_prev_rate_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Degraded input: the sidecar has the sample + epoch but NO
    ``session_cpu_rate_cores`` key (interrupted warmup shape) — prev rate is
    no-signal, the veto cannot fire, the override fires (current behavior)."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5102.0",  # material THIS tick — still not enough alone
        zombie_pids="1262130",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="4000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_cpu_veto_missing_sample_epoch_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Degraded input: NO ``session_cpu_sample_epoch`` key — exactly the
    pre-#951 / fresh-sidecar-restart shape — so the current rate is not
    computable and the override fires even though a (stale) prev rate is
    present. Doubles as proof the existing seeded-state test semantics are
    preserved by key-absence."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5102.0",
        zombie_pids="1262130",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="4000.0",
            zombie_streak="1",
            session_cpu_rate_cores="1.83",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_cpu_veto_negative_delta_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Degraded input: the current sample DROPPED below the prior one (a
    run-restart / child-exit de-count) — the negative rate sits below the
    threshold, the veto cannot fire, the override fires. The sub-max drop
    arm of the #518/#658 rescue (|delta| > 0.5 s) still holds ``running``
    into the zombie block. Since #1477 this holds when pcpu is unavailable
    (this probe carries no ``SESSION_PCPU_TOTAL`` line, so the negative-rate
    confirm veto stays inert)."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="3000.0",  # < prev 4000.0 -> negative rate
        zombie_pids="1262130",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="4000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="1.9",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_cpu_veto_then_streak_restarts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reset-not-hold pin: tick A is material on both ticks -> veto (running,
    streak "0"); tick B re-polls immediately (dt~0 -> rate None, a degraded
    CPU read) with the candidate still stale+present -> the run gets a FULL
    fresh 2-tick persistence window: it DEFERS (running, streak "1") instead
    of firing off a held streak."""
    now = int(time.time())
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="4000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="1.83",
        )
    )
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5102.0",  # ~2.04 cores vs the seeded epoch
        zombie_pids="1262130",
    )
    tick_a = _poll_once_9664(state_file)
    assert tick_a.status == "running"
    assert tick_a.stall_reason is None
    assert _saved_zombie_streak(state_file) == "0"

    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="6000.0",  # advancing (rescue holds running); rate None (dt~0)
        zombie_pids="1262130",
    )
    tick_b = _poll_once_9664(state_file)
    assert tick_b.status == "running"
    assert tick_b.stall_reason is None
    assert _saved_zombie_streak(state_file) == "1"


def test_zombie_direct_call_rate_none_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct ``_apply_zombie_override`` call WITHOUT the new kwarg (mirrors
    the pre-#951 call style): the ``session_cpu_rate_cores=None`` default
    keeps outputs identical to today — the override fires on a persisted
    stale-log candidate even though the sidecar carries a material prev
    rate."""
    out = pp._apply_zombie_override(
        status="running",
        zombie_gpu_pids=["1262130"],
        stall_sec=900,
        last_mtime_ago=2000.0,
        phase_log_mtime_ago=10**9,
        shard_log_mtime_ago=10**9,
        prev_state={"zombie_streak": "1", "session_cpu_rate_cores": "1.83"},
        pod="pod-9664",
        cpu_override_active=True,
    )
    assert out == ("stalled", "vllm_worker_dead_zombie_gpu", False, 2)


def test_save_state_persists_cpu_sample_epoch_and_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_save_state`` persists the #951 pair. Fresh sidecar (tick 1): the
    sample epoch lands ~now and the rate is ``"unknown"`` (no prior sample —
    the warmup, fail-safe). To observe a FORMATTED rate the state is
    re-seeded with a BACKDATED epoch (dt past the floor) + a prev sample and
    ONE further ``poll_once`` asserts the persisted rate parses to the
    expected float (a naive back-to-back second call persists ``"unknown"``
    again — dt~0 < floor, the helper-level pin in the dt-floor unit test)."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 5,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="95",
        session_cpu="4000.0",
        zombie_pids="",
    )
    state_file = tmp_path / "poll-state.json"
    result = _poll_once_9664(state_file)
    assert result.status == "running"
    saved = json.loads(state_file.read_text())["9664"]
    assert abs(int(saved["session_cpu_sample_epoch"]) - now) < 120
    assert saved["session_cpu_rate_cores"] == "unknown"

    # Backdated re-seed (revision item 8): dt ~540 s >= the floor.
    state_file.write_text(
        _stale_state(now, prev_cpu="4000.0", session_cpu_sample_epoch=str(now - 540))
    )
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 5,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="95",
        session_cpu="5102.0",
        zombie_pids="",
    )
    result = _poll_once_9664(state_file)
    assert result.status == "running"
    saved = json.loads(state_file.read_text())["9664"]
    assert float(saved["session_cpu_rate_cores"]) == pytest.approx(1102.0 / 540.0, rel=0.05)


def test_zombie_prev_material_now_low_still_fires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BOTH-ticks pin, prev-material direction (the dropped-conjunct guard):
    the PRIOR tick burned 1.83 cores but THIS tick churns only ~0.22 — a
    "computed last tick, churning this tick" run must NOT be vetoed on the
    stale prev rate alone; the override fires. An accidentally-dropped
    current-rate conjunct would veto here and fail this test."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="4119.0",  # +119 s over ~540 s ~= 0.22 cores now
        zombie_pids="1262130",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="4000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="1.83",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_cpu_rate_exact_threshold_vetoes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Boundary pin: rate exactly ``ZOMBIE_OVERRIDE_CPU_CORES_MIN`` (0.5000)
    on BOTH ticks -> the veto fires (``>=``, not ``>`` — a future ``>`` typo
    fails this test). ``poll_once``'s clock is frozen at ``now`` so the
    engineered +270 s / 540 s delta lands EXACTLY on the threshold."""
    from datetime import UTC as _utc
    from datetime import datetime as _dt

    now = int(time.time())

    class _FrozenDatetime:
        """Freeze ``pp.datetime.now`` so dt is exactly 540 s (no wall jitter)."""

        @staticmethod
        def now(tz: Any = None) -> Any:
            return _dt.fromtimestamp(now, tz=_utc)

    monkeypatch.setattr(pp, "datetime", _FrozenDatetime)
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-06-27 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="4270.0",  # +270 s / 540 s = exactly 0.5 cores
        zombie_pids="1262130",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="4000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="0.5000",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "running"
    assert result.stall_reason is None
    assert _saved_zombie_streak(state_file) == "0"


# ── #1477 negative-rate pcpu confirm veto ─────────────────────────────────────
#
# Root cause (#1345): procps cumulative-CPU ``time=`` is ``[DD-]HH:MM:SS``;
# the old inline ``n==3`` awk branch numerically coerced ``"1-02"`` -> ``1``
# (strtod prefix), so a process crossing 86400 cumulative CPU-sec collapsed
# from ~86400+s to ~D*3600 + MM*60 + SS in the session sum — sawtoothing the
# cross-tick rate NEGATIVE on a healthy 20-core worker (running-max 83824 ->
# 9351; now=-0.88/prev=-0.18). The awk tests run the REAL system awk
# (subprocess, parametrized over resolvable variants); the replay tests
# drive the real ``poll_once`` -> ``_parse_probe_stdout`` ->
# ``_apply_zombie_override`` path with only the SSH boundary faked.


def _available_awk_bins() -> list[str]:
    """awk variants resolvable on this machine. Plain ``awk`` is the floor
    (Ubuntu's default is mawk); ``mawk`` + ``gawk`` are parametrized in when
    they resolve so the constants stay variant-portable (no gawk
    extensions)."""
    import shutil

    return [v for v in ("awk", "mawk", "gawk") if shutil.which(v)]


_AWK_BINS = _available_awk_bins()


def _awk_sum(awk_bin: str, program: str, stdin: str) -> str:
    """Run one awk variant over ``<sess> <value>`` rows with ``-v s=1``
    (the probe's session filter); returns the stripped stdout."""
    import subprocess

    out = subprocess.run(
        [awk_bin, "-v", "s=1", program],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.mark.skipif(not _AWK_BINS, reason="no awk variant resolvable on this machine")
@pytest.mark.parametrize("awk_bin", _AWK_BINS or ["awk"])
def test_session_cpu_awk_day_format_monotone(awk_bin: str) -> None:
    """The #1345 root-cause pin against the REAL awk: a ``D-HH:MM:SS``
    cputime parses to D*86400 + HH*3600 + MM*60 + SS, so crossing the
    86400-s day boundary never DECREASES the parsed value (pre-fix:
    ``1-02:35:51`` collapsed to 1*3600 + 35*60 + 51 = 5751)."""
    awk = pp._SESSION_CPU_TIME_AWK
    pre_boundary = _awk_sum(awk_bin, awk, "1 23:17:04\n")
    assert pre_boundary == "83824.0"  # the #1345 pre-collapse running-max
    post_boundary = _awk_sum(awk_bin, awk, "1 1-02:35:51\n")
    assert post_boundary == "95751.0"  # 1*86400 + 2*3600 + 35*60 + 51
    assert float(post_boundary) > float(pre_boundary)  # monotone across the day boundary
    assert _awk_sum(awk_bin, awk, "1 1-00:00:00\n") == "86400.0"
    assert _awk_sum(awk_bin, awk, "1 05:00\n") == "300.0"  # MM:SS branch
    # Session filtering: the s=2 row is excluded from the s=1 sum.
    assert _awk_sum(awk_bin, awk, "1 05:00\n2 99:00:00\n") == "300.0"
    # Empty input -> "unknown" (NR==0).
    assert _awk_sum(awk_bin, awk, "") == "unknown"


@pytest.mark.skipif(not _AWK_BINS, reason="no awk variant resolvable on this machine")
@pytest.mark.parametrize("awk_bin", _AWK_BINS or ["awk"])
def test_session_cpu_awk_legacy_formats_unchanged(awk_bin: str) -> None:
    """No-day regression guard: HH:MM:SS and MM:SS inputs produce sums
    identical to the pre-fix parse (a[1]*3600 + a[2]*60 + a[3] /
    a[1]*60 + a[2]); plus the defensive n==1 forms (bare seconds; the
    etime-style ``D-HH`` branch units-corrected in passing — D8)."""
    awk = pp._SESSION_CPU_TIME_AWK
    # 00:05:00 -> 300; 10:30 -> 630; 123:45:06 -> 445506 (pre-fix arithmetic).
    assert _awk_sum(awk_bin, awk, "1 00:05:00\n1 10:30\n1 123:45:06\n") == "446436.0"
    assert _awk_sum(awk_bin, awk, "1 500\n") == "500.0"  # n==1 bare seconds
    assert _awk_sum(awk_bin, awk, "1 2-05\n") == "190800.0"  # 2*86400 + 5*3600 (D8)


def test_zombie_negative_rate_high_pcpu_vetoes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The #1345 replay (arbiter arm): stale logs + idle GPUs + zombie
    candidate at streak "1" (would fire under pure #826) + cross-tick rate
    ~ -0.93 cores (probe 9500.0 vs raw sample 10000.0 over ~540 s — the
    parse-collapse sawtooth; an IMPOSSIBLE value on a monotone counter) +
    persisted prev rate -0.18, BUT the SAME tick's session pcpu reads
    2012.5% = 20.125 cores (>= 0.5) — a demonstrably-computing worker.
    #1477 veto: running, no reason, streak reset."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-07-15 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="9500.0",  # < prev 10000.0 -> rate ~ -0.93 (scope reset)
        zombie_pids="184938",
        session_pcpu="2012.5",  # 20.125 cores
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="10000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="-0.18",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "running"
    assert result.stall_reason is None
    assert _saved_zombie_streak(state_file) == "0"


def test_zombie_prev_negative_current_positive_pcpu_vetoes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second observed #1345 pair: current rate ~ +0.82 cores (probe
    10442.8 vs raw 10000.0 over ~540 s) but the PERSISTED prev rate is
    -0.56 — a negative sample on EITHER tick invalidates the pair (the
    #951 both-ticks->=T veto could not have fired: prev < T). High pcpu
    => #1477 veto."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-07-15 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="10442.8",  # +442.8 s over ~540 s ~ +0.82 cores
        zombie_pids="184938",
        session_pcpu="2012.5",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="10000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="-0.56",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "running"
    assert result.stall_reason is None
    assert _saved_zombie_streak(state_file) == "0"


def test_zombie_prev_negative_rate_none_pcpu_vetoes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The (rate_now None, prev < 0) cell — §4.4 normative-predicate pin
    (round-1 Statistics Must-Fix 2): the sidecar persists a NEGATIVE prev
    rate ("-0.56") but NO ``session_cpu_sample_epoch``, so the current
    rate is uncomputable (None). The persisted negative affirmatively
    PROVES scope inconsistency, so the confirm still arms: high pcpu =>
    veto (running, streak reset)."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-07-15 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5102.0",  # advancing vs max 4000.0 (rescue holds running)
        zombie_pids="184938",
        session_pcpu="2012.5",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="4000.0",
            zombie_streak="1",
            session_cpu_rate_cores="-0.56",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "running"
    assert result.stall_reason is None
    assert _saved_zombie_streak(state_file) == "0"


def test_zombie_negative_rate_no_pcpu_still_fires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail-safe fall-through (older pod probe / degraded ps): the same
    negative-rate shape as the #1345 replay but ``SESSION_PCPU_TOTAL``
    reads ``unknown`` -> pcpu None -> the #1477 confirm cannot arm and
    today's streak-2 fire happens byte-identically (#864 law: every
    degraded read fails toward CURRENT behavior)."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-07-15 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="9500.0",
        zombie_pids="184938",
        session_pcpu="unknown",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="10000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="-0.18",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_negative_rate_low_pcpu_still_fires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """True-positive calibration pin: a negative-rate tick (a worker-death
    de-count) whose surviving session churns only the #664 idle-EngineCore
    ~0.22 cores (pcpu "22.0" percent < 0.5-core threshold) must NOT be
    shielded — the override fires. Bounded-delay contract note: a
    heavy-history dead worker (high LIFETIME pcpu at death) legitimately
    costs one veto tick on its de-count tick — afterwards cur ~ prev
    (rate small non-negative), pcpu is no longer consulted, and the
    normal #951/streak path fires (<=2 ticks total delay)."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-07-15 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="9500.0",
        zombie_pids="184938",
        session_pcpu="22.0",  # 0.22 cores — the #664 churn, as percent
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="10000.0",
            zombie_streak="1",
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="-0.18",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_nonnegative_rate_high_pcpu_still_fires(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unreapability guard — §4.4 row 6 with pcpu PRESENT and HIGH (round-1
    Statistics Must-Fix 1): a frozen-counter hung run reads rate EXACTLY
    0.0 (probe sample EQUAL to the persisted raw sample) with a low
    non-negative prev rate, while a heavy-history session can read high
    LIFETIME pcpu (2012.5%). With NO negative sample on either tick, pcpu
    must NOT be consulted — the override fires. A ``<= 0``
    mis-implementation or an inverted comparison passes every other test
    but vetoes here and fails this one. ``max_cpu_secs`` is seeded BELOW
    the frozen sample so the #518/#658 rescue still holds ``running``
    into the zombie block (the integration-seeding rule above)."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-07-15 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5102.0",  # == persisted raw sample -> rate 0.0
        zombie_pids="184938",
        session_pcpu="2012.5",  # heavy-history lifetime average
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(
        _stale_state(
            now,
            prev_cpu="5102.0",
            zombie_streak="1",
            max_cpu="5000.0",  # cpu_advancing True (|5102-5000| > 0.5)
            session_cpu_sample_epoch=str(now - 540),
            session_cpu_rate_cores="0.10",
        )
    )
    result = _poll_once_9664(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


def test_zombie_warmup_rate_none_pcpu_not_consulted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Warmup pin (§11 D4 scoping as disambiguated in §4.4): no persisted
    ``session_cpu_sample_epoch`` AND no persisted ``session_cpu_rate_cores``
    -> current rate None with NO negative evidence on either tick. pcpu
    (even 2012.5%) is NOT consulted; today's streak path fires
    byte-identically. Widening the trigger to bare ``None`` would shield
    freshly-launched hung runs whose init burn inflates the lifetime pcpu
    average — the #664-unreapability direction."""
    now = int(time.time())
    _patch_pod(
        monkeypatch,
        mtime_epoch=now - 2000,
        tail="2026-07-15 00:00:01 [phase=training step=5/100]",
        gpu_util="0,0,0,0,0,0,0,0",
        session_cpu="5102.0",
        zombie_pids="184938",
        session_pcpu="2012.5",
    )
    state_file = tmp_path / "poll-state.json"
    state_file.write_text(_stale_state(now, prev_cpu="4000.0", zombie_streak="1"))
    result = _poll_once_9664(state_file)
    assert result.status == "stalled"
    assert result.stall_reason == "vllm_worker_dead_zombie_gpu"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2012.5", 20.125),
        ("50.0", 0.5),
        ("0.0", 0.0),
        ("unknown", None),
        ("", None),
        ("garbage", None),
        (None, None),
        ("-5.0", None),  # negative %cpu is malformed -> fail-safe None
    ],
)
def test_parse_session_pcpu_cores_units(value: str | None, expected: float | None) -> None:
    """``_parse_session_pcpu_cores``: percent -> cores; unknown / malformed /
    negative -> None (the #1477 confirm veto stays inert)."""
    got = pp._parse_session_pcpu_cores(value)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_parse_probe_stdout_lifts_session_pcpu() -> None:
    """The parser dispatches the ``SESSION_PCPU_TOTAL=`` line; an absent
    line (older pod probe) defaults to ``unknown``."""
    parsed = pp._parse_probe_stdout("PID_ALIVE=1\nSESSION_PCPU_TOTAL=2012.5\n")
    assert parsed["session_pcpu_total"] == "2012.5"
    assert pp._parse_probe_stdout("PID_ALIVE=1\n")["session_pcpu_total"] == "unknown"


def test_probe_heredoc_emits_session_pcpu_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """Producer-side emission / key-parity pin (#1477; mirrors
    ``test_gpu_probe_emits_namespace_count_keys``): captures the REAL
    composed remote command and asserts (a) the new field's emission on the
    success branch + its ``=unknown`` twins on BOTH degraded branches, (b)
    the SEPARATE ``sess=,pcpu=`` ps pipeline, (c) the literal fixed-awk
    day-format fragment — pinning the f-string interpolation of the
    constants (a doubled-brace regression corrupts exactly this text), and
    (d) key parity across ``_PROBE_SCALAR_KEYS`` / parser default /
    ssh-failed fallback."""
    import subprocess as _subprocess

    captured: dict[str, str] = {}

    def _fake_run(cmd: list[str], **kwargs: Any):
        captured["remote"] = cmd[-1]
        return _subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    pp._ssh_probe(
        "pod-9664",
        "/workspace/logs/issue-9664.log",
        "/workspace/logs/issue-9664.pid",
        9664,
    )
    remote = captured["remote"]

    # (a) emission tokens: success branch + the two degraded-branch twins.
    assert 'echo "SESSION_PCPU_TOTAL=${PCPU_SUM:-unknown}"' in remote
    assert remote.count('echo "SESSION_PCPU_TOTAL=unknown"') == 2
    # (b) the SEPARATE pcpu pipeline (failure isolation from the time= sum).
    assert "ps -e -o sess=,pcpu=" in remote
    assert "ps -e -o sess=,time=" in remote
    # (c) the fixed-awk day-format fragment lands verbatim (single braces).
    assert "b[1]*86400 + b[2]*3600" in remote
    assert "{_SESSION_CPU_TIME_AWK}" not in remote  # constants interpolated, not literal

    # (d) key parity: _PROBE_SCALAR_KEYS -> parser default -> ssh-failed
    # fallback (lowercased mapping).
    assert "SESSION_PCPU_TOTAL" in pp._PROBE_SCALAR_KEYS
    assert pp._parse_probe_stdout("")["session_pcpu_total"] == "unknown"

    def _fake_run_fail(cmd: list[str], **kwargs: Any):
        return _subprocess.CompletedProcess(args=cmd, returncode=255, stdout="", stderr="down")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run_fail)
    fallback = pp._ssh_probe(
        "pod-9664",
        "/workspace/logs/issue-9664.log",
        "/workspace/logs/issue-9664.pid",
        9664,
    )
    assert fallback["session_pcpu_total"] == "unknown"


def test_zombie_direct_call_pcpu_default_none() -> None:
    """Direct ``_apply_zombie_override`` call WITHOUT the new kwarg on a
    NEGATIVE-rate fire shape (mirrors
    ``test_zombie_direct_call_rate_none_default``): the
    ``session_pcpu_cores=None`` default is inert — the override fires
    exactly as pre-#1477 even though both rates are negative."""
    out = pp._apply_zombie_override(
        status="running",
        zombie_gpu_pids=["184938"],
        stall_sec=900,
        last_mtime_ago=2000.0,
        phase_log_mtime_ago=10**9,
        shard_log_mtime_ago=10**9,
        prev_state={"zombie_streak": "1", "session_cpu_rate_cores": "-0.18"},
        pod="pod-9664",
        cpu_override_active=True,
        session_cpu_rate_cores=-0.93,
    )
    assert out == ("stalled", "vllm_worker_dead_zombie_gpu", False, 2)
