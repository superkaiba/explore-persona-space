"""Tests for the adaptive bg-poll interval (anti-stall redesign §7).

The orchestrator's Step 6d.2 bg-Bash sleep-chain re-invokes a full
orchestrator turn on every poll exit, so the chain interval is the
dominant per-run cost over multi-hour runs (issue-601: 2,561 turns).
``scripts/poll_pipeline.py`` therefore emits a recommended
``next_interval`` with each tick's verdict: the long QUIET interval
(1800s) ONLY on a healthy, quiet, post-early-run ``running`` tick far
from any phase boundary; the short DEFAULT (540s) on everything else.

These tests pin:

* the pure decision table (``recommend_next_interval``) — healthy-quiet
  -> 1800; EACH non-quiet condition independently -> 540; early-run ->
  540; unknown signals fail toward the short interval;
* the window boundary semantics (>= window is quiet, < window is not);
* the ``poll_once`` wiring — the field is threaded into the returned
  ``PollResult`` and the boundary epoch persists in the state file;
* the JSON-surface contract — ``poll_pipeline.main``'s JSON,
  ``backend_poll._serialize_poll_result`` (including the duck-typed
  older-result fallback), ``backend_poll._missing_sidecar_json``, and
  the ``backends.base.PollResult`` / RunPod passthrough;
* the orchestrator-side clamp + quiet-wait branch documented in
  ``.claude/skills/issue/SKILL.md`` Step 6d.2 (#1818: the sleep is FIXED
  at 540s per bg-Bash call — the Bash tool kills any call at its
  600000 ms ceiling, background included, so a composed ``sleep 1800``
  dies mid-sleep (#1768); the old sleep-honor branch stays gone, and the
  §7 quiet cadence is instead realized as a one-wake Monitor
  wait-then-poll branch keyed on ``next_interval == 1800`` (#1924)).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(filename: str, alias: str):
    """Load a ``scripts/*.py`` file as a module (mirrors
    ``tests/test_poll_pipeline_sentinels.py``'s loader)."""
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


pp = _load_script_module("poll_pipeline.py", "poll_pipeline_next_interval_under_test")
bp = _load_script_module("backend_poll.py", "backend_poll_next_interval_under_test")


# Keyword set for a fully healthy-quiet tick; tests override one signal at
# a time to prove each condition independently forces the short interval.
QUIET_KWARGS: dict[str, Any] = {
    "status": "running",
    "gate": None,
    "sentinels_processed": 0,
    "phase_transitioned": False,
    "ssh_failed": False,
    "gpu_idle_advisory_posted": False,
    "cpu_override_active": False,
    "run_age_sec": 7200.0,
    "phase_changed_ago_sec": 7200.0,
}


# ── decision table ───────────────────────────────────────────────────────────


def test_constants_match_design() -> None:
    """§7 pins the two interval values: 540s default, 1800s quiet."""
    assert pp.POLL_INTERVAL_DEFAULT_SEC == 540
    assert pp.POLL_INTERVAL_QUIET_SEC == 1800
    assert pp.POLL_INTERVAL_QUIET_SEC > pp.POLL_INTERVAL_DEFAULT_SEC


def test_healthy_quiet_running_gets_quiet_interval() -> None:
    assert pp.recommend_next_interval(**QUIET_KWARGS) == pp.POLL_INTERVAL_QUIET_SEC


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"status": "done"}, id="status-done"),
        pytest.param({"status": "gate"}, id="status-gate"),
        pytest.param({"status": "stalled"}, id="status-stalled"),
        pytest.param({"status": "dead"}, id="status-dead"),
        pytest.param({"gate": "fact-candidates"}, id="gate-sentinel"),
        pytest.param({"sentinels_processed": 1}, id="sentinel-activity"),
        pytest.param({"phase_transitioned": True}, id="phase-transition-this-tick"),
        pytest.param({"ssh_failed": True}, id="ssh-transport-failure"),
        pytest.param({"gpu_idle_advisory_posted": True}, id="gpu-idle-advisory"),
        pytest.param({"gpu_idle_escalation_posted": True}, id="gpu-idle-escalation"),
        pytest.param({"cpu_override_active": True}, id="cpu-override-stall-rescue"),
        pytest.param({"run_age_sec": 600.0}, id="early-run"),
        pytest.param({"run_age_sec": None}, id="unknown-launch-age"),
        pytest.param({"phase_changed_ago_sec": 60.0}, id="recent-phase-change"),
        pytest.param({"phase_changed_ago_sec": None}, id="unknown-phase-change-time"),
    ],
)
def test_each_non_quiet_condition_forces_short_interval(override: dict[str, Any]) -> None:
    """EVERY non-quiet condition independently pins the short interval —
    the long interval must never delay a gate or mask a fresh anomaly."""
    kwargs = {**QUIET_KWARGS, **override}
    assert pp.recommend_next_interval(**kwargs) == pp.POLL_INTERVAL_DEFAULT_SEC


def test_window_boundaries_are_inclusive_at_the_window() -> None:
    """Exactly AT each window the run counts as quiet; one second under
    does not (>= semantics, pinned so a refactor can't drift them)."""
    at_windows = {
        **QUIET_KWARGS,
        "run_age_sec": float(pp.EARLY_RUN_WINDOW_SEC),
        "phase_changed_ago_sec": float(pp.RECENT_PHASE_CHANGE_WINDOW_SEC),
    }
    assert pp.recommend_next_interval(**at_windows) == pp.POLL_INTERVAL_QUIET_SEC
    just_under_run_age = {**at_windows, "run_age_sec": float(pp.EARLY_RUN_WINDOW_SEC - 1)}
    assert pp.recommend_next_interval(**just_under_run_age) == pp.POLL_INTERVAL_DEFAULT_SEC
    just_under_phase = {
        **at_windows,
        "phase_changed_ago_sec": float(pp.RECENT_PHASE_CHANGE_WINDOW_SEC - 1),
    }
    assert pp.recommend_next_interval(**just_under_phase) == pp.POLL_INTERVAL_DEFAULT_SEC


def test_gpu_idleness_alone_is_documented_as_not_quiet_blocking() -> None:
    """Considered-and-rejected (decision 2026-06-12): raw GPU idleness on
    a healthy tick does NOT force the short interval — idle GPUs are
    routine during long CPU-bound phases, exactly the high-savings quiet
    stretches. The rejection is recorded in the decision core's docstring
    so it cannot be silently re-litigated; behaviorally, a healthy quiet
    tick stays quiet regardless of GPU utilization (gpu_util is not an
    input to the decision core — only the advisory-post event is)."""
    assert "Deliberately NOT in the set" in (pp.recommend_next_interval.__doc__ or "")
    assert pp.recommend_next_interval(**QUIET_KWARGS) == pp.POLL_INTERVAL_QUIET_SEC


# ── lane-shared quiet subset (SLURM + GCP lanes) ─────────────────────────────


def _backends_base():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from explore_persona_space.backends import base as backends_base

    return backends_base


# Keyword set for a fully quiet lane tick; tests override one signal at a
# time to prove each condition independently forces the short interval.
LANE_QUIET_KWARGS: dict[str, Any] = {
    "status": "running",
    "gate": None,
    "sentinels_processed": 0,
    "new_milestone": False,
    "run_age_sec": 7200.0,
    "lane_anomaly": False,
}


def test_lane_constants_stay_in_sync_with_poll_pipeline() -> None:
    """``backends.base`` mirrors poll_pipeline's interval constants as
    local literals (no heavy scripts/ import at package import time);
    this pin is what keeps the two sets from drifting."""
    base = _backends_base()
    assert base.POLL_INTERVAL_DEFAULT_SEC == pp.POLL_INTERVAL_DEFAULT_SEC == 540
    assert base.POLL_INTERVAL_QUIET_SEC == pp.POLL_INTERVAL_QUIET_SEC == 1800
    assert base.EARLY_RUN_WINDOW_SEC == pp.EARLY_RUN_WINDOW_SEC == 1800


def test_lane_healthy_quiet_running_gets_quiet_interval() -> None:
    base = _backends_base()
    assert base.recommend_lane_next_interval(**LANE_QUIET_KWARGS) == base.POLL_INTERVAL_QUIET_SEC


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"status": "done"}, id="status-done"),
        pytest.param({"status": "gate"}, id="status-gate"),
        pytest.param({"status": "stalled"}, id="status-stalled"),
        pytest.param({"status": "dead"}, id="status-dead"),
        pytest.param({"gate": "fact-candidates"}, id="gate-sentinel"),
        pytest.param({"sentinels_processed": 1}, id="sentinel-activity"),
        pytest.param({"new_milestone": True}, id="milestone-this-tick"),
        pytest.param({"lane_anomaly": True}, id="lane-anomaly"),
        pytest.param({"run_age_sec": 600.0}, id="early-run"),
        pytest.param({"run_age_sec": None}, id="unknown-launch-age"),
    ],
)
def test_lane_each_non_quiet_condition_forces_short_interval(override: dict[str, Any]) -> None:
    """EVERY lane non-quiet condition independently pins the short
    interval — same never-delay-a-gate contract as the RunPod core."""
    base = _backends_base()
    kwargs = {**LANE_QUIET_KWARGS, **override}
    assert base.recommend_lane_next_interval(**kwargs) == base.POLL_INTERVAL_DEFAULT_SEC


def test_lane_window_boundary_is_inclusive_at_the_window() -> None:
    base = _backends_base()
    at_window = {**LANE_QUIET_KWARGS, "run_age_sec": float(base.EARLY_RUN_WINDOW_SEC)}
    assert base.recommend_lane_next_interval(**at_window) == base.POLL_INTERVAL_QUIET_SEC
    just_under = {**LANE_QUIET_KWARGS, "run_age_sec": float(base.EARLY_RUN_WINDOW_SEC - 1)}
    assert base.recommend_lane_next_interval(**just_under) == base.POLL_INTERVAL_DEFAULT_SEC


# ── poll_once wiring ─────────────────────────────────────────────────────────


def _probe_stdout(*, mtime_epoch: int, tail: str) -> str:
    """Minimal healthy-probe stdout in the shape ``_parse_probe_stdout``
    expects (pid alive, fresh main log, no cell/phase/shard logs)."""
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
            "PHASE_LOG_MTIME_EPOCH=0",
            "SHARD_LOG_MTIME_EPOCH=0",
            "GPU_UTIL=95",
            "SESSION_CPU_SECS=unknown",
            "RESULTS_SENTINEL_PRESENT=0",
        ]
    )


def _patch_healthy_pod(monkeypatch: pytest.MonkeyPatch, *, run_age_sec: float) -> None:
    """Monkeypatch poll_pipeline's I/O boundary for a healthy running pod."""
    now = int(time.time())

    def _fake_run(cmd: list[str], **kwargs: Any):
        import subprocess

        remote = cmd[-1]
        stdout = (
            ""
            if "SENTINEL_START" in remote
            else _probe_stdout(
                mtime_epoch=now, tail="2026-06-12 00:00:01 [phase=training step=5/100]"
            )
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(pp.subprocess, "run", _fake_run)
    monkeypatch.setattr(pp, "post_event", MagicMock())
    monkeypatch.setattr(pp, "_marker_pid", lambda issue: None)
    monkeypatch.setattr(pp, "_run_launched_age_sec", lambda issue, now_epoch: run_age_sec)


def test_poll_once_quiet_tick_emits_quiet_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A healthy running tick on a long-stable phase, past the early-run
    window, threads the quiet interval into the PollResult.

    ``run_age_sec`` (10800) deliberately exceeds the seeded boundary age
    (7200) so the relaunch clamp sees a boundary AFTER the launch and
    leaves it alone — and stays robust to the seconds of skew between
    the test's ``now`` and poll_once's own ``now_epoch``."""
    _patch_healthy_pod(monkeypatch, run_age_sec=10800.0)
    state_file = tmp_path / "poll-state.json"
    now = int(time.time())
    state_file.write_text(
        json.dumps({"9999": {"phase": "training", "last_phase_change_epoch": str(now - 7200)}})
    )
    result = pp.poll_once(
        issue=9999,
        pod="pod-9999",
        log_path="/workspace/logs/issue-9999.log",
        pid_file="/workspace/logs/issue-9999.pid",
        state_file=state_file,
    )
    assert result.status == "running"
    assert result.next_interval == pp.POLL_INTERVAL_QUIET_SEC
    # The boundary epoch persists unchanged (no transition this tick).
    saved = json.loads(state_file.read_text())["9999"]
    assert saved["last_phase_change_epoch"] == str(now - 7200)


def test_poll_once_stale_boundary_from_before_relaunch_is_clamped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Relaunch clamp (code-review 2026-06-12): a boundary epoch recorded
    BEFORE the current run's launch is stale state from a previous run on
    the same issue — it must NOT satisfy the recent-phase-change guard.
    Past the early-run window + same phase name as the stale state, the
    tick stays SHORT and the stale epoch is reset to 0 in the state file."""
    _patch_healthy_pod(monkeypatch, run_age_sec=7200.0)
    state_file = tmp_path / "poll-state.json"
    now = int(time.time())
    # Boundary 20000s ago, launch 7200s ago -> boundary predates launch.
    state_file.write_text(
        json.dumps({"9999": {"phase": "training", "last_phase_change_epoch": str(now - 20000)}})
    )
    result = pp.poll_once(
        issue=9999,
        pod="pod-9999",
        log_path="/workspace/logs/issue-9999.log",
        pid_file="/workspace/logs/issue-9999.pid",
        state_file=state_file,
    )
    assert result.status == "running"
    assert result.next_interval == pp.POLL_INTERVAL_DEFAULT_SEC
    saved = json.loads(state_file.read_text())["9999"]
    assert saved["last_phase_change_epoch"] == "0"


def test_poll_once_first_tick_stays_short_and_records_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The first tick sees a phase transition ('' -> training): short
    interval, and the boundary epoch lands in the state file so the
    recent-change window starts counting."""
    _patch_healthy_pod(monkeypatch, run_age_sec=7200.0)
    state_file = tmp_path / "poll-state.json"
    result = pp.poll_once(
        issue=9999,
        pod="pod-9999",
        log_path="/workspace/logs/issue-9999.log",
        pid_file="/workspace/logs/issue-9999.pid",
        state_file=state_file,
    )
    assert result.status == "running"
    assert result.next_interval == pp.POLL_INTERVAL_DEFAULT_SEC
    saved = json.loads(state_file.read_text())["9999"]
    assert int(saved["last_phase_change_epoch"]) > 0


def test_poll_once_early_run_stays_short(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Quiet phase but inside the early-run window -> short interval."""
    _patch_healthy_pod(monkeypatch, run_age_sec=300.0)
    state_file = tmp_path / "poll-state.json"
    now = int(time.time())
    state_file.write_text(
        json.dumps({"9999": {"phase": "training", "last_phase_change_epoch": str(now - 7200)}})
    )
    result = pp.poll_once(
        issue=9999,
        pod="pod-9999",
        log_path="/workspace/logs/issue-9999.log",
        pid_file="/workspace/logs/issue-9999.pid",
        state_file=state_file,
    )
    assert result.status == "running"
    assert result.next_interval == pp.POLL_INTERVAL_DEFAULT_SEC


# ── JSON-surface contract ────────────────────────────────────────────────────


def test_poll_result_dataclass_defaults_to_short_interval() -> None:
    result = pp.PollResult(
        status="running",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=True,
        pid_file_missing=False,
        log_tail_excerpt="",
    )
    assert result.next_interval == pp.POLL_INTERVAL_DEFAULT_SEC


def test_backend_poll_serializes_next_interval_passthrough() -> None:
    """``_serialize_poll_result`` carries the field through verbatim."""
    result = SimpleNamespace(
        status="running",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=True,
        log_tail_excerpt="",
        gate=None,
        sentinels_processed=0,
        phase_log_mtime_sec_ago=10**9,
        shard_log_mtime_sec_ago=10**9,
        gpu_util="95",
        next_interval=1800,
    )
    assert bp._serialize_poll_result(result)["next_interval"] == 1800


def test_backend_poll_serializer_falls_back_for_older_results() -> None:
    """A duck-typed result without the field (older module copy in a
    mixed-version worktree) degrades to the short default — never a
    crash, never a lengthened interval."""
    result = SimpleNamespace(
        status="running",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=True,
        log_tail_excerpt="",
        gate=None,
        sentinels_processed=0,
        phase_log_mtime_sec_ago=10**9,
        shard_log_mtime_sec_ago=10**9,
        gpu_util="95",
    )
    assert bp._serialize_poll_result(result)["next_interval"] == 540


def test_missing_sidecar_json_carries_short_interval() -> None:
    payload = bp._missing_sidecar_json(123, Path("/tmp/none.json"), "sidecar not found")
    assert payload["next_interval"] == 540


def test_backends_base_pollresult_defaults_short_and_runpod_passes_through() -> None:
    """Lane contract: lanes that don't compute the quiet heuristic keep
    the short default; the RunPod rebuild copies the value through."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from explore_persona_space.backends.base import PollResult as BasePollResult

    base_result = BasePollResult(
        status="running",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=True,
        log_tail_excerpt="",
    )
    assert base_result.next_interval == 540
    runpod_src = (REPO_ROOT / "src/explore_persona_space/backends/runpod.py").read_text()
    assert "next_interval=raw.next_interval" in runpod_src


# ── #664 zombie-GPU stall_reason passthrough (RunPod lane) ───────────────────


def test_backends_base_pollresult_defaults_stall_reason_to_none() -> None:
    """``stall_reason`` defaults to ``None`` so SLURM / GCP (which never set
    it) serialize identically and the field is absent from a healthy tick."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from explore_persona_space.backends.base import PollResult as BasePollResult

    base_result = BasePollResult(
        status="running",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=True,
        log_tail_excerpt="",
    )
    assert base_result.stall_reason is None


def test_backend_poll_serializes_zombie_stall_reason_passthrough() -> None:
    """``_serialize_poll_result`` carries the #664 zombie-GPU stall reason
    through verbatim (the RunPod lane sets it when poll_once overrides a
    masked ``running`` to ``stalled``)."""
    result = SimpleNamespace(
        status="stalled",
        current_phase="eval",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=True,
        log_tail_excerpt="",
        gate=None,
        sentinels_processed=0,
        phase_log_mtime_sec_ago=10**9,
        shard_log_mtime_sec_ago=10**9,
        gpu_util="95",
        next_interval=540,
        stall_reason="vllm_worker_dead_zombie_gpu",
    )
    assert bp._serialize_poll_result(result)["stall_reason"] == "vllm_worker_dead_zombie_gpu"


def test_runpod_poll_copies_stall_reason_through_from_poll_once() -> None:
    """The slice-6 RunPod rewrap MUST copy ``stall_reason`` from
    ``poll_once``'s result. This rewrap is the only seam between the
    zombie-GPU probe (in ``poll_once``) and
    ``backend_poll._serialize_poll_result``; without the copy the
    distinguishing reason is silently dropped on the slice-6 RunPod path
    even though the serializer already reads it (#664)."""
    runpod_src = (REPO_ROOT / "src/explore_persona_space/backends/runpod.py").read_text()
    assert "stall_reason=raw.stall_reason" in runpod_src


# ── orchestrator-fallback semantics (SKILL.md) ───────────────────────────────


def test_skill_sleep_chain_clamped_at_540_per_call() -> None:
    """Step 6d.2's sleep-chain is HARD-CLAMPED at 540s per bg-Bash call
    (#1818): the Bash tool kills ANY call at its 600000 ms ceiling —
    background calls included — so a composed ``sleep 1800`` dies
    mid-sleep and reads as a stale poll on the next wake (#1768). The
    old sleep-honor branch stays gone; the §7 quiet cadence is instead
    realized as a ONE-wake Monitor wait-then-poll branch keyed on
    ``next_interval == 1800`` (#1924) — the wait+poll run in one unit,
    so the terminal stdout line IS the tick JSON."""
    skill = (REPO_ROOT / ".claude/skills/issue/SKILL.md").read_text()
    assert "ADAPTIVE POLL INTERVAL" in skill
    assert 'f"sleep {interval} && uv run python scripts/backend_poll.py --issue {N}"' in skill
    assert "sleep 540 && uv run python scripts/backend_poll.py" not in skill
    # The SLEEP-honor branch stays REMOVED — next_interval never sets a
    # bg-Bash sleep length.
    assert "in (540, 1800)" not in skill
    assert 'interval = result["next_interval"]' not in skill
    # The fixed clamp + its rationale are stated explicitly.
    assert "interval = 540" in skill
    assert "NEVER compose a sleep longer than 540s" in skill
    assert "#1768" in skill
    # The quiet-wait Monitor branch is wired (#1924): keyed on the
    # poller's own 1800 recommendation, hard-bounded, one-wake shape.
    assert 'result.get("next_interval") == 1800' in skill
    assert "timeout_ms=2400000" in skill
    assert "quiet-wait issue" in skill
    # Ordering pin: the quiet-wait branch PRECEDES the `interval = 540`
    # else-arm, so a future reword cannot invert the fail-toward-coverage
    # default (non-1800 / missing / unparseable -> the fixed 540s chain).
    assert skill.index('result.get("next_interval") == 1800') < skill.index("interval = 540")
    # Both-sources phrasing: after #1924 the tick JSON arrives from either
    # a bg-Bash exit OR the quiet-wait Monitor notification. The Step 6d.2
    # prose must name BOTH sources so a byte-precise reader cannot
    # conclude `result` is only set from bg-Bash ticks (#1996).
    assert "quiet-wait Monitor" in skill
    assert (
        "bg-Bash exit OR quiet-wait Monitor" in skill
        or "bg-Bash exit or quiet-wait Monitor" in skill
    )


def test_main_json_line_includes_next_interval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``poll_pipeline.main`` emits the field on its JSON line — the
    actual surface the orchestrator's legacy parent-pod-reuse fallback
    parses."""
    fake = pp.PollResult(
        status="running",
        current_phase="training",
        new_milestone=False,
        last_log_mtime_sec_ago=10,
        pid_alive=True,
        pid_file_missing=False,
        log_tail_excerpt="",
        next_interval=1800,
    )
    monkeypatch.setattr(pp, "poll_once", lambda **kwargs: fake)
    rc = pp.main(
        [
            "--issue",
            "9999",
            "--pod",
            "pod-9999",
            "--log",
            "/workspace/logs/issue-9999.log",
            "--pid-file",
            "/workspace/logs/issue-9999.pid",
        ]
    )
    assert rc == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(line)["next_interval"] == 1800
