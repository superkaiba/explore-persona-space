"""SLURM monitor — state-mapping + stall semantics.

The monitor builds a :class:`PollResult` from three signals:

1. SLURM job state (``scontrol show job`` / ``squeue -j``).
2. Rsync'd ``status.json`` (heartbeat + phase + gpu_busy).
3. Rsync'd ``job.out`` (log tail, ``[phase=...]`` grep).

These tests cover the state→status mapping table, the stall threshold,
the scontrol parser, and the preflight-failure shortcut. They run
without a cluster (every shell-out is dependency-injected).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from explore_persona_space.backends.base import (
    POLL_INTERVAL_DEFAULT_SEC,
    POLL_INTERVAL_QUIET_SEC,
)
from explore_persona_space.backends.slurm import get_cluster_config
from explore_persona_space.backends.slurm_monitor import (
    FRESHNESS_SKEW_MARGIN_SEC,
    SLURM_STATE_TO_STATUS,
    STALL_CONSECUTIVE_TICKS,
    STALL_SEC,
    SlurmProbeError,
    _parse_scontrol_show_job,
    _parse_slurm_runtime,
    _read_stall_streak,
    _scrub_secret_tokens,
    build_poll_result,
    fetch_started_evidence,
    query_by_name,
    query_slurm_state,
    rsync_status_and_log,
)


def _nibi():
    return get_cluster_config("nibi")


@pytest.fixture(autouse=True)
def _no_real_marker_posts(monkeypatch):
    """Defense in depth: never let a monitor test shell out to the real
    ``task.py post-marker`` (it would pollute a real tasks/<N>/events.jsonl,
    as happened to #137). Patches the default poster to a no-op; tests that
    assert on posts inject ``marker_poster=`` explicitly.
    """
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm.post_marker_via_task_py",
        lambda **_kw: None,
    )


@pytest.fixture(autouse=True)
def _isolated_local_state_dir(tmp_path, monkeypatch):
    """Round-6 Mn3: route ``_local_state_dir`` under pytest's ``tmp_path``.

    The pre-fix tests wrote to the REAL ``/tmp/slurm-<id>`` with fixed
    job ids, so parallel pytest runs (or a test and a live monitor)
    could collide on the same files.
    """
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm_monitor._local_state_dir",
        lambda job_id: tmp_path / f"slurm-{job_id}",
    )


# ---------------------------------------------------------------------------
# scontrol parser
# ---------------------------------------------------------------------------


def test_parse_scontrol_show_job_extracts_jobstate_and_exit() -> None:
    sample = """\
JobId=9001 JobName=eps-issue-137
   UserId=alice GroupId=alice Account=rrg-bengioy-ad_gpu
   JobState=COMPLETED Reason=None Dependency=(null)
   ExitCode=0:0 RunTime=00:42:13
   NodeList=ng17302
"""
    parsed = _parse_scontrol_show_job(sample)
    assert parsed["status"] == "COMPLETED"
    assert parsed["exit_code"] == "0:0"
    assert parsed["node"] == "ng17302"
    assert parsed["run_time_sec"] == 42 * 60 + 13


def test_parse_scontrol_show_job_handles_missing_fields() -> None:
    """Garbage / partial scontrol output must NOT crash; surface
    UNKNOWN so the caller can route to the marker-trail lookup."""
    parsed = _parse_scontrol_show_job("(unhelpful)")
    assert parsed["status"] == "UNKNOWN"
    assert parsed["exit_code"] is None
    assert parsed["run_time_sec"] is None


@pytest.mark.parametrize(
    ("runtime_val", "expected_sec"),
    [
        ("00:00:00", 0),
        ("00:05:00", 300),
        ("00:42:13", 42 * 60 + 13),
        ("1-00:00:05", 86_405),
        ("12:00:00", 12 * 3600),
        ("INVALID", None),  # scontrol's non-time value → fall back to submit age
        ("UNKNOWN", None),
    ],
)
def test_parse_slurm_runtime(runtime_val: str, expected_sec: int | None) -> None:
    """``RunTime=[days-]HH:MM:SS`` → seconds; non-time values → None."""
    assert _parse_slurm_runtime(runtime_val) == expected_sec


# ---------------------------------------------------------------------------
# State-mapping table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("slurm_state", "expected_status"),
    [
        ("PENDING", "running"),
        ("RUNNING", "running"),
        ("CONFIGURING", "running"),
        ("COMPLETING", "running"),
        ("COMPLETED", "done"),
        ("FAILED", "dead"),
        ("TIMEOUT", "dead"),
        ("PREEMPTED", "dead"),
        ("NODE_FAIL", "dead"),
        ("CANCELLED", "dead"),
        ("CANCELLED+", "dead"),
        ("OUT_OF_MEMORY", "dead"),
        ("SUSPENDED", "stalled"),
    ],
)
def test_slurm_state_table(slurm_state: str, expected_status: str) -> None:
    assert SLURM_STATE_TO_STATUS[slurm_state] == expected_status


# ---------------------------------------------------------------------------
# build_poll_result — happy path: SLURM RUNNING + fresh heartbeat = running
# ---------------------------------------------------------------------------


def _seed_local_state(
    tmp_path: Path,
    job_id: str,
    *,
    status_json_body: dict | None,
    job_out_lines: list[str] | None,
) -> Path:
    """Seed the (tmp_path-isolated) slurm-<id>/ dir with status.json + job.out."""
    local_dir = tmp_path / f"slurm-{job_id}"
    local_dir.mkdir(parents=True, exist_ok=True)
    status_path = local_dir / "status.json"
    job_out_path = local_dir / "job.out"
    if status_json_body is None:
        if status_path.exists():
            status_path.unlink()
    else:
        status_path.write_text(json.dumps(status_json_body))
    if job_out_lines is None:
        if job_out_path.exists():
            job_out_path.unlink()
    else:
        job_out_path.write_text("\n".join(job_out_lines))
    return local_dir


def test_build_poll_result_running_with_fresh_heartbeat(tmp_path: Path) -> None:
    job_id = "9101"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "sft",
            "heartbeat_ts": fresh_ts,
            "gpu_busy": True,
            "exit_code": "",
        },
        job_out_lines=["[phase=sft]", "step 100 loss=1.23"],
    )

    def fake_state(*, robot_alias, job_id):
        return {"status": "RUNNING", "exit_code": None}

    def fake_rsync(*, robot_alias, scratch_dir, job_id):
        return None  # files already seeded

    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=fake_state,
        rsyncer=fake_rsync,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **_kw: None,
        event_reader=lambda _issue: [],
    )
    assert poll.status == "running"
    assert poll.current_phase == "sft"
    assert poll.gpu_util == "busy"
    assert poll.pid_alive is True
    assert poll.new_milestone is True
    assert "[phase=sft]" in poll.log_tail_excerpt


def test_build_poll_result_running_when_heartbeat_stale_but_log_fresh(tmp_path: Path) -> None:
    """#1969 log-freshness veto — the #1900 false-read replay: SLURM
    RUNNING + heartbeat STALL_SEC+ stale while job.out is FRESH (the
    job is demonstrably alive by its own log) ⇒ running, streak 0.
    Pre-#1969 the heartbeat-only predicate read this tick stalled."""
    job_id = "9102"
    now = datetime.now(tz=UTC)
    stale_ts = (now - timedelta(seconds=STALL_SEC + 217)).isoformat().replace("+00:00", "Z")
    local = _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "sft",
            "heartbeat_ts": stale_ts,
            "gpu_busy": False,
            "exit_code": "",
        },
        job_out_lines=["[phase=sft]", "INFO fresh progress line"],  # fresh mtime
    )

    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **_kw: None,
        event_reader=lambda _issue: [],
    )
    assert poll.status == "running"
    assert poll.stall_reason is None
    assert not (local / "stall_streak.json").exists()


def test_build_poll_result_pending_is_running_not_stalled(tmp_path: Path) -> None:
    """A PENDING job that's writing nothing must NOT be reported as
    stalled — the selector's submit-and-park watchdog owns that logic."""
    job_id = "9103"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=None)
    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "PENDING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **_kw: None,
        event_reader=lambda _issue: [],
    )
    assert poll.status == "running"  # PENDING is treated as running, not stalled


def test_build_poll_result_terminal_states(tmp_path: Path) -> None:
    job_id = "9104"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "done",
            "heartbeat_ts": fresh_ts,
            "gpu_busy": False,
            "exit_code": "0",
        },
        job_out_lines=["[phase=done]"],
    )
    for slurm_state, expected in [
        ("COMPLETED", "done"),
        ("FAILED", "dead"),
        ("TIMEOUT", "dead"),
        ("PREEMPTED", "dead"),
        ("OUT_OF_MEMORY", "dead"),
    ]:
        poll = build_poll_result(
            issue=137,
            job_id=job_id,
            cluster=_nibi(),
            scratch_dir="/scratch/tjiral/eps/issue-137",
            log_path="/scratch/tjiral/eps/issue-137/job.out",
            state_querier=lambda *, robot_alias, job_id, _s=slurm_state: {
                "status": _s,
                "exit_code": None,
            },
            rsyncer=lambda **_: None,
            now_fn=lambda: now.timestamp(),
            marker_poster=lambda **_kw: None,
            event_reader=lambda _issue: [],
        )
        assert poll.status == expected, f"{slurm_state} -> {poll.status} (expected {expected})"


def test_build_poll_result_preflight_failure_shortcut(tmp_path: Path) -> None:
    """The sbatch echoes ``[phase=preflight-failed]`` then exits non-zero.
    The monitor flips to ``dead`` even before SLURM transitions to FAILED."""
    job_id = "9105"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "preflight",
            "heartbeat_ts": fresh_ts,
            "gpu_busy": False,
            "exit_code": "",
        },
        job_out_lines=[
            "[phase=preflight]",
            "[FAIL] HF_TOKEN missing",
            "[phase=preflight-failed]",
        ],
    )
    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        # SLURM may still report RUNNING for a moment before reaping.
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **_kw: None,
        event_reader=lambda _issue: [],
    )
    assert poll.status == "dead"
    assert poll.current_phase == "preflight-failed"


def test_build_poll_result_missing_status_json_treats_as_stalled(tmp_path: Path) -> None:
    """SLURM RUNNING + status.json absent ⇒ heartbeat infinitely old.
    With the job.out ALSO stale, #1969 reads a suspect at tick 1 and
    stalled at tick 2 (one extra tick vs the pre-#1969 predicate)."""
    job_id = "9106"
    now = datetime.now(tz=UTC)
    local = _seed_local_state(
        tmp_path, job_id, status_json_body=None, job_out_lines=["random output"]
    )
    old_epoch = now.timestamp() - 3600
    os.utime(local / "job.out", (old_epoch, old_epoch))

    def _tick(at: float):
        return build_poll_result(
            issue=137,
            job_id=job_id,
            cluster=_nibi(),
            scratch_dir="/scratch/tjiral/eps/issue-137",
            log_path="/scratch/tjiral/eps/issue-137/job.out",
            state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
            rsyncer=lambda **_: None,
            now_fn=lambda: at,
            marker_poster=lambda **_kw: None,
            event_reader=lambda _issue: [],
        )

    assert _tick(now.timestamp()).status == "running"  # suspect tick
    assert _tick(now.timestamp() + STALL_SEC + 60).status == "stalled"


# ---------------------------------------------------------------------------
# §7 lane extension — adaptive bg-poll interval on the SLURM lane
# ---------------------------------------------------------------------------


def _quiet_poll_kwargs(tmp_path: Path, job_id: str, now: datetime) -> dict:
    """build_poll_result kwargs for a fully quiet tick: SLURM RUNNING,
    fresh heartbeat, NO ``[phase=...]`` line left in the job.out tail
    (phase comes from status.json), submit 2 h ago."""
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "sft",
            "heartbeat_ts": fresh_ts,
            "gpu_busy": True,
            "exit_code": "",
        },
        job_out_lines=["step 100 loss=1.23", "step 200 loss=1.10"],
    )
    return {
        "issue": 137,
        "job_id": job_id,
        "cluster": _nibi(),
        "scratch_dir": "/scratch/tjiral/eps/issue-137",
        "log_path": "/scratch/tjiral/eps/issue-137/job.out",
        "state_querier": lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        "rsyncer": lambda **_: None,
        "now_fn": lambda: now.timestamp(),
        "marker_poster": lambda **_kw: None,
        "event_reader": lambda _issue: [],
        "submitted_at": now.timestamp() - 7200,
    }


def test_build_poll_result_quiet_running_emits_quiet_interval(tmp_path: Path) -> None:
    """§7 lane extension: SLURM RUNNING, fresh heartbeat, phase line
    scrolled out of the tail, past the early-run window ⇒ quiet interval."""
    now = datetime.now(tz=UTC)
    poll = build_poll_result(**_quiet_poll_kwargs(tmp_path, "9301", now))
    assert poll.status == "running"
    assert poll.new_milestone is False
    assert poll.next_interval == POLL_INTERVAL_QUIET_SEC


def test_build_poll_result_milestone_in_tail_keeps_short_interval(tmp_path: Path) -> None:
    """A ``[phase=...]`` line still in the 16 KiB tail is the lane's
    milestone signal — sticky-conservative, so the tick stays short."""
    now = datetime.now(tz=UTC)
    kwargs = _quiet_poll_kwargs(tmp_path, "9302", now)
    _seed_local_state(
        tmp_path,
        "9302",
        status_json_body={
            "phase": "sft",
            "heartbeat_ts": now.isoformat().replace("+00:00", "Z"),
            "gpu_busy": True,
            "exit_code": "",
        },
        job_out_lines=["[phase=sft]", "step 100 loss=1.23"],
    )
    poll = build_poll_result(**kwargs)
    assert poll.status == "running"
    assert poll.new_milestone is True
    assert poll.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_build_poll_result_early_run_keeps_short_interval(tmp_path: Path) -> None:
    """Inside the early-run window (submit < ~30 min ago) ⇒ short interval."""
    now = datetime.now(tz=UTC)
    kwargs = _quiet_poll_kwargs(tmp_path, "9303", now)
    kwargs["submitted_at"] = now.timestamp() - 600
    poll = build_poll_result(**kwargs)
    assert poll.status == "running"
    assert poll.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_build_poll_result_without_submitted_at_keeps_short_interval(tmp_path: Path) -> None:
    """A legacy handle without ``submitted_at`` has unknown run age ⇒
    counts as early-run (fail toward coverage) ⇒ short interval."""
    now = datetime.now(tz=UTC)
    kwargs = _quiet_poll_kwargs(tmp_path, "9304", now)
    kwargs["submitted_at"] = None
    poll = build_poll_result(**kwargs)
    assert poll.status == "running"
    assert poll.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_build_poll_result_long_pending_fresh_start_keeps_short_interval(
    tmp_path: Path,
) -> None:
    """Long-PENDING regression: submitted 2 h ago but SLURM's RunTime says
    the job STARTED 5 min ago (queue time ≠ run time). The early-run guard
    must read the RunTime, not the inflated submit age — pre-fix, the first
    RUNNING ticks of a long-queued job could go quiet (1800 s) the moment
    real output scrolled the ``[phase=...]`` line out of the tail."""
    now = datetime.now(tz=UTC)
    kwargs = _quiet_poll_kwargs(tmp_path, "9307", now)  # submitted_at = now - 7200
    kwargs["state_querier"] = lambda *, robot_alias, job_id: {
        "status": "RUNNING",
        "exit_code": None,
        "run_time_sec": 300,
    }
    poll = build_poll_result(**kwargs)
    assert poll.status == "running"
    assert poll.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_build_poll_result_runtime_past_window_goes_quiet(tmp_path: Path) -> None:
    """Counterpart: once SLURM's RunTime itself clears the early-run
    window, a fully quiet tick still earns the long interval (the
    RunTime preference must not pin the lane short forever)."""
    now = datetime.now(tz=UTC)
    kwargs = _quiet_poll_kwargs(tmp_path, "9308", now)
    kwargs["state_querier"] = lambda *, robot_alias, job_id: {
        "status": "RUNNING",
        "exit_code": None,
        "run_time_sec": 2400,
    }
    poll = build_poll_result(**kwargs)
    assert poll.status == "running"
    assert poll.next_interval == POLL_INTERVAL_QUIET_SEC


def test_build_poll_result_fresh_start_after_long_queue_not_stalled(tmp_path: Path) -> None:
    """C2 stall-floor sibling of the long-PENDING regression: a job that
    started 60 s ago after a 2 h queue has no heartbeat yet (status.json
    not written/rsynced). The stall clock must floor at the RUN age —
    pre-fix the floor used ``now - submitted_at`` (7200 s > STALL_SEC),
    so the first RUNNING tick read a LIVE just-started job as stalled."""
    job_id = "9309"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=["booting"])
    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {
            "status": "RUNNING",
            "exit_code": None,
            "run_time_sec": 60,
        },
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **_kw: None,
        event_reader=lambda _issue: [],
        submitted_at=now.timestamp() - 7200,
    )
    assert poll.status == "running"


def test_build_poll_result_pending_keeps_short_interval(tmp_path: Path) -> None:
    """PENDING maps to ``running`` for the orchestrator, but a non-RUNNING
    SLURM state is the lane anomaly — transitional ticks never go quiet."""
    now = datetime.now(tz=UTC)
    kwargs = _quiet_poll_kwargs(tmp_path, "9305", now)
    kwargs["state_querier"] = lambda *, robot_alias, job_id: {
        "status": "PENDING",
        "exit_code": None,
    }
    poll = build_poll_result(**kwargs)
    assert poll.status == "running"
    assert poll.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_build_poll_result_stalled_keeps_short_interval(tmp_path: Path) -> None:
    """Neither a suspect tick nor a stalled verdict ever goes quiet
    (#1969: suspect via the lane anomaly; stalled via status)."""
    now = datetime.now(tz=UTC)
    kwargs = _quiet_poll_kwargs(tmp_path, "9306", now)
    stale_ts = (now - timedelta(seconds=3600)).isoformat().replace("+00:00", "Z")
    local = _seed_local_state(
        tmp_path,
        "9306",
        status_json_body={
            "phase": "sft",
            "heartbeat_ts": stale_ts,
            "gpu_busy": False,
            "exit_code": "",
        },
        job_out_lines=["step 100 loss=1.23"],
    )
    old_epoch = now.timestamp() - 3600
    os.utime(local / "job.out", (old_epoch, old_epoch))
    first = build_poll_result(**kwargs)
    assert first.status == "running"
    assert first.stall_reason == "slurm_stall_suspect"
    assert first.next_interval == POLL_INTERVAL_DEFAULT_SEC
    kwargs["now_fn"] = lambda: now.timestamp() + STALL_SEC + 60
    second = build_poll_result(**kwargs)
    assert second.status == "stalled"
    assert second.next_interval == POLL_INTERVAL_DEFAULT_SEC


# ---------------------------------------------------------------------------
# Blocker 2: monitor posts epm:cluster-poll on transition + epm:cluster-terminal
# exactly once + idempotent reconnect reads the persisted terminal marker.
# ---------------------------------------------------------------------------


def _capture_markers(captured: list[dict]):
    def fake(**kwargs):
        captured.append(kwargs)

    return fake


def test_monitor_posts_cluster_poll_on_first_observation(tmp_path: Path) -> None:
    """First poll for a job MUST post epm:cluster-poll v1 (no prior
    cluster-poll in events.jsonl to dedup against)."""
    job_id = "9201"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "sft", "heartbeat_ts": fresh_ts, "gpu_busy": True},
        job_out_lines=["[phase=sft]"],
    )

    posted: list[dict] = []
    build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=_capture_markers(posted),
        event_reader=lambda _issue: [],
    )
    polls = [m for m in posted if m["marker"] == "epm:cluster-poll"]
    assert len(polls) == 1
    body = json.loads(polls[0]["note"])
    assert body["job_id"] == "9201"
    assert body["status"] == "running"
    assert body["current_phase"] == "sft"
    assert body["slurm_state"] == "RUNNING"
    assert body["gpu_util"] == "busy"
    # Also asserts issue is threaded for the dashboard.
    assert polls[0]["issue"] == 137


def test_monitor_dedups_cluster_poll_when_status_unchanged(tmp_path: Path) -> None:
    """Status + phase + slurm_state unchanged vs the last cluster-poll
    for this job_id MUST NOT post a fresh marker (keeps the trail
    readable on long hours-stable phases)."""
    job_id = "9202"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "sft", "heartbeat_ts": fresh_ts, "gpu_busy": True},
        job_out_lines=["[phase=sft]"],
    )

    prior_event = {
        "kind": "epm:cluster-poll",
        "note": json.dumps(
            {
                "job_id": "9202",
                "status": "running",
                "current_phase": "sft",
                "slurm_state": "RUNNING",
            }
        ),
    }

    posted: list[dict] = []
    build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=_capture_markers(posted),
        event_reader=lambda _issue: [prior_event],
    )
    polls = [m for m in posted if m["marker"] == "epm:cluster-poll"]
    assert polls == [], "duplicate cluster-poll posted despite identical status/phase"


def test_monitor_posts_cluster_terminal_first_time_on_completed(tmp_path: Path) -> None:
    """First COMPLETED observation MUST post epm:cluster-terminal v1
    with next_action='interpret'."""
    job_id = "9203"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "done",
            "heartbeat_ts": fresh_ts,
            "gpu_busy": False,
            "exit_code": "0",
        },
        job_out_lines=["[phase=done]"],
    )

    posted: list[dict] = []
    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {
            "status": "COMPLETED",
            "exit_code": "0:0",
        },
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=_capture_markers(posted),
        event_reader=lambda _issue: [],
    )
    assert poll.status == "done"
    terminals = [m for m in posted if m["marker"] == "epm:cluster-terminal"]
    assert len(terminals) == 1
    body = json.loads(terminals[0]["note"])
    assert body["job_id"] == "9203"
    assert body["slurm_state"] == "COMPLETED"
    assert body["next_action"] == "interpret"
    assert body["exit_code"] == "0:0"


def test_monitor_does_not_double_post_cluster_terminal(tmp_path: Path) -> None:
    """If a terminal marker already exists for this job_id, a second
    terminal observation MUST NOT post another (idempotent)."""
    job_id = "9204"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "done", "heartbeat_ts": fresh_ts, "gpu_busy": False},
        job_out_lines=["[phase=done]"],
    )
    prior_terminal = {
        "kind": "epm:cluster-terminal",
        "note": json.dumps(
            {
                "job_id": "9204",
                "cluster": "nibi",
                "slurm_state": "COMPLETED",
                "exit_code": "0:0",
                "observed_at": "2026-06-08T01:02:03Z",
                "next_action": "interpret",
                "status": "done",
            }
        ),
    }
    posted: list[dict] = []
    build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {
            "status": "COMPLETED",
            "exit_code": "0:0",
        },
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=_capture_markers(posted),
        event_reader=lambda _issue: [prior_terminal],
    )
    terminals = [m for m in posted if m["marker"] == "epm:cluster-terminal"]
    assert terminals == [], "double-posted epm:cluster-terminal on already-terminal job"


def test_monitor_reads_persisted_terminal_on_slurm_unknown(tmp_path: Path) -> None:
    """When squeue/scontrol both age out (status=UNKNOWN), the monitor
    MUST synthesize the PollResult from the persisted epm:cluster-terminal
    v1 body — NOT default to running and loop forever."""
    job_id = "9205"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=None)

    prior_terminal = {
        "kind": "epm:cluster-terminal",
        "note": json.dumps(
            {
                "job_id": "9205",
                "cluster": "nibi",
                "slurm_state": "COMPLETED",
                "exit_code": "0:0",
                "observed_at": "2026-06-08T01:02:03Z",
                "next_action": "interpret",
                "status": "done",
            }
        ),
    }
    posted: list[dict] = []
    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "UNKNOWN", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=_capture_markers(posted),
        event_reader=lambda _issue: [prior_terminal],
    )
    # Authoritative answer comes from the persisted marker.
    assert poll.status == "done"
    assert poll.current_phase == "completed"
    # No duplicate posts on the reconnect path.
    assert posted == []


def test_monitor_filters_events_by_job_id(tmp_path: Path) -> None:
    """A task that ran on the cluster twice (two job_ids) MUST NOT
    inherit the first job's terminal verdict on the second run."""
    job_id = "9206"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "sft", "heartbeat_ts": fresh_ts, "gpu_busy": True},
        job_out_lines=["[phase=sft]"],
    )

    other_job_terminal = {
        "kind": "epm:cluster-terminal",
        "note": json.dumps(
            {
                "job_id": "9999",
                "cluster": "nibi",
                "slurm_state": "FAILED",
                "exit_code": "1:0",
                "observed_at": "2026-06-08T00:00:00Z",
                "next_action": "investigate",
                "status": "dead",
            }
        ),
    }
    posted: list[dict] = []
    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=_capture_markers(posted),
        event_reader=lambda _issue: [other_job_terminal],
    )
    # The other job's terminal MUST NOT short-circuit this job's poll.
    assert poll.status == "running"
    # And we DO post a fresh cluster-poll for this job_id.
    polls = [m for m in posted if m["marker"] == "epm:cluster-poll"]
    assert len(polls) == 1


def test_monitor_posts_cluster_poll_again_on_phase_transition(tmp_path: Path) -> None:
    """Same status but a NEW phase MUST trigger a fresh cluster-poll."""
    job_id = "9207"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "dpo", "heartbeat_ts": fresh_ts, "gpu_busy": True},
        job_out_lines=["[phase=sft]", "[phase=dpo]"],
    )
    prior_poll = {
        "kind": "epm:cluster-poll",
        "note": json.dumps(
            {
                "job_id": "9207",
                "status": "running",
                "current_phase": "sft",
                "slurm_state": "RUNNING",
            }
        ),
    }
    posted: list[dict] = []
    build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=_capture_markers(posted),
        event_reader=lambda _issue: [prior_poll],
    )
    polls = [m for m in posted if m["marker"] == "epm:cluster-poll"]
    assert len(polls) == 1
    body = json.loads(polls[0]["note"])
    assert body["current_phase"] == "dpo"


# ---------------------------------------------------------------------------
# fetch_started_evidence — terminal-before-running workload classification
# ---------------------------------------------------------------------------


def test_fetch_started_evidence_returns_phase_and_tail(tmp_path: Path) -> None:
    """Runtime artifacts in the scratch dir (status.json / job.out) prove
    the job STARTED — the router uses this to classify a fast-failing
    job as a workload failure instead of no-compute.

    Files are seeded INSIDE the injected rsyncer: the probe clears its
    local cache at start (round-6 C2), so pre-seeded files simulate the
    wrong thing (a stale prior tick, which must be wiped)."""
    job_id = "9501"

    def seeding_rsync(**_kw) -> None:
        _seed_local_state(
            tmp_path,
            job_id,
            status_json_body={"phase": "preflight-failed", "exit_code": "1"},
            job_out_lines=[
                "[FAIL] secrets file /scratch/tjiral/eps/issue-535/secrets.env not found",
                "[phase=preflight-failed]",
            ],
        )

    evidence = fetch_started_evidence(
        robot_alias="robot-nibi",
        scratch_dir="/scratch/tjiral/eps/issue-535",
        job_id=job_id,
        rsyncer=seeding_rsync,
    )
    assert evidence is not None
    assert evidence["phase"] == "preflight-failed"
    assert "[FAIL] secrets file" in evidence["job_out_tail"]
    assert evidence["status_json"]["exit_code"] == "1"


def test_fetch_started_evidence_returns_none_when_no_artifacts(tmp_path: Path) -> None:
    """No status.json AND no job.out = the job never started — the
    router's legacy no_compute classification stands."""

    job_id = "9502"
    evidence = fetch_started_evidence(
        robot_alias="robot-nibi",
        scratch_dir="/scratch/tjiral/eps/issue-999",
        job_id=job_id,
        rsyncer=lambda **_kw: None,  # rsync "succeeded" but pulled nothing
    )
    assert evidence is None


def test_fetch_started_evidence_job_out_alone_counts(tmp_path: Path) -> None:
    """A job.out with no status.json still proves the job ran (the
    sbatch writes job.out via --output the moment the job starts)."""
    job_id = "9503"

    def seeding_rsync(**_kw) -> None:
        _seed_local_state(
            tmp_path,
            job_id,
            status_json_body=None,
            job_out_lines=["early crash before status.json writer armed"],
        )

    evidence = fetch_started_evidence(
        robot_alias="robot-nibi",
        scratch_dir="/scratch/tjiral/eps/issue-998",
        job_id=job_id,
        rsyncer=seeding_rsync,
    )
    assert evidence is not None
    assert "early crash" in evidence["job_out_tail"]


def test_fetch_started_evidence_clears_stale_local_cache(tmp_path: Path) -> None:
    """Round-6 C2(3): files left by a PREVIOUS tick (or a colliding
    job id from another cluster) are wiped at probe start — a no-op
    rsync must yield None, never the stale files."""
    job_id = "9506"
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "sft"},
        job_out_lines=["stale prior-tick content"],
    )
    evidence = fetch_started_evidence(
        robot_alias="robot-nibi",
        scratch_dir="/scratch/tjiral/eps/issue-997",
        job_id=job_id,
        rsyncer=lambda **_kw: None,  # transport failure / nothing pulled
    )
    assert evidence is None


def test_fetch_started_evidence_stale_artifacts_gated_out(tmp_path: Path) -> None:
    """Round-6 C2(1): artifacts older than THIS attempt's submit time are
    the PREVIOUS attempt's (per-issue scratch dir; SLURM truncates
    --output only when the new job starts) — they must NOT classify the
    new job as a workload failure. Live shape: attempt-1 heartbeat
    20:26Z vs attempt-2 submit 20:57Z."""
    job_id = "9507"
    now = time.time()
    stale_epoch = now - 1860  # 31 min ago
    stale_iso = datetime.fromtimestamp(stale_epoch, tz=UTC).isoformat().replace("+00:00", "Z")

    def seeding_rsync(**_kw) -> None:
        local = _seed_local_state(
            tmp_path,
            job_id,
            status_json_body={"phase": "sft", "heartbeat_ts": stale_iso},
            job_out_lines=["attempt-1 output", "[phase=sft]"],
        )
        os.utime(local / "job.out", (stale_epoch, stale_epoch))
        os.utime(local / "status.json", (stale_epoch, stale_epoch))

    evidence = fetch_started_evidence(
        robot_alias="robot-nibi",
        scratch_dir="/scratch/tjiral/eps/issue-535",
        job_id=job_id,
        rsyncer=seeding_rsync,
        min_artifact_ts=now,  # this attempt submitted NOW
    )
    assert evidence is None


def test_fetch_started_evidence_fresh_artifacts_pass_the_gate(tmp_path: Path) -> None:
    """Artifacts written AFTER this attempt's submit ARE evidence."""
    job_id = "9508"
    now = time.time()
    fresh_iso = datetime.fromtimestamp(now, tz=UTC).isoformat().replace("+00:00", "Z")

    def seeding_rsync(**_kw) -> None:
        _seed_local_state(
            tmp_path,
            job_id,
            status_json_body={"phase": "preflight-failed", "heartbeat_ts": fresh_iso},
            job_out_lines=["[phase=preflight-failed]"],
        )

    evidence = fetch_started_evidence(
        robot_alias="robot-nibi",
        scratch_dir="/scratch/tjiral/eps/issue-535",
        job_id=job_id,
        rsyncer=seeding_rsync,
        min_artifact_ts=now - 600,  # submitted 10 min ago; artifacts written now
    )
    assert evidence is not None
    assert evidence["phase"] == "preflight-failed"


def test_fetch_started_evidence_scrubs_tokens_from_tail(tmp_path: Path) -> None:
    """Round-6 C1: the evidence tail lands in git-committed markers
    (epm:backend-selected extra.evidence, epm:failure evidence) — secret
    tokens must be redacted BEFORE truncation."""
    job_id = "9509"
    hf_token = "hf_" + "A" * 30
    wandb_token = "wandb_v1_" + "b" * 28

    def seeding_rsync(**_kw) -> None:
        _seed_local_state(
            tmp_path,
            job_id,
            status_json_body=None,
            job_out_lines=[
                f"+ : {hf_token}",
                f"+ : {wandb_token}",
                "[phase=preflight-failed]",
            ],
        )

    evidence = fetch_started_evidence(
        robot_alias="robot-nibi",
        scratch_dir="/scratch/tjiral/eps/issue-535",
        job_id=job_id,
        rsyncer=seeding_rsync,
    )
    assert evidence is not None
    assert hf_token not in evidence["job_out_tail"]
    assert wandb_token not in evidence["job_out_tail"]
    assert "«REDACTED»" in evidence["job_out_tail"]
    assert "[phase=preflight-failed]" in evidence["job_out_tail"]


# ---------------------------------------------------------------------------
# Secret-token scrubber (round-6 C1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    [
        "hf_" + "Ab1" * 10,  # HF token
        "wandb_v1_" + "x_9" * 8,  # WandB v1 key
        "sk-proj-" + "Z" * 24,  # OpenAI project key
        "sk-" + "a" * 24,  # OpenAI classic key
        "0123456789abcdef" * 2 + "01234567",  # 40-hex (legacy WandB key)
    ],
)
def test_scrub_secret_tokens_redacts_known_shapes(token: str) -> None:
    text = f"+ : {token}\nsome surrounding line\n"
    out = _scrub_secret_tokens(text)
    assert token not in out
    assert "«REDACTED»" in out
    assert "some surrounding line" in out


def test_scrub_secret_tokens_leaves_normal_log_lines_alone() -> None:
    text = "[phase=sft]\nstep 100 loss=1.23\nhf_short\nsaving to /scratch/eps\n"
    assert _scrub_secret_tokens(text) == text


def test_cluster_poll_marker_tail_is_scrubbed(tmp_path: Path) -> None:
    """The epm:cluster-poll log_tail_excerpt is committed to git — the
    monitor must redact tokens that leaked into job.out (the issue-535
    live run traced both HF and WandB tokens via xtrace)."""
    job_id = "9510"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    hf_token = "hf_" + "C" * 30
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "sft", "heartbeat_ts": fresh_ts, "gpu_busy": True},
        job_out_lines=[f"+ : {hf_token}", "[phase=sft]"],
    )
    posted: list[dict] = []
    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=_capture_markers(posted),
        event_reader=lambda _issue: [],
    )
    assert hf_token not in poll.log_tail_excerpt
    polls = [m for m in posted if m["marker"] == "epm:cluster-poll"]
    assert len(polls) == 1
    assert hf_token not in polls[0]["note"]
    # json.dumps escapes the guillemets («…) — parse before checking.
    assert "«REDACTED»" in json.loads(polls[0]["note"])["log_tail_excerpt"]


# ---------------------------------------------------------------------------
# Monitor attempt-freshness gate (round-6 C2 — the live failure chain)
# ---------------------------------------------------------------------------


def test_monitor_ignores_prior_attempt_heartbeat_just_after_submit(tmp_path: Path) -> None:
    """The issue-535 attempt-2 chain: SLURM RUNNING + a 31-min-old
    PRIOR-attempt heartbeat, one minute after submit → the stall clock
    is floored at now-submit, so the poll reports running, NOT stalled."""
    job_id = "9601"
    now = datetime.now(tz=UTC)
    stale_ts = (now - timedelta(minutes=31)).isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "sft", "heartbeat_ts": stale_ts, "gpu_busy": False},
        job_out_lines=None,
    )
    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **_kw: None,
        event_reader=lambda _issue: [],
        submitted_at=now.timestamp() - 60,  # submitted one minute ago
    )
    assert poll.status == "running"


def test_monitor_still_stalls_long_after_submit_without_fresh_heartbeat(tmp_path: Path) -> None:
    """The floor only protects the young-job window: a job submitted
    well past STALL_SEC ago with no fresh heartbeat AND no log output is
    still stalled — suspect at tick 1, stalled at tick 2 (#1969)."""
    job_id = "9602"
    now = datetime.now(tz=UTC)
    stale_ts = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "sft", "heartbeat_ts": stale_ts, "gpu_busy": False},
        job_out_lines=None,
    )

    def _tick(at: float):
        return build_poll_result(
            issue=137,
            job_id=job_id,
            cluster=_nibi(),
            scratch_dir="/scratch/tjiral/eps/issue-137",
            log_path="/scratch/tjiral/eps/issue-137/job.out",
            state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
            rsyncer=lambda **_: None,
            now_fn=lambda: at,
            marker_poster=lambda **_kw: None,
            event_reader=lambda _issue: [],
            submitted_at=now.timestamp() - (STALL_SEC + FRESHNESS_SKEW_MARGIN_SEC + 120),
        )

    first = _tick(now.timestamp())
    assert first.status == "running"
    assert first.stall_reason == "slurm_stall_suspect"
    assert _tick(now.timestamp() + STALL_SEC + 60).status == "stalled"


def test_monitor_ignores_prior_attempt_preflight_marker(tmp_path: Path) -> None:
    """A stale job.out carrying ``[phase=preflight-failed]`` from the
    PREVIOUS attempt must not flip the NEW job to dead."""
    job_id = "9603"
    now = datetime.now(tz=UTC)
    stale_epoch = now.timestamp() - 1800
    local = _seed_local_state(
        tmp_path,
        job_id,
        status_json_body=None,
        job_out_lines=["[FAIL] secrets file not found", "[phase=preflight-failed]"],
    )
    os.utime(local / "job.out", (stale_epoch, stale_epoch))
    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **_kw: None,
        event_reader=lambda _issue: [],
        submitted_at=now.timestamp() - 60,
    )
    assert poll.status == "running"
    assert poll.current_phase != "preflight-failed"


def test_monitor_without_submitted_at_keeps_legacy_behavior(tmp_path: Path) -> None:
    """Back-compat: handles without a ``submitted_at`` stamp (pre-fix
    sidecars, reconnect handles) keep the ungated C2 stall semantics —
    still stalled once the #1969 streak reaches two ticks."""
    job_id = "9604"
    now = datetime.now(tz=UTC)
    stale_ts = (now - timedelta(seconds=STALL_SEC + 60)).isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "sft", "heartbeat_ts": stale_ts, "gpu_busy": False},
        job_out_lines=None,
    )

    def _tick(at: float):
        return build_poll_result(
            issue=137,
            job_id=job_id,
            cluster=_nibi(),
            scratch_dir="/scratch/tjiral/eps/issue-137",
            log_path="/scratch/tjiral/eps/issue-137/job.out",
            state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
            rsyncer=lambda **_: None,
            now_fn=lambda: at,
            marker_poster=lambda **_kw: None,
            event_reader=lambda _issue: [],
        )

    assert _tick(now.timestamp()).status == "running"  # suspect tick
    assert _tick(now.timestamp() + STALL_SEC + 60).status == "stalled"


# ---------------------------------------------------------------------------
# Probe-failure vs job-absent distinction (round-6 B1)
# ---------------------------------------------------------------------------


def _fake_run_factory(results: list[subprocess.CompletedProcess]):
    """Sequential subprocess.run stub: pops one CompletedProcess per call."""
    queue = list(results)

    def fake_run(argv, **_kw):
        return queue.pop(0)

    return fake_run


def _proc(rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["ssh"], returncode=rc, stdout=stdout, stderr=stderr)


def test_query_by_name_rc_nonzero_raises_probe_error(monkeypatch) -> None:
    """rc != 0 = the PROBE failed (wrapper rejection / ssh transport) —
    must raise, never read as "job gone" (the live diagnosis: the
    quote-stripping wrapper failed multi-token formats with
    ``Unrecognized option: %T`` and the failure read as absent)."""
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm_monitor.subprocess.run",
        _fake_run_factory([_proc(1, stderr="Unrecognized option: %T")]),
    )
    with pytest.raises(SlurmProbeError):
        query_by_name(robot_alias="robot-nibi", job_name="eps-issue-137")


def test_query_by_name_timeout_raises_probe_error(monkeypatch) -> None:
    """A HUNG squeue (wedged slurmctld; TimeoutExpired) is a PROBE
    failure, not "job gone" — pre-fix it bypassed the typed-error
    contract and the reconnect path blind-double-submitted over a
    possibly-live job's scratch (round-7 M1)."""

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=30)

    monkeypatch.setattr(
        "explore_persona_space.backends.slurm_monitor.subprocess.run",
        _raise_timeout,
    )
    with pytest.raises(SlurmProbeError):
        query_by_name(robot_alias="robot-nibi", job_name="eps-issue-137")


def test_query_slurm_state_timeout_raises_probe_error(monkeypatch) -> None:
    """Same hang-shape contract for the scontrol/squeue state probe."""

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=30)

    monkeypatch.setattr(
        "explore_persona_space.backends.slurm_monitor.subprocess.run",
        _raise_timeout,
    )
    with pytest.raises(SlurmProbeError):
        query_slurm_state(robot_alias="robot-nibi", job_id="15859991")


def test_query_by_name_rc_zero_empty_means_absent(monkeypatch) -> None:
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm_monitor.subprocess.run",
        _fake_run_factory([_proc(0, stdout="")]),
    )
    assert query_by_name(robot_alias="robot-nibi", job_name="eps-issue-137") is None


def test_query_by_name_rc_zero_with_id_returns_it(monkeypatch) -> None:
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm_monitor.subprocess.run",
        _fake_run_factory([_proc(0, stdout="15859991\n")]),
    )
    assert query_by_name(robot_alias="robot-nibi", job_name="eps-issue-137") == "15859991"


def test_query_slurm_state_transport_failure_raises_probe_error(monkeypatch) -> None:
    """Both scontrol and squeue failing with a NON-"invalid job id"
    stderr = transport down → typed probe error, not UNKNOWN."""
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm_monitor.subprocess.run",
        _fake_run_factory(
            [
                _proc(255, stderr="ssh: connect to host nibi port 22: Connection refused"),
                _proc(255, stderr="ssh: connect to host nibi port 22: Connection refused"),
            ]
        ),
    )
    with pytest.raises(SlurmProbeError):
        query_slurm_state(robot_alias="robot-nibi", job_id="15859991")


def test_query_slurm_state_invalid_job_id_is_unknown(monkeypatch) -> None:
    """SLURM's explicit "Invalid job id specified" = genuinely absent
    (aged out) → UNKNOWN, so the persisted-terminal lookup resolves it."""
    monkeypatch.setattr(
        "explore_persona_space.backends.slurm_monitor.subprocess.run",
        _fake_run_factory(
            [
                _proc(1, stderr="slurm_load_jobs error: Invalid job id specified"),
                _proc(1, stderr="slurm_load_jobs error: Invalid job id specified"),
            ]
        ),
    )
    state = query_slurm_state(robot_alias="robot-nibi", job_id="15859991")
    assert state["status"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# Fellows sentinel drain (#1898) — drain_cluster_sentinels + threading
# ---------------------------------------------------------------------------
#
# The drain reuses scripts/poll_pipeline.py's transport-agnostic helpers
# (sentinel_drain_shell / parse_sentinel_stream / drain_sentinels_via /
# _ssh_mark_processed) over plain `ssh charmander`. The drain-list ssh is
# faked at the injected ``runner=`` boundary; the VM-side post + the
# `.processed` rename are intercepted on the ``scripts.poll_pipeline``
# module object (the SAME module drain_cluster_sentinels lazy-imports
# from, so call-time name lookups resolve to the patches — the
# test_poll_pipeline_sentinels.py convention).


def _fellows():
    return get_cluster_config("fellows")


def _pp_module():
    """The scripts.poll_pipeline module object drain_cluster_sentinels
    lazy-imports from (repo root is on sys.path under pytest)."""
    import scripts.poll_pipeline as pp

    return pp


def _drain_body(
    *,
    kind: str = "epm:progress",
    gate: str | None = None,
    blocks_pipeline: bool | None = None,
    note: str = "fellows drain test",
) -> str:
    body = {
        "sentinel_schema_version": 1,
        "task_id": 9999,
        "kind": kind,
        "version": 1,
        "note": note,
        "by": "pod-sentinel",
        "ts": "2026-07-30T00:00:00+00:00",
    }
    if gate is not None:
        body["gate"] = gate
    if blocks_pipeline is not None:
        body["blocks_pipeline"] = blocks_pipeline
    return json.dumps(body)


def _drain_stream(*pairs: tuple[str, str]) -> str:
    """The SENTINEL_START/END stdout shape sentinel_drain_shell emits."""
    out: list[str] = []
    for path, body in pairs:
        out.append(f"SENTINEL_START {path}")
        out.append(body)
        out.append("SENTINEL_END")
    return "\n".join(out) + ("\n" if out else "")


class _DrainRunner:
    """Fake for the drain-list ssh (the ``runner=`` seam)."""

    def __init__(self, *, stdout: str = "", returncode: int = 0, raise_timeout: bool = False):
        self.stdout = stdout
        self.returncode = returncode
        self.raise_timeout = raise_timeout
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        assert cmd[0] == "ssh", f"expected ssh argv, got {cmd!r}"
        self.calls.append(cmd)
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(cmd, 60)
        return subprocess.CompletedProcess(
            args=cmd, returncode=self.returncode, stdout=self.stdout, stderr="boom"
        )


def _patch_pp(monkeypatch):
    """Intercept the VM-side post + rename on the poll_pipeline module.

    Returns (posts, mv_calls). ``list_events`` is stubbed to "no prior
    events" so the #1084 fp-dedupe read never touches real task state
    (the test_poll_pipeline_sentinels.py ``_stub_no_prior_events``
    convention).
    """
    pp = _pp_module()
    posts: list[dict] = []
    mv_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(pp, "post_event", lambda *a, **kw: posts.append(kw) or {})
    monkeypatch.setattr(pp, "list_events", lambda _issue: [])
    monkeypatch.setattr(
        pp, "_ssh_mark_processed", lambda host, path: mv_calls.append((host, path)) or True
    )
    return posts, mv_calls


def test_drain_parses_and_posts_multi_sentinel_stream(monkeypatch) -> None:
    from explore_persona_space.backends.slurm_monitor import drain_cluster_sentinels

    posts, mv_calls = _patch_pp(monkeypatch)
    p1 = "/workspace/logs/issue-9999-epm_progress-100.json"
    p2 = "/workspace/logs/issue-9999-epm_progress-200.json"
    runner = _DrainRunner(stdout=_drain_stream((p1, _drain_body()), (p2, _drain_body())))
    processed, gate = drain_cluster_sentinels(
        9999, _fellows(), "/workspace/superkaiba/eps/issue-9999", runner=runner
    )
    assert (processed, gate) == (2, None)
    assert len(posts) == 2
    assert [c[1] for c in mv_calls] == [p1, p2]
    assert all(host == "charmander" for host, _ in mv_calls)


def test_drain_gate_field_threads_and_flips_status(tmp_path: Path, monkeypatch) -> None:
    """#1898 critic round-1 Must-Fix 1: a drained BLOCKING gate flips the
    tick to status "gate" (the gcp.py merge, Step 6d.4 park); a benign
    ``gate=phase`` / ``blocks_pipeline: False`` sentinel never flips."""
    from explore_persona_space.backends.slurm_monitor import drain_cluster_sentinels

    # (a) drain level: blocks_pipeline=True surfaces the gate; the benign
    # phase-progress signal posts but is NOT surfaced.
    posts, _ = _patch_pp(monkeypatch)
    blocking = _drain_body(kind="epm:fact-candidates", gate="confirm-x", blocks_pipeline=True)
    benign = _drain_body(kind="epm:progress", gate="phase", blocks_pipeline=False)
    runner = _DrainRunner(
        stdout=_drain_stream(
            ("/workspace/logs/issue-9999-epm_fact-candidates-1.json", blocking),
            ("/workspace/logs/issue-9999-epm_progress-2.json", benign),
        )
    )
    processed, gate = drain_cluster_sentinels(
        9999, _fellows(), "/workspace/superkaiba/eps/issue-9999", runner=runner
    )
    assert (processed, gate) == (2, "confirm-x")
    assert len(posts) == 2

    # (b) build_poll_result threads the gate AND flips status to "gate".
    job_id = "9401"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={"phase": "sft", "heartbeat_ts": fresh_ts, "gpu_busy": True},
        job_out_lines=["[phase=sft]"],
    )
    common = dict(
        issue=9999,
        job_id=job_id,
        cluster=_fellows(),
        scratch_dir="/workspace/superkaiba/eps/issue-9999",
        log_path="/workspace/superkaiba/eps/issue-9999/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **_kw: None,
        event_reader=lambda _issue: [],
    )
    poll = build_poll_result(**common, sentinel_drainer=lambda i, c, s: (1, "confirm-x"))
    assert poll.status == "gate"
    assert poll.gate == "confirm-x"
    assert poll.sentinels_processed == 1

    # (c) a benign drain round (gate never surfaced by drain_sentinels_via)
    # leaves the base status untouched.
    poll = build_poll_result(**common, sentinel_drainer=lambda i, c, s: (1, None))
    assert poll.status == "running"
    assert poll.gate is None
    assert poll.sentinels_processed == 1


def test_drain_transport_failure_fail_soft(tmp_path: Path, monkeypatch) -> None:
    """rc!=0 AND TimeoutExpired both log + return (0, None); a drainer
    that RAISES is belted by build_poll_result (normal PollResult)."""
    from explore_persona_space.backends.slurm_monitor import drain_cluster_sentinels

    _patch_pp(monkeypatch)
    processed, gate = drain_cluster_sentinels(
        9999,
        _fellows(),
        "/workspace/superkaiba/eps/issue-9999",
        runner=_DrainRunner(returncode=255),
    )
    assert (processed, gate) == (0, None)
    processed, gate = drain_cluster_sentinels(
        9999,
        _fellows(),
        "/workspace/superkaiba/eps/issue-9999",
        runner=_DrainRunner(raise_timeout=True),
    )
    assert (processed, gate) == (0, None)

    def _raising_drainer(i, c, s):
        raise RuntimeError("drain bug")

    job_id = "9402"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=["[phase=sft]"])
    poll = build_poll_result(
        issue=9999,
        job_id=job_id,
        cluster=_fellows(),
        scratch_dir="/workspace/superkaiba/eps/issue-9999",
        log_path="/workspace/superkaiba/eps/issue-9999/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **_kw: None,
        event_reader=lambda _issue: [],
        sentinel_drainer=_raising_drainer,
    )
    assert poll.sentinels_processed == 0
    assert poll.gate is None


def test_drain_mark_processed_failure_leaves_sentinel_for_retry(monkeypatch) -> None:
    """A False rename still counts per drain_sentinels_via's contract (the
    marker POSTED; the next tick's #1084 fp-dedupe replay retries the
    rename only — pinned in poll_pipeline's own tests)."""
    from explore_persona_space.backends.slurm_monitor import drain_cluster_sentinels

    pp = _pp_module()
    posts, _ = _patch_pp(monkeypatch)
    monkeypatch.setattr(pp, "_ssh_mark_processed", lambda host, path: False)
    runner = _DrainRunner(
        stdout=_drain_stream(("/workspace/logs/issue-9999-epm_progress-1.json", _drain_body()))
    )
    processed, gate = drain_cluster_sentinels(
        9999, _fellows(), "/workspace/superkaiba/eps/issue-9999", runner=runner
    )
    assert (processed, gate) == (1, None)
    assert len(posts) == 1


def test_non_sentinel_drain_cluster_makes_no_ssh_call(tmp_path: Path) -> None:
    """Acceptance criterion 3 (#1898): sentinel_drain=False clusters make
    ZERO drain ssh calls and keep sentinels_processed=0 exactly as today
    — no poll_pipeline import either (the gate precedes the lazy import)."""
    from explore_persona_space.backends.slurm_monitor import drain_cluster_sentinels

    def _forbidden_runner(cmd, **kwargs):
        raise AssertionError(f"drain ssh must not run for this cluster: {cmd!r}")

    for name in ("nibi", "mila"):
        processed, gate = drain_cluster_sentinels(
            9999, get_cluster_config(name), "/scratch/x/eps/issue-9999", runner=_forbidden_runner
        )
        assert (processed, gate) == (0, None)

    # build_poll_result on a DRAC cluster (default drainer): no ssh, 0/None.
    job_id = "9403"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=["[phase=sft]"])
    poll = build_poll_result(
        issue=9999,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-9999",
        log_path="/scratch/tjiral/eps/issue-9999/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=lambda **_kw: None,
        event_reader=lambda _issue: [],
    )
    assert poll.sentinels_processed == 0
    assert poll.gate is None
    # No heartbeat but a FRESH job.out — #1969's log-freshness veto reads
    # running; the drain path left the base stall semantics untouched.
    assert poll.status == "running"
    assert poll.stall_reason is None


def test_drain_shell_includes_scratch_fallback_glob(monkeypatch) -> None:
    """The composed drain shell carries BOTH the canonical /workspace/logs
    glob AND the scratch-dir out_root fallback (the GCP #610 belt), and is
    wrapped in `bash -c` — charmander's login shell is zsh, where the
    drain loop's bash-isms fail (`shopt: command not found`; the
    2026-07-30 live-acceptance finding)."""
    from explore_persona_space.backends.slurm_monitor import drain_cluster_sentinels

    _patch_pp(monkeypatch)
    runner = _DrainRunner(stdout="")
    drain_cluster_sentinels(9999, _fellows(), "/workspace/superkaiba/eps/issue-9999", runner=runner)
    assert len(runner.calls) == 1
    argv = runner.calls[0]
    assert argv[-2] == "charmander"
    shell = argv[-1]
    assert shell.startswith("bash -c ")  # zsh-proof remote invocation
    assert "/workspace/logs/issue-9999-*.json" in shell
    assert (
        "/workspace/superkaiba/eps/issue-9999/eval_results/issue_9999/logs/issue-9999-*.json"
        in shell
    )


def test_fellows_config_sentinel_drain_true_drac_mila_false() -> None:
    """Config pin (#1898): the drain capability is fellows-ONLY (fir read
    off the raw table — get_cluster_config raises on available=False)."""
    from explore_persona_space.backends.slurm import CLUSTER_CONFIGS

    assert get_cluster_config("fellows").sentinel_drain is True
    for name in ("nibi", "fir", "mila"):
        assert CLUSTER_CONFIGS[name].sentinel_drain is False, name


# ---------------------------------------------------------------------------
# Done-evidence disambiguation (#1866): dead-class SLURM state + attempt-fresh
# status.json {"phase":"done","exit_code":"0"} reads as done, not dead. Every
# fail-safe arm (stale evidence, no submitted_at, non-done phase, non-"0"
# exit) stays byte-identical to the pre-#1866 dead verdict.
# ---------------------------------------------------------------------------


def _run_dead_class_poll(
    tmp_path: Path,
    *,
    job_id: str,
    slurm_state: str,
    status_json_text: str | None,
    job_out_lines: list[str],
    submitted_at: float | None,
    now: datetime,
    posted: list[dict] | None = None,
    slurm_exit_code: str | None = "1:0",
):
    """Run ``build_poll_result`` against a stubbed dead-class SLURM state.

    Seeds the tmp_path-isolated ``slurm-<id>/`` dir with a RAW status.json
    string (so tests can pin the sbatch template's exact printf shape, not
    just a json.dumps re-encoding) + job.out, and returns the PollResult.
    """
    local_dir = tmp_path / f"slurm-{job_id}"
    local_dir.mkdir(parents=True, exist_ok=True)
    status_path = local_dir / "status.json"
    if status_json_text is None:
        if status_path.exists():
            status_path.unlink()
    else:
        status_path.write_text(status_json_text)
    (local_dir / "job.out").write_text("\n".join(job_out_lines))
    return build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id, _s=slurm_state, _e=slurm_exit_code: {
            "status": _s,
            "exit_code": _e,
        },
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=_capture_markers(posted) if posted is not None else (lambda **_kw: None),
        event_reader=lambda _issue: [],
        submitted_at=submitted_at,
    )


def test_build_poll_result_failed_with_fresh_done_status_reads_done(tmp_path: Path) -> None:
    """FAILED + attempt-fresh ``{"phase":"done","exit_code":"0"}`` ⇒ the
    workload finished and the dead-class label is a teardown artifact
    (#1866): status flips to done, the terminal marker keeps slurm_state
    FAILED verbatim, routes next_action=interpret, and carries the flag."""
    job_id = "9301"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    posted: list[dict] = []
    poll = _run_dead_class_poll(
        tmp_path,
        job_id=job_id,
        slurm_state="FAILED",
        status_json_text=json.dumps(
            {"phase": "done", "heartbeat_ts": fresh_ts, "gpu_busy": False, "exit_code": "0"}
        ),
        job_out_lines=["[phase=done]"],
        submitted_at=now.timestamp() - 3600,
        now=now,
        posted=posted,
    )
    assert poll.status == "done"
    assert poll.current_phase == "done"
    terminals = [m for m in posted if m["marker"] == "epm:cluster-terminal"]
    assert len(terminals) == 1
    body = json.loads(terminals[0]["note"])
    assert body["slurm_state"] == "FAILED"  # audit trail keeps the true SLURM verdict
    assert body["next_action"] == "interpret"
    assert body["workload_done_despite_slurm"] is True
    assert body["status"] == "done"


def test_build_poll_result_failed_with_stale_done_status_stays_dead(tmp_path: Path) -> None:
    """done-0 evidence PREdating ``submitted_at - FRESHNESS_SKEW_MARGIN_SEC``
    is blanked by the C2 attempt-freshness gate — a PRIOR attempt's terminal
    record must never flip THIS attempt's crash to done (fail-safe)."""
    job_id = "9302"
    now = datetime.now(tz=UTC)
    stale_ts = (
        (now - timedelta(seconds=FRESHNESS_SKEW_MARGIN_SEC + 7200))
        .isoformat()
        .replace("+00:00", "Z")
    )
    posted: list[dict] = []
    poll = _run_dead_class_poll(
        tmp_path,
        job_id=job_id,
        slurm_state="FAILED",
        status_json_text=json.dumps(
            {"phase": "done", "heartbeat_ts": stale_ts, "gpu_busy": False, "exit_code": "0"}
        ),
        job_out_lines=["[phase=workload]"],
        submitted_at=now.timestamp() - 60,
        now=now,
        posted=posted,
    )
    assert poll.status == "dead"
    terminals = [m for m in posted if m["marker"] == "epm:cluster-terminal"]
    assert len(terminals) == 1
    body = json.loads(terminals[0]["note"])
    assert body["next_action"] == "investigate"
    assert "workload_done_despite_slurm" not in body


def test_build_poll_result_failed_without_submitted_at_stays_dead(tmp_path: Path) -> None:
    """Legacy handle without ``submitted_at``: attempt-freshness is
    unprovable, so even fresh-looking done-0 evidence fails toward the
    pre-#1866 dead verdict (fail-safe)."""
    job_id = "9303"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    poll = _run_dead_class_poll(
        tmp_path,
        job_id=job_id,
        slurm_state="FAILED",
        status_json_text=json.dumps(
            {"phase": "done", "heartbeat_ts": fresh_ts, "gpu_busy": False, "exit_code": "0"}
        ),
        job_out_lines=["[phase=done]"],
        submitted_at=None,
        now=now,
    )
    assert poll.status == "dead"


def test_build_poll_result_failed_mid_workload_stays_dead(tmp_path: Path) -> None:
    """The REAL crash shape (fellows job 15194 died in-workload): FAILED with
    phase 'workload' / exit_code '' stays dead → investigate, no flag."""
    job_id = "9304"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    posted: list[dict] = []
    poll = _run_dead_class_poll(
        tmp_path,
        job_id=job_id,
        slurm_state="FAILED",
        status_json_text=json.dumps(
            {"phase": "workload", "heartbeat_ts": fresh_ts, "gpu_busy": True, "exit_code": ""}
        ),
        job_out_lines=["[phase=workload]"],
        submitted_at=now.timestamp() - 3600,
        now=now,
        posted=posted,
    )
    assert poll.status == "dead"
    terminals = [m for m in posted if m["marker"] == "epm:cluster-terminal"]
    assert len(terminals) == 1
    body = json.loads(terminals[0]["note"])
    assert body["next_action"] == "investigate"
    assert "workload_done_despite_slurm" not in body


def test_build_poll_result_failed_done_nonzero_exit_stays_dead(tmp_path: Path) -> None:
    """Defensive predicate pin: phase 'done' with exit_code '' or '1' is NOT
    done-evidence. Unreachable from the current template (its terminal block
    only ever writes done WITH 0) — pins the predicate so a future template
    shape change cannot silently widen the flip."""
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    for job_id, bad_exit in (("9305", ""), ("9306", "1")):
        poll = _run_dead_class_poll(
            tmp_path,
            job_id=job_id,
            slurm_state="FAILED",
            status_json_text=json.dumps(
                {
                    "phase": "done",
                    "heartbeat_ts": fresh_ts,
                    "gpu_busy": False,
                    "exit_code": bad_exit,
                }
            ),
            job_out_lines=["[phase=done]"],
            submitted_at=now.timestamp() - 3600,
            now=now,
        )
        assert poll.status == "dead", f"exit_code={bad_exit!r} -> {poll.status}"


def test_build_poll_result_timeout_with_fresh_done_status_reads_done(tmp_path: Path) -> None:
    """Dead-class uniformity: the evidence predicate is state-agnostic —
    TIMEOUT (a fallback_runpod state pre-#1866) flips exactly like FAILED
    when the workload's own terminal record proves a clean finish."""
    job_id = "9307"
    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    posted: list[dict] = []
    poll = _run_dead_class_poll(
        tmp_path,
        job_id=job_id,
        slurm_state="TIMEOUT",
        status_json_text=json.dumps(
            {"phase": "done", "heartbeat_ts": fresh_ts, "gpu_busy": False, "exit_code": "0"}
        ),
        job_out_lines=["[phase=done]"],
        submitted_at=now.timestamp() - 3600,
        now=now,
        posted=posted,
        slurm_exit_code=None,
    )
    assert poll.status == "done"
    terminals = [m for m in posted if m["marker"] == "epm:cluster-terminal"]
    assert len(terminals) == 1
    body = json.loads(terminals[0]["note"])
    assert body["slurm_state"] == "TIMEOUT"
    assert body["next_action"] == "interpret"
    assert body["workload_done_despite_slurm"] is True


def test_persisted_terminal_with_workload_done_flag_reads_done_phase(tmp_path: Path) -> None:
    """UNKNOWN-reconnect synthesis of a persisted #1866 flip: status already
    round-trips as done via the persisted body; the phase must ALSO read
    'done' (not the dead-class slurm_state lowercased)."""
    job_id = "9308"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=None)
    prior_terminal = {
        "kind": "epm:cluster-terminal",
        "note": json.dumps(
            {
                "job_id": "9308",
                "cluster": "nibi",
                "slurm_state": "FAILED",
                "exit_code": "1:0",
                "observed_at": "2026-07-30T01:02:03Z",
                "next_action": "interpret",
                "status": "done",
                "workload_done_despite_slurm": True,
            }
        ),
    }
    posted: list[dict] = []
    poll = build_poll_result(
        issue=137,
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/tjiral/eps/issue-137",
        log_path="/scratch/tjiral/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "UNKNOWN", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
        marker_poster=_capture_markers(posted),
        event_reader=lambda _issue: [prior_terminal],
    )
    assert poll.status == "done"
    assert poll.current_phase == "done"
    assert posted == []  # reconnect path never re-posts


def test_done_evidence_predicate_matches_template_producer(tmp_path: Path) -> None:
    """Producer-parity pin (#1866): (a) the rendered sbatch terminal block
    still emits ``_write_status "done" 0``; (b) the template's exact printf
    JSON shape satisfies the predicate through the REAL build_poll_result,
    while the heartbeat/stage shape (``exit_code:""``) does not. A future
    ``_write_status`` shape change (int exit_code, renamed phase) fails THIS
    test instead of silently disarming the flip with tests green."""
    from explore_persona_space.backends import RunSpec, render_sbatch, stages_for_spec

    spec = RunSpec(
        issue=137,
        intent="lora-7b",
        backend="cluster",
        cluster="nibi",
        hydra_args=("condition=c1_evil_wrong_em", "seed=42"),
    )
    script = render_sbatch(
        spec=spec,
        cluster=_nibi(),
        plan=stages_for_spec(spec),
        scratch_dir="/scratch/tjiral/eps/issue-137",
    )
    assert '_write_status "done" 0' in script

    now = datetime.now(tz=UTC)
    fresh_ts = now.isoformat().replace("+00:00", "Z")
    # The template's _write_status printf shape, rendered verbatim:
    #   {"phase":"%s","heartbeat_ts":"%s","gpu_busy":%s,"exit_code":"%s"}
    template_done = (
        f'{{"phase":"done","heartbeat_ts":"{fresh_ts}","gpu_busy":false,"exit_code":"0"}}\n'
    )
    poll = _run_dead_class_poll(
        tmp_path,
        job_id="9309",
        slurm_state="FAILED",
        status_json_text=template_done,
        job_out_lines=["[phase=done]"],
        submitted_at=now.timestamp() - 3600,
        now=now,
    )
    assert poll.status == "done"

    # Heartbeat/stage writes call _write_status with NO second arg → the
    # printf renders exit_code:"" — must NOT satisfy the predicate.
    template_heartbeat = (
        f'{{"phase":"done","heartbeat_ts":"{fresh_ts}","gpu_busy":false,"exit_code":""}}\n'
    )
    poll = _run_dead_class_poll(
        tmp_path,
        job_id="9310",
        slurm_state="FAILED",
        status_json_text=template_heartbeat,
        job_out_lines=["[phase=done]"],
        submitted_at=now.timestamp() - 3600,
        now=now,
    )
    assert poll.status == "dead"


# ---------------------------------------------------------------------------
# #1969 — stall grace: log-freshness veto + consecutive-tick streak +
# transport-degraded skip (plan pin tests; see slurm_monitor module
# docstring § Stall semantics)
# ---------------------------------------------------------------------------


def _stall_kwargs(
    job_id: str,
    *,
    at: float,
    run_time_sec: int | None = 7200,
    rsyncer=None,
    marker_poster=None,
    event_reader=None,
) -> dict:
    """build_poll_result kwargs for the #1969 stall pins: SLURM RUNNING,
    RunTime past the early-run window (so the quiet interval is live and
    the interval asserts are real pins), no gate / sentinel activity."""
    state: dict = {"status": "RUNNING", "exit_code": None}
    if run_time_sec is not None:
        state["run_time_sec"] = run_time_sec
    return {
        "issue": 137,
        "job_id": job_id,
        "cluster": _nibi(),
        "scratch_dir": "/scratch/tjiral/eps/issue-137",
        "log_path": "/scratch/tjiral/eps/issue-137/job.out",
        "state_querier": lambda *, robot_alias, job_id: dict(state),
        "rsyncer": rsyncer or (lambda **_: None),
        "now_fn": lambda: at,
        "marker_poster": marker_poster or (lambda **_kw: None),
        "event_reader": event_reader or (lambda _issue: []),
    }


def _seed_stale_both(tmp_path: Path, job_id: str, now: datetime, *, age_sec: int = 3600) -> Path:
    """Seed status.json + job.out with BOTH signals ``age_sec`` stale.

    The job.out carries NO ``[phase=...]`` line so ``new_milestone`` is
    False and the quiet-interval asserts are real pins.
    """
    stale_ts = (now - timedelta(seconds=age_sec)).isoformat().replace("+00:00", "Z")
    local = _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "sft",
            "heartbeat_ts": stale_ts,
            "gpu_busy": False,
            "exit_code": "",
        },
        job_out_lines=["step 100 loss=1.23"],
    )
    old_epoch = now.timestamp() - age_sec
    os.utime(local / "job.out", (old_epoch, old_epoch))
    return local


def test_stall_consecutive_ticks_constant_pinned() -> None:
    """N=2: one tick is 540 s > STALL_SEC, so one silently-failed rsync
    ages both artifacts past the threshold — a single tick must never
    kill the run's orchestration (#1969 plan constant)."""
    assert STALL_CONSECUTIVE_TICKS == 2


def test_stall_suspect_first_tick_running_short_interval_streak_one(tmp_path: Path) -> None:
    """Pin 2: hb stale + log stale, tick 1 ⇒ running with
    stall_reason="slurm_stall_suspect", streak file == 1, and the SHORT
    interval (the suspect anomaly vetoes the otherwise-quiet tick)."""
    job_id = "9701"
    now = datetime.now(tz=UTC)
    _seed_stale_both(tmp_path, job_id, now)
    poll = build_poll_result(**_stall_kwargs(job_id, at=now.timestamp()))
    assert poll.status == "running"
    assert poll.stall_reason == "slurm_stall_suspect"
    assert _read_stall_streak(job_id) == (1, pytest.approx(now.timestamp()))
    assert poll.new_milestone is False
    assert poll.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_stall_second_tick_after_stall_sec_reports_stalled(tmp_path: Path) -> None:
    """Pin 3: the SAME condition on a second tick >= STALL_SEC later ⇒
    stalled, with the machine-readable infra-routing reason."""
    job_id = "9702"
    now = datetime.now(tz=UTC)
    _seed_stale_both(tmp_path, job_id, now)
    first = build_poll_result(**_stall_kwargs(job_id, at=now.timestamp()))
    assert first.status == "running"
    second = build_poll_result(**_stall_kwargs(job_id, at=now.timestamp() + STALL_SEC + 60))
    assert second.status == "stalled"
    assert second.stall_reason == "slurm_heartbeat_and_log_stale"
    assert second.pid_alive is False
    assert second.next_interval == POLL_INTERVAL_DEFAULT_SEC


def test_stall_rapid_second_poll_does_not_reach_stalled(tmp_path: Path) -> None:
    """Pin 3b (time gate): a second poll SECONDS after the first — a
    manual poll beside the cron tick — must NOT reach streak 2 on one
    silently-failed rsync; still running/suspect, streak stays 1."""
    job_id = "9703"
    now = datetime.now(tz=UTC)
    _seed_stale_both(tmp_path, job_id, now)
    build_poll_result(**_stall_kwargs(job_id, at=now.timestamp()))
    rapid = build_poll_result(**_stall_kwargs(job_id, at=now.timestamp() + 5))
    assert rapid.status == "running"
    assert rapid.stall_reason == "slurm_stall_suspect"
    assert _read_stall_streak(job_id)[0] == 1


def test_stall_healthy_tick_resets_streak(tmp_path: Path) -> None:
    """Pin 4: a healthy tick after a suspect tick resets the streak to
    0, so the NEXT stale tick is suspect again — never stalled."""
    job_id = "9704"
    now = datetime.now(tz=UTC)
    _seed_stale_both(tmp_path, job_id, now)
    build_poll_result(**_stall_kwargs(job_id, at=now.timestamp()))
    assert _read_stall_streak(job_id)[0] == 1

    # Healthy tick: fresh heartbeat + fresh log at t2.
    t2 = now + timedelta(seconds=600)
    fresh_ts = t2.isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "sft",
            "heartbeat_ts": fresh_ts,
            "gpu_busy": True,
            "exit_code": "",
        },
        job_out_lines=["step 200 loss=1.10"],
    )
    local = tmp_path / f"slurm-{job_id}"
    os.utime(local / "job.out", (t2.timestamp(), t2.timestamp()))
    healthy = build_poll_result(**_stall_kwargs(job_id, at=t2.timestamp()))
    assert healthy.status == "running"
    assert healthy.stall_reason is None
    assert _read_stall_streak(job_id)[0] == 0

    # Stale again at t3: suspect (streak restarts at 1), NOT stalled.
    t3 = t2 + timedelta(seconds=600)
    again = build_poll_result(**_stall_kwargs(job_id, at=t3.timestamp()))
    assert again.status == "running"
    assert again.stall_reason == "slurm_stall_suspect"
    assert _read_stall_streak(job_id)[0] == 1


def test_stall_transport_degraded_skips_evaluation(tmp_path: Path) -> None:
    """Pin 5: rsyncer returns False (transport-class failure) ⇒ NO stall
    evaluation this tick — status per SLURM state, streak neither
    incremented nor reset, short interval, and the cluster-poll note
    records transport_degraded=true + the untouched streak."""
    job_id = "9705"
    now = datetime.now(tz=UTC)
    _seed_stale_both(tmp_path, job_id, now)
    # Pre-seed a live streak from a prior suspect tick.
    local = tmp_path / f"slurm-{job_id}"
    (local / "stall_streak.json").write_text(
        json.dumps({"streak": 1, "updated_ts": now.timestamp() - 600})
    )
    captured: list[dict] = []
    poll = build_poll_result(
        **_stall_kwargs(
            job_id,
            at=now.timestamp(),
            rsyncer=lambda **_: False,
            marker_poster=_capture_markers(captured),
        )
    )
    assert poll.status == "running"  # per SLURM state — no stall read
    assert poll.stall_reason is None
    assert _read_stall_streak(job_id) == (1, pytest.approx(now.timestamp() - 600))
    assert poll.next_interval == POLL_INTERVAL_DEFAULT_SEC
    polls = [m for m in captured if m.get("marker") == "epm:cluster-poll"]
    assert polls, "first observation must post epm:cluster-poll"
    body = json.loads(polls[0]["note"])
    assert body["transport_degraded"] is True
    assert body["stall_streak"] == 1


def test_stall_legacy_none_rsyncer_evaluates_normally(tmp_path: Path) -> None:
    """Pin 6 (back-compat): a legacy/stub rsyncer returning None is NOT
    transport-degraded — the monitor-side check is ``ret is False`` —
    so stall evaluation runs normally (streak file written)."""
    job_id = "9706"
    now = datetime.now(tz=UTC)
    _seed_stale_both(tmp_path, job_id, now)
    poll = build_poll_result(**_stall_kwargs(job_id, at=now.timestamp(), rsyncer=lambda **_: None))
    assert poll.stall_reason == "slurm_stall_suspect"
    assert (tmp_path / f"slurm-{job_id}" / "stall_streak.json").exists()


def test_stall_both_artifacts_missing_stalls_at_second_tick(tmp_path: Path) -> None:
    """Pin 7 (weakening preserved, +1 tick): a job writing nothing
    anywhere — status.json AND job.out both missing — with run age past
    STALL_SEC is suspect at tick 1 and stalled at tick 2."""
    job_id = "9707"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=None)
    first = build_poll_result(**_stall_kwargs(job_id, at=now.timestamp()))
    assert first.status == "running"
    assert first.stall_reason == "slurm_stall_suspect"
    second = build_poll_result(**_stall_kwargs(job_id, at=now.timestamp() + STALL_SEC + 60))
    assert second.status == "stalled"
    assert second.stall_reason == "slurm_heartbeat_and_log_stale"


def test_stall_young_job_both_missing_stays_running(tmp_path: Path) -> None:
    """Pin 8 (C2 floor kept): a job that has only RUN for 60 s can be at
    most 60 s stale — both artifacts missing reads running, streak 0."""
    job_id = "9708"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=None)
    poll = build_poll_result(**_stall_kwargs(job_id, at=now.timestamp(), run_time_sec=60))
    assert poll.status == "running"
    assert poll.stall_reason is None
    assert not (tmp_path / f"slurm-{job_id}" / "stall_streak.json").exists()


# ---------------------------------------------------------------------------
# #1969 — rsync_status_and_log transport-verdict unit pins (plan test 9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rc", "expected"),
    [
        (0, True),
        (23, True),  # source file missing — never-writing job, NOT transport
        (24, True),  # source file vanished mid-transfer — same class
        (255, False),  # ssh error
        (30, False),  # rsync timeout in data send/receive
        (12, False),  # protocol-stream error — the unlisted-rc DEFAULT pin
        (1, False),  # any other non-zero rc defaults to transport-class
    ],
)
def test_rsync_status_and_log_rc_verdicts(monkeypatch, tmp_path, rc: int, expected: bool) -> None:
    """Plan test 9: rc → transport verdict, incl. the unlisted-rc
    fail-closed default (rc 12 / rc 1 → False)."""
    monkeypatch.setattr(subprocess, "run", _fake_run_factory([_proc(rc), _proc(rc)]))
    ok = rsync_status_and_log(
        robot_alias="nibi-robot", scratch_dir="/scratch/x/eps/issue-137", job_id="9801"
    )
    assert ok is expected


def test_rsync_status_and_log_timeout_returns_false(monkeypatch, tmp_path) -> None:
    """Plan test 9: a subprocess.TimeoutExpired is CAUGHT (pre-#1969 it
    crashed the tick) and reads as transport-degraded."""

    def _raise(argv, **_kw):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=30)

    monkeypatch.setattr(subprocess, "run", _raise)
    ok = rsync_status_and_log(
        robot_alias="nibi-robot", scratch_dir="/scratch/x/eps/issue-137", job_id="9802"
    )
    assert ok is False


def test_rsync_status_and_log_mixed_ok_then_transport_fails(monkeypatch, tmp_path) -> None:
    """ANY transport-class failure among the pulls degrades the tick,
    even when the other pull succeeded."""
    monkeypatch.setattr(subprocess, "run", _fake_run_factory([_proc(0), _proc(255)]))
    ok = rsync_status_and_log(
        robot_alias="nibi-robot", scratch_dir="/scratch/x/eps/issue-137", job_id="9803"
    )
    assert ok is False
