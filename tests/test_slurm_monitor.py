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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from explore_persona_space.backends.slurm import get_cluster_config
from explore_persona_space.backends.slurm_monitor import (
    SLURM_STATE_TO_STATUS,
    STALL_SEC,
    _parse_scontrol_show_job,
    build_poll_result,
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


def test_parse_scontrol_show_job_handles_missing_fields() -> None:
    """Garbage / partial scontrol output must NOT crash; surface
    UNKNOWN so the caller can route to the marker-trail lookup."""
    parsed = _parse_scontrol_show_job("(unhelpful)")
    assert parsed["status"] == "UNKNOWN"
    assert parsed["exit_code"] is None


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
    """Pre-seed /tmp/slurm-<id>/ with status.json + job.out."""
    local_dir = Path("/tmp") / f"slurm-{job_id}"
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


def test_build_poll_result_stalled_when_heartbeat_stale(tmp_path: Path) -> None:
    """SLURM RUNNING + heartbeat older than STALL_SEC ⇒ stalled."""
    job_id = "9102"
    now = datetime.now(tz=UTC)
    stale_ts = (now - timedelta(seconds=STALL_SEC + 60)).isoformat().replace("+00:00", "Z")
    _seed_local_state(
        tmp_path,
        job_id,
        status_json_body={
            "phase": "sft",
            "heartbeat_ts": stale_ts,
            "gpu_busy": False,
            "exit_code": "",
        },
        job_out_lines=["[phase=sft]"],
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
    assert poll.status == "stalled"


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
    """SLURM RUNNING + status.json absent ⇒ heartbeat infinitely old ⇒ stalled."""
    job_id = "9106"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=["random output"])
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
    assert poll.status == "stalled"


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
    job as a workload failure instead of no-compute."""
    from explore_persona_space.backends.slurm_monitor import fetch_started_evidence

    job_id = "9501"
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
        rsyncer=lambda **_kw: None,  # files already seeded locally
    )
    assert evidence is not None
    assert evidence["phase"] == "preflight-failed"
    assert "[FAIL] secrets file" in evidence["job_out_tail"]
    assert evidence["status_json"]["exit_code"] == "1"


def test_fetch_started_evidence_returns_none_when_no_artifacts(tmp_path: Path) -> None:
    """No status.json AND no job.out = the job never started — the
    router's legacy no_compute classification stands."""
    from explore_persona_space.backends.slurm_monitor import fetch_started_evidence

    job_id = "9502"
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=None)
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
    from explore_persona_space.backends.slurm_monitor import fetch_started_evidence

    job_id = "9503"
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
        rsyncer=lambda **_kw: None,
    )
    assert evidence is not None
    assert "early crash" in evidence["job_out_tail"]
