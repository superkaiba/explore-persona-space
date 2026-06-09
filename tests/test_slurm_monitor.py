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
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/eps/issue-137",
        log_path="/scratch/eps/issue-137/job.out",
        state_querier=fake_state,
        rsyncer=fake_rsync,
        now_fn=lambda: now.timestamp(),
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
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/eps/issue-137",
        log_path="/scratch/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
    )
    assert poll.status == "stalled"


def test_build_poll_result_pending_is_running_not_stalled(tmp_path: Path) -> None:
    """A PENDING job that's writing nothing must NOT be reported as
    stalled — the selector's submit-and-park watchdog owns that logic."""
    job_id = "9103"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=None)
    poll = build_poll_result(
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/eps/issue-137",
        log_path="/scratch/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "PENDING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
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
            job_id=job_id,
            cluster=_nibi(),
            scratch_dir="/scratch/eps/issue-137",
            log_path="/scratch/eps/issue-137/job.out",
            state_querier=lambda *, robot_alias, job_id, _s=slurm_state: {
                "status": _s,
                "exit_code": None,
            },
            rsyncer=lambda **_: None,
            now_fn=lambda: now.timestamp(),
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
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/eps/issue-137",
        log_path="/scratch/eps/issue-137/job.out",
        # SLURM may still report RUNNING for a moment before reaping.
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
    )
    assert poll.status == "dead"
    assert poll.current_phase == "preflight-failed"


def test_build_poll_result_missing_status_json_treats_as_stalled(tmp_path: Path) -> None:
    """SLURM RUNNING + status.json absent ⇒ heartbeat infinitely old ⇒ stalled."""
    job_id = "9106"
    now = datetime.now(tz=UTC)
    _seed_local_state(tmp_path, job_id, status_json_body=None, job_out_lines=["random output"])
    poll = build_poll_result(
        job_id=job_id,
        cluster=_nibi(),
        scratch_dir="/scratch/eps/issue-137",
        log_path="/scratch/eps/issue-137/job.out",
        state_querier=lambda *, robot_alias, job_id: {"status": "RUNNING", "exit_code": None},
        rsyncer=lambda **_: None,
        now_fn=lambda: now.timestamp(),
    )
    assert poll.status == "stalled"
