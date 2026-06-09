"""SLURM cluster monitor — builds :class:`PollResult` from cluster state.

The orchestrator's bg-Bash polling loop drives this for any
``backend: cluster`` run. The forced-command robot wrapper FORBIDS
``cat`` / ``tail`` / ``ps`` / ``nvidia-smi`` from the SSH side, so a
"fake-pod SSH shim" mirroring ``poll_pipeline.py`` is impossible. The
monitor instead composes three legal signals:

1. **SLURM job state** — ``scontrol show job <id>`` / ``squeue -j <id>``
   over the robot SSH alias (allowed by the wrapper). Args must be
   single-token (P0 finding: quoted multi-token ``-o "%i %j"`` gets
   split by the SSH forced-command flattening and errors).
2. **Heartbeat** — ``status.json`` rsync'd from
   ``$SCRATCH_JOB_DIR/status.json``; the sbatch writes a fresh row
   every :data:`slurm.HEARTBEAT_INTERVAL_SECONDS`. A live ``RUNNING``
   SLURM state PLUS a recent heartbeat = ``running``; a live
   ``RUNNING`` state with a STALE heartbeat = ``stalled``.
3. **Log tail** — ``job.out`` rsync'd from
   ``$SCRATCH_JOB_DIR/job.out``; grepped for ``[phase=<name>]`` lines
   to set ``current_phase`` and ``new_milestone``. The rsync interval
   is set BELOW ``STALL_SEC`` so the heartbeat read stays accurate.

The shape :class:`PollResult` returns is BYTE-COMPATIBLE with the JSON
``scripts/poll_pipeline.py`` emits, so the orchestrator's existing
JSON-line parser does not change.

Idempotent reconnect
--------------------

When the in-process state vanishes (orchestrator re-spawn, new shell)
the monitor falls back to ``squeue --name <job_name>`` to disambiguate
"ageout" from "really gone". If both ``squeue -j <id>`` and ``squeue
--name <name>`` show nothing AND the persisted ``epm:cluster-terminal``
marker exists, treat the job as ``done`` / ``dead`` per the marker. If
the marker is absent, post a ``epm:cluster-terminal v1`` ``unknown``
verdict so a future call doesn't infinitely retry.

Stall semantics
---------------

``STALL = SLURM state RUNNING but heartbeat_ts older than STALL_SEC``.
This is weaker than the pod poller's 4-way check (PID alive, log mtime,
GPU util, sentinels) because we cannot run remote ``ps``/``nvidia-smi``
via the forced-command wrapper. The documented weakening: a job that
write nothing to status.json (e.g. an early-init crash that hangs
before the heartbeat loop starts) shows as ``stalled`` until SLURM
itself reaps it. Operators can grep the rsync'd job.out for
``[phase=preflight-failed]`` to disambiguate.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explore_persona_space.backends.base import PollResult
from explore_persona_space.backends.slurm import (
    PREFLIGHT_FAIL_MARKER,
    ClusterConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Stall threshold (seconds). A SLURM-RUNNING job whose status.json
# heartbeat is older than this is reported as ``stalled``. Must sit
# safely above ``slurm.HEARTBEAT_INTERVAL_SECONDS`` (default 60s) so a
# healthy job's natural pause between heartbeats is NOT a false stall.
# Default 5 min matches the pod poller's STALL_SEC.
STALL_SEC = 300

# How far back to read job.out (bytes) when building the log_tail_excerpt.
LOG_TAIL_BYTES = 16_384


# Local dir under /tmp where rsync'd status.json / job.out files land.
# Per-job to avoid cross-contamination across concurrent monitors.
def _local_state_dir(job_id: str) -> Path:
    return Path("/tmp") / f"slurm-{job_id}"


# Mapping from SLURM JobState (per ``scontrol show job`` / ``squeue -h
# -o %T``) to the orchestrator's PollResult.status enum. Anything not
# in the map defaults to ``running`` (pessimistic: don't reap a job we
# don't recognize yet — a future SLURM version's new state name should
# NOT mass-cancel jobs).
SLURM_STATE_TO_STATUS: dict[str, str] = {
    "PENDING": "running",  # selector watchdog handles the PENDING->RUNNING wait
    "CONFIGURING": "running",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "done",
    "CANCELLED": "dead",
    "CANCELLED+": "dead",  # CANCELLED by a different uid surfaces as CANCELLED+
    "FAILED": "dead",
    "TIMEOUT": "dead",
    "PREEMPTED": "dead",
    "NODE_FAIL": "dead",
    "BOOT_FAIL": "dead",
    "OUT_OF_MEMORY": "dead",
    "DEADLINE": "dead",
    "SUSPENDED": "stalled",
}


# Regex that matches ``[phase=<name>]`` log lines (the sbatch writes
# these between stages).
_PHASE_LINE_RE = re.compile(r"\[phase=([a-zA-Z0-9_\-]+)\]")


# ---------------------------------------------------------------------------
# Public entrypoint: build_poll_result
# ---------------------------------------------------------------------------


def build_poll_result(
    *,
    job_id: str,
    cluster: ClusterConfig,
    scratch_dir: str,
    log_path: str,
    state_querier=None,
    rsyncer=None,
    now_fn=time.time,
) -> PollResult:
    """One-tick poll → :class:`PollResult`.

    Composes:

    * ``query_slurm_state`` over SSH for the SLURM JobState + exit code.
    * ``rsync_status_and_log`` for the heartbeat + log tail.
    * Stall detection: SLURM=RUNNING but heartbeat older than
      :data:`STALL_SEC`.

    Test seams:

    * ``state_querier`` — defaults to :func:`query_slurm_state`. Tests
      pass a stub returning a parsed state dict.
    * ``rsyncer`` — defaults to :func:`rsync_status_and_log`. Tests
      pass a no-op + pre-seeded local files.
    * ``now_fn`` — for the stall clock; tests pin it.

    Returns:
        A :class:`PollResult` with the SAME shape ``poll_pipeline.py``
        produces, so the orchestrator's JSON-line parser keeps working.
    """
    state_querier = state_querier or query_slurm_state
    rsyncer = rsyncer or rsync_status_and_log

    state = state_querier(robot_alias=cluster.robot_alias, job_id=job_id)
    rsyncer(
        robot_alias=cluster.robot_alias,
        scratch_dir=scratch_dir,
        job_id=job_id,
    )

    local_state_dir = _local_state_dir(job_id)
    status_json = local_state_dir / "status.json"
    job_out = local_state_dir / "job.out"

    status_data = _read_status_json(status_json)
    log_tail, current_phase, new_milestone, log_mtime_sec_ago = _read_job_out(
        job_out, now_fn=now_fn
    )

    slurm_status = state.get("status", "RUNNING")
    base_status = SLURM_STATE_TO_STATUS.get(slurm_status, "running")

    # If we have a fresher phase from status.json, prefer it (the sbatch
    # writes status BEFORE its echo of the phase to stdout, so the JSON
    # tends to be one tick ahead).
    json_phase = status_data.get("phase")
    if json_phase:
        current_phase = json_phase

    # Heartbeat freshness (seconds-ago). If status.json is missing
    # entirely, treat the heartbeat as infinitely old so the stall path
    # fires for a job that's RUNNING but writing nothing.
    heartbeat_sec_ago = _heartbeat_sec_ago(status_data, now_fn=now_fn)

    # Stall detection (only meaningful while SLURM still says RUNNING).
    # Don't flag PENDING as stalled — the selector watchdog handles that.
    if base_status == "running" and heartbeat_sec_ago > STALL_SEC and slurm_status != "PENDING":
        base_status = "stalled"

    # Preflight failure detection — the sbatch echoes
    # ``[phase=preflight-failed]`` then exit's non-zero. Even before SLURM
    # transitions to FAILED, we can spot it in the log.
    if PREFLIGHT_FAIL_MARKER in log_tail:
        base_status = "dead"
        current_phase = "preflight-failed"

    return PollResult(
        status=base_status,
        current_phase=current_phase or slurm_status.lower(),
        new_milestone=new_milestone,
        last_log_mtime_sec_ago=log_mtime_sec_ago,
        pid_alive=base_status == "running",
        log_tail_excerpt=log_tail[-2000:],
        gate=None,
        sentinels_processed=0,
        phase_log_mtime_sec_ago=log_mtime_sec_ago,
        shard_log_mtime_sec_ago=log_mtime_sec_ago,
        gpu_util="busy" if status_data.get("gpu_busy") else "idle",
    )


# ---------------------------------------------------------------------------
# SLURM state query (scontrol / squeue)
# ---------------------------------------------------------------------------


def query_slurm_state(
    *,
    robot_alias: str,
    job_id: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Query SLURM for ``job_id``'s state via ``scontrol show job``.

    Returns a dict with at least ``{"status": <STATE>, "exit_code":
    <"N:M"|None>, "node": <node|None>}``. On scontrol "no such job"
    falls back to ``squeue -j`` (same disambiguation as the pod poller's
    fallback). If both report nothing, returns ``{"status":
    "UNKNOWN"}`` — the caller's idempotent-reconnect path handles that
    by reading the persisted ``epm:cluster-terminal`` marker.

    Args MUST be single-token (P0 finding: the forced-command wrapper
    flattens quoted multi-token args like ``-o "%i %j"`` and errors).
    """
    # Try scontrol first — it carries the most detail (JobState,
    # ExitCode, NodeList, RunTime).
    proc = subprocess.run(
        ["ssh", robot_alias, "scontrol", "show", "job", job_id],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return _parse_scontrol_show_job(proc.stdout)

    # Fallback: squeue -j <id> -h -o %T (single-token format).
    proc = subprocess.run(
        ["ssh", robot_alias, "squeue", "-j", job_id, "-h", "-o", "%T"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return {"status": proc.stdout.strip().splitlines()[0].strip(), "exit_code": None}

    # Both empty → job aged out of the active queue. The caller's
    # epm:cluster-terminal lookup is the authoritative truth here.
    return {"status": "UNKNOWN", "exit_code": None}


def _parse_scontrol_show_job(stdout: str) -> dict[str, Any]:
    """Parse ``scontrol show job <id>`` output into a dict.

    The output is ``key=value`` pairs whitespace-separated. We extract
    JobState, ExitCode, NodeList; everything else is noise for the
    monitor.
    """
    out: dict[str, Any] = {"status": "UNKNOWN", "exit_code": None, "node": None}
    # scontrol emits both ``key=value`` and ``key=value key=value`` on
    # the same line. Use a regex over the whole blob.
    for match in re.finditer(r"([A-Za-z]+)=([^\s]+)", stdout):
        key, val = match.group(1), match.group(2)
        if key == "JobState":
            out["status"] = val
        elif key == "ExitCode":
            out["exit_code"] = val
        elif key == "NodeList" and val != "(null)":
            out["node"] = val
    return out


def query_by_name(
    *,
    robot_alias: str,
    job_name: str,
    timeout: int = 30,
) -> str | None:
    """Reconnect helper: ``squeue --name <job_name> -h -o %i``.

    Used when the in-process state has no job id (orchestrator
    re-spawn) but the persisted launch marker named the job. Returns
    the numeric job id of the most recent matching live job, or
    ``None`` if none exists (job aged out / never landed).
    """
    proc = subprocess.run(
        ["ssh", robot_alias, "squeue", "--name", job_name, "-h", "-o", "%i"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1]


# ---------------------------------------------------------------------------
# Rsync of status.json + job.out
# ---------------------------------------------------------------------------


def rsync_status_and_log(
    *,
    robot_alias: str,
    scratch_dir: str,
    job_id: str,
    timeout: int = 30,
) -> None:
    """Pull ``status.json`` + ``job.out`` from the cluster scratch dir.

    Lands them under ``/tmp/slurm-<job_id>/`` so concurrent monitors on
    different jobs don't clobber each other. ``--partial`` + ``--mkpath``
    keep the cost low.

    Non-fatal on rsync failure — a transient SSH hiccup shouldn't crash
    the polling loop; the next tick will retry and the local files
    (still readable from the previous tick) keep the monitor honest.
    """
    local_dir = _local_state_dir(job_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("status.json", "job.out"):
        argv = [
            "rsync",
            "-a",
            "--partial",
            "--mkpath",
            f"{robot_alias}:{scratch_dir}/{filename}",
            str(local_dir / filename),
        ]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        if proc.returncode != 0:
            logger.debug(
                "rsync %s/%s returned %d: %s",
                scratch_dir,
                filename,
                proc.returncode,
                proc.stderr.strip(),
            )


# ---------------------------------------------------------------------------
# Local-file readers
# ---------------------------------------------------------------------------


def _read_status_json(path: Path) -> dict[str, Any]:
    """Read the rsync'd ``status.json``. Returns ``{}`` if absent / malformed."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("could not read status.json at %s: %s", path, exc)
        return {}


def _read_job_out(path: Path, *, now_fn=time.time) -> tuple[str, str, bool, int]:
    """Read the rsync'd ``job.out`` tail.

    Returns ``(tail_text, current_phase, new_milestone, mtime_sec_ago)``.

    * ``tail_text``: last :data:`LOG_TAIL_BYTES` decoded UTF-8 with
      replacement for malformed bytes.
    * ``current_phase``: most recent ``[phase=<name>]`` capture from
      the tail (empty string if none).
    * ``new_milestone``: True iff a phase line appears in the LAST
      ``LOG_TAIL_BYTES`` (the orchestrator uses this for the polling
      back-off heuristic).
    * ``mtime_sec_ago``: seconds since the file was last modified.
      ``10**9`` when the file is missing (treated as "infinitely old"
      so the stall path can fire).
    """
    if not path.exists():
        return "", "", False, 10**9
    try:
        stat = path.stat()
    except OSError:
        return "", "", False, 10**9
    mtime_sec_ago = max(0, int(now_fn() - stat.st_mtime))
    with path.open("rb") as fh:
        if stat.st_size > LOG_TAIL_BYTES:
            fh.seek(-LOG_TAIL_BYTES, 2)
        data = fh.read()
    tail = data.decode("utf-8", errors="replace")
    matches = _PHASE_LINE_RE.findall(tail)
    current_phase = matches[-1] if matches else ""
    new_milestone = bool(matches)
    return tail, current_phase, new_milestone, mtime_sec_ago


def _heartbeat_sec_ago(status_data: dict[str, Any], *, now_fn=time.time) -> int:
    """Seconds since the most recent heartbeat in ``status.json``.

    ``10**9`` when status.json is missing or has no parseable timestamp
    (so the stall path fires for a job writing nothing).
    """
    ts = status_data.get("heartbeat_ts")
    if not ts:
        return 10**9
    try:
        # ISO-8601 with trailing 'Z' (UTC). datetime.fromisoformat handles
        # 'Z' on Python 3.11+.
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 10**9
    now_utc = datetime.fromtimestamp(now_fn(), tz=UTC)
    delta = (now_utc - parsed).total_seconds()
    return max(0, int(delta))


__all__ = [
    "LOG_TAIL_BYTES",
    "SLURM_STATE_TO_STATUS",
    "STALL_SEC",
    "build_poll_result",
    "query_by_name",
    "query_slurm_state",
    "rsync_status_and_log",
]
