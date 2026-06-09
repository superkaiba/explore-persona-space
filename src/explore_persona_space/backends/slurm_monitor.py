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


# Status enum values that count as terminal for the orchestrator.
# ``done`` = clean exit; ``dead`` = any non-zero terminal state (FAILED /
# TIMEOUT / PREEMPTED / NODE_FAIL / CANCELLED / OOM).
_TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "dead"})


def build_poll_result(
    *,
    issue: int,
    job_id: str,
    cluster: ClusterConfig,
    scratch_dir: str,
    log_path: str,
    state_querier=None,
    rsyncer=None,
    now_fn=time.time,
    marker_poster=None,
    event_reader=None,
) -> PollResult:
    """One-tick poll → :class:`PollResult`.

    Composes:

    * ``query_slurm_state`` over SSH for the SLURM JobState + exit code.
    * ``rsync_status_and_log`` for the heartbeat + log tail.
    * Stall detection: SLURM=RUNNING but heartbeat older than
      :data:`STALL_SEC`.

    Posts (per ``workflow.yaml § markers``):

    * ``epm:cluster-poll v1`` on every status / phase transition
      (deduplicated against the last posted cluster-poll for this job
      by reading events.jsonl). Keeps the trail readable; a long-
      running job that stays in the same phase doesn't spam markers.
    * ``epm:cluster-terminal v1`` the FIRST time terminal state is
      observed (``status in {"done", "dead"}``). Persists the
      authoritative breadcrumb so idempotent reconnect after squeue /
      scontrol ageout finds the verdict here when SLURM returns
      ``UNKNOWN``.

    Idempotent reconnect: when ``state_querier`` returns
    ``status == "UNKNOWN"`` (job aged out of the active queue), the
    monitor reads the persisted ``epm:cluster-terminal v1`` for this
    job_id and synthesizes a PollResult from it. The dead/done verdict
    survives the SLURM cache TTL.

    Test seams:

    * ``state_querier`` — defaults to :func:`query_slurm_state`. Tests
      pass a stub returning a parsed state dict.
    * ``rsyncer`` — defaults to :func:`rsync_status_and_log`. Tests
      pass a no-op + pre-seeded local files.
    * ``now_fn`` — for the stall clock; tests pin it.
    * ``marker_poster`` — defaults to
      :func:`backends.slurm.post_marker_via_task_py`. Tests pass a
      list-appender to capture which markers were posted.
    * ``event_reader`` — defaults to
      :func:`task_workflow.list_events`. Tests pass a stub returning a
      pre-seeded event trail.

    Returns:
        A :class:`PollResult` with the SAME shape ``poll_pipeline.py``
        produces, so the orchestrator's JSON-line parser keeps working.
    """
    state_querier = state_querier or query_slurm_state
    rsyncer = rsyncer or rsync_status_and_log
    if marker_poster is None:
        from explore_persona_space.backends.slurm import post_marker_via_task_py

        marker_poster = post_marker_via_task_py
    if event_reader is None:
        from explore_persona_space.task_workflow import list_events

        event_reader = list_events

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

    # ---- Idempotent-reconnect path: SLURM said UNKNOWN ----
    # When squeue + scontrol both age out (~5 min on most CC clusters),
    # the only authoritative record is the persisted epm:cluster-terminal
    # marker. Reach for it BEFORE falling through to the default-RUNNING
    # safety net; otherwise a stale handle would loop forever reading
    # "running".
    if slurm_status == "UNKNOWN":
        persisted = _read_persisted_terminal(issue=issue, job_id=job_id, event_reader=event_reader)
        if persisted is not None:
            return _poll_result_from_persisted_terminal(
                persisted=persisted, log_tail=log_tail, log_mtime_sec_ago=log_mtime_sec_ago
            )
        # No marker either — we genuinely don't know. Default to running
        # so the orchestrator doesn't reap a job we haven't proven dead.

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

    final_phase = current_phase or slurm_status.lower()

    # ---- Post epm:cluster-poll v1 on transition ----
    _maybe_post_cluster_poll(
        issue=issue,
        job_id=job_id,
        status=base_status,
        current_phase=final_phase,
        slurm_state=slurm_status,
        heartbeat_sec_ago=heartbeat_sec_ago,
        gpu_busy=bool(status_data.get("gpu_busy")),
        log_tail_excerpt=log_tail[-2000:],
        marker_poster=marker_poster,
        event_reader=event_reader,
    )

    # ---- Post epm:cluster-terminal v1 the first time terminal observed ----
    if base_status in _TERMINAL_STATUSES:
        _maybe_post_cluster_terminal(
            issue=issue,
            job_id=job_id,
            cluster_name=cluster.name,
            slurm_state=slurm_status,
            exit_code=state.get("exit_code"),
            base_status=base_status,
            marker_poster=marker_poster,
            event_reader=event_reader,
            now_fn=now_fn,
        )

    return PollResult(
        status=base_status,
        current_phase=final_phase,
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
# Marker posting helpers (read events.jsonl for dedup; post via task.py)
# ---------------------------------------------------------------------------


def _events_for_job(*, issue: int, job_id: str, kind: str, event_reader) -> list[dict[str, Any]]:
    """Return prior events for this job_id of a given marker kind.

    events.jsonl is shared across all jobs on a single task; we filter
    by the embedded ``job_id`` in the marker body so two attempts on
    the same task don't cross-contaminate. The body is JSON inside the
    event ``note`` field per :func:`backends.slurm.post_marker_via_task_py`.
    """
    out: list[dict[str, Any]] = []
    try:
        events = event_reader(issue)
    except Exception:
        # If the events file is missing / unreadable we cannot dedup;
        # treat as "no prior events" and post fresh. We don't want a
        # missing file to silently DROP the marker (the post itself
        # will create it).
        return out
    for ev in events:
        if ev.get("kind") != kind:
            continue
        note = ev.get("note", "")
        try:
            body = json.loads(note) if isinstance(note, str) and note.startswith("{") else None
        except (json.JSONDecodeError, ValueError):
            body = None
        if isinstance(body, dict) and body.get("job_id") == job_id:
            out.append({"event": ev, "body": body})
    return out


def _maybe_post_cluster_poll(
    *,
    issue: int,
    job_id: str,
    status: str,
    current_phase: str,
    slurm_state: str,
    heartbeat_sec_ago: int,
    gpu_busy: bool,
    log_tail_excerpt: str,
    marker_poster,
    event_reader,
) -> None:
    """Post ``epm:cluster-poll v1`` only when status or phase changed.

    Dedup against the most recent prior cluster-poll for this job_id;
    if status AND phase are unchanged, skip (keeps the events.jsonl tail
    readable on a long full-FT that stays in the same phase for hours).
    """
    prior = _events_for_job(
        issue=issue, job_id=job_id, kind="epm:cluster-poll", event_reader=event_reader
    )
    if prior:
        last_body = prior[-1]["body"]
        if (
            last_body.get("status") == status
            and last_body.get("current_phase") == current_phase
            and last_body.get("slurm_state") == slurm_state
        ):
            return
    body = {
        "job_id": job_id,
        "status": status,
        "current_phase": current_phase,
        "slurm_state": slurm_state,
        "heartbeat_sec_ago": heartbeat_sec_ago,
        "gpu_util": "busy" if gpu_busy else "idle",
        "log_tail_excerpt": log_tail_excerpt[-2000:],
    }
    note = json.dumps(body, sort_keys=True)
    # post-marker enforces the 50_000-char cap on note; the log tail is
    # already capped at 2000 chars above so this is well within bounds.
    marker_poster(
        issue=issue,
        marker="epm:cluster-poll",
        note=note,
        version=1,
        by="backends.slurm_monitor",
    )


def _maybe_post_cluster_terminal(
    *,
    issue: int,
    job_id: str,
    cluster_name: str,
    slurm_state: str,
    exit_code: str | None,
    base_status: str,
    marker_poster,
    event_reader,
    now_fn,
) -> None:
    """Post ``epm:cluster-terminal v1`` exactly once per job_id.

    Subsequent ticks read the persisted marker via
    :func:`_read_persisted_terminal` and short-circuit, so a job that
    re-emerges briefly as FAILED across two ticks only writes ONE
    terminal-state row.
    """
    prior = _events_for_job(
        issue=issue, job_id=job_id, kind="epm:cluster-terminal", event_reader=event_reader
    )
    if prior:
        return
    # next_action per workflow.yaml § markers:
    # COMPLETED -> interpret; FAILED/OOM -> investigate; rest -> fallback_runpod.
    if slurm_state == "COMPLETED":
        next_action = "interpret"
    elif slurm_state in {"FAILED", "OUT_OF_MEMORY"}:
        next_action = "investigate"
    elif slurm_state in {"TIMEOUT", "PREEMPTED", "NODE_FAIL"} or slurm_state in {
        "CANCELLED",
        "CANCELLED+",
        "BOOT_FAIL",
        "DEADLINE",
    }:
        next_action = "fallback_runpod"
    else:
        # Defensive: ``preflight-failed`` short-circuit fires before
        # SLURM has flipped to FAILED; the in-job failure means the
        # next attempt belongs on RunPod.
        next_action = "fallback_runpod"

    observed_at = datetime.fromtimestamp(now_fn(), tz=UTC).isoformat().replace("+00:00", "Z")
    body = {
        "job_id": job_id,
        "cluster": cluster_name,
        "slurm_state": slurm_state,
        "exit_code": exit_code,
        "observed_at": observed_at,
        "next_action": next_action,
        "status": base_status,
    }
    note = json.dumps(body, sort_keys=True)
    marker_poster(
        issue=issue,
        marker="epm:cluster-terminal",
        note=note,
        version=1,
        by="backends.slurm_monitor",
    )


def _read_persisted_terminal(*, issue: int, job_id: str, event_reader) -> dict[str, Any] | None:
    """Read the persisted ``epm:cluster-terminal v1`` body for this job_id.

    Returns the parsed marker body, or ``None`` if no terminal marker
    exists yet for this job_id. The body shape matches
    :func:`_maybe_post_cluster_terminal`.
    """
    prior = _events_for_job(
        issue=issue, job_id=job_id, kind="epm:cluster-terminal", event_reader=event_reader
    )
    if not prior:
        return None
    return prior[-1]["body"]


def _poll_result_from_persisted_terminal(
    *, persisted: dict[str, Any], log_tail: str, log_mtime_sec_ago: int
) -> PollResult:
    """Synthesize a :class:`PollResult` from a persisted terminal marker.

    Used by the idempotent-reconnect path when SLURM returned
    ``UNKNOWN`` (squeue + scontrol ageout) but a terminal marker exists.
    The synthesized result carries the persisted status so the
    orchestrator's polling loop reaches its terminal branch instead of
    looping on a stale "running".
    """
    base_status = persisted.get("status", "dead")
    current_phase = persisted.get("slurm_state", "done").lower()
    return PollResult(
        status=base_status,
        current_phase=current_phase,
        new_milestone=False,
        last_log_mtime_sec_ago=log_mtime_sec_ago,
        pid_alive=False,
        log_tail_excerpt=log_tail[-2000:],
        gate=None,
        sentinels_processed=0,
        phase_log_mtime_sec_ago=log_mtime_sec_ago,
        shard_log_mtime_sec_ago=log_mtime_sec_ago,
        gpu_util="unknown",
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
