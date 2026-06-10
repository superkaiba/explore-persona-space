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
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from explore_persona_space.backends.base import BackendProbeError, PollResult
from explore_persona_space.backends.slurm import (
    PREFLIGHT_FAIL_MARKER,
    ClusterConfig,
)

logger = logging.getLogger(__name__)


class SlurmProbeError(BackendProbeError):
    """A squeue/scontrol probe FAILED (ssh rc != 0) — state UNKNOWN, NOT absent.

    Raised when the SSH transport / forced-command wrapper rejected the
    probe itself. Distinct from "the probe succeeded and the job is not
    in the queue" (genuinely absent → ``None`` / ``UNKNOWN``). Live
    incident (issue 535 attempt 2): the DRAC robot wrapper STRIPS
    QUOTING, so a quoted multi-token ``-o "%i %T ..."`` format fails
    with ``Unrecognized option: %T`` — pre-fix that rc!=0 read as "job
    gone" and let the router orphan a live job. Callers treat this as
    UNKNOWN-retry under a consecutive-failure budget, never as
    terminal.
    """


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

# Clock-skew margin (seconds) for the artifact-freshness gates. The
# submit timestamp is VM wall-clock while job.out mtimes / status.json
# heartbeats are written cluster-side; a modest skew must not gate out
# a genuinely-fresh artifact written seconds after submit. Stale
# prior-attempt artifacts are typically MANY minutes older than the new
# submit (issue 535 attempt 2: 31 min), so 120 s keeps the gate sharp.
FRESHNESS_SKEW_MARGIN_SEC = 120

# Secret-token patterns scrubbed out of every log tail BEFORE it can
# reach a git-committed marker (epm:cluster-poll log_tail_excerpt,
# epm:backend-selected evidence, epm:failure evidence). The sbatch
# preflight leaked expanded `${HF_TOKEN:?}` / `${WANDB_API_KEY:?}`
# values into job.out under xtrace (round-6 C1); the render now keeps
# those checks outside xtrace, and this scrubber is the
# defense-in-depth for any OTHER token that lands in a log.
_SECRET_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"wandb_v1_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(
        r"\b[0-9a-fA-F]{40}\b"
    ),  # 40-hex (WANDB legacy keys; also matches git SHAs — safe side)
)

_REDACTED = "«REDACTED»"


def _scrub_secret_tokens(text: str) -> str:
    """Replace secret-shaped tokens with ``«REDACTED»``.

    MUST run on the FULL text BEFORE any truncation: a ``[-2000:]`` cut
    can split a token so its tail no longer matches the prefix pattern
    yet still carries most of the secret.
    """
    for pattern in _SECRET_TOKEN_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return text


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
    submitted_at: float | None = None,
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
    * ``submitted_at`` — Unix timestamp of THIS attempt's sbatch submit
      (``handle.extra["submitted_at"]``, stamped by
      ``SlurmBackend.launch``). When provided, artifacts that PREDATE
      it (minus :data:`FRESHNESS_SKEW_MARGIN_SEC`) are ignored: the
      scratch dir is per-ISSUE and reused across attempts, so a stale
      prior-attempt ``status.json`` heartbeat / ``job.out`` preflight
      marker would otherwise mark the NEW job stalled/dead within one
      tick of submit (issue 535 attempt 2). The stall clock is floored
      at ``now - submitted_at`` — a job that has only existed for 60 s
      can be at most 60 s stale.

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

    state = state_querier(robot_alias=cluster.ssh_host, job_id=job_id)
    rsyncer(
        robot_alias=cluster.ssh_host,
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

    # ---- Attempt-freshness gate (C2) ----
    # The scratch dir is per-ISSUE and reused across attempts; SLURM only
    # truncates --output when the NEW job STARTS, so during the PENDING
    # window the rsync'd artifacts are the PREVIOUS attempt's. An
    # artifact older than this attempt's submit time must not feed the
    # stalled-detector, the phase read, or the preflight-failed
    # shortcut.
    if submitted_at is not None:
        min_artifact_epoch = float(submitted_at) - FRESHNESS_SKEW_MARGIN_SEC
        job_out_mtime_epoch = now_fn() - log_mtime_sec_ago
        if log_mtime_sec_ago >= 10**9 or job_out_mtime_epoch < min_artifact_epoch:
            log_tail, current_phase, new_milestone, log_mtime_sec_ago = "", "", False, 10**9
        if status_data and _status_artifact_epoch(status_data, status_json) < min_artifact_epoch:
            status_data = {}

    # Scrub secret-shaped tokens BEFORE any truncation — the tail feeds
    # git-committed markers (epm:cluster-poll) and the PollResult the
    # orchestrator may quote (round-6 C1).
    log_tail = _scrub_secret_tokens(log_tail)

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

    # Stall-clock floor (C2): a job submitted T seconds ago can be at
    # most T seconds stale — without this floor, a missing/gated-out
    # heartbeat reads as infinitely old and a tick one minute after
    # submit declares a LIVE job stalled (the live failure chain on
    # issue 535 attempt 2).
    if submitted_at is not None:
        heartbeat_sec_ago = min(heartbeat_sec_ago, max(0, int(now_fn() - float(submitted_at))))

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
    # Pass the FULL (already-scrubbed) tail; the poster scrubs again and
    # truncates itself so scrub-before-truncate holds for every caller.
    _maybe_post_cluster_poll(
        issue=issue,
        job_id=job_id,
        status=base_status,
        current_phase=final_phase,
        slurm_state=slurm_status,
        heartbeat_sec_ago=heartbeat_sec_ago,
        gpu_busy=bool(status_data.get("gpu_busy")),
        log_tail_excerpt=log_tail,
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

    ``log_tail_excerpt`` is scrubbed for secret-shaped tokens BEFORE the
    final 2000-char truncation (round-6 C1) — the marker note is
    committed to git, and a truncation that splits a token would leave
    an unmatchable-but-mostly-intact secret behind.
    """
    log_tail_excerpt = _scrub_secret_tokens(log_tail_excerpt)
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
    fallback). If both report the job as NOT FOUND, returns
    ``{"status": "UNKNOWN"}`` — the caller's idempotent-reconnect path
    handles that by reading the persisted ``epm:cluster-terminal``
    marker.

    Raises :class:`SlurmProbeError` when the PROBE ITSELF failed (ssh
    transport rc != 0 that is NOT a SLURM "Invalid job id" reply) —
    "couldn't ask" must never read as "job gone" (round-6 B1; a live
    job was orphaned when a probe failure classified as terminal).

    Args MUST be single-token: the DRAC robot forced-command wrapper
    STRIPS QUOTING before re-splitting, so a quoted multi-token format
    like ``-o "%i %T %M"`` arrives as ``-o %i %T %M`` and squeue errors
    with ``Unrecognized option: %T`` (verified live on robot-nibi).
    Every ``-o``/``--format`` value here is a single space-free token.
    """
    # Try scontrol first — it carries the most detail (JobState,
    # ExitCode, NodeList, RunTime). scontrol exits non-zero BOTH on a
    # genuinely-absent job ("Invalid job id specified") and on transport
    # failure, so its rc alone is ambiguous — the squeue fallback below
    # is the disambiguator. A HANG (TimeoutExpired) is a probe failure,
    # never "job gone" — same wrap as query_by_name (round-7 M1).
    try:
        proc = subprocess.run(
            ["ssh", robot_alias, "scontrol", "show", "job", job_id],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SlurmProbeError(
            f"scontrol show job {job_id} probe timed out after {timeout}s on {robot_alias}"
        ) from exc
    if proc.returncode == 0 and proc.stdout.strip():
        return _parse_scontrol_show_job(proc.stdout)
    scontrol_stderr = (proc.stderr or "").strip()

    # Fallback: squeue -j <id> -h -o %T (single-token format — see the
    # wrapper quote-stripping note in the docstring).
    try:
        proc = subprocess.run(
            ["ssh", robot_alias, "squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SlurmProbeError(
            f"squeue -j {job_id} probe timed out after {timeout}s on {robot_alias}"
        ) from exc
    if proc.returncode == 0 and proc.stdout.strip():
        return {"status": proc.stdout.strip().splitlines()[0].strip(), "exit_code": None}
    if proc.returncode == 0 or _is_job_not_found_stderr(proc.stderr):
        # squeue answered: the job is not in the active queue (empty
        # output, or the explicit "Invalid job id specified" reply for
        # an aged-out id). Genuinely absent → UNKNOWN; the caller's
        # epm:cluster-terminal lookup is the authoritative truth.
        return {"status": "UNKNOWN", "exit_code": None}

    # Both probes failed for non-"job not found" reasons — transport /
    # wrapper failure. State is UNKNOWN-because-unaskable, NOT absent.
    raise SlurmProbeError(
        f"SLURM state probe failed for job {job_id} on {robot_alias}: "
        f"squeue rc={proc.returncode} stderr={(proc.stderr or '').strip()[:200]!r}; "
        f"scontrol stderr={scontrol_stderr[:200]!r}"
    )


# SLURM's "the job id is not in the active queue" reply (squeue and
# scontrol both phrase it as "Invalid job id specified"). Matched
# case-insensitively so a wording tweak across SLURM versions degrades
# to a probe error (loud) rather than a silent misclassification.
_JOB_NOT_FOUND_RE = re.compile(r"invalid job id", re.IGNORECASE)


def _is_job_not_found_stderr(stderr: str | None) -> bool:
    """True iff stderr is SLURM's explicit job-not-found reply."""
    return bool(_JOB_NOT_FOUND_RE.search(stderr or ""))


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
    ``None`` ONLY when the probe SUCCEEDED (rc=0) and showed no match
    (job aged out / never landed).

    Raises :class:`SlurmProbeError` on rc != 0 — a transient SSH
    failure or a forced-command wrapper rejection must NOT read as
    "job gone" (round-6 B1): pre-fix, a probe failure during the cancel
    state machine returned "cancelled" on a still-live job, and during
    reconnect it triggered a blind double-submit. The ``-o %i`` format
    is a single space-free token by design — the wrapper strips quoting
    and re-splits, so multi-token formats fail with ``Unrecognized
    option`` (verified live on robot-nibi).

    A HANG (wedged slurmctld; connection up, command stuck) raises
    ``subprocess.TimeoutExpired`` — wrapped into the SAME
    :class:`SlurmProbeError` so the reconnect path's type discrimination
    (``except BackendProbeError: raise``) can't misread a hang as
    "no live job" and blind-double-submit (round-7 M1).
    """
    try:
        proc = subprocess.run(
            ["ssh", robot_alias, "squeue", "--name", job_name, "-h", "-o", "%i"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SlurmProbeError(
            f"squeue --name {job_name} probe timed out after {timeout}s on {robot_alias}"
        ) from exc
    if proc.returncode != 0:
        # ``squeue --name`` with zero matches exits 0 with empty output,
        # so ANY non-zero rc here is the probe failing, not the job
        # being absent.
        raise SlurmProbeError(
            f"squeue --name {job_name} probe failed on {robot_alias}: "
            f"rc={proc.returncode} stderr={(proc.stderr or '').strip()[:200]!r}"
        )
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


def fetch_started_evidence(
    *,
    robot_alias: str,
    scratch_dir: str,
    job_id: str,
    timeout: int = 30,
    rsyncer=None,
    min_artifact_ts: float | None = None,
) -> dict[str, Any] | None:
    """Probe the scratch dir for runtime artifacts proving the job STARTED.

    Used by the router's terminal-before-running classification: a job
    that vanished from ``squeue`` before it was ever observed RUNNING
    may have fast-failed (PD→R→exit between polls — e.g. an in-job
    preflight failure). The sbatch writes ``status.json`` + ``job.out``
    into the scratch dir the moment it starts, so their existence is
    the "it actually ran" signal that distinguishes a WORKLOAD failure
    from genuine no-compute.

    ``min_artifact_ts`` (Unix epoch — THIS attempt's submit time, from
    ``handle.extra["submitted_at"]``): when provided, an artifact only
    counts as evidence if it is at least as new as the submit (minus
    :data:`FRESHNESS_SKEW_MARGIN_SEC`). The scratch dir is per-ISSUE
    and reused across attempts, and SLURM truncates ``--output`` only
    when the new job STARTS — exactly the never-started window this
    probe targets — so without the gate a re-run reads the PREVIOUS
    attempt's artifacts as a guaranteed false "workload failure"
    (issue 535 attempt 2). ``job.out`` is gated on its rsync-preserved
    mtime; ``status.json`` on its ``heartbeat_ts`` (file mtime
    fallback).

    The local ``/tmp/slurm-<job_id>/`` cache is CLEARED before the
    rsync so fail-open actually holds: a transport failure leaves no
    files (→ ``None``), never the previous tick's possibly
    prior-attempt files — and a cross-cluster job-id collision cannot
    read another job's cache.

    Transport is rsync (allowlisted by the DRAC robot forced-command
    wrapper; ``ssh <alias> cat`` is NOT) via :func:`rsync_status_and_log`.
    Fail-open by design: a transport failure leaves the local files
    absent and this returns ``None``, so the router falls back to its
    legacy ``no_compute_available`` classification rather than gaining
    a new crash path.

    Returns an evidence dict (``phase`` / ``job_out_tail`` /
    ``status_json``) built from FRESH artifacts only, else ``None``.
    The tail is scrubbed for secret-shaped tokens BEFORE truncation
    (round-6 C1 — the evidence lands in git-committed markers).
    """
    sync = rsyncer or rsync_status_and_log
    local_dir = _local_state_dir(job_id)
    shutil.rmtree(local_dir, ignore_errors=True)
    sync(robot_alias=robot_alias, scratch_dir=scratch_dir, job_id=job_id, timeout=timeout)
    status_json_path = local_dir / "status.json"
    status_data = _read_status_json(status_json_path)
    job_out = local_dir / "job.out"
    tail, phase, _new_milestone, _mtime_sec_ago = _read_job_out(job_out)

    min_epoch = (
        float(min_artifact_ts) - FRESHNESS_SKEW_MARGIN_SEC if min_artifact_ts is not None else None
    )
    job_out_fresh = job_out.exists() and (min_epoch is None or job_out.stat().st_mtime >= min_epoch)
    status_fresh = bool(status_data) and (
        min_epoch is None or _status_artifact_epoch(status_data, status_json_path) >= min_epoch
    )
    if not job_out_fresh:
        tail, phase = "", ""
    if not status_fresh:
        status_data = {}
    if not job_out_fresh and not status_fresh:
        return None
    return {
        "phase": phase or str(status_data.get("phase", "")),
        "job_out_tail": _scrub_secret_tokens(tail)[-2000:],
        "status_json": status_data,
    }


def _status_artifact_epoch(status_data: dict[str, Any], path: Path) -> float:
    """Best-known write epoch of a rsync'd ``status.json``.

    Prefers the in-band ``heartbeat_ts`` (written by the compute node);
    falls back to the rsync-preserved file mtime when the timestamp is
    missing / unparseable. ``-inf`` when neither is available so the
    freshness gates treat the artifact as arbitrarily old.
    """
    ts = status_data.get("heartbeat_ts")
    if ts:
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            pass
    try:
        return path.stat().st_mtime
    except OSError:
        return float("-inf")


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
    "FRESHNESS_SKEW_MARGIN_SEC",
    "LOG_TAIL_BYTES",
    "SLURM_STATE_TO_STATUS",
    "STALL_SEC",
    "SlurmProbeError",
    "build_poll_result",
    "fetch_started_evidence",
    "query_by_name",
    "query_slurm_state",
    "rsync_status_and_log",
]
