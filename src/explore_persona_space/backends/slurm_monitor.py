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
   SLURM state PLUS a recent heartbeat = ``running``; a STALE
   heartbeat alone is only one stall SIGNAL — see "Stall semantics"
   below (#1969: the log mtime must be stale too, across consecutive
   ticks).
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

``STALL = SLURM state RUNNING AND heartbeat_ts older than STALL_SEC AND
job.out mtime older than STALL_SEC, sustained across >=
STALL_CONSECUTIVE_TICKS consecutive stall-condition ticks`` (#1969).
Three composable defenses over the pre-#1969 heartbeat-only predicate
(whose false reads #1900 recorded: heartbeat 397-517 s stale while the
captured log tail carried INFO lines written seconds before the tick):

* **Log-freshness veto** — BOTH signals must be stale. The log mtime is
  attempt-freshness-gated (C2, blanked to ``10**9`` for prior-attempt
  artifacts) and floored at the run age exactly like the heartbeat, so
  a missing/blanked log still reads stale and a job writing nothing
  anywhere still stalls.
* **Consecutive-tick streak (N = :data:`STALL_CONSECUTIVE_TICKS`)** —
  a single stall-condition tick reports ``running`` with
  ``stall_reason="slurm_stall_suspect"`` (a suspect tick keeps the
  short poll interval via the lane anomaly); ``stalled`` requires >= 2
  consecutive stall-condition ticks, with the streak increment
  TIME-GATED (>= :data:`STALL_SEC` elapsed since the last increment) so
  two rapid overlapping polls cannot reach the threshold seconds apart
  on one silently-failed rsync. The streak persists at
  ``/tmp/slurm-<job_id>/stall_streak.json`` — the same volatility class
  as the rsync'd artifacts; a VM reboot resets it, failing toward
  ``running`` for one extra tick.
* **Transport-degraded skip** — when :func:`rsync_status_and_log`
  reports a transport-class failure (``False`` return), the local
  copies are stale BY TRANSPORT, not evidence about the job: the tick
  skips stall evaluation entirely (streak neither incremented nor
  reset), records ``transport_degraded=true`` in the cluster-poll note,
  and keeps the short poll interval via the lane anomaly.

This remains weaker than the pod poller's 4-way check (PID alive, log
mtime, GPU util, sentinels) because we cannot run remote
``ps``/``nvidia-smi`` via the forced-command wrapper. The documented
weakening: a job that writes nothing to status.json OR job.out (e.g. an
early-init crash that hangs before the heartbeat loop starts) shows as
``stalled`` — one tick later than pre-#1969 (streak must reach 2) —
until SLURM itself reaps it. Operators can grep the rsync'd job.out for
``[phase=preflight-failed]`` to disambiguate. A ``stalled`` verdict
carries ``stall_reason="slurm_heartbeat_and_log_stale"`` (routes
``infra`` via ``scripts/failure_classifier.STALL_REASON_INFRA``). The
SLURM state-map path (``SUSPENDED -> stalled``,
:data:`SLURM_STATE_TO_STATUS`) is a different, correct signal and is
deliberately untouched by all three defenses.

Sentinel drain: fellows only (#1898)
------------------------------------

``sentinels_processed`` is drained ONLY on a cluster whose
:class:`ClusterConfig` carries ``sentinel_drain=True`` (the fellows /
charmander row): :func:`drain_cluster_sentinels` runs the shared
``poll_pipeline.sentinel_drain_shell`` loop over plain
``ssh <cluster.ssh_host>`` once per poll tick (the exact pattern
``GcpBackend._drain_sentinels`` established), posts parsed sentinels
VM-side via the transport-agnostic ``poll_pipeline.drain_sentinels_via``
(inheriting the ``.processed`` rename idempotency, the ``sentinel_fp``
dedupe #1084, the ``epm:results`` version rewrite #1095, the envelope
rescue #899, and the oversize-pointer path), and threads
``(sentinels_processed, gate)`` into every :class:`PollResult` this
module builds — a drained BLOCKING gate flips the tick status to
``"gate"`` (the ``backends/gcp.py`` merge semantics; benign
``gate=phase``/``smoke``/``dryrun`` signals never flip). NOTE the
fellows ``/workspace/logs`` is cluster-SHARED and PERSISTENT: a prior
crashed run's undrained sentinel posts late on the next same-issue
launch (correct-by-design), and any file matching ``issue-<N>-*.json``
is schema-parsed, posted, and ``mv -n``-renamed regardless of author
(documented trust surface — see ``.claude/rules/pod-side-reporting.md``).

On every OTHER SLURM cluster (``sentinel_drain=False`` — DRAC + Mila)
``sentinels_processed`` stays ``0``. That is the lane CONTRACT for
those clusters, not a missing drain (#608 follow-up verdict,
2026-06-11). The RunPod/GCP sentinel channel
(``/workspace/logs/issue-<N>-*.json``) cannot exist there:

* Compute nodes have no ``/workspace`` and unprivileged jobs cannot
  create one, so the contract's hardcoded sentinel dir is unwritable.
* The robot forced-command wrapper allowlists only ``sbatch`` /
  ``scancel`` / ``squeue`` / ``scp`` / ``rsync`` — the drain's
  list+cat shell (``poll_pipeline.sentinel_drain_shell``) and the
  ``.processed`` rename are both unexecutable over this transport.

A workload-cmd dispatch script written to the ``/workspace/logs``
contract fails LOUD there (``mkdir -p /workspace/logs`` → permission
denied → non-zero exit under ``set -euo pipefail`` → SLURM ``FAILED``
→ ``epm:cluster-terminal``), so unlike pre-#608 GCP those lanes cannot
silently drop markers from a COMPLETED run. Markers flow via the
rsync'd ``status.json`` + ``[phase=...]`` log lines, posted VM-side by
this monitor; a dispatcher that depends on sentinel-carried markers
(``epm:results`` payloads, ``gate`` fields) must be routed to a
drained lane (GCP, RunPod, or fellows) at plan time.

Workload-cmd must block (deliberate; #601 follow-up)
----------------------------------------------------

``COMPLETED → done`` here means "the batch script exited 0", nothing
more. The custom-stage renderer (``slurm.render_sbatch``) runs the
terminal ``[phase=done]`` block the moment the workload command
returns, and SLURM's job-exit cgroup teardown kills any detached
(setsid-forked) children at that same moment — so a self-daemonizing
``--workload-cmd`` yields a premature ``done`` verdict over a KILLED
run (surfacing as missing artifacts at fetch_results), never the
pre-#601-fix GCP failure of a terminal-success posted while a healthy
run continues in the background. There is no detached-pid wait on this
lane to "fix" that with: the GCP contract's ``/workspace/logs/*.pid``
dir is unwritable here for the same reason as the sentinel channel
above, and no SLURM-side pid-file convention exists. Workload commands
MUST block; detached patterns route to the GCP or RunPod lane at plan
time.

Dead-class disambiguation (#1866): the INVERSE terminal read — a
dead-class SLURM state (FAILED / TIMEOUT / PREEMPTED / ...) over a
FINISHED workload — is disambiguated by the job's own attempt-fresh
``status.json`` terminal-success record (``{"phase": "done",
"exit_code": "0"}``, written ONLY by the sbatch terminal block after
every stage exited clean under ``set -euo pipefail``; heartbeat and
stage writes carry ``exit_code: ""``). Post-done teardown artifacts
(slurmstepd cgroup errors, EXIT-trap / epilog failures) then read as
``done`` → ``next_action: interpret`` (the ``epm:cluster-terminal``
body keeps ``slurm_state`` verbatim and adds
``workload_done_despite_slurm: true``) instead of a false ``dead`` →
``investigate``. Fail-safe: evidence missing, stale (blanked by the C2
attempt-freshness gate), non-``done`` phase, non-``"0"`` exit_code, or
a legacy handle without ``submitted_at`` keeps the ``dead`` verdict
byte-identical to the pre-#1866 behavior.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Direct-submodule import (NOT ``from explore_persona_space.backends import
# ...``): this module is itself imported DURING the package __init__'s
# execution, so the submodule form is the cycle-safe binding (#1574 — the
# same shape as the .base / .slurm imports below). Bound as the MODULE so
# tests can monkeypatch ``excerpt_digest.issue_trigger_dense`` at the shared
# module object.
import explore_persona_space.backends.excerpt_digest as excerpt_digest
from explore_persona_space.backends.base import (
    BackendProbeError,
    PollResult,
    recommend_lane_next_interval,
)
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
# heartbeat AND job.out mtime are BOTH older than this (post the C2
# run-age floor) accrues stall-streak ticks toward a ``stalled``
# verdict (#1969 — module docstring § Stall semantics). Must sit
# safely above ``slurm.HEARTBEAT_INTERVAL_SECONDS`` (default 60s) so a
# healthy job's natural pause between heartbeats is NOT a false stall.
# Default 5 min matches the pod poller's STALL_SEC. Doubles as the time
# gate on consecutive stall-streak increments (STALL_CONSECUTIVE_TICKS
# below).
STALL_SEC = 300

# Consecutive stall-condition ticks required before ``stalled`` is
# reported (#1969). One poll tick is 540 s > STALL_SEC = 300 s, so ONE
# silently-failed rsync ages both local artifacts past the threshold —
# a single-tick transient must not kill the run's orchestration. N=2
# bounds the added genuinely-dead detection latency to ~9 min (one
# extra tick) while absorbing every single-tick transient (the observed
# #1900 class); N=3 would add ~18 min for no observed benefit.
STALL_CONSECUTIVE_TICKS = 2

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


def _stall_streak_path(job_id: str) -> Path:
    """Per-job stall-streak state file (#1969) — same volatility class
    as the rsync'd artifacts beside it (a VM reboot resets the streak,
    which fails toward ``running`` for one extra tick)."""
    return _local_state_dir(job_id) / "stall_streak.json"


def _read_stall_streak(job_id: str) -> tuple[int, float]:
    """Read the persisted stall streak → ``(streak, updated_ts)``.

    ``(0, 0.0)`` when absent / malformed, so the first stall-condition
    tick of a fresh job increments from zero.
    """
    try:
        data = json.loads(_stall_streak_path(job_id).read_text())
        return int(data["streak"]), float(data["updated_ts"])
    except (OSError, ValueError, TypeError, KeyError):
        return 0, 0.0


def _write_stall_streak(job_id: str, streak: int, *, now_fn=time.time) -> None:
    """Persist the stall streak; ``updated_ts`` feeds the time gate on
    the NEXT increment (module docstring § Stall semantics)."""
    path = _stall_streak_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"streak": int(streak), "updated_ts": float(now_fn())}))


def _evaluate_stall(
    *,
    job_id: str,
    base_status: str,
    slurm_status: str,
    heartbeat_sec_ago: int,
    log_stale_sec: int,
    transport_degraded: bool,
    now_fn=time.time,
) -> tuple[str, str | None, bool, int]:
    """One tick of #1969 stall evaluation (module docstring § Stall semantics).

    Returns ``(base_status, stall_reason, stall_anomaly, stall_streak)``:

    * Only meaningful while SLURM still says RUNNING; PENDING is never
      flagged (the selector's submit-and-park watchdog owns that wait).
    * **Log-freshness veto**: the stall condition requires the heartbeat
      AND the (run-age-floored) log mtime BOTH older than
      :data:`STALL_SEC`.
    * **Consecutive-tick streak**: the condition must hold on
      >= :data:`STALL_CONSECUTIVE_TICKS` ticks before ``stalled`` is
      reported; a sub-threshold tick reports ``running`` with
      ``stall_reason="slurm_stall_suspect"`` and a lane anomaly (keeps
      the short poll interval). Increments are TIME-GATED: the streak
      measures ELAPSED staleness, not poll count — two rapid overlapping
      polls (a manual poll beside the cron tick) must not reach the
      threshold seconds apart on one silently-failed rsync. The 0 -> 1
      transition is ungated (a first suspect tick is informational only;
      only increments TOWARD the stalled threshold are gated).
    * **Transport-degraded skip**: a failed rsync means the local copies
      are stale BY TRANSPORT — no evidence about the job. The tick skips
      evaluation entirely (streak neither incremented nor reset) and
      contributes a lane anomaly.
    * Any evaluated tick where the condition does NOT hold resets the
      streak to 0.
    """
    stall_reason: str | None = None
    stall_anomaly = False
    stall_streak, streak_updated_ts = _read_stall_streak(job_id)
    if transport_degraded:
        stall_anomaly = True
    elif (
        base_status == "running"
        and slurm_status != "PENDING"
        and heartbeat_sec_ago > STALL_SEC
        and log_stale_sec > STALL_SEC
    ):
        if stall_streak == 0 or float(now_fn()) - streak_updated_ts >= STALL_SEC:
            stall_streak += 1
            _write_stall_streak(job_id, stall_streak, now_fn=now_fn)
        if stall_streak >= STALL_CONSECUTIVE_TICKS:
            base_status = "stalled"
            stall_reason = "slurm_heartbeat_and_log_stale"
        else:
            # Suspect tick: never kill the run's orchestration on one
            # tick — report ``running`` and carry the suspicion.
            stall_reason = "slurm_stall_suspect"
            stall_anomaly = True
    elif stall_streak:
        _write_stall_streak(job_id, 0, now_fn=now_fn)
        stall_streak = 0
    return base_status, stall_reason, stall_anomaly, stall_streak


# Mapping from SLURM JobState (per ``scontrol show job`` / ``squeue -h
# -o %T``) to the orchestrator's PollResult.status enum. Anything not
# in the map defaults to ``running`` (pessimistic: don't reap a job we
# don't recognize yet — a future SLURM version's new state name should
# NOT mass-cancel jobs). Every ``dead`` mapping is additionally subject
# to the done-evidence disambiguation in :func:`build_poll_result`
# (#1866): a dead-class state whose attempt-fresh status.json reads
# ``{"phase": "done", "exit_code": "0"}`` flips to ``done`` (the
# workload finished; the dead label is a teardown artifact).
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


def _emit_tail(
    log_tail: str,
    *,
    trigger_dense: bool,
    status: str,
    current_phase: str,
    pid_alive: bool,
    log_path: str,
    log_mtime_sec_ago: int,
) -> str:
    """#1574: the emitted tail for one SLURM emission surface.

    The shared #1556 structural digest when the workload is trigger-dense,
    else the raw (already-scrubbed) tail returned UNTOUCHED — the untagged
    arm is byte-identical to pre-#1574 output by construction.
    """
    if not trigger_dense:
        return log_tail
    return excerpt_digest.digest_tail_excerpt(
        log_tail,
        status=status,
        current_phase=current_phase,
        pid_alive=pid_alive,
        source="slurm_job_out",
        log_path=log_path,
        mtime_sec_ago=min(log_mtime_sec_ago, 10**9),
    )


def drain_cluster_sentinels(
    issue: int, cluster: ClusterConfig, scratch_dir: str, *, runner=subprocess.run
) -> tuple[int, str | None]:
    """Drain ``/workspace/logs`` sentinels on a ``sentinel_drain`` cluster (#1898).

    Returns ``(sentinels_processed, gate)``. Fail-soft by contract: any
    transport failure (rc!=0, ``TimeoutExpired``) logs and returns
    ``(0, None)`` — the poll loop is never blocked (MooseFS read-wedge
    tolerance rides the local ``timeout=60`` bound). Not-``sentinel_drain``
    clusters (DRAC/Mila): no-op ``(0, None)`` with no ssh call and no
    ``scripts.poll_pipeline`` import (their #608 no-drain contract).

    Besides the canonical ``/workspace/logs/issue-<N>-*.json`` glob, the
    drain also scans the job's scratch-dir out_root logs dir as a fallback
    (the GCP #610 belt — covers a future perms change on the shared root).
    Posting is VM-side through ``poll_pipeline.drain_sentinels_via`` →
    ``task_workflow.post_event`` (legal: nothing runs cluster-side, so the
    pod-side ``task.py``-shellout ban does not apply); the ``.processed``
    rename runs over the same ssh alias via
    ``poll_pipeline._ssh_mark_processed``.

    ``runner`` is the test seam for the drain-list ssh (matches the file's
    dependency-injection convention; ``_ssh_mark_processed`` keeps its own
    ``subprocess.run`` — tests inject ``mark_processed`` behavior by faking
    the whole transport at the ``runner`` boundary or monkeypatching the
    ``poll_pipeline`` module).
    """
    if not cluster.sentinel_drain:
        return 0, None
    # Lazy import — the gcp.py::_drain_sentinels pattern (production
    # entrypoints put the repo root on sys.path; __file__-derived fallback
    # for direct library use).
    try:
        from scripts.poll_pipeline import (
            _ssh_mark_processed,
            drain_sentinels_via,
            parse_sentinel_stream,
            sentinel_drain_shell,
        )
    except ModuleNotFoundError:
        import sys

        repo_root = str(Path(__file__).resolve().parents[3])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.poll_pipeline import (
            _ssh_mark_processed,
            drain_sentinels_via,
            parse_sentinel_stream,
            sentinel_drain_shell,
        )
    fallback = f"{scratch_dir}/eval_results/issue_{issue}/logs/issue-{issue}-*.json"
    # Wrap in `bash -c` (the GCP transport's idiom, minus its sudo — fellows
    # files are our own user's): charmander's login shell is ZSH, where the
    # drain loop's bash-isms fail (`shopt: command not found`) and an empty
    # glob is a hard `no matches found` error (live acceptance, 2026-07-30).
    shell = "bash -c " + shlex.quote(sentinel_drain_shell(issue, extra_globs=(fallback,)))
    try:
        res = runner(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", cluster.ssh_host, shell],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        logger.error("fellows sentinel drain ssh timed out (60s) — skipping this tick")
        return 0, None
    if res.returncode != 0:
        logger.error(
            "fellows sentinel drain FAILED (rc=%d): %s",
            res.returncode,
            (res.stderr or "").strip()[:300],
        )
        return 0, None
    sentinels = parse_sentinel_stream(res.stdout or "")
    processed, gate = drain_sentinels_via(
        issue=issue,
        list_sentinels=lambda: sentinels,
        mark_processed=lambda p: _ssh_mark_processed(cluster.ssh_host, p),
    )
    if sentinels and processed == 0:
        logger.error(  # the #608 fail-loud line — never a silent 0
            "fellows sentinel drain: %d matched, 0 processed — inspect "
            "/workspace/logs on %s + poller stderr",
            len(sentinels),
            cluster.ssh_host,
        )
    return processed, gate


def _entry_drain(
    sentinel_drainer, issue: int, cluster: ClusterConfig, scratch_dir: str
) -> tuple[int, str | None]:
    """Belt around the #1898 entry drain: default the seam + never raise.

    A drain bug must never block the poll loop — this wraps the helper's
    own fail-soft contract in one more ``except Exception`` and returns
    the no-drain ``(0, None)`` on a raise.
    """
    drainer = sentinel_drainer if sentinel_drainer is not None else drain_cluster_sentinels
    try:
        return drainer(issue, cluster, scratch_dir)
    except Exception:
        logger.exception("sentinel drain raised (fail-soft) — continuing poll tick")
        return 0, None


def _apply_done_evidence(
    base_status: str,
    *,
    submitted_at: float | None,
    status_data: dict[str, Any],
) -> tuple[str, bool]:
    """Done-evidence disambiguation (#1866) — returns ``(status, evidence)``.

    The sbatch terminal block writes status.json ``{"phase": "done",
    "exit_code": "0"}`` ONLY after every stage exited clean under
    ``set -euo pipefail`` (heartbeat/stage writes carry ``exit_code: ""``),
    so a dead-class SLURM state with ATTEMPT-FRESH done-0 evidence is a
    teardown/epilog artifact (cgroup errors, trap failures), not a crashed
    workload — flip to ``done``. Requires ``submitted_at`` (the caller's C2
    gate blanked stale artifacts); a legacy handle without it fails toward
    the pre-#1866 ``dead`` verdict.
    """
    evidence = (
        base_status == "dead"
        and submitted_at is not None
        and status_data.get("phase") == "done"
        and status_data.get("exit_code") == "0"
    )
    return ("done" if evidence else base_status), evidence


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
    sentinel_drainer=None,
) -> PollResult:
    """One-tick poll → :class:`PollResult`.

    Composes:

    * ``query_slurm_state`` over SSH for the SLURM JobState + exit code.
    * ``rsync_status_and_log`` for the heartbeat + log tail.
    * Stall detection (#1969): SLURM=RUNNING with heartbeat AND job.out
      mtime BOTH older than :data:`STALL_SEC` (each floored at the run
      age), sustained across :data:`STALL_CONSECUTIVE_TICKS`
      consecutive stall-condition ticks — module docstring § Stall
      semantics. A sub-threshold stall-condition tick reports
      ``running`` with ``stall_reason="slurm_stall_suspect"``; a
      transport-degraded tick (rsyncer returned ``False``) skips stall
      evaluation entirely.

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
      pass a no-op + pre-seeded local files. A ``False`` return marks
      the tick transport-degraded (stall evaluation skipped); the check
      is ``is False``, so a legacy/stub seam returning ``None``
      evaluates normally (#1969).
    * ``now_fn`` — for the stall clock; tests pin it.
    * ``marker_poster`` — defaults to
      :func:`backends.slurm.post_marker_via_task_py`. Tests pass a
      list-appender to capture which markers were posted.
    * ``event_reader`` — defaults to
      :func:`task_workflow.list_events`. Tests pass a stub returning a
      pre-seeded event trail.
    * ``sentinel_drainer`` — defaults to
      :func:`drain_cluster_sentinels` (#1898). Tests pass a stub
      returning a canned ``(processed, gate)`` tuple. A no-op on
      ``sentinel_drain=False`` clusters (DRAC/Mila keep
      ``sentinels_processed=0`` byte-identically).
    * ``submitted_at`` — Unix timestamp of THIS attempt's sbatch submit
      (``handle.extra["submitted_at"]``, stamped by
      ``SlurmBackend.launch``). When provided, artifacts that PREDATE
      it (minus :data:`FRESHNESS_SKEW_MARGIN_SEC`) are ignored: the
      scratch dir is per-ISSUE and reused across attempts, so a stale
      prior-attempt ``status.json`` heartbeat / ``job.out`` preflight
      marker would otherwise mark the NEW job stalled/dead within one
      tick of submit (issue 535 attempt 2). The stall clock is floored
      at the run age — a job that has only RUN for 60 s can be at most
      60 s stale. Run age prefers SLURM's own ``RunTime`` from the
      scontrol probe (queue time excluded — a long PENDING wait must
      not push a just-started job past the §7 early-run window or
      defeat this floor); ``now - submitted_at`` is the fallback when
      the probe carried no RunTime (squeue fallback / UNKNOWN).

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
    # ---- Sentinel drain (#1898, fellows only) ----
    # Runs at ENTRY so the TERMINAL tick drains too (the final epm:results
    # sentinel is drained on the same tick that reads COMPLETED) and the
    # UNKNOWN persisted-terminal early return below still threads the
    # drained values. ``_entry_drain`` defaults the seam + never raises.
    drain_processed, drain_gate = _entry_drain(sentinel_drainer, issue, cluster, scratch_dir)

    state = state_querier(robot_alias=cluster.ssh_host, job_id=job_id)
    rsync_ok = rsyncer(
        robot_alias=cluster.ssh_host,
        scratch_dir=scratch_dir,
        job_id=job_id,
    )
    # #1969: ``is False`` deliberately — only the production
    # :func:`rsync_status_and_log` (or a stub modeling it) can mark the
    # tick transport-degraded; legacy test stubs returning ``None``
    # evaluate normally.
    transport_degraded = rsync_ok is False

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

    # #1574: per-tick trigger-dense declaration read (#1556 semantics — one
    # fresh read per tick, so a mid-run `task.py add-tag <N> trigger-dense`
    # takes effect on the next tick). VM-side by construction: this monitor
    # composes SSH/rsync from the orchestrator VM — the cluster node has no
    # git checkout (module docstring § "Sentinel drain: fellows only").
    # Gates ONLY the emitted excerpt surfaces below (the returned PollResult,
    # the epm:cluster-poll note, the persisted-terminal synthesis); detection
    # — the C2 freshness gate above, PREFLIGHT_FAIL_MARKER, phase parsing —
    # always reads the raw ``log_tail``.
    trigger_dense = excerpt_digest.issue_trigger_dense(issue, log=logger)

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
                persisted=persisted,
                log_tail=log_tail,
                log_mtime_sec_ago=log_mtime_sec_ago,
                trigger_dense=trigger_dense,
                sentinels_processed=drain_processed,
                gate=drain_gate,
            )
        # No marker either — we genuinely don't know. Default to running
        # so the orchestrator doesn't reap a job we haven't proven dead.

    base_status = SLURM_STATE_TO_STATUS.get(slurm_status, "running")

    # Done-evidence disambiguation (#1866): a dead-class SLURM state over
    # an attempt-fresh done-0 status.json flips to `done` (see
    # :func:`_apply_done_evidence`). Ordering is fail-safe: the C2
    # attempt-freshness gate above already blanked stale artifacts, and
    # the PREFLIGHT_FAIL_MARKER shortcut below retains its override.
    base_status, workload_done_evidence = _apply_done_evidence(
        base_status, submitted_at=submitted_at, status_data=status_data
    )

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

    # Run age — feeds the stall-clock floor below AND the §7 early-run
    # guard. PREFER SLURM's own ``RunTime`` (parsed from the same
    # scontrol probe): it measures time since the job STARTED, in the
    # cluster's own accounting, so a long PENDING queue does not count
    # toward it. The submit-time age is the fallback (squeue-fallback /
    # UNKNOWN ticks carry no RunTime) — it over-counts queue time, the
    # pre-fix behavior. ``StartTime`` is deliberately NOT used: scontrol
    # prints it as a naive cluster-LOCAL timestamp, so differencing it
    # against the VM clock injects multi-hour timezone error.
    run_time_sec = state.get("run_time_sec")
    if run_time_sec is not None:
        run_age_sec: float | None = float(run_time_sec)
    elif submitted_at is not None:
        run_age_sec = max(0.0, float(now_fn()) - float(submitted_at))
    else:
        run_age_sec = None

    # Stall-clock floor (C2): a job that has RUN for T seconds can be at
    # most T seconds stale — without this floor, a missing/gated-out
    # heartbeat reads as infinitely old and a tick one minute after
    # submit declares a LIVE job stalled (the live failure chain on
    # issue 535 attempt 2). RUN age, not submit age: after a long
    # PENDING queue the submit-time floor is no floor at all, and the
    # first RUNNING tick of a just-started job (no heartbeat written
    # yet) would read a LIVE job as stalled.
    # The SAME floor applies to the log-staleness read (#1969 symmetry):
    # a missing/blanked job.out reads 10**9 and would otherwise defeat
    # the young-job protection the heartbeat floor provides.
    log_stale_sec = log_mtime_sec_ago
    if run_age_sec is not None:
        heartbeat_sec_ago = min(heartbeat_sec_ago, max(0, int(run_age_sec)))
        log_stale_sec = min(log_stale_sec, max(0, int(run_age_sec)))

    # Stall detection (#1969) — module docstring § Stall semantics. The
    # SUSPENDED->stalled STATE-MAP path above is a different, correct
    # signal and bypasses the stall defenses by design.
    base_status, stall_reason, stall_anomaly, stall_streak = _evaluate_stall(
        job_id=job_id,
        base_status=base_status,
        slurm_status=slurm_status,
        heartbeat_sec_ago=heartbeat_sec_ago,
        log_stale_sec=log_stale_sec,
        transport_degraded=transport_degraded,
        now_fn=now_fn,
    )

    # Preflight failure detection — the sbatch echoes
    # ``[phase=preflight-failed]`` then exit's non-zero. Even before SLURM
    # transitions to FAILED, we can spot it in the log.
    if PREFLIGHT_FAIL_MARKER in log_tail:
        base_status = "dead"
        current_phase = "preflight-failed"

    final_phase = current_phase or slurm_status.lower()

    # #1574: on a trigger-dense workload BOTH emitted excerpt surfaces below
    # (the epm:cluster-poll note and the returned PollResult) carry the
    # shared #1556 structural digest instead of raw tail lines. The raw arm
    # passes ``log_tail`` through untouched (``emit_tail is log_tail`` —
    # byte-identical untagged output); the digest is a short single line, so
    # the existing ``[-2000:]`` slices below are no-ops on it.
    emit_tail = _emit_tail(
        log_tail,
        trigger_dense=trigger_dense,
        status=base_status,
        current_phase=final_phase,
        pid_alive=base_status == "running",
        log_path=str(job_out),
        log_mtime_sec_ago=log_mtime_sec_ago,
    )

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
        log_tail_excerpt=emit_tail,
        marker_poster=marker_poster,
        event_reader=event_reader,
        stall_streak=stall_streak,
        transport_degraded=transport_degraded,
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
            workload_done=workload_done_evidence,
        )

    # ---- Adaptive bg-poll interval (§7) — the SLURM lane's quiet subset ----
    # ``recommend_lane_next_interval`` goes quiet ONLY while SLURM itself
    # says RUNNING: PENDING / CONFIGURING / COMPLETING / UNKNOWN-defaulted
    # ticks are transitional or ambiguous and stay on the short interval
    # (fail toward coverage). Run age prefers SLURM's ``RunTime`` (computed
    # above), so a long PENDING queue no longer pushes a just-started job
    # past the 30-min early-run window on its first RUNNING ticks; the
    # fresh ``[phase=...]`` line in the 16 KiB tail (``new_milestone``)
    # remains the defense-in-depth on RunTime-less fallback ticks.
    # #1969: a stall-suspect or transport-degraded tick is OR'd into the
    # lane anomaly (never replacing the existing non-RUNNING condition)
    # so such a tick can never draw the 1800 s quiet interval.
    next_interval = recommend_lane_next_interval(
        status=base_status,
        gate=drain_gate,
        sentinels_processed=drain_processed,
        new_milestone=new_milestone,
        run_age_sec=run_age_sec,
        lane_anomaly=(slurm_status != "RUNNING") or stall_anomaly,
    )

    return PollResult(
        # Gate precedence mirrors backends/gcp.py's drain merge: a drained
        # BLOCKING gate sentinel wins over every other status — the
        # orchestrator must park at the user gate (Step 6d.4) before
        # advancing. ``drain_sentinels_via`` surfaces ``gate`` only for a
        # ``blocks_pipeline: True`` sentinel, so benign ``gate=phase`` /
        # ``smoke`` / ``dryrun`` signals never flip status (#1898). On
        # ``sentinel_drain=False`` clusters (DRAC/Mila) the drain is a
        # no-op and this stays ``base_status`` with ``sentinels_processed=0``
        # (module docstring § "Sentinel drain: fellows only").
        status="gate" if drain_gate else base_status,
        current_phase=final_phase,
        new_milestone=new_milestone,
        last_log_mtime_sec_ago=log_mtime_sec_ago,
        pid_alive=base_status == "running",
        log_tail_excerpt=emit_tail[-2000:],
        gate=drain_gate,
        sentinels_processed=drain_processed,
        phase_log_mtime_sec_ago=log_mtime_sec_ago,
        shard_log_mtime_sec_ago=log_mtime_sec_ago,
        gpu_util="busy" if status_data.get("gpu_busy") else "idle",
        next_interval=next_interval,
        # #1969: "slurm_heartbeat_and_log_stale" on a stalled verdict
        # (routes infra via failure_classifier.STALL_REASON_INFRA);
        # "slurm_stall_suspect" on a sub-threshold stall-condition tick
        # (status stays "running"); None otherwise.
        stall_reason=stall_reason,
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
    stall_streak: int = 0,
    transport_degraded: bool = False,
) -> None:
    """Post ``epm:cluster-poll v1`` only when status or phase changed.

    ``stall_streak`` / ``transport_degraded`` (#1969) are ADDITIVE note
    keys — the consecutive-tick stall streak after this tick's
    evaluation, and whether the tick's rsync failed transport-class
    (stall evaluation skipped). The dedup predicate is unchanged.

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
        "stall_streak": stall_streak,
        "transport_degraded": transport_degraded,
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
    workload_done: bool = False,
) -> None:
    """Post ``epm:cluster-terminal v1`` exactly once per job_id.

    Subsequent ticks read the persisted marker via
    :func:`_read_persisted_terminal` and short-circuit, so a job that
    re-emerges briefly as FAILED across two ticks only writes ONE
    terminal-state row.

    ``workload_done`` (#1866): True when :func:`build_poll_result`'s
    done-evidence disambiguation fired (dead-class SLURM state +
    attempt-fresh ``{"phase": "done", "exit_code": "0"}`` status.json).
    On a non-COMPLETED slurm_state that routes ``next_action`` to
    ``interpret`` and adds ``workload_done_despite_slurm: true`` to the
    marker body; ``slurm_state`` stays verbatim for the audit trail.
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

    # #1866: done-evidence disambiguation — the workload's own terminal
    # status.json record proves every stage exited clean, so the
    # dead-class SLURM label is a teardown/epilog artifact. Route to
    # `interpret` (Step 6d.3 upload verification stays the hard artifact
    # gate) and flag the marker body; slurm_state rides verbatim.
    workload_done_despite_slurm = workload_done and slurm_state != "COMPLETED"
    if workload_done_despite_slurm:
        next_action = "interpret"

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
    if workload_done_despite_slurm:
        body["workload_done_despite_slurm"] = True
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
    *,
    persisted: dict[str, Any],
    log_tail: str,
    log_mtime_sec_ago: int,
    trigger_dense: bool = False,
    sentinels_processed: int = 0,
    gate: str | None = None,
) -> PollResult:
    """Synthesize a :class:`PollResult` from a persisted terminal marker.

    Used by the idempotent-reconnect path when SLURM returned
    ``UNKNOWN`` (squeue + scontrol ageout) but a terminal marker exists.
    The synthesized result carries the persisted status so the
    orchestrator's polling loop reaches its terminal branch instead of
    looping on a stale "running".

    ``trigger_dense`` (#1574): when the workload is declared trigger-dense
    the synthesized excerpt is the shared #1556 structural digest of the
    (possibly stale) local tail; the additive default keeps every
    pre-#1574 caller byte-identical.

    ``sentinels_processed`` / ``gate`` (#1898): the entry drain's values,
    threaded so the fellows lane's UNKNOWN-reconnect tick still reports a
    drain that ran this tick. The additive defaults (``0`` / ``None``) keep
    the ``sentinel_drain=False`` clusters byte-identical (module docstring
    § "Sentinel drain: fellows only"), and a drained BLOCKING gate flips
    status exactly as the main path (the gcp.py merge semantics).
    """
    base_status = persisted.get("status", "dead")
    current_phase = persisted.get("slurm_state", "done").lower()
    # #1866: a persisted done-evidence flip carries a dead-class slurm_state
    # (teardown artifact) with status "done" — the reconnect tick reports
    # phase "done" to match the original tick's read, not "failed"/"timeout".
    if persisted.get("workload_done_despite_slurm"):
        current_phase = "done"
    emit_tail = _emit_tail(
        log_tail,
        trigger_dense=trigger_dense,
        status=base_status,
        current_phase=current_phase,
        pid_alive=False,
        log_path="",
        log_mtime_sec_ago=log_mtime_sec_ago,
    )
    return PollResult(
        status="gate" if gate else base_status,
        current_phase=current_phase,
        new_milestone=False,
        last_log_mtime_sec_ago=log_mtime_sec_ago,
        pid_alive=False,
        log_tail_excerpt=emit_tail[-2000:],
        gate=gate,
        sentinels_processed=sentinels_processed,
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
    <"N:M"|None>, "node": <node|None>}``; the scontrol path additionally
    carries ``run_time_sec`` (SLURM's elapsed RUN time, queue time
    excluded — see :func:`_parse_scontrol_show_job`). On scontrol "no
    such job" falls back to ``squeue -j`` (same disambiguation as the
    pod poller's fallback). If both report the job as NOT FOUND, returns
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
            encoding="utf-8",
            errors="replace",
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
            encoding="utf-8",
            errors="replace",
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
    JobState, ExitCode, NodeList, RunTime; everything else is noise for
    the monitor. ``run_time_sec`` (from ``RunTime``) is the cluster's
    own elapsed RUN time — queue time excluded — and is preferred over
    the VM-stamped submit age for the §7 early-run guard and the C2
    stall-clock floor. (``StartTime`` is deliberately not parsed: it is
    a naive cluster-LOCAL timestamp, useless against the VM clock.)
    """
    out: dict[str, Any] = {
        "status": "UNKNOWN",
        "exit_code": None,
        "node": None,
        "run_time_sec": None,
    }
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
        elif key == "RunTime":
            out["run_time_sec"] = _parse_slurm_runtime(val)
    return out


# scontrol's elapsed-time format: ``[days-]HH:MM:SS``. Non-time values
# (``INVALID``, ``UNKNOWN``) deliberately don't match → ``None`` → the
# caller falls back to the submit-time age.
_SLURM_RUNTIME_RE = re.compile(r"^(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})$")


def _parse_slurm_runtime(val: str) -> int | None:
    """Parse scontrol's ``RunTime=[days-]HH:MM:SS`` into whole seconds.

    Returns ``None`` for non-time values (e.g. ``INVALID``) so the
    caller's run-age computation falls back to ``submitted_at``.
    """
    match = _SLURM_RUNTIME_RE.match(val)
    if match is None:
        return None
    days, hours, minutes, seconds = match.groups()
    return (int(days or 0) * 24 + int(hours)) * 3600 + int(minutes) * 60 + int(seconds)


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
            encoding="utf-8",
            errors="replace",
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
) -> bool:
    """Pull ``status.json`` + ``job.out`` from the cluster scratch dir.

    Lands them under ``/tmp/slurm-<job_id>/`` so concurrent monitors on
    different jobs don't clobber each other. ``--partial`` + ``--mkpath``
    keep the cost low.

    Non-fatal on rsync failure — a transient SSH hiccup shouldn't crash
    the polling loop; the next tick will retry and the local files
    (still readable from the previous tick) keep the monitor honest.

    Returns the per-tick transport verdict (#1969):

    * ``True`` — every pull exited 0, or failed ONLY with rsync rc
      23/24 (partial transfer / source file vanished): the file
      genuinely does not exist remotely, so the never-writing-job stall
      weakening is preserved and stall evaluation proceeds normally.
    * ``False`` — ANY other non-zero rc (255 ssh error, 30/35 timeouts,
      10/11/12/13 socket / file-I/O / protocol errors — rc 12 is common
      when ssh dies mid-stream), or a :class:`subprocess.TimeoutExpired`
      (caught here — pre-#1969 it would crash the tick): a
      transport-class failure means the local copies cannot be trusted
      this tick, and :func:`build_poll_result` skips stall evaluation.
      DEFAULT policy is deliberately fail-closed: mapping an unlisted rc
      to ``True`` would re-create a 2-tick false-stall path under
      sustained degradation, while over-suppression stays bounded by
      SLURM dead-class states + job time limits.
    """
    local_dir = _local_state_dir(job_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    transport_ok = True
    for filename in ("status.json", "job.out"):
        argv = [
            "rsync",
            "-a",
            "--partial",
            "--mkpath",
            f"{robot_alias}:{scratch_dir}/{filename}",
            str(local_dir / filename),
        ]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.debug("rsync %s/%s timed out after %ss", scratch_dir, filename, timeout)
            transport_ok = False
            continue
        if proc.returncode != 0:
            logger.debug(
                "rsync %s/%s returned %d: %s",
                scratch_dir,
                filename,
                proc.returncode,
                proc.stderr.strip(),
            )
            # rc 23/24 = source missing/vanished — NORMAL for a job
            # that has not written the artifact yet; not transport.
            if proc.returncode not in (23, 24):
                transport_ok = False
    return transport_ok


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
    "STALL_CONSECUTIVE_TICKS",
    "STALL_SEC",
    "SlurmProbeError",
    "build_poll_result",
    "fetch_started_evidence",
    "query_by_name",
    "query_slurm_state",
    "rsync_status_and_log",
]
