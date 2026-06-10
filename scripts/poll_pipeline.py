"""poll_pipeline.py — one-tick poller for a running experiment pod.

Invoked by the `/issue` orchestrator's bg-Bash sleep-chain (see
`.claude/skills/issue/SKILL.md` Step 6d.2). Performs ONE poll then exits
— the orchestrator chains successive `Bash(sleep 540 && uv run python
scripts/poll_pipeline.py ..., run_in_background=true)` calls and is
re-invoked by the harness when each bg Bash returns.

Why orchestrator-owned: subagents have ONE turn — they are NOT
auto-re-invoked when a bg Bash finishes. The orchestrator IS. See
`CLAUDE.md` § "Subagent vs orchestrator re-invocation semantics" and
the deprecated memory `feedback_subagent_sleep_chain.md` for context.

Per tick:

1. Drain pod-side sentinel files (`/workspace/logs/issue-<N>-*.json`,
   skipping `*.processed`). Each sentinel was written by a pod-side
   dispatcher that cannot shell out to `scripts/task.py` (CLAUDE.md
   "Pod-side code NEVER shells out" rule). The poller parses each
   sentinel, posts the carried `epm:<kind>` marker from the local VM
   via `task_workflow.post_event`, then renames the sentinel to
   `<path>.processed` so it posts exactly once. If a sentinel carries a
   non-empty ``gate`` field, the poll returns ``status=gate`` with that
   gate name in the JSON output so the orchestrator parks at a user
   gate instead of continuing the polling loop.
2. SSH to the pod (one heredoc batching: PID liveness, log mtime, log tail).
3. Parse the latest `[phase=...]` line from the log tail.
4. If new milestone vs the cached previous phase, post `epm:progress`
   to the task's events.jsonl via the local-VM `task_workflow.post_event`
   library (NOT on the pod).
5. Decide status: `done` | `gate` | `stalled` | `dead` | `running`.
6. Print one JSON line summary to stdout. Exit 0 on successful poll
   regardless of `status`. Exit non-zero only on caller-error (bad args,
   library import failure).

Stall threshold: ALL of (a) `last_log_mtime_sec_ago > stall_sec`
(default 900s via ``DEFAULT_STALL_SEC``, overridable per-tick via
``--stall-sec`` CLI or ``EPM_POLL_STALL_SEC`` env var for workloads
with sparse log cadence, e.g. checkpoint-only logging at >15min
intervals; taken over BOTH the top-level log and the freshest cell
log), (b) every per-phase log under
``/workspace/logs/issue-<N>-*.log`` is also quiet for >stall_sec,
(c) every shard / repo-rooted phase log under
``/workspace/explore-persona-space/logs/issue_<N>{,_*}/*.log`` AND
every dispatcher per-job log under
``/workspace/explore-persona-space/eval_results/issue_<N>{,_*}/logs/*.log``
is also quiet for >stall_sec, and (d) the GPUs are idle. Only when all
four signals agree does the poll declare `stalled`; any fresh log
OR a busy GPU keeps the run in `running`.

CPU-advancing override (#518): even with the stall conjunction met, a
launcher whose process session (`setsid` group) has accrued more
cumulative CPU since the previous tick is doing CPU-bound work and is
NOT stalled. The probe sums `time` across every process sharing the
launcher PID's SID via `ps -e -o sess=,time=` and persists the sample
to the local state file; on the next tick the delta is compared to a
small epsilon (`SESSION_CPU_ADVANCE_EPSILON_SECS`). If CPU advanced,
the verdict flips to `running`. If CPU is flat OR unknown (first tick
after launch, launcher dead, `ps` unavailable), the older arbiters
keep the verdict — fail-safe to the pre-#518 behavior. Incident: task
#518 scoring_syco phase, 2026-06-10 — a healthy CPU-bound aggregation
phase wrote nothing to the log for ~7.8h while the python child was
at 100% CPU; the poller falsely declared `stalled`.

Staleness folds in cell-log mtimes (incident #405 smoke-first): when the
dispatcher is blocked in ``proc.wait()`` on a sequential smoke cell, the
main sweep log goes silent for ~15-18 min while the smoke cell actively
trains+evals and writes to its own per-cell log
(``<main_log_no_ext>/cell_*.log``). The probe therefore reports the
freshest mtime across (main log, newest cell log) so a healthy single-
cell phase reads as `running`, not false-`stalled` / false-`dead`. When
a cell log is the fresher source, its tail is also surfaced in
``log_tail_excerpt`` for the orchestrator's progress notifications.

Staleness ALSO folds in per-phase logs + GPU utilization (incident
#468 multi-phase training-sweep): a launcher that writes
``[phase=X]`` to the top-level log only at phase boundaries and
redirects the long phase's stdout to a separate
``/workspace/logs/issue-<N>-<phase>.log`` keeps the top-level log
silent for the full phase while the workload is actively writing to
the per-phase log AND keeping a GPU busy. Declaring `stalled` from
the top-level mtime alone false-fails the healthy run and strands a
billing pod. The probe therefore also reports (a) the max mtime over
``/workspace/logs/issue-<N>-*.log`` (excluding the top-level log and
``*.json`` / ``*.processed`` sentinels) and (b) per-GPU
``utilization.gpu`` integers via ``nvidia-smi``. The GPU check fails
safe: ``nvidia-smi`` unavailable / errors -> ``unknown`` (NOT idle),
so a healthy run is NEVER declared stalled purely from an nvidia-smi
failure — the per-phase-log mtime signal still carries the verdict.

Staleness ALSO folds in per-shard / repo-rooted phase logs (incident
#488 multi-GPU shard fan-out): some launchers write per-GPU shard
logs under a subdirectory like
``/workspace/explore-persona-space/logs/issue_<N>/phase*_g*.log``
(8 shard files under a nested directory, underscore separator), and
the #331 family of multi-phase scripts writes flat repo-rooted phase
logs like ``/workspace/explore-persona-space/logs/issue_<N>_phase<X>.log``.
Both layouts are invisible to the #468 ``/workspace/logs/issue-<N>-*.log``
glob — the i488 Pass B inner loop (~3 min between shard-log writes
across 57 cells per shard) silently tripped the 36-min main-log
threshold on 2026-06-07 while the pipeline was healthy. The probe
therefore ALSO reports the max mtime across both shard layouts so a
healthy multi-GPU run reads as `running`, not false-`stalled`. The
match remains intentionally narrow (only paths embedding ``issue_<N>``
or ``issue-<N>`` under the repo logs dir; not a broad recursive scan)
to avoid coupling other pods' background writes to the verdict.

Staleness ALSO folds in dispatcher per-job logs (incident #521
judge-batch wait): the issue_519/521-style dispatcher writes one log
per job under ``<output_dir>/logs/*.log``, with ``output_dir``
typically ``/workspace/explore-persona-space/eval_results/issue_<N>``.
During a CPU-bound phase that polls an external judge batch the GPUs
are idle BY DESIGN and the main log is quiet, while the per-job log
appends every 30-60s — the only liveness signal. On 2026-06-10 a #521
tick declared the healthy EM-steering job ``stalled`` (pid alive,
GPUs all 0, main log 1302s stale) because no probe reached
``eval_results/issue_<N>/logs/``. The shard-log probe therefore ALSO
globs ``eval_results/issue_<N>{,_*}/logs/*.log`` into the same
max-mtime reduction. The match stays narrow on purpose: the directory
must be exactly ``issue_<N>`` or ``issue_<N>_<suffix>`` (a bare
``issue_<N>*`` glob would let issue 5 match issue 521's directories).

GPU-idle advisory (incidents #518 + #537): the stall verdict treats an
idle GPU only as CORROBORATION — a run that is alive and logging on a
CPU-only phase with every GPU at 0% is (correctly) classified healthy,
and before this advisory it burned silently (#518 ran a single-core CPU
scoring phase ~14h on an idle 8xH100; #537 polled an external judge
batch 2.5h+ the same day, 2026-06-10). The poller therefore ALSO tracks
the sustained span of "healthy verdict + every GPU idle" across ticks
(state-file backed, like ``ssh_fail_count``) and, once the span exceeds
``EPM_GPU_IDLE_ADVISORY_MIN`` minutes (default 30; ``0`` disables),
posts a NON-BLOCKING ``epm:progress`` advisory marker (note prefixed
``[gpu-idle-advisory]``, riding the same marker channel as the phase-
transition posts — no new marker schema) suggesting the CPU phase move
off-pod per CLAUDE.md "CPU-only phases don't hold GPU pods". At most
one advisory per phase name (de-dup persisted in the state file); the
advisory NEVER changes the status verdict and never stops anything.
Fail-safe semantics carry over from ``_gpu_idle``: an ``unknown`` /
unparsable GPU sample resets the span rather than counting as idle.

Dead: PID not alive AND last phase line is NOT `done` (clean exit
should always end with `[phase=done]`).

Phase-line shape expected from the entry script:
    2026-05-21 14:32:18 [phase=training step=1000/2000 loss=2.1]
    2026-05-21 14:55:02 [phase=eval]
    2026-05-21 15:10:44 [phase=done]

Anything matching the regex `\\[phase=([a-z0-9_]+)` will be picked up; the
token immediately after `phase=` is the milestone name. Digits are part of
the token (`[phase=p0_render]` parses as `p0_render`, not `p`), so numbered
phase-naming schemes (p0/p1/p2) work without spelling digits out.

Sentinel schema (v1) — written by pod-side dispatchers, drained here:

    filename: /workspace/logs/issue-<N>-<kind_slug>-<epoch_seconds>.json
        kind_slug = kind with `:` -> `_` (e.g. ``epm_fact_candidates``).
    payload (JSON, dict):
        {
          "sentinel_schema_version": 1,                  # required, must be 1
          "task_id": <int>,                              # informational
          "kind": "<full kind, e.g. 'epm:fact-candidates'>",
          "version": <int>,                              # marker version
          "gate": "<gate name>" | null,                  # if set, poll returns status=gate
          "blocks_pipeline": true|false,                 # informational
          "note": "<marker note body>",                  # may also be sent as 'payload'
          "by": "<author>",
          "ts": "<ISO-8601 UTC>",
        }

Unknown schema versions are logged + skipped (not renamed) so a future
poller can re-process them. Malformed JSON / missing required fields are
logged + skipped likewise — the sentinel is left in place so the next
poller (or a human) can inspect it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make src/ importable so we can call task_workflow.post_event directly.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from explore_persona_space.task_workflow import (  # noqa: E402
    EVENT_NOTE_MAX,
    find_task_path,
    latest_event,
    post_event,
)

log = logging.getLogger("poll_pipeline")

# Default seconds of log-mtime silence before declaring the run stalled.
# Workloads with sparse log cadence (e.g. checkpoint-only logging at >15min
# intervals — task #522 builds a 16x16 JS-distance matrix that logs only
# every ~106min per partial-cache checkpoint) override this at the CLI via
# ``--stall-sec`` or env via ``EPM_POLL_STALL_SEC`` so the poller does not
# false-positive ``stalled`` during normal inter-checkpoint quiet windows.
# ``STALL_SEC`` is preserved as a module-level alias of the default so
# existing tests that read ``pp.STALL_SEC`` as a reference threshold keep
# working without modification.
DEFAULT_STALL_SEC = 900
STALL_SEC = DEFAULT_STALL_SEC
# Substring of the ValueError message raised by ``task_workflow.post_event``
# when ``note`` exceeds ``EVENT_NOTE_MAX``. Matched against ``str(exc)`` so
# we route exactly that failure to graceful-degradation (persist + pointer
# marker) instead of leaving the sentinel un-renamed and retrying forever.
# See ``src/explore_persona_space/task_workflow.py`` ``post_event``: the
# message format is ``"event note exceeds {EVENT_NOTE_MAX} chars (<len>); ..."``.
_OVERSIZE_NOTE_ERROR_SUBSTR = "event note exceeds"
PHASE_RE = re.compile(r"\[phase=([a-z0-9_]+)")
# The epm:run-launched marker note is free-form `key=value` tokens plus
# trailing prose (see .claude/agents/experimenter.md "Post epm:run-launched").
# `pid=<int>` is the resolved python child PID the experimenter posted.
MARKER_PID_RE = re.compile(r"\bpid=(\d+)")
DEFAULT_STATE_DIR = _REPO_ROOT / ".claude" / "cache"

# How many consecutive SSH-probe failures must accumulate before the poller
# auto-fires ``pod.py config --refresh-from-api <pod>`` as a stale-port
# self-heal. Set to 10 (~3-4 min at the orchestrator's typical 20s spacing) so
# a transient SSH hiccup never burns a refresh call, but a sustained
# connection-refused stretch — the #488 stale-port pattern — does. After the
# refresh attempt the counter resets so we never hot-loop refresh calls; the
# next ``SSH_FAIL_REFRESH_THRESHOLD`` consecutive failures will trigger a
# re-try.
SSH_FAIL_REFRESH_THRESHOLD = int(os.environ.get("EPM_POLL_SSH_FAIL_REFRESH_THRESHOLD", "10"))


def _try_refresh_pods_conf_from_api(pod: str) -> bool:
    """Best-effort ``pod.py config --refresh-from-api <pod>`` self-heal.

    Fires after :data:`SSH_FAIL_REFRESH_THRESHOLD` consecutive ``_ssh_probe``
    failures on the same pod — the #488 stale-port pattern, where a
    SUPPLY_CONSTRAINT-blocked resume eventually brought the pod back at a NEW
    SSH port via a retry path that bypassed ``_upsert_pods_conf`` and
    ``pods.conf`` stayed stale while the SSH polling loop spun indefinitely.

    Fail-soft: any failure (subprocess timeout, non-zero exit, missing
    binary, oserror) is logged and the function returns False. The polling
    loop never crashes on this auto-heal; the caller resets the failure
    counter regardless so we don't hot-loop refresh calls back-to-back.

    Returns True on success (refresh-from-api exited 0), False otherwise.
    """
    cmd = [
        "uv",
        "run",
        "python",
        str(_REPO_ROOT / "scripts" / "pod.py"),
        "config",
        "--refresh-from-api",
        pod,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning(
            "auto-heal: pod.py config --refresh-from-api %s raised %s; "
            "polling loop continues (next %d consecutive failures will retry)",
            pod,
            type(exc).__name__,
            SSH_FAIL_REFRESH_THRESHOLD,
        )
        return False
    if result.returncode != 0:
        log.warning(
            "auto-heal: pod.py config --refresh-from-api %s exited rc=%d; stderr=%s",
            pod,
            result.returncode,
            (result.stderr or "").strip(),
        )
        return False
    log.info(
        "auto-heal: pod.py config --refresh-from-api %s OK; pods.conf "
        "+ ~/.ssh/config refreshed against the live RunPod API after %d "
        "consecutive SSH-probe failures (#488 stale-port pattern)",
        pod,
        SSH_FAIL_REFRESH_THRESHOLD,
    )
    return True


def _marker_pid(issue: int) -> int | None:
    """Return the `pid=` from the latest epm:run-launched marker, or None.

    Self-correction source when the on-pod pidfile is stale: the marker
    the experimenter posts on every (re)launch carries the live python
    child PID. Reading it is a pure, branch-guarded library read on the
    VM (no commit), so it is safe from poll_pipeline's bg-Bash context.
    """
    try:
        ev = latest_event(issue, prefix="epm:run-launched")
    except Exception as exc:
        log.warning("could not read epm:run-launched for #%d: %s", issue, exc)
        return None
    if ev is None:
        return None
    m = MARKER_PID_RE.search(ev.get("note", "") or "")
    return int(m.group(1)) if m else None


# Schema version the poller knows how to parse. Bump in lockstep with the
# pod-side writer (currently ``run_experiment_<N>.py::SENTINEL_SCHEMA_VERSION``).
# Newer schemas are skipped + logged, never silently mis-parsed.
SENTINEL_SCHEMA_VERSION_SUPPORTED = 1

# Required keys in every parsed sentinel payload. ``payload`` is accepted as
# a synonym for ``note`` for forward-compat with sentinels that put the
# marker body under that key.
_SENTINEL_REQUIRED_KEYS: tuple[str, ...] = (
    "sentinel_schema_version",
    "kind",
    "version",
)


@dataclass(frozen=True)
class PollResult:
    status: str  # running | done | gate | stalled | dead
    current_phase: str
    new_milestone: bool
    last_log_mtime_sec_ago: int
    pid_alive: bool
    # True when the pod-side pid FILE did not exist at probe time —
    # ``pid_alive=False`` then means "no pidfile to probe" (possibly with
    # a live marker-pid fallback carrying liveness), NOT "pid probed
    # dead". Observability only; status routing is unchanged (#521).
    pid_file_missing: bool
    log_tail_excerpt: str
    gate: str | None = None  # set when a drained sentinel carried a non-empty gate
    sentinels_processed: int = 0
    # Broadened liveness signals (see ``_ssh_probe`` docstring + the
    # module-level "Staleness ALSO folds in per-phase logs + GPU
    # utilization" paragraph). Surfaced so the orchestrator's JSON-line
    # summary records WHY a healthy long-phase run stayed in `running`
    # despite a quiet top-level + cell log.
    phase_log_mtime_sec_ago: int = 10**9
    # Shard / repo-rooted phase log freshness (#488 shard layouts +
    # #521 dispatcher per-job logs). ``10**9`` means no covered layout
    # exists yet (defaults to "very old" so the absence never by itself
    # keeps a stalled verdict from firing).
    shard_log_mtime_sec_ago: int = 10**9
    gpu_util: str = "unknown"
    # True when THIS tick posted the [gpu-idle-advisory] marker (#518/#537).
    # Observability only; the advisory never changes ``status``.
    gpu_idle_advisory_posted: bool = False
    # Session-CPU signal (#518). ``session_cpu_secs`` is the literal probe
    # output: a float string like ``"4271.5"`` or ``"unknown"``.
    # ``cpu_advancing`` is the ternary decision: True (session advanced
    # since previous tick), False (session flat), None (no signal — first
    # tick, launcher dead, or ps unavailable). Surfaced in the JSON line
    # so operators can see WHY a long-quiet run stayed in ``running`` (or
    # WHY a stall verdict landed despite a CPU-bound phase).
    session_cpu_secs: str = "unknown"
    cpu_advancing: bool | None = None


def _ssh_probe(
    pod: str,
    log_path: str,
    pid_file: str,
    issue: int,
    marker_pid: int | None = None,
) -> dict[str, str]:
    """One SSH round-trip — returns dict with keys pid_alive,
    marker_pid_alive, mtime_epoch, cell_mtime_epoch, log_tail,
    cell_log_tail, phase_log_mtime_epoch, gpu_util.

    Batches into a single heredoc to keep the SSH cost to one connection.

    Liveness keys:
    * ``pid_alive`` — liveness of the PID stored in ``pid_file``.
    * ``pid_file_missing`` — ``"1"`` when ``pid_file`` does not exist on
      the pod; ``PID_ALIVE=0`` then means "no pidfile to probe", NOT
      "pid probed dead". Observability-only (incident #521: a false
      ``status=dead`` on a healthy run was hard to diagnose because the
      tick collapsed "file absent, marker-pid fallback in effect" into
      a bare ``pid_alive=False``). ``"0"`` on the SSH-failure fail-safe
      path — transport failure means "unknown", not "missing".
    * ``marker_pid_alive`` — liveness of ``marker_pid`` (the PID carried
      by the latest epm:run-launched marker) when one is supplied. The
      marker-pid probe is the self-correction path for a stale pidfile.

    Liveness-of-output keys (used to broaden the stall verdict so a long
    healthy phase that writes only to a per-cell or per-phase log is not
    false-failed as stalled, incidents #405 + #468):

    * ``mtime_epoch`` — top-level log mtime (still drives the milestone /
      phase-line parse).
    * ``cell_mtime_epoch`` — mtime of the freshest per-cell log under
      ``<log_path stripped of .log>/cell_*.log`` (the smoke-first /
      sequential-cell convention; #405). ``"0"`` when no cell logs exist.
    * ``cell_log_tail`` — tail of that same freshest cell log; used by
      ``poll_once`` as the ``log_tail_excerpt`` source when the cell log
      is fresher than the main log. Permission / nullglob /
      no-such-directory cases silently degrade to ``0`` + empty tail
      (and the caller falls back to the main-log mtime alone) — matching
      how the existing main-log probe degrades on a missing log file.
    * ``phase_log_mtime_epoch`` — max mtime over per-phase logs matching
      ``/workspace/logs/issue-<issue>-*.log`` (excluding ``*.json`` /
      ``*.processed`` sentinels, and the top-level
      ``issue-<issue>.log`` itself). ``"0"`` when no per-phase log
      exists yet. Complements ``cell_mtime_epoch``: cell logs live
      under ``<log_path%.log>/cell_*.log`` (nested), per-phase logs
      live flat at ``/workspace/logs/issue-<N>-<phase>.log``; the two
      globs don't overlap.
    * ``shard_log_mtime_epoch`` — max mtime over repo-rooted shard /
      phase / per-job logs (incidents #488 + #521). Covers three extra
      layouts neither the cell-log nor the per-phase-log probe sees:
      (1) ``/workspace/explore-persona-space/logs/issue_<issue>/*.log``
      — nested subdirectory holding per-GPU shard logs (e.g.
      ``phase1_g0.log``..``phase1_g7.log``);
      (2) ``/workspace/explore-persona-space/logs/issue_<issue>_*.log``
      — flat repo-rooted phase logs (e.g. ``issue_<N>_phase0.log``,
      the #331 / #444 family layout);
      (3) ``/workspace/explore-persona-space/eval_results/
      issue_<issue>{,_*}/logs/*.log`` — dispatcher per-job logs
      (``<output_dir>/logs/<job>.log``, the issue_519/521 dispatcher
      convention; #521), the only fresh signal during a CPU-bound
      judge-batch wait with GPUs idle by design.
      Excludes ``*.json`` / ``*.processed`` sentinels. ``"0"`` when no
      covered layout exists. All patterns share an mtime reduction
      (max), so a healthy run keeping ANY layout fresh stays in
      ``running``.
    * ``gpu_util`` — comma-separated per-GPU ``utilization.gpu``
      integers (e.g. ``"95,87,42,90"``). ``"unknown"`` when
      ``nvidia-smi`` is unavailable or errors (fail-safe — see
      ``_gpu_idle``).
    * ``session_cpu_secs`` — cumulative CPU seconds (as a float string,
      e.g. ``"4271.5"``) summed across every process in the launcher
      PID's process SESSION (`setsid` group). The launcher itself
      accrues ~no CPU — its children carry the work — so summing over
      the session captures every descendant regardless of how the
      python child re-execs. ``"unknown"`` when the launcher PID is
      not alive (no session to probe) or when ``ps`` is unavailable /
      errors (fail-safe — see ``_session_cpu_advancing``). Used as a
      defense against false-stalled verdicts on silent CPU-bound
      phases: even when every log mtime exceeds the stall threshold
      AND the GPUs are idle, a session whose cumulative CPU time has
      advanced since the previous tick is doing work, not hanging
      (incident #518 scoring_syco phase, 2026-06-10 — a healthy run
      with cumulative CPU time advancing 1:1 with wall time was
      false-declared stalled because no log line appeared for ~7.8h).
    """
    marker_probe = ""
    if marker_pid is not None:
        marker_probe = (
            f"if ps -p {marker_pid} > /dev/null 2>&1; "
            f"then echo MARKER_PID_ALIVE=1; else echo MARKER_PID_ALIVE=0; fi; "
        )
    # Cell-log probe: strip a trailing `.log` from log_path to get the
    # per-cell log directory (the dispatch_sweep convention used since
    # #405 smoke-first runs). `shopt -s nullglob` makes the empty case
    # expand to nothing rather than the literal pattern. We pick the
    # single freshest cell log via `stat -c '%Y %n'` + `sort -n` and
    # emit its mtime + its tail, so the caller has both the staleness
    # signal AND a tail to surface when the main log is the stale one.
    cell_probe = (
        'CELL_LOG_DIR="${LOG_PATH%.log}"; '
        "shopt -s nullglob; "
        'CELL_FILES=("$CELL_LOG_DIR"/cell_*.log); '
        "if [ ${#CELL_FILES[@]} -gt 0 ]; then "
        '  FRESHEST=$(stat -c "%Y %n" "${CELL_FILES[@]}" 2>/dev/null | sort -n | tail -1); '
        '  CELL_MTIME="${FRESHEST%% *}"; '
        '  CELL_PATH="${FRESHEST#* }"; '
        '  echo "CELL_MTIME_EPOCH=${CELL_MTIME:-0}"; '
        "  echo CELL_TAIL_START; "
        '  if [ -n "$CELL_PATH" ] && [ -f "$CELL_PATH" ]; then tail -500 "$CELL_PATH"; fi; '
        "  echo CELL_TAIL_END; "
        "else "
        "  echo CELL_MTIME_EPOCH=0; echo CELL_TAIL_START; echo CELL_TAIL_END; "
        "fi; "
    )
    # Per-phase-log probe (#468): max mtime across
    # `/workspace/logs/issue-<issue>-*.log`, excluding the top-level
    # `issue-<issue>.log` itself and any `*.json` / `*.processed`
    # sentinels (the sentinel naming uses `-*-*.json[.processed]` so
    # `*.log` already excludes them; the explicit `case` defends against
    # accidental `.log.json` etc.). `shopt -s nullglob` makes an empty
    # glob expand to nothing rather than the literal pattern. `sort -n
    # | tail -1` yields the max epoch, or "" when no per-phase log
    # exists; the `echo` then prints "PHASE_LOG_MTIME_EPOCH=0" (parsed
    # as 0 by the caller).
    phase_log_probe = (
        f"PHASE_LOG_MAX=$("
        f"shopt -s nullglob; "
        f"for f in /workspace/logs/issue-{issue}-*.log; do "
        f'  case "$f" in *.processed|*.json) continue ;; esac; '
        f'  case "$f" in /workspace/logs/issue-{issue}.log) continue ;; esac; '
        f'  stat -c %Y "$f" 2>/dev/null; '
        f"done | sort -n | tail -1); "
        f'echo "PHASE_LOG_MTIME_EPOCH=${{PHASE_LOG_MAX:-0}}"; '
    )
    # Shard-log probe (#488): the i488 multi-GPU layout writes per-GPU
    # shard logs under `/workspace/explore-persona-space/logs/issue_<N>/
    # phase*_g*.log` (nested subdirectory, underscore separator), and the
    # #331/#444 family writes flat repo-rooted phase logs at
    # `/workspace/explore-persona-space/logs/issue_<N>_*.log`. Neither
    # pattern is reached by the `phase_log_probe` glob above, so the
    # i488 Pass B inner loop (~3 min between shard writes across 57
    # cells per shard) silently tripped the 36-min main-log threshold
    # while every shard log was actively being written (2026-06-07).
    # We probe BOTH layouts and reduce to the max mtime; either layout
    # being fresh keeps the verdict in `running`. The match is narrow on
    # purpose — paths must embed `issue_<N>` (underscore) under the repo
    # logs directory, so unrelated logs from other pods don't pollute
    # the freshness signal.
    #
    # Dispatcher per-job logs (#521): the issue_519/521-style dispatcher
    # writes one log per job under `<output_dir>/logs/*.log`, with
    # `output_dir` typically `eval_results/issue_<N>` under the repo
    # root. During a CPU-bound judge-batch wait (GPUs idle by design,
    # main log quiet) the per-job log is the ONLY fresh signal — a #521
    # tick false-declared `stalled` on a healthy EM-steering job
    # (2026-06-10) because no probe reached it. Folded into the same
    # SHARD_LOG max. The two extra globs keep the issue-number match
    # exact (`issue_<N>` or `issue_<N>_<suffix>`; a bare `issue_<N>*`
    # would let issue 5 match issue 521's directories).
    shard_log_probe = (
        f"SHARD_LOG_MAX=$("
        f"shopt -s nullglob; "
        f"for f in /workspace/explore-persona-space/logs/issue_{issue}/*.log "
        f"         /workspace/explore-persona-space/logs/issue_{issue}_*.log "
        f"         /workspace/explore-persona-space/eval_results/issue_{issue}/logs/*.log "
        f"         /workspace/explore-persona-space/eval_results/issue_{issue}_*/logs/*.log; do "
        f'  case "$f" in *.processed|*.json) continue ;; esac; '
        f'  stat -c %Y "$f" 2>/dev/null; '
        f"done | sort -n | tail -1); "
        f'echo "SHARD_LOG_MTIME_EPOCH=${{SHARD_LOG_MAX:-0}}"; '
    )
    # GPU util probe (#468): fail-safe to "unknown" so a missing /
    # erroring nvidia-smi never declares stalled by itself (the
    # per-phase-log + cell-log signals still protect long phases). See
    # `_gpu_idle` for the threshold + fail-safe semantics.
    gpu_probe = (
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "  GPU_OUT=$(nvidia-smi --query-gpu=utilization.gpu "
        "    --format=csv,noheader,nounits 2>/dev/null | paste -sd, -); "
        '  echo "GPU_UTIL=${GPU_OUT:-unknown}"; '
        'else echo "GPU_UTIL=unknown"; fi; '
    )
    # Session CPU probe (#518): cumulative CPU seconds summed across
    # every process sharing the launcher PID's session id (SID). The
    # launcher is started with `setsid nohup bash <launcher>` (see
    # `.claude/agents/experimenter.md` "Launch") so every descendant
    # — the python child, vLLM workers, judge subprocesses, etc. —
    # carries the same SID as the launcher PID itself. `ps -o sess=`
    # reads that SID; `etime` field is wall-clock; `time` field is
    # cumulative CPU. We filter the full `ps -e` output by SID and
    # sum `time` (HH:MM:SS, or D-HH:MM:SS for >1 day) into seconds.
    #
    # ``unknown`` when (a) the pidfile is missing / pid is dead — no
    # session to probe; the launcher exiting clean is `phase=done` /
    # `dead` territory and the stall arbiter never reaches this
    # signal — or (b) `ps` is unavailable / errors. The
    # `_session_cpu_advancing` decision fails safe to "no signal" in
    # those cases (the older log + GPU arbiters then carry the
    # verdict, preserving the pre-#518 behavior).
    session_cpu_probe = (
        f"if [ -f {pid_file} ]; then "
        f"  LPID=$(cat {pid_file}); "
        f"  SID=$(ps -o sess= -p $LPID 2>/dev/null | tr -d ' '); "
        f'  if [ -n "$SID" ] && [ "$SID" != "0" ]; then '
        f"    CPU_SUM=$(ps -e -o sess=,time= 2>/dev/null | "
        f'      awk -v s="$SID" \'$1==s {{ '
        f'        n=split($2,a,":"); '
        f"        if (n==3) {{ secs += a[1]*3600 + a[2]*60 + a[3] }} "
        f"        else if (n==2) {{ secs += a[1]*60 + a[2] }} "
        f"        else if (n==1) {{ "
        f'          m=split(a[1],b,"-"); '
        f"          if (m==2) {{ secs += b[1]*86400 + b[2] }} "
        f"          else {{ secs += a[1] }} "
        f"        }} "
        f"      }} END {{ "
        f'        if (NR==0) {{ print "unknown" }} '
        f'        else {{ printf "%.1f", secs }} '
        f"      }}'); "
        f'    echo "SESSION_CPU_SECS=${{CPU_SUM:-unknown}}"; '
        f'  else echo "SESSION_CPU_SECS=unknown"; fi; '
        f'else echo "SESSION_CPU_SECS=unknown"; fi; '
    )
    heredoc = (
        f"LOG_PATH={log_path}; "
        f"if [ -f {pid_file} ]; then "
        f"  echo PID_FILE_MISSING=0; PID=$(cat {pid_file}); "
        f"  if ps -p $PID > /dev/null 2>&1; then echo PID_ALIVE=1; else echo PID_ALIVE=0; fi; "
        f"else echo PID_FILE_MISSING=1; echo PID_ALIVE=0; fi; "
        f"{marker_probe}"
        f"if [ -f $LOG_PATH ]; then "
        f"  echo MTIME_EPOCH=$(stat -c %Y $LOG_PATH); "
        f"  echo TAIL_START; tail -500 $LOG_PATH; echo TAIL_END; "
        f"else echo MTIME_EPOCH=0; echo TAIL_START; echo TAIL_END; fi; "
        f"{cell_probe}"
        f"{phase_log_probe}"
        f"{shard_log_probe}"
        f"{gpu_probe}"
        f"{session_cpu_probe}"
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod, heredoc],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        log.error("ssh failed (rc=%d): %s", result.returncode, result.stderr.strip())
        # ``ssh_failed`` is the explicit caller signal so ``poll_once`` can
        # count consecutive transport failures (#488 stale-port auto-heal)
        # WITHOUT having to infer "ssh down" from the zeroed values below
        # (which can also legitimately mean "log file does not exist yet").
        return {
            "pid_alive": "0",
            "pid_file_missing": "0",
            "marker_pid_alive": "0",
            "mtime_epoch": "0",
            "cell_mtime_epoch": "0",
            "log_tail": "",
            "cell_log_tail": "",
            "phase_log_mtime_epoch": "0",
            "shard_log_mtime_epoch": "0",
            "gpu_util": "unknown",
            "session_cpu_secs": "unknown",
            "ssh_failed": "1",
        }
    parsed = _parse_probe_stdout(result.stdout)
    parsed["ssh_failed"] = "0"
    return parsed


# Scalar `KEY=value` lines the probe heredoc emits. Order is irrelevant —
# the parser dispatches on the prefix and stores the trailing value.
_PROBE_SCALAR_KEYS: tuple[str, ...] = (
    "PID_ALIVE",
    "PID_FILE_MISSING",
    "MARKER_PID_ALIVE",
    "MTIME_EPOCH",
    "CELL_MTIME_EPOCH",
    "PHASE_LOG_MTIME_EPOCH",
    "SHARD_LOG_MTIME_EPOCH",
    "GPU_UTIL",
    "SESSION_CPU_SECS",
)


def _parse_probe_stdout(stdout: str) -> dict[str, str]:
    """Parse the probe heredoc's stdout into the dict ``_ssh_probe`` returns.

    Factored out of ``_ssh_probe`` to keep the SSH call site simple and
    drive the parser's complexity below the C901 cap. Pure / stdout-only;
    no I/O.
    """
    parsed: dict[str, str] = {
        "pid_alive": "0",
        "pid_file_missing": "0",
        "marker_pid_alive": "0",
        "mtime_epoch": "0",
        "cell_mtime_epoch": "0",
        "log_tail": "",
        "cell_log_tail": "",
        "phase_log_mtime_epoch": "0",
        "shard_log_mtime_epoch": "0",
        "gpu_util": "unknown",
        "session_cpu_secs": "unknown",
    }
    tail_lines: list[str] = []
    cell_tail_lines: list[str] = []
    in_tail = False
    in_cell_tail = False
    for line in stdout.splitlines():
        if line == "TAIL_START":
            in_tail = True
            continue
        if line == "TAIL_END":
            in_tail = False
            continue
        if line == "CELL_TAIL_START":
            in_cell_tail = True
            continue
        if line == "CELL_TAIL_END":
            in_cell_tail = False
            continue
        if in_tail:
            tail_lines.append(line)
            continue
        if in_cell_tail:
            cell_tail_lines.append(line)
            continue
        # Dispatch on the `KEY=value` prefix; store under the lowercased key.
        for key in _PROBE_SCALAR_KEYS:
            if line.startswith(f"{key}="):
                parsed[key.lower()] = line.split("=", 1)[1].strip()
                break
    parsed["log_tail"] = "\n".join(tail_lines)
    parsed["cell_log_tail"] = "\n".join(cell_tail_lines)
    return parsed


def _ssh_drain_sentinels(pod: str, issue: int) -> list[tuple[str, str]]:
    """List + cat unprocessed sentinels in one SSH round-trip.

    Globs ``/workspace/logs/issue-<issue>-*.json`` (skipping ``*.processed``),
    emits each as ``SENTINEL_START <path>\\n<body>\\nSENTINEL_END`` so the
    caller can parse multiple sentinels from one stdout blob. Files are NOT
    renamed here — the rename happens via ``_ssh_mark_processed`` only after
    the marker post succeeds, so a mid-tick crash leaves the sentinel
    un-renamed and the next poll retries it (idempotent).

    Returns a list of ``(remote_path, body)`` pairs (possibly empty). On
    SSH failure returns an empty list and logs the error.
    """
    # The glob is path-terminal `.json` and explicitly excludes `.processed`.
    # ``shopt -s nullglob`` makes an empty glob expand to nothing instead of
    # the literal pattern so we don't accidentally cat a path called e.g.
    # ``/workspace/logs/issue-444-*.json``.
    heredoc = (
        f"shopt -s nullglob; "
        f"for f in /workspace/logs/issue-{issue}-*.json; do "
        f'  case "$f" in *.processed) continue ;; esac; '
        f'  echo "SENTINEL_START $f"; '
        f'  cat "$f"; '
        f'  echo ""; echo "SENTINEL_END"; '
        f"done"
    )
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod, heredoc],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        log.error("ssh drain failed (rc=%d): %s", result.returncode, result.stderr.strip())
        return []
    sentinels: list[tuple[str, str]] = []
    current_path: str | None = None
    current_body: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("SENTINEL_START "):
            current_path = line[len("SENTINEL_START ") :].strip()
            current_body = []
        elif line == "SENTINEL_END":
            if current_path is not None:
                sentinels.append((current_path, "\n".join(current_body).strip()))
            current_path = None
            current_body = []
        elif current_path is not None:
            current_body.append(line)
    return sentinels


def _ssh_mark_processed(pod: str, remote_path: str) -> bool:
    """Rename ``remote_path`` -> ``remote_path + '.processed'`` on the pod.

    Returns True on success. Logs + returns False on failure (the sentinel
    is left in place; next poll tick will re-attempt). We use ``mv -n`` (no
    clobber) so a pre-existing ``.processed`` file is preserved — the
    sentinel writer never reuses epoch-tagged filenames, so a collision
    here would itself be a bug worth surfacing.
    """
    # Single-quote the remote path to neutralise shell metacharacters; the
    # writer's filename is ``issue-<N>-<kind_slug>-<epoch>.json`` so it's
    # safe by construction, but defence-in-depth costs nothing.
    quoted = "'" + remote_path.replace("'", "'\\''") + "'"
    cmd = f"mv -n {quoted} {quoted}.processed"
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod, cmd],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        log.error(
            "ssh mv failed for %s (rc=%d): %s",
            remote_path,
            result.returncode,
            result.stderr.strip(),
        )
        return False
    return True


def _slugify_kind(kind: str) -> str:
    """Match the pod-side sentinel writer's `kind` slug (``:`` -> ``_``).

    Used to name the persisted oversize-note artifact (``sentinel-note-<slug>-
    <epoch>.txt``) so the artifact file's stem mirrors the sentinel filename
    convention. Pure / no I/O.
    """
    return kind.replace(":", "_")


def _persist_oversize_note(
    *,
    issue: int,
    remote_path: str,
    kind: str,
    version: int,
    by: str,
    full_note: str,
    original_extras: dict[str, Any] | None = None,
) -> bool:
    """Graceful-degradation for an oversize sentinel ``note``.

    Triggered when ``task_workflow.post_event`` raises ``ValueError`` because
    ``note`` exceeds ``EVENT_NOTE_MAX`` (currently 50,000 chars). Without
    this fallback, ``_drain_sentinels`` would leave the sentinel un-renamed
    and every poll tick would re-post + re-fail the same oversize payload
    forever (incident 2026-06-04 task #477: a 52001-char
    ``epm:progress`` aggregate sentinel cycled indefinitely).

    Strategy:

    1. Write ``full_note`` to ``<task>/artifacts/sentinel-note-<kind_slug>-
       <epoch>.txt`` (task folder resolved via ``find_task_path``, so the
       branch-guarded ``main`` resolver picks the correct path even when
       the poller is invoked from elsewhere).
    2. Post a SHORT pointer marker of the same ``(kind, version)`` whose
       ``note`` (a) cites the artifact path, (b) records original length,
       and (c) is a leading excerpt of the original. The excerpt is
       hard-bounded under ``EVENT_NOTE_MAX`` so the pointer post itself
       cannot trip the same cap. ``artifacts=[<rel_path>]`` and
       ``oversize=True`` are carried as marker extras so the dashboard /
       downstream consumers can locate the full payload.

    Returns ``True`` on success (artifact written + pointer marker posted).
    Returns ``False`` (and logs) on any failure — caller must NOT rename
    the sentinel in that case so a future tick can retry. Carries through
    the original sentinel's ``gate`` / ``blocks_pipeline`` semantics by
    asking the caller to forward those via ``original_extras``.
    """
    try:
        task_dir = find_task_path(issue)
    except Exception as exc:
        log.error(
            "could not resolve task #%d for oversize-note persistence (sentinel %s, kind=%s): %s",
            issue,
            remote_path,
            kind,
            exc,
        )
        return False

    artifacts_dir = task_dir / "artifacts"
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.error(
            "could not create artifacts/ for task #%d (sentinel %s): %s",
            issue,
            remote_path,
            exc,
        )
        return False

    epoch = int(datetime.now(tz=UTC).timestamp())
    artifact_name = f"sentinel-note-{_slugify_kind(kind)}-{epoch}.txt"
    artifact_path = artifacts_dir / artifact_name
    try:
        artifact_path.write_text(full_note, encoding="utf-8")
    except OSError as exc:
        log.error(
            "could not write oversize-note artifact %s (sentinel %s): %s",
            artifact_path,
            remote_path,
            exc,
        )
        return False

    # Compute repo-relative artifact path for the marker. Falls back to the
    # absolute path if relative resolution fails (e.g. unusual mounts).
    try:
        rel_artifact = str(artifact_path.relative_to(task_dir.parents[2]))
    except ValueError:
        rel_artifact = str(artifact_path)

    # Build the pointer-marker note. It MUST fit under EVENT_NOTE_MAX. We
    # reserve ~512 chars for the pointer header and use the remainder for
    # a leading excerpt of the original, so operators see the start of the
    # payload inline without needing to open the artifact.
    header = (
        f"[oversize note persisted; original {len(full_note)} chars > "
        f"{EVENT_NOTE_MAX} cap]\n"
        f"Full payload: {rel_artifact}\n"
        f"Original kind={kind} version={version} by={by}\n"
        f"--- leading excerpt ---\n"
    )
    excerpt_budget = max(0, EVENT_NOTE_MAX - len(header) - 32)  # 32-byte safety
    excerpt = full_note[:excerpt_budget]
    pointer_note = header + excerpt
    # Belt-and-suspenders: hard truncate if any accounting drift would push
    # the pointer marker itself over the cap.
    if len(pointer_note) > EVENT_NOTE_MAX:
        pointer_note = pointer_note[:EVENT_NOTE_MAX]

    extras: dict[str, Any] = {"oversize": True, "oversize_orig_len": len(full_note)}
    if original_extras:
        # Forward operationally-meaningful sentinel fields (notably ``gate``
        # and ``blocks_pipeline``) so the pointer marker preserves the
        # semantics of the original.
        for key in ("gate", "blocks_pipeline"):
            if key in original_extras and original_extras[key] is not None:
                extras[key] = original_extras[key]

    try:
        post_event(
            issue,
            kind,
            version=version,
            by=by,
            note=pointer_note,
            artifacts=[rel_artifact],
            **extras,
        )
    except Exception as exc:
        log.error(
            "pointer-marker post failed for oversize sentinel %s (kind=%s): %s",
            remote_path,
            kind,
            exc,
        )
        return False

    log.warning(
        "sentinel %s carried %d-char note (> %d cap); persisted to %s and "
        "posted truncated pointer marker (kind=%s).",
        remote_path,
        len(full_note),
        EVENT_NOTE_MAX,
        rel_artifact,
        kind,
    )
    return True


def _parse_sentinel(remote_path: str, body: str) -> dict[str, Any] | None:
    """Decode + validate one sentinel body. Returns the dict on success.

    Returns None (and logs) for any of: empty body, JSON decode error,
    non-dict payload, missing required keys, unsupported schema version.
    The sentinel is left un-renamed in these cases so a future poller (or
    a human) can inspect it.
    """
    if not body:
        log.warning("sentinel %s is empty; skipping", remote_path)
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        log.warning("sentinel %s has invalid JSON (%s); skipping", remote_path, exc)
        return None
    if not isinstance(data, dict):
        log.warning("sentinel %s is not a JSON object; skipping", remote_path)
        return None
    missing = [k for k in _SENTINEL_REQUIRED_KEYS if k not in data]
    if missing:
        log.warning("sentinel %s missing required keys %s; skipping", remote_path, missing)
        return None
    schema_version = data.get("sentinel_schema_version")
    if schema_version != SENTINEL_SCHEMA_VERSION_SUPPORTED:
        log.warning(
            "sentinel %s has unsupported schema_version=%r (supported: %d); skipping",
            remote_path,
            schema_version,
            SENTINEL_SCHEMA_VERSION_SUPPORTED,
        )
        return None
    return data


def _drain_sentinels(*, issue: int, pod: str) -> tuple[int, str | None]:
    """Drain pod-side sentinels for this task; post markers from the VM.

    Returns ``(processed_count, gate_name_or_None)``. ``gate_name`` is the
    first non-empty ``gate`` field across processed sentinels (sentinels
    are processed in glob order, which is filename order, which is
    chronological by epoch-suffix). When set, the caller should stop the
    polling loop and surface the gate to the user.

    Each successfully-posted sentinel is renamed to ``<path>.processed``
    so the next tick won't re-post the same marker. If the marker post or
    the rename fails for an individual sentinel, the sentinel is left in
    place and a warning is logged; subsequent ticks will retry.

    Exception: an oversize-``note`` ``ValueError`` from ``post_event`` (note
    exceeds ``EVENT_NOTE_MAX``) is NOT a retryable failure — re-posting the
    same oversize payload next tick will fail identically, looping forever
    (incident 2026-06-04 task #477: a 52001-char ``epm:progress`` aggregate
    sentinel cycled indefinitely). It is degraded gracefully via
    ``_persist_oversize_note`` (full note -> ``<task>/artifacts/sentinel-
    note-*.txt`` + a truncated pointer marker of the same ``(kind, version)``
    that cites the artifact) and the sentinel is renamed ``.processed`` to
    end the loop. Any OTHER ``post_event`` exception (transient infra,
    schema bug, etc.) keeps the original retry-on-next-tick semantics.
    """
    sentinels = _ssh_drain_sentinels(pod, issue)
    processed = 0
    gate: str | None = None
    for remote_path, body in sentinels:
        data = _parse_sentinel(remote_path, body)
        if data is None:
            continue
        kind = data["kind"]
        version = int(data["version"])
        note = data.get("note")
        if note is None:
            note = data.get("payload")
        if note is not None and not isinstance(note, str):
            note = json.dumps(note, ensure_ascii=False)
        by = data.get("by") or "pod-sentinel"
        try:
            post_event(issue, kind, version=version, by=by, note=note)
        except ValueError as exc:
            # Oversize-note guard: match the EXACT message ``post_event``
            # raises (``"event note exceeds {N} chars (...)"``). Routing
            # any-old ``ValueError`` to graceful-degradation would
            # silently swallow real schema bugs, so the substring match
            # stays narrow.
            if _OVERSIZE_NOTE_ERROR_SUBSTR not in str(exc) or note is None:
                log.error(
                    "post_event failed for sentinel %s (kind=%s): %s",
                    remote_path,
                    kind,
                    exc,
                )
                continue
            if not _persist_oversize_note(
                issue=issue,
                remote_path=remote_path,
                kind=kind,
                version=version,
                by=by,
                full_note=note,
                original_extras=data,
            ):
                # Persistence / pointer-post failed — leave sentinel
                # un-renamed so the next tick can retry the whole path
                # (e.g. transient disk-write failure).
                continue
            # Pointer marker posted from the persisted artifact; fall
            # through to the rename + accounting block below so this
            # sentinel stops being re-attempted.
        except Exception as exc:
            # Don't rename on post failure — next tick will retry. We log
            # at error so an operator can see repeated failures.
            log.error(
                "post_event failed for sentinel %s (kind=%s): %s",
                remote_path,
                kind,
                exc,
            )
            continue
        if not _ssh_mark_processed(pod, remote_path):
            # Marker is posted but rename failed; on the next tick we'd
            # re-post and create a duplicate event. Surface loudly so the
            # operator can rename manually.
            log.error(
                "marker %s posted from sentinel %s but rename failed; "
                "future ticks may duplicate. Rename manually with: "
                "ssh %s mv %s %s.processed",
                kind,
                remote_path,
                pod,
                remote_path,
                remote_path,
            )
            # Still count as processed so the caller's accounting is honest.
        processed += 1
        sentinel_gate = data.get("gate")
        if gate is None and isinstance(sentinel_gate, str) and sentinel_gate:
            gate = sentinel_gate
    return processed, gate


def _latest_phase(log_tail: str) -> str:
    """Return the milestone name from the most recent `[phase=...]` line, or 'unknown'."""
    for line in reversed(log_tail.splitlines()):
        m = PHASE_RE.search(line)
        if m:
            return m.group(1)
    return "unknown"


# A GPU is considered idle when its `utilization.gpu` is at or below this
# percent. A real training / vLLM-generation workload reads >>5% on any
# GPU it is using (typically 80-100%); the threshold is a conservative
# floor that tolerates briefly-idle GPUs during inter-step bookkeeping
# without admitting a truly idle pod.
GPU_IDLE_UTIL_THRESHOLD = 5


def _gpu_idle(gpu_util: str) -> bool:
    """Return True iff every parsed GPU's utilization is <= IDLE threshold.

    Fail-safe: returns False (NOT idle) when ``gpu_util`` is the literal
    sentinel ``"unknown"``, is empty, or any token fails to parse as an
    int. The stall verdict requires ``gpu_idle == True``, so a missing /
    erroring ``nvidia-smi`` will NEVER by itself declare a healthy
    long-phase run stalled — the per-phase-log + cell-log mtime signals
    then carry the verdict.
    """
    if not gpu_util or gpu_util == "unknown":
        return False
    try:
        utils = [int(tok.strip()) for tok in gpu_util.split(",") if tok.strip()]
    except ValueError:
        return False
    if not utils:
        return False
    return all(u <= GPU_IDLE_UTIL_THRESHOLD for u in utils)


# ── GPU-idle advisory (incidents #518 + #537) ───────────────────────────────
#
# Minutes of sustained "healthy verdict + every GPU idle" before the poller
# posts a one-time, non-blocking [gpu-idle-advisory] epm:progress marker.
# ``0`` (or negative) disables the advisory entirely. Read at import time to
# mirror ``SSH_FAIL_REFRESH_THRESHOLD``; tests pass ``advisory_min``
# explicitly to the pure decision core instead of mutating the env.
GPU_IDLE_ADVISORY_MIN = int(os.environ.get("EPM_GPU_IDLE_ADVISORY_MIN", "30"))


@dataclass(frozen=True)
class GpuIdleAdvisoryUpdate:
    """Outcome of one advisory-counter tick (``_gpu_idle_advisory_update``)."""

    should_post: bool
    idle_since_epoch: int  # 0 = no active all-idle span
    idle_span_sec: int  # length of the current span; 0 when no span


def _gpu_idle_advisory_update(
    *,
    status: str,
    gpu_util: str,
    current_phase: str,
    prev_phase: str,
    prev_idle_since_epoch: int,
    advised_phases: set[str],
    now_epoch: int,
    advisory_min: int,
) -> GpuIdleAdvisoryUpdate:
    """Pure decision core for the GPU-idle advisory (incidents #518 + #537).

    Tracks the sustained span of "healthy verdict + every GPU idle" across
    poll ticks. The span RESETS (``idle_since_epoch`` -> 0) whenever the
    verdict is not ``running``, any GPU is busy, or the GPU sample is
    ``unknown`` / unparsable — the idle predicate is ``_gpu_idle`` itself
    (<= ``GPU_IDLE_UTIL_THRESHOLD``% on every card), so the stall verdict's
    fail-safe semantics carry over unchanged: a missing / erroring
    nvidia-smi never accumulates toward an advisory. A phase change
    RESTARTS the span at the current tick so each phase is judged on its
    own idle window.

    ``should_post`` is True only when the span has lasted at least
    ``advisory_min`` minutes AND ``current_phase`` is not already in
    ``advised_phases`` (at-most-once-per-phase de-dup). ``advisory_min <= 0``
    disables the advisory. Pure / no I/O — the caller owns state
    persistence and the marker post.
    """
    if advisory_min <= 0:
        return GpuIdleAdvisoryUpdate(should_post=False, idle_since_epoch=0, idle_span_sec=0)
    if status != "running" or not _gpu_idle(gpu_util):
        return GpuIdleAdvisoryUpdate(should_post=False, idle_since_epoch=0, idle_span_sec=0)
    if current_phase != prev_phase or prev_idle_since_epoch <= 0:
        idle_since = now_epoch
    else:
        idle_since = prev_idle_since_epoch
    span = max(0, now_epoch - idle_since)
    should_post = span >= advisory_min * 60 and current_phase not in advised_phases
    return GpuIdleAdvisoryUpdate(
        should_post=should_post, idle_since_epoch=idle_since, idle_span_sec=span
    )


def _maybe_post_gpu_idle_advisory(
    *,
    issue: int,
    pod: str,
    status: str,
    gpu_util: str,
    current_phase: str,
    prev_state: dict[str, str],
    now_epoch: int,
) -> tuple[int, set[str], bool]:
    """Advisory wiring for ``poll_once``: parse state, decide, maybe post.

    Returns ``(idle_since_epoch, advised_phases, posted)`` for the caller to
    persist via ``_save_state``. Posting rides the SAME ``epm:progress``
    marker channel as the phase-transition posts (note prefixed
    ``[gpu-idle-advisory]``, plus a ``gpu_idle_advisory=True`` extra for
    downstream consumers) — no new marker schema. A post failure is logged
    and the phase is NOT recorded as advised, so the next tick retries; the
    advisory never affects the status verdict and never stops anything.
    """
    try:
        prev_idle_since = int(prev_state.get("gpu_idle_since_epoch", "0"))
    except (TypeError, ValueError):
        prev_idle_since = 0
    advised_phases = {
        p for p in (prev_state.get("gpu_idle_advised_phases", "") or "").split(",") if p
    }
    update = _gpu_idle_advisory_update(
        status=status,
        gpu_util=gpu_util,
        current_phase=current_phase,
        prev_phase=prev_state.get("phase", ""),
        prev_idle_since_epoch=prev_idle_since,
        advised_phases=advised_phases,
        now_epoch=now_epoch,
        advisory_min=GPU_IDLE_ADVISORY_MIN,
    )
    if not update.should_post:
        return update.idle_since_epoch, advised_phases, False
    n_gpus = len([tok for tok in gpu_util.split(",") if tok.strip()])
    idle_min = update.idle_span_sec // 60
    note = (
        f"[gpu-idle-advisory] all {n_gpus} GPUs <= {GPU_IDLE_UTIL_THRESHOLD}% util for "
        f"{idle_min} min while the run is healthy (phase={current_phase}, "
        f"gpu_util={gpu_util}). Likely a CPU-only phase holding a GPU pod — consider "
        "moving the phase off-pod to the VM or stopping the pod after a checkpoint "
        "(CLAUDE.md: CPU-only phases don't hold GPU pods). Advisory only: the stall "
        "verdict is unchanged and nothing was stopped."
    )
    try:
        post_event(
            issue,
            "epm:progress",
            by="poll_pipeline",
            note=note,
            phase=current_phase,
            pod=pod,
            gpu_idle_advisory=True,
        )
    except Exception as exc:
        log.error("gpu-idle advisory post failed (next tick will retry): %s", exc)
        return update.idle_since_epoch, advised_phases, False
    log.warning(
        "posted gpu-idle advisory for #%d: all %d GPUs idle %d min during healthy phase=%s",
        issue,
        n_gpus,
        idle_min,
        current_phase,
    )
    advised_phases.add(current_phase)
    return update.idle_since_epoch, advised_phases, True


# Minimum cumulative CPU-seconds delta between consecutive ticks before
# declaring the launcher's process session "advancing". Set conservatively
# so a single accounting quantum or a brief sleep across ticks does not
# false-fire "advancing" on a truly hung session. A real CPU-bound phase
# accrues many seconds per minute of wall time across its process tree;
# even a half-second delta over a 9-minute poll interval is well above
# the noise floor of `ps` rounding.
SESSION_CPU_ADVANCE_EPSILON_SECS = 0.5


def _parse_session_cpu(value: str) -> float | None:
    """Parse a SESSION_CPU_SECS probe value to seconds, or None if unknown.

    The probe heredoc emits one of: a float like ``"4271.5"`` (success),
    ``"unknown"`` (pidfile missing, pid dead, ps unavailable, or ``ps``
    errored). Any other input (empty, malformed) is treated as unknown so
    the caller fails safe to "no signal" — never to "advancing".
    """
    if not value or value == "unknown":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _session_cpu_advancing(prev: str | None, current: str) -> bool | None:
    """Return True / False / None for the session-CPU "advancing" decision.

    * ``True``  — both samples parse AND current > prev + epsilon. The
      session is doing CPU work; a stalled-on-logs verdict should flip
      to running.
    * ``False`` — both samples parse AND current is at or below prev +
      epsilon. The session is truly idle; stalled stands.
    * ``None``  — at least one sample is unknown. NO signal; the caller
      preserves whatever the older log + GPU arbiters decided. This is
      the fail-safe path on (a) first tick after launch (no prior
      observation), (b) launcher dead (no session to probe — the
      pid-alive arbiter already routed to `dead`), or (c) `ps`
      unavailable.

    Returning None on first-tick prevents an immediate false-stalled →
    epm:failure cascade on a freshly-launched run; the next tick will
    have a prior observation and the decision flips to True / False.
    """
    cur = _parse_session_cpu(current)
    if cur is None:
        return None
    prv = _parse_session_cpu(prev) if prev is not None else None
    if prv is None:
        return None
    return cur > prv + SESSION_CPU_ADVANCE_EPSILON_SECS


def _load_state(state_file: Path, issue: int) -> dict[str, str]:
    if not state_file.exists():
        return {}
    try:
        data = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("state file %s unreadable; treating as empty", state_file)
        return {}
    return data.get(str(issue), {})


def _save_state(state_file: Path, issue: int, payload: dict[str, str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    all_state: dict[str, dict[str, str]] = {}
    if state_file.exists():
        try:
            all_state = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            all_state = {}
    all_state[str(issue)] = payload
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    tmp.write_text(json.dumps(all_state, indent=2, sort_keys=True))
    tmp.replace(state_file)


def poll_once(
    *,
    issue: int,
    pod: str,
    log_path: str,
    pid_file: str,
    state_file: Path,
    stall_sec: int = DEFAULT_STALL_SEC,
) -> PollResult:
    # Drain pod-side sentinels FIRST — posting any pending markers from the
    # VM. A user-gate sentinel (e.g. epm:fact-candidates) takes precedence
    # over the phase=done check so the orchestrator parks at the gate even
    # if the pipeline subsequently reached done.
    sentinels_processed, gate = _drain_sentinels(issue=issue, pod=pod)

    # Self-correction for a stale pidfile (incident #451): on a re-launch
    # the on-pod pidfile can hold the dead first-run PID while the live
    # python child runs under a new PID carried by the latest
    # epm:run-launched marker. Cross-check the marker pid so a healthy
    # re-run is not misreported as dead.
    marker_pid = _marker_pid(issue)
    probe = _ssh_probe(pod, log_path, pid_file, issue, marker_pid)

    # ── #488 stale-port self-heal ────────────────────────────────────────
    # Track consecutive SSH-probe failures across ticks. When the live API
    # has moved a pod's SSH endpoint to a new port but ``pods.conf`` still
    # holds the pre-stop value, every probe lands on a dead address and
    # this counter accumulates. Once it crosses
    # ``SSH_FAIL_REFRESH_THRESHOLD`` we shell out to ``pod.py config
    # --refresh-from-api <pod>`` once (fail-soft) to pull the current
    # host/port from the live API into ``pods.conf`` + ``~/.ssh/config``,
    # then reset the counter so the NEXT N consecutive failures will
    # retry. This is the auto-heal that closes the gap left by the
    # #488 manual recovery (the new ``--refresh-from-api`` subcommand
    # already exists; this is the wiring that uses it without a human in
    # the loop).
    prev_state = _load_state(state_file, issue)
    prev_ssh_fail_count_raw = prev_state.get("ssh_fail_count", "0")
    try:
        prev_ssh_fail_count = int(prev_ssh_fail_count_raw)
    except (TypeError, ValueError):
        prev_ssh_fail_count = 0
    ssh_failed = probe.get("ssh_failed") == "1"
    if ssh_failed:
        ssh_fail_count = prev_ssh_fail_count + 1
        if ssh_fail_count >= SSH_FAIL_REFRESH_THRESHOLD:
            log.warning(
                "SSH probe failed %d consecutive ticks for pod %s; "
                "firing pod.py config --refresh-from-api %s "
                "(#488 stale-port auto-heal)",
                ssh_fail_count,
                pod,
                pod,
            )
            _try_refresh_pods_conf_from_api(pod)
            # Reset after the attempt regardless of outcome so we don't
            # hot-loop refresh calls every tick.
            ssh_fail_count = 0
    else:
        ssh_fail_count = 0

    pidfile_pid_alive = probe["pid_alive"] == "1"
    marker_pid_alive = marker_pid is not None and probe["marker_pid_alive"] == "1"
    pid_alive = pidfile_pid_alive or marker_pid_alive
    # Observability for the #521 false-dead diagnosis: surface "the pid
    # FILE was absent" (vs "the pid probed dead") in the tick JSON, and
    # warn when the epm:run-launched marker pid is the fallback standing
    # in for it. Status routing is deliberately untouched — ``pid_alive``
    # already ORs in the marker pid.
    pid_file_missing = probe.get("pid_file_missing") == "1"
    if pid_file_missing and marker_pid is not None:
        log.warning(
            "pid file %s absent on pod %s; using epm:run-launched marker pid %d fallback",
            pid_file,
            pod,
            marker_pid,
        )
    mtime_epoch = int(probe["mtime_epoch"] or "0")
    cell_mtime_epoch = int(probe["cell_mtime_epoch"] or "0")
    phase_log_mtime_epoch = int(probe["phase_log_mtime_epoch"] or "0")
    shard_log_mtime_epoch = int(probe.get("shard_log_mtime_epoch") or "0")
    # Staleness folds in the newest cell log (#405). A sequential smoke
    # cell blocks the dispatcher in `proc.wait()` for ~15-18 min while the
    # cell process actively trains+evals and writes to its own log; the
    # main log goes silent for that window. Take the freshest of (main,
    # newest cell) so a healthy single-cell phase reads as running, not
    # false-stalled / false-dead. Phase detection stays on the MAIN log
    # because the `[phase=...]` line is written by the dispatcher (cells
    # log training steps, not phase transitions).
    freshest_mtime_epoch = max(mtime_epoch, cell_mtime_epoch)
    now_epoch = int(datetime.now(tz=UTC).timestamp())
    last_mtime_ago = now_epoch - freshest_mtime_epoch if freshest_mtime_epoch > 0 else 10**9
    phase_log_mtime_ago = now_epoch - phase_log_mtime_epoch if phase_log_mtime_epoch > 0 else 10**9
    shard_log_mtime_ago = now_epoch - shard_log_mtime_epoch if shard_log_mtime_epoch > 0 else 10**9
    gpu_util = probe.get("gpu_util", "unknown")
    gpu_idle = _gpu_idle(gpu_util)
    current_phase = _latest_phase(probe["log_tail"])

    # Decide status. Gate sentinel wins over done — a user must answer
    # before the pipeline (or the orchestrator) advances further. The
    # phase=done check still runs (we want to know the pipeline finished)
    # but ``status`` reflects the gate so the orchestrator parks.
    # `dead` requires BOTH the pidfile PID and the marker PID to be dead
    # (pid_alive is their OR) AND the log not to show completion — a stale
    # pidfile alone never declares a live marker-PID run dead. The
    # `current_phase == "done"` precedence already covers the
    # "log-shows-completion" half: a completed run is `done`, never `dead`.
    #
    # `stalled` requires ALL FIVE liveness-of-output signals to agree:
    # the top-level log AND the freshest cell log (folded together as
    # `last_mtime_ago`, #405) AND every per-phase log under
    # `/workspace/logs/issue-<N>-*.log` (#468) AND every shard /
    # repo-rooted phase log under `/workspace/explore-persona-space/
    # logs/issue_<N>{,_*}/*.log` (#488) plus every dispatcher per-job
    # log under `/workspace/explore-persona-space/eval_results/
    # issue_<N>{,_*}/logs/*.log` (#521, folded into the same shard-log
    # max) AND the GPUs must ALL be quiet/idle for >STALL_SEC. The
    # shard-log conjunction (#488)
    # prevents a false stall when a multi-GPU launcher fans out per-GPU
    # shard logs under a subdirectory and the inner loop's per-shard
    # write cadence (e.g. ~3 min between writes for i488 Pass B across
    # 57 cells) exceeds the 30-min threshold on the main log alone —
    # in that pattern the main + cell + per-phase logs all go silent
    # while the shard logs are actively appended. `_gpu_idle` remains
    # fail-safe (returns False on nvidia-smi error / unknown), so a
    # healthy long phase whose shard log OR per-phase log is fresh OR
    # whose GPU is busy will stay in `running` even if nvidia-smi is
    # unavailable.
    # Session-CPU advancing check (#518): even when every log-mtime
    # signal AND the GPU-idle signal agree on "stalled", a launcher
    # whose process session has accrued more cumulative CPU since the
    # previous tick is doing CPU-bound work (e.g. the scoring_syco
    # phase that polls a judge batch and aggregates results — silent
    # on logs for hours, GPUs idle by design, but the python child is
    # at 100% CPU). Override `stalled` -> `running` when CPU is
    # advancing; preserve `stalled` when CPU is flat or unknown
    # (fail-safe). The very first tick after launch has no prior
    # observation, so `_session_cpu_advancing` returns None and the
    # decision falls back to the older log+GPU arbiters: a freshly-
    # launched run cannot meet the >stall_sec mtime conjunction on
    # the first tick (the logs ARE fresh), so this code path doesn't
    # change first-tick semantics. From the second tick onward, a
    # truly hung session (CPU flat AND logs stale AND GPUs idle)
    # still routes to `stalled` and the orchestrator still fires
    # epm:failure.
    current_session_cpu = probe.get("session_cpu_secs", "unknown")
    prev_session_cpu = prev_state.get("session_cpu_secs")
    cpu_advancing = _session_cpu_advancing(prev_session_cpu, current_session_cpu)

    if gate is not None:
        status = "gate"
    elif current_phase == "done":
        status = "done"
    elif not pid_alive:
        status = "dead"
    elif (
        last_mtime_ago > stall_sec
        and phase_log_mtime_ago > stall_sec
        and shard_log_mtime_ago > stall_sec
        and gpu_idle
    ):
        if cpu_advancing is True:
            log.info(
                "stall conjunction met (logs >%ds + GPUs idle) BUT session CPU "
                "advanced %s -> %s on pod %s (#518 silent CPU-bound override); "
                "reporting status=running",
                stall_sec,
                prev_session_cpu,
                current_session_cpu,
                pod,
            )
            status = "running"
        else:
            status = "stalled"
    else:
        status = "running"

    # ── #518/#537 GPU-idle advisory ──────────────────────────────────────
    # The stall verdict above treats an idle GPU only as corroboration, so
    # a HEALTHY run on a long CPU-only phase (fresh logs, every GPU at 0%)
    # burns pod-hours silently. Track the sustained healthy-and-all-idle
    # span across ticks (state-file backed, like ssh_fail_count) and post
    # a one-per-phase, non-blocking advisory marker once it exceeds
    # GPU_IDLE_ADVISORY_MIN minutes. Never flips ``status``.
    gpu_idle_since_epoch, gpu_idle_advised_phases, gpu_idle_advisory_posted = (
        _maybe_post_gpu_idle_advisory(
            issue=issue,
            pod=pod,
            status=status,
            gpu_util=gpu_util,
            current_phase=current_phase,
            prev_state=prev_state,
            now_epoch=now_epoch,
        )
    )

    # New milestone? (re-uses ``prev_state`` loaded above for the
    # ssh_fail_count tracking — we only read state once per tick.)
    prev_phase = prev_state.get("phase", "")
    new_milestone = current_phase != prev_phase and current_phase != "unknown"

    if new_milestone:
        try:
            post_event(
                issue,
                "epm:progress",
                by="poll_pipeline",
                note=f"phase transition: {prev_phase or '(start)'} -> {current_phase}",
                phase=current_phase,
                pod=pod,
            )
        except Exception as exc:
            log.error("post_event failed: %s", exc)
            new_milestone = False  # Don't claim we recorded it.

    _save_state(
        state_file,
        issue,
        {
            "phase": current_phase,
            "last_mtime_epoch": str(mtime_epoch),
            "ssh_fail_count": str(ssh_fail_count),
            # GPU-idle advisory span + per-phase de-dup (#518/#537). Phase
            # names match PHASE_RE ([a-z0-9_]+) so the comma join is safe.
            "gpu_idle_since_epoch": str(gpu_idle_since_epoch),
            "gpu_idle_advised_phases": ",".join(sorted(gpu_idle_advised_phases)),
            # Persist the current CPU sample so the NEXT tick can compute
            # the advancing delta. Stored as the literal probe string
            # (``"unknown"`` or a float-as-string) so `_parse_session_cpu`
            # treats it consistently with the live probe value.
            "session_cpu_secs": current_session_cpu,
        },
    )

    # Pick the tail excerpt from whichever log is the fresher signal: if
    # cell logs exist AND are fresher than the main log, surface the cell
    # tail so operators see what's actually happening (training-step
    # output, eval progress) rather than the stale dispatcher tail. When
    # both are zero (no logs yet) or the main log is fresher, fall back
    # to the main-log tail (preserves prior behavior for non-cell runs).
    if cell_mtime_epoch > 0 and cell_mtime_epoch > mtime_epoch:
        tail_excerpt = "\n".join(probe["cell_log_tail"].splitlines()[-5:])
    else:
        tail_excerpt = "\n".join(probe["log_tail"].splitlines()[-5:])
    return PollResult(
        status=status,
        current_phase=current_phase,
        new_milestone=new_milestone,
        last_log_mtime_sec_ago=min(last_mtime_ago, 10**9),
        pid_alive=pid_alive,
        pid_file_missing=pid_file_missing,
        log_tail_excerpt=tail_excerpt,
        gate=gate,
        sentinels_processed=sentinels_processed,
        phase_log_mtime_sec_ago=min(phase_log_mtime_ago, 10**9),
        shard_log_mtime_sec_ago=min(shard_log_mtime_ago, 10**9),
        gpu_util=gpu_util,
        gpu_idle_advisory_posted=gpu_idle_advisory_posted,
        session_cpu_secs=current_session_cpu,
        cpu_advancing=cpu_advancing,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--issue", type=int, required=True, help="Task / issue number.")
    parser.add_argument("--pod", required=True, help="SSH host alias (e.g. epm-issue-137).")
    parser.add_argument("--log", required=True, help="Remote log file path.")
    parser.add_argument("--pid-file", required=True, help="Remote PID file path.")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Local cache JSON (default: .claude/cache/poll-pipeline-<N>.json).",
    )
    parser.add_argument(
        "--stall-sec",
        type=int,
        default=int(os.environ.get("EPM_POLL_STALL_SEC", DEFAULT_STALL_SEC)),
        help=(
            "Seconds of log-mtime silence before declaring the run stalled "
            f"(default {DEFAULT_STALL_SEC}). Raise for workloads with sparse "
            "log cadence (e.g. checkpoint-cadence-only logging at >15min "
            "intervals). Falls back to the EPM_POLL_STALL_SEC env var when "
            "the flag is not set."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Log to stderr at DEBUG level.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    state_file = args.state_file or (DEFAULT_STATE_DIR / f"poll-pipeline-{args.issue}.json")

    result = poll_once(
        issue=args.issue,
        pod=args.pod,
        log_path=args.log,
        pid_file=args.pid_file,
        state_file=state_file,
        stall_sec=args.stall_sec,
    )

    print(
        json.dumps(
            {
                "status": result.status,
                "current_phase": result.current_phase,
                "new_milestone": result.new_milestone,
                "last_log_mtime_sec_ago": result.last_log_mtime_sec_ago,
                "pid_alive": result.pid_alive,
                "pid_file_missing": result.pid_file_missing,
                "log_tail_excerpt": result.log_tail_excerpt,
                "gate": result.gate,
                "sentinels_processed": result.sentinels_processed,
                "phase_log_mtime_sec_ago": result.phase_log_mtime_sec_ago,
                "shard_log_mtime_sec_ago": result.shard_log_mtime_sec_ago,
                "gpu_util": result.gpu_util,
                "gpu_idle_advisory_posted": result.gpu_idle_advisory_posted,
                "session_cpu_secs": result.session_cpu_secs,
                "cpu_advancing": result.cpu_advancing,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
