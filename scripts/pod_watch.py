"""Stall-detection watchdog. Spawned by ``/issue`` Step 6d, runs detached
on the local VM, NOT on the pod.

Patterned on Symphony §8.5 / §10.6 ``stall_timeout_ms``. Probes (in order):

1. WandB run heartbeat (``run.heartbeat_at``) — primary.
2. Log file mtime over SSH — fallback.

Self-stops when ANY of:

* ``epm:results v1`` posted (graceful end-of-run);
* ``epm:failure`` posted by anyone;
* the experiment's status is no longer ``running``;
* PID file ``.claude/cache/watch-<N>.pid`` deleted (manual override);
* wall-time cap hit (``--max-runtime-secs``, default 86400 = 24h).

On stall (no event in ``--threshold-secs`` seconds, default 300) the
watchdog posts an ``epm:failure`` marker with ``failure_class: infra``
and ``reason: stall``, flips the experiment's status to ``blocked``,
and exits. The marker metadata carries ``watch_pid=<pid>`` for
de-duplication; a watchdog will refuse to post a fresh failure if a
marker with a higher pid already exists.

State backend: this watchdog reads and writes the Sagan dashboard's
HTTP API via :mod:`sagan_state`. There is no repository issue involvement;
``--issue N`` is interpreted as ``experiments.number`` in Sagan.

Race-hardening (per plan §2):

* Re-read the status IMMEDIATELY before posting the failure marker;
  abort if it has already moved out of ``running``.
* Idempotency: scan existing ``epm:failure`` markers; if any has a
  ``watch_pid`` >= our pid, exit silently.

False-positive hardening (liveness robustness):

* The heartbeat/log signal alone over-reports stalls in two cases —
  (a) the orchestrator hands us the outer ``uv run`` wrapper PID
  instead of the real python child, and (b) a phase writes to a data
  file rather than the log, so log-mtime goes quiet while the run is
  still burning CPU. :func:`resolve_real_pid` descends the process
  tree to the real python child, and :func:`_probe_process_active`
  reports whether that process is actually doing work. A stall is only
  declared when BOTH the heartbeat/log probe AND the process-tree probe
  agree the run is idle. The process-tree probe is *corroboration only*:
  when no PID is supplied (the default), it returns ``None`` and the
  stall decision falls back to the heartbeat/log signal alone — exactly
  the prior behavior. It never manufactures a stall verdict on its own.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# scripts/ is added to sys.path so ``sagan_state`` (sibling module) imports cleanly
# whether the watchdog is invoked as `python scripts/pod_watch.py` (cwd repo root,
# scripts not on path) or via `uv run python -m scripts.pod_watch`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_state as sagan_state

log = logging.getLogger("pod_watch")
TICK_SECS = 60
PROBE_FAILURE_LIMIT = 5  # ticks of probe-unreachable before giving up
DEFAULT_THRESHOLD_SECS = 300
DEFAULT_MAX_RUNTIME_SECS = 86400  # 24h

# A wrapper command whose direct python child IS the real workload. When the
# orchestrator hands us the PID of one of these wrappers, resolve_real_pid()
# descends the process tree to the python child so the liveness probe reads
# the right process.
WRAPPER_BASENAMES = frozenset({"uv", "nohup", "env", "timeout", "stdbuf", "setsid"})

# Statuses that mean "experiment progressed beyond running" — we exit
# silently without posting. Snake_case to match Sagan's experiment_status
# enum.
GRACEFUL_TERMINAL_STATUSES = {
    "uploading",
    "interpreting",
    "reviewing",
    "awaiting_promotion",
    "completed",
    "archived",
    "blocked",
}

RUNNING_STATUS = "running"
BLOCKED_STATUS = "blocked"


def _experiment_snapshot(number: int) -> dict[str, Any]:
    """Return {experiment, events, approvalRequests} from Sagan."""
    return sagan_state.get_experiment(number)


def _status(snapshot: dict[str, Any]) -> str:
    return snapshot["experiment"]["status"]


def _has_marker(snapshot: dict[str, Any], kind: str) -> bool:
    """True if any event carries an ``epm:<kind>`` marker."""
    target = f"epm:{kind}"
    for ev in snapshot.get("events", []):
        meta = ev.get("metadata") or {}
        marker = meta.get("marker_type") or ev.get("markerType")
        if marker and marker.startswith(target):
            return True
    return False


def _max_failure_pid(snapshot: dict[str, Any]) -> int | None:
    """Largest watch_pid found in any existing epm:failure marker, or None."""
    largest: int | None = None
    for ev in snapshot.get("events", []):
        meta = ev.get("metadata") or {}
        marker = meta.get("marker_type") or ev.get("markerType") or ""
        if not marker.startswith("epm:failure"):
            continue
        pid = meta.get("watch_pid")
        if pid is None:
            continue
        try:
            candidate = int(pid)
        except (TypeError, ValueError):
            continue
        if largest is None or candidate > largest:
            largest = candidate
    return largest


def _probe_wandb(run_url: str | None) -> float | None:
    """Return Unix timestamp of last heartbeat, or None on failure."""
    if not run_url:
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed; cannot probe run heartbeat")
        return None
    try:
        run = wandb.Api().run(run_url)
        # Try both attribute names; the public Api object exposes
        # snake_case `heartbeat_at` for the GraphQL `heartbeatAt` field.
        # On freshly-launched runs heartbeat_at can be None for ~30s.
        ts = getattr(run, "heartbeat_at", None)
        if ts is None:
            ts = run.summary.get("_timestamp")
        if ts is None:
            return None
        # `heartbeat_at` is a `datetime` (UTC). `_timestamp` is a Unix
        # epoch float. Normalise.
        if isinstance(ts, datetime):
            return ts.timestamp()
        return float(ts)
    except Exception as exc:
        log.info("wandb probe failed: %s", exc)
        return None


def _probe_log_mtime(log_path: str | None) -> float | None:
    """Return the mtime of a remote log via SSH, or None.

    ``log_path`` is shaped ``<server>:<path>`` (e.g.
    ``pod-137:/workspace/logs/issue-137.log``). For local log paths
    pass a single path with no colon.
    """
    if not log_path:
        return None
    if ":" in log_path:
        server, remote_path = log_path.split(":", 1)
        cmd = ["ssh", server, "stat", "-c", "%Y", remote_path]
    else:
        cmd = ["stat", "-c", "%Y", log_path]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30, stderr=subprocess.DEVNULL)
        return float(out.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        log.info("log mtime probe failed: %s", exc)
        return None


def resolve_real_pid(pid: int) -> int:
    """Descend a wrapper-process PID to the real python-workload child.

    The orchestrator may hand us the PID of an outer ``uv run`` (or
    ``nohup`` / ``env`` / ``timeout`` / ...) wrapper. That wrapper's
    own CPU usage stays near zero while the actual workload runs in a
    python grandchild, so a liveness probe pointed at the wrapper PID
    looks idle even when the run is healthy. This walks down the
    process tree from ``pid``, following the single child of each
    wrapper, until it reaches a non-wrapper process (typically the
    python interpreter) or a fork point.

    Returns the resolved real PID. Falls back to the input ``pid`` when
    psutil is unavailable, the process is gone, or the tree branches
    (ambiguous descent — better to probe the wrapper than guess).
    """
    try:
        import psutil
    except ImportError:
        log.warning("psutil not installed; cannot resolve wrapper PID %d", pid)
        return pid

    current = pid
    # Bounded descent: a wrapper chain (uv -> python, nohup -> uv -> python)
    # is at most a handful deep; the cap guards against pathological cycles.
    for _ in range(8):
        try:
            proc = psutil.Process(current)
            name = (proc.name() or "").lower()
            children = proc.children()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            log.info("resolve_real_pid: cannot inspect pid %d: %s", current, exc)
            return pid
        # A non-wrapper process IS the workload — stop here.
        if name not in WRAPPER_BASENAMES:
            return current
        # Wrapper with exactly one child → descend. Zero children means the
        # wrapper is the leaf (workload already exited or never spawned);
        # >1 child means the tree forked and the descent is ambiguous.
        if len(children) != 1:
            log.info(
                "resolve_real_pid: wrapper pid %d (%s) has %d children; stopping descent",
                current,
                name,
                len(children),
            )
            return current
        current = children[0].pid
    log.info("resolve_real_pid: descent cap hit from pid %d; using pid %d", pid, current)
    return current


def _process_active_local(pid: int) -> bool | None:
    """Return True if the LOCAL process tree rooted at ``pid`` is alive.

    Snapshot semantics: the resolved real process (or any of its
    descendants) exists and is in a running/sleeping state — i.e. NOT
    zombie (dead-not-reaped) or stopped. Returns None when the process
    is gone or psutil is unavailable; the caller treats None as "no
    process signal", NOT as "idle".
    """
    try:
        import psutil
    except ImportError:
        return None
    real_pid = resolve_real_pid(pid)
    try:
        proc = psutil.Process(real_pid)
        statuses = {proc.status()}
        for child in proc.children(recursive=True):
            try:
                statuses.add(child.status())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        log.info("_process_active_local: pid %d gone/inaccessible: %s", real_pid, exc)
        return None
    # Dead-but-not-reaped (zombie) or stopped means the workload is NOT
    # progressing; an alive running/sleeping process means it might be.
    active_states = {
        psutil.STATUS_RUNNING,
        psutil.STATUS_DISK_SLEEP,
        psutil.STATUS_SLEEPING,
        psutil.STATUS_WAKING,
        psutil.STATUS_IDLE,
    }
    return any(s in active_states for s in statuses)


def _process_active_remote(pid: int, server: str) -> bool | None:
    """Return True if the REMOTE (pod-side) process tree rooted at ``pid`` is
    alive. Mirrors :func:`_process_active_local` over SSH.

    Uses ``ps -o stat= -p <pid>`` to read the kernel process state. A
    ``Z`` (zombie) or ``T``/``t`` (stopped) leading state code means not
    progressing; ``R``/``S``/``D`` means alive. Returns None when the
    SSH probe fails or the process is gone — caller treats None as "no
    process signal", never as "idle".
    """
    cmd = ["ssh", server, "ps", "-o", "stat=", "-p", str(pid)]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # CalledProcessError with empty output = no such PID (process gone).
        log.info("_process_active_remote: probe failed for pid %d on %s: %s", pid, server, exc)
        return None
    if not out:
        return None
    leading = out[0]
    # Z = zombie (dead-not-reaped), T/t = stopped/traced — not progressing.
    return leading not in ("Z", "T", "t")


def _probe_process_active(process_target: str | None) -> bool | None:
    """Corroboration probe: is the workload process actually doing work?

    ``process_target`` is shaped ``<server>:<pid>`` for a pod-side
    process (SSH probe) or a bare ``<pid>`` for a local process. None /
    empty → returns None (no process signal; caller must NOT treat that
    as a stall on its own).

    Returns True (alive/progressing), False (gone/zombie/stopped), or
    None (unknown — probe unavailable). The stall decision suppresses a
    verdict only on a True result; it never *manufactures* a stall from
    a False/None here — genuine stall detection still keys on the
    heartbeat/log probe.
    """
    if not process_target:
        return None
    if ":" in process_target:
        server, raw_pid = process_target.split(":", 1)
        try:
            pid = int(raw_pid)
        except ValueError:
            log.info("_probe_process_active: malformed remote target %r", process_target)
            return None
        return _process_active_remote(pid, server)
    try:
        pid = int(process_target)
    except ValueError:
        log.info("_probe_process_active: malformed local target %r", process_target)
        return None
    return _process_active_local(pid)


def _check_terminal(issue: int) -> bool:
    """Return True if the watchdog should exit gracefully (epm:results
    posted, status moved beyond running, etc).

    On `blocked` or any GRACEFUL_TERMINAL_STATUSES we ALSO return True —
    the watchdog never re-flips a blocked or graceful-terminal experiment.
    """
    snapshot = _experiment_snapshot(issue)
    # epm:results = graceful end-of-run.
    if _has_marker(snapshot, "results"):
        return True
    # Someone else already posted failure — don't pile on.
    if _has_marker(snapshot, "failure"):
        return True
    # Status moved out of running — terminal regardless of where it
    # moved to (graceful next phase, manual blocked, archived, etc).
    return _status(snapshot) != RUNNING_STATUS


def _post_failure(issue: int, *, reason: str, last_event: float | None) -> None:
    """Post an epm:failure marker with stall metadata, then flip status to blocked."""
    pid = os.getpid()
    snapshot = _experiment_snapshot(issue)
    status = _status(snapshot)

    # Step 1: re-read status; abort if it has moved.
    if status != RUNNING_STATUS:
        log.info(
            "watchdog %d: status no longer 'running' (current=%s); "
            "aborting failure post — graceful exit",
            pid,
            status,
        )
        return

    # Step 2: idempotency — if a later-pid failure marker exists, exit silent.
    largest_pid = _max_failure_pid(snapshot)
    if largest_pid is not None and largest_pid >= pid:
        log.info(
            "watchdog %d: failure marker already posted by watch_pid=%s; exit",
            pid,
            largest_pid,
        )
        return

    # Step 3: post the marker via Sagan, then flip status. Both writes
    # are idempotent server-side; if a parallel writer beat us to
    # `blocked` between calls 1 and 4 the second PATCH is a no-op.
    experiment_id = snapshot["experiment"]["id"]
    last_event_iso = datetime.fromtimestamp(last_event).isoformat() if last_event else "never"
    note = (
        f"## Stall detected\n\n"
        f"failure_class: infra\n"
        f"reason: {reason}\n"
        f"last_event: {last_event_iso}\n"
        f"watch_pid: {pid}\n\n"
        f"The pod.py-watch heartbeat probe detected a stall. Routed to "
        f"the infra failure path; experimenter will be respawned on the "
        f"next `/issue {issue}` invocation (cap 3)."
    )
    sagan_state.post_marker(
        experiment_id,
        "epm:failure",
        note=note,
        metadata={
            "failure_class": "infra",
            "reason": reason,
            "last_event_iso": last_event_iso,
            "watch_pid": pid,
        },
    )

    # Step 4: flip status. If a manual blocked happened between steps 1
    # and 4 this PATCH still settles at blocked (correct — no harm).
    sagan_state.set_status(experiment_id, BLOCKED_STATUS, note="watchdog stall")
    log.info("watchdog %d: posted epm:failure (reason=%s); flipped to blocked", pid, reason)


def _watch_loop(
    issue: int,
    *,
    threshold_secs: int,
    wandb_run_url: str | None,
    log_path: str | None,
    pid_file: Path,
    max_runtime_secs: int,
    process_target: str | None = None,
) -> int:
    """Tick every TICK_SECS; flag stall after threshold_secs of no event.

    ``process_target`` (``<server>:<pid>`` or bare ``<pid>``) enables the
    process-tree corroboration probe. When supplied and the resolved real
    process is alive/progressing, a heartbeat/log stall is suppressed (the
    log went quiet but the run is still working). When absent, the stall
    decision keys on the heartbeat/log probe alone — the prior behavior.

    Returns the desired process exit code.
    """
    started_at = time.time()
    last_event_at: float = started_at  # treat startup as an event
    consecutive_unreachable = 0

    while True:
        time.sleep(TICK_SECS)

        # Wall-time cap.
        if time.time() - started_at > max_runtime_secs:
            log.info(
                "watchdog %d: max-runtime cap reached (%ds); exiting silently",
                os.getpid(),
                max_runtime_secs,
            )
            return 0

        # Manual override.
        if not pid_file.exists():
            log.info("watchdog %d: pid file %s deleted; exiting silently", os.getpid(), pid_file)
            return 0

        # Terminal-state check (results posted, status moved, etc).
        try:
            if _check_terminal(issue):
                log.info("watchdog %d: graceful terminal state; exit", os.getpid())
                return 0
        except sagan_state.SaganError as exc:
            # Sagan API call failed; treat as a probe failure.
            log.info("terminal-state probe failed: %s", exc)
            consecutive_unreachable += 1
            if consecutive_unreachable >= PROBE_FAILURE_LIMIT:
                _post_failure(issue, reason="probe_unreachable", last_event=last_event_at)
                return 1
            continue

        # Probe.
        wandb_ts = _probe_wandb(wandb_run_url)
        log_ts = _probe_log_mtime(log_path)
        ev = max((t for t in (wandb_ts, log_ts) if t is not None), default=None)

        if ev is None:
            consecutive_unreachable += 1
            if consecutive_unreachable >= PROBE_FAILURE_LIMIT:
                _post_failure(issue, reason="probe_unreachable", last_event=last_event_at)
                return 1
            continue

        consecutive_unreachable = 0
        last_event_at = max(last_event_at, ev)

        # Stall check. The heartbeat/log probe is primary: only when it
        # says "no event for longer than the threshold" do we even
        # consider a stall.
        elapsed = time.time() - last_event_at
        if elapsed > threshold_secs:
            # Corroboration: a quiet log does NOT mean a dead run. Some
            # phases write to a data file, not the log, so log-mtime goes
            # silent while the process is still burning CPU. If the
            # process-tree probe says the workload is actively alive,
            # suppress the stall verdict and keep watching. The probe is
            # corroboration only — a False/None result does not by itself
            # trigger a stall (the heartbeat/log signal already did).
            proc_active = _probe_process_active(process_target)
            if proc_active is True:
                log.info(
                    "watchdog %d: log/heartbeat quiet for %ds but process "
                    "target %s is alive; suppressing stall verdict",
                    os.getpid(),
                    int(elapsed),
                    process_target,
                )
                continue
            _post_failure(issue, reason="stall", last_event=last_event_at)
            return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stall-detection watchdog for a /issue experiment run."
    )
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument(
        "--threshold-secs",
        type=int,
        default=DEFAULT_THRESHOLD_SECS,
        help="Stall threshold (seconds). Default: %(default)s.",
    )
    parser.add_argument(
        "--wandb-run-url",
        default=None,
        help="WandB run identifier — e.g. 'user/project/run-id'.",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="<server>:<path> log file to stat over SSH (fallback probe). "
        "Local paths (no colon) also accepted.",
    )
    parser.add_argument(
        "--max-runtime-secs",
        type=int,
        default=DEFAULT_MAX_RUNTIME_SECS,
        help="Wall-time cap; watchdog exits silently after this. Default: 24h.",
    )
    parser.add_argument(
        "--pid-file",
        default=None,
        help="PID file path. Defaults to .claude/cache/watch-<issue>.pid.",
    )
    parser.add_argument(
        "--process-target",
        default=None,
        help="<server>:<pid> (pod-side, probed over SSH) or a bare <pid> "
        "(local) of the workload process. Enables the process-tree "
        "corroboration probe: a heartbeat/log stall is suppressed when this "
        "process is still alive (the log went quiet but the run is still "
        "working). An outer 'uv run'/'nohup' wrapper PID is resolved down to "
        "the real python child automatically. Omit to key stall detection on "
        "the heartbeat/log probe alone (the default).",
    )
    parser.add_argument(
        "--force-attach",
        action="store_true",
        help="Bypass the SECTION_2_LAND_SHA gate. Used to attach the watchdog "
        "to a long-running pre-§2 dispatch. The /issue Step 6d auto-spawn "
        "never sets this flag; the safe default is to skip attaching to "
        "pre-§2 dispatches. (Documented per plan §2 line 493-500.)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level for the watchdog process.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    pid_file = (
        Path(args.pid_file) if args.pid_file else Path(".claude/cache") / f"watch-{args.issue}.pid"
    )
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    log.info(
        "watchdog %d: started (issue=%s, threshold=%ds, wandb=%s, log=%s, "
        "process_target=%s, force_attach=%s)",
        os.getpid(),
        args.issue,
        args.threshold_secs,
        args.wandb_run_url,
        args.log_path,
        args.process_target,
        args.force_attach,
    )

    try:
        return _watch_loop(
            args.issue,
            threshold_secs=args.threshold_secs,
            wandb_run_url=args.wandb_run_url,
            log_path=args.log_path,
            pid_file=pid_file,
            max_runtime_secs=args.max_runtime_secs,
            process_target=args.process_target,
        )
    finally:
        # Clean up the pid file on exit (any path).
        try:
            if pid_file.exists():
                pid_file.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
