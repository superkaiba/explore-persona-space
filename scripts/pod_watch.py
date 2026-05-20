"""Stall-detection watchdog. Spawned by ``/issue`` Step 6d, runs detached
on the local VM, NOT on the pod.

Patterned on Symphony §8.5 / §10.6 ``stall_timeout_ms``. Probes (in order):

1. WandB run heartbeat (``run.heartbeat_at``) — primary.
2. Log file mtime over SSH — fallback.

Self-stops when ANY of:

* ``epm:results v1`` posted (graceful end-of-run);
* ``epm:failure`` posted by anyone;
* the issue's status label is no longer ``status:running``;
* PID file ``.claude/cache/watch-<N>.pid`` deleted (manual override);
* wall-time cap hit (``--max-runtime-secs``, default 86400 = 24h).

On stall (no event in ``--threshold-secs`` seconds, default 300) the
watchdog posts an ``epm:failure`` marker with ``failure_class: infra`` and
``reason: stall``, flips the label to ``status:blocked``, and exits. The
Marker title carries ``watch-pid=<pid>`` for de-duplication; a watchdog
will refuse to post a fresh failure if a marker with a higher pid already
exists.

Race-hardening (per plan §2):

* Re-read the status label IMMEDIATELY before posting the failure marker;
  abort if the label has already moved out of ``status:running``.
* Idempotency: scan existing ``epm:failure`` markers; if any has a
  ``watch-pid`` >= our pid, exit silently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("pod_watch")
TICK_SECS = 60
PROBE_FAILURE_LIMIT = 5  # ticks of probe-unreachable before giving up
DEFAULT_THRESHOLD_SECS = 300
DEFAULT_MAX_RUNTIME_SECS = 86400  # 24h

# Status labels that mean "experiment progressed beyond running" — we exit
# silently without posting.
GRACEFUL_TERMINAL_LABELS = {
    "status:uploading",
    "status:interpreting",
    "status:reviewing",
    "status:under-review",
    "status:awaiting-promotion",
    "status:done-experiment",
    "status:done-impl",
    "status:followups-running",
    "status:archived",
}

# Already-blocked: another code path beat us; refuse to layer a duplicate.
BLOCKED_LABEL = "status:blocked"
RUNNING_LABEL = "status:running"

WATCH_PID_RE = re.compile(r"watch-pid=(\d+)")


def _gh_view(issue: int) -> dict:
    """Return parsed `gh issue view` JSON."""
    out = subprocess.check_output(
        ["gh", "issue", "view", str(issue), "--json", "labels,comments"],
        text=True,
    )
    return json.loads(out)


def _label_names(snapshot: dict) -> set[str]:
    return {label_obj["name"] for label_obj in snapshot.get("labels", [])}


def _has_marker(snapshot: dict, kind: str) -> bool:
    """True if any comment carries an ``<!-- epm:<kind> v* -->`` opener."""
    needle = f"<!-- epm:{kind} v"
    return any(needle in c.get("body", "") for c in snapshot.get("comments", []))


def _max_failure_pid(snapshot: dict) -> int | None:
    """Largest watch-pid found in any existing epm:failure marker, or None."""
    largest: int | None = None
    for c in snapshot.get("comments", []):
        body = c.get("body", "")
        if "<!-- epm:failure v" not in body:
            continue
        m = WATCH_PID_RE.search(body)
        if m is None:
            continue
        candidate = int(m.group(1))
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


def _check_terminal(issue: int) -> bool:
    """Return True if the watchdog should exit gracefully (epm:results
    posted, status moved beyond running, etc).

    On `status:blocked` or any GRACEFUL_TERMINAL_LABELS we ALSO return
    True — the watchdog never re-flips a blocked or graceful-terminal
    issue.
    """
    snapshot = _gh_view(issue)
    labels = _label_names(snapshot)
    # epm:results = graceful end-of-run.
    if _has_marker(snapshot, "results"):
        return True
    # Someone else already posted failure — don't pile on.
    if _has_marker(snapshot, "failure"):
        return True
    # Status moved out of running — terminal regardless of which label it
    # moved to (graceful next phase, manual blocked, archived, etc).
    return RUNNING_LABEL not in labels


def _post_failure(issue: int, *, reason: str, last_event: float | None) -> None:
    """Post an epm:failure marker with stall metadata, then flip the label."""
    pid = os.getpid()
    snapshot = _gh_view(issue)
    labels = _label_names(snapshot)

    # Step 1: re-read status; abort if it has moved.
    if RUNNING_LABEL not in labels:
        log.info(
            "watchdog %d: status no longer 'running' (labels=%s); "
            "aborting failure post — graceful exit",
            pid,
            sorted(labels),
        )
        return

    # Step 2: idempotency — if a later-pid failure marker exists, exit silent.
    largest_pid = _max_failure_pid(snapshot)
    if largest_pid is not None and largest_pid >= pid:
        log.info(
            "watchdog %d: failure marker already posted by watch-pid=%s; exit",
            pid,
            largest_pid,
        )
        return

    # Step 3: post the marker.
    last_event_iso = datetime.fromtimestamp(last_event).isoformat() if last_event else "never"
    body = (
        f"<!-- epm:failure v1 (watch-pid={pid}) -->\n"
        f"## Stall detected\n\n"
        f"failure_class: infra\n"
        f"reason: {reason}\n"
        f"last_event: {last_event_iso}\n"
        f"watchdog_pid: {pid}\n\n"
        f"The pod.py-watch heartbeat probe detected a stall. Routed to the "
        f"infra failure path; experimenter will be respawned on the next "
        f"`/issue {issue}` invocation (cap 3).\n"
        f"<!-- /epm:failure -->"
    )
    subprocess.check_call(["gh", "issue", "comment", str(issue), "--body", body])

    # Step 4: flip the label. Two-step gh edit isn't atomic on GitHub's
    # side; if the user manually flipped to status:blocked between steps
    # 1 and 4 the --remove is a no-op and --add is idempotent. Worst case
    # the issue stays at blocked (correct) — no harm.
    subprocess.check_call(
        [
            "gh",
            "issue",
            "edit",
            str(issue),
            "--remove-label",
            "status:running",
            "--add-label",
            "status:blocked",
        ]
    )
    log.info("watchdog %d: posted epm:failure (reason=%s); flipped to blocked", pid, reason)


def _watch_loop(
    issue: int,
    *,
    threshold_secs: int,
    wandb_run_url: str | None,
    log_path: str | None,
    pid_file: Path,
    max_runtime_secs: int,
) -> int:
    """Tick every TICK_SECS; flag stall after threshold_secs of no event.

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

        # Terminal-state check (results posted, label moved, etc).
        try:
            if _check_terminal(issue):
                log.info("watchdog %d: graceful terminal state; exit", os.getpid())
                return 0
        except subprocess.CalledProcessError as exc:
            # gh failed; treat as a probe failure.
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

        # Stall check.
        elapsed = time.time() - last_event_at
        if elapsed > threshold_secs:
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
        "watchdog %d: started (issue=%s, threshold=%ds, wandb=%s, log=%s, force_attach=%s)",
        os.getpid(),
        args.issue,
        args.threshold_secs,
        args.wandb_run_url,
        args.log_path,
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
