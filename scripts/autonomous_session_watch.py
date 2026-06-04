"""Crash-recovery watcher for autonomous (`--auto`) issue sessions.

The `/loop 10m /issue <N>` driver and any `CronCreate(durable=False)` backstop
live *inside* the session's Claude process, so they die with it — a process
crash / OOM / VM reboot leaves an autonomous experiment stalled until someone
manually `happy resume`s it. This watcher closes that gap: it runs OUT of
process (a real VM crontab line, like cron_worktree_audit.sh) and re-spawns an
autonomous session whose driver has died.

Mechanism
---------
`spawn_session.py spawn-issue --auto` writes one registry file per issue at
``~/.eps-autonomous/issue-<N>.json`` recording the Happy session id + cwd + the
GPU-hour cap. This watcher, each run:

  * reads the task's current status (via `task.py view --json`);
  * decides per :func:`decide` whether to RESPAWN / KEEP / DELETE the entry;
  * a session is "alive" iff its recorded id is in the daemon's live set OR a
    live session sits in the issue's worktree (`.claude/worktrees/issue-<N>`);
  * a dead session is only re-spawned after ``--threshold`` (default 2)
    consecutive misses, so a transient daemon-list glitch never double-spawns;
  * single-flight via flock so two overlapping cron fires can't race.

RESPAWN re-invokes `spawn_session.py spawn-issue --auto`, which rewrites the
registry with the new id and ``missed=0``. Parked/terminal tasks are never
re-spawned (see the status sets below); awaiting_promotion is a human gate.

Run: ``uv run python scripts/autonomous_session_watch.py [--dry-run] [--threshold N]``
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

# scripts/ is sys.path[0] when run as `python scripts/autonomous_session_watch.py`,
# so spawn_session (its sibling) imports directly. Reuse its daemon readers +
# registry constants rather than duplicating them.
from spawn_session import (
    AUTONOMOUS_REGISTRY_DIR,
    PROJECT_ROOT,
    _live_session_ids,
    _load_session_meta,
)

# Active-drive statuses: a dead session here SHOULD be resurrected.
ACTIVE = {"planning", "approved", "running", "verifying", "interpreting", "reviewing"}
# Park statuses: legitimately waiting on the user or a gate — never re-spawn,
# but keep the entry (it may flip back to ACTIVE, e.g. plan_pending -> approved).
PARK = {"proposed", "clarifying", "plan_pending", "blocked"}
# Terminal statuses: the autonomous run is done — drop the entry.
# awaiting_promotion is terminal HERE (experiment finished; the user promotes
# manually — no more auto-driving needed).
TERMINAL = {"awaiting_promotion", "completed", "archived"}

# Hard backstop: drop a registry entry whose task has not progressed in this
# long, so a stuck/unknown-status entry cannot linger and re-spawn forever.
MAX_ENTRY_AGE_S = 14 * 24 * 3600


def decide(status: str, alive: bool, missed: int, threshold: int = 2) -> tuple[str, int]:
    """Pure decision: given a task's status, whether its session is alive, and
    the consecutive-miss count, return ``(action, new_missed)`` where action is
    ``"respawn"`` | ``"keep"`` | ``"delete"``.

    Safety: only an ACTIVE status with a session confirmed dead on
    ``threshold`` consecutive checks (default 2 = ~20 min at a 10-min cron)
    yields ``"respawn"``. Parked tasks reset the miss count and are kept;
    terminal tasks are deleted; an unknown status is kept without ever spawning.
    """
    if status in TERMINAL:
        return ("delete", 0)
    if status in PARK:
        return ("keep", 0)
    if status in ACTIVE:
        if alive:
            return ("keep", 0)
        new_missed = missed + 1
        if new_missed >= threshold:
            return ("respawn", 0)
        return ("keep", new_missed)
    # Unknown status (e.g. a renamed enum): do nothing, keep the entry so a
    # human notices rather than silently dropping or spawning.
    return ("keep", missed)


def _task_status(issue: int) -> str | None:
    """Current status of task ``issue`` via `task.py view --json`, or ``None``
    if the task no longer exists / cannot be read."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "view", str(issue), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    status = data.get("status") or (data.get("frontmatter") or {}).get("status")
    return status if isinstance(status, str) else None


def _daemon_reachable() -> bool:
    """True iff the Happy daemon's control server answers /list.

    Critical guard: ``_live_session_ids()`` returns an empty set BOTH when the
    daemon is up with zero sessions AND when it is unreachable. Without
    distinguishing them, a daemon outage would make every recorded session look
    dead and trigger a mass re-spawn (-> duplicate pods). So the watcher probes
    reachability first and skips the whole run if the daemon is down."""
    try:
        import urllib.error
        import urllib.request

        from spawn_session import daemon_port

        url = f"http://127.0.0.1:{daemon_port()}/list"
        req = urllib.request.Request(
            url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            json.loads(resp.read())
        return True
    except (SystemExit, urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


def _session_alive(entry: dict, live_ids: set[str], live_cwds: set[str]) -> bool:
    """A session counts as alive if its recorded Happy id is still tracked by
    the daemon, OR a live session occupies the issue's worktree dir (covers a
    manual / PM re-spawn that replaced the recorded id)."""
    if entry.get("happy_session_id") in live_ids:
        return True
    issue = entry.get("issue")
    return any(p.rstrip("/").endswith(f"/issue-{issue}") for p in live_cwds)


def _respawn(entry: dict, dry_run: bool) -> bool:
    """Re-spawn the autonomous session for this entry. Returns True on success.
    spawn_session rewrites the registry (new id, missed=0) as a side effect."""
    issue = entry["issue"]
    cap = entry.get("auto_approve_gpu_hours", 24.0)
    cmd = [
        "uv", "run", "python", "scripts/spawn_session.py", "spawn-issue",
        "--issue", str(issue), "--auto", "--auto-approve-gpu-hours", str(cap),
    ]  # fmt: skip
    if dry_run:
        print(f"  [dry-run] would respawn: {' '.join(cmd)}")
        return False
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        print(f"  RESPAWN FAILED issue #{issue}: {res.stderr.strip()[:300]}", file=sys.stderr)
        return False
    first_line = (res.stdout.strip().splitlines() or [""])[0]
    print(f"  RESPAWNED issue #{issue} (session was dead): {first_line}")
    return True


def _acquire_lock() -> object | None:
    """Single-flight: hold a non-blocking flock so overlapping cron fires don't
    race (a race could double-spawn -> two pods). Returns the held fd, or None
    if another watcher run holds it (caller should exit cleanly)."""
    AUTONOMOUS_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    # Held for the whole run (released on process exit) — a context manager
    # would close it and drop the lock, so the bare open is deliberate.
    fd = open(AUTONOMOUS_REGISTRY_DIR / "watch.lock", "w")  # noqa: SIM115
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fd.close()
        return None
    return fd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="log decisions; do not respawn or mutate entries"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="consecutive dead-checks before re-spawning (default 2 = ~20 min at a 10-min cron)",
    )
    args = parser.parse_args(argv)

    lock = _acquire_lock()
    if lock is None:
        print("another autonomous_session_watch run holds the lock; exiting")
        return 0

    entries = sorted(AUTONOMOUS_REGISTRY_DIR.glob("issue-*.json"))
    if not entries:
        print("no autonomous sessions registered")
        return 0

    if not _daemon_reachable():
        print("Happy daemon unreachable; skipping run (won't mass-respawn on a daemon outage)")
        return 0

    live_ids = _live_session_ids()
    meta = _load_session_meta()
    live_cwds = {m.get("path", "") for sid, m in meta.items() if sid in live_ids}
    print(f"{len(entries)} registered, {len(live_ids)} live session(s)")

    for path in entries:
        _process_entry(path, live_ids, live_cwds, args.dry_run, args.threshold)

    return 0


def _process_entry(
    path: Path, live_ids: set[str], live_cwds: set[str], dry_run: bool, threshold: int
) -> None:
    """Apply one registry entry's decision (read status -> decide -> act).

    Removes the entry on unreadable/missing-task/backstop-age; respawns a dead
    ACTIVE session; otherwise persists an updated miss count. Honours dry_run
    (logs but never mutates / spawns)."""
    try:
        entry = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"  {path.name}: unreadable; removing")
        if not dry_run:
            path.unlink(missing_ok=True)
        return

    issue = entry.get("issue")
    status = _task_status(issue)
    if status is None:
        print(f"  issue #{issue}: task not found / unreadable; removing entry")
        if not dry_run:
            path.unlink(missing_ok=True)
        return

    if time.time() - entry.get("spawned_at", 0) > MAX_ENTRY_AGE_S and status not in ACTIVE:
        print(f"  issue #{issue}: entry older than backstop + not active ({status}); removing")
        if not dry_run:
            path.unlink(missing_ok=True)
        return

    alive = _session_alive(entry, live_ids, live_cwds)
    action, new_missed = decide(status, alive, entry.get("missed", 0), threshold)
    print(
        f"  issue #{issue}: status={status} alive={alive} "
        f"missed={entry.get('missed', 0)}->{new_missed} action={action}"
    )

    if action == "delete":
        if not dry_run:
            path.unlink(missing_ok=True)
    elif action == "respawn":
        _respawn(entry, dry_run)  # rewrites the registry on success
    elif action == "keep" and new_missed != entry.get("missed", 0):
        entry["missed"] = new_missed
        if not dry_run:
            path.write_text(json.dumps(entry, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
