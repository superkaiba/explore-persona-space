#!/usr/bin/env python3
"""Per-task orchestrator lockfile management.

Each EPS task can have at most one `claude` CLI subprocess (a "body")
acting on it at any time. The lockfile is `tasks/<status>/<N>/.orchestrator.pid`
written at body start and deleted on clean exit. If a body crashes, the
next attempt detects the stale PID (no live process) and reclaims.

Subcommands: acquire | release | status
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from explore_persona_space.task_workflow import find_task_path  # noqa: E402


def _is_alive(pid: int) -> bool:
    """Return True if the given PID is currently a live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it — still "alive".
        return True
    return True


def _lock_path(task_n: int) -> Path:
    return find_task_path(task_n) / ".orchestrator.pid"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cmd_acquire(args: argparse.Namespace) -> int:
    lock = _lock_path(args.number)
    if lock.exists():
        try:
            content = lock.read_text().strip().splitlines()
            owner_pid = int(content[0])
        except (ValueError, IndexError):
            owner_pid = -1
        if owner_pid > 0 and _is_alive(owner_pid):
            print(f"locked by pid={owner_pid}", file=sys.stderr)
            return 1
        # Stale — reclaim.
        lock.write_text(f"{os.getpid()}\n{_now()}\n")
        print(f"reclaimed (was pid={owner_pid})")
        return 0
    lock.write_text(f"{os.getpid()}\n{_now()}\n")
    print(f"acquired pid={os.getpid()}")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    lock = _lock_path(args.number)
    if not lock.exists():
        print("not locked", file=sys.stderr)
        return 0
    try:
        owner_pid = int(lock.read_text().strip().splitlines()[0])
    except (ValueError, IndexError):
        owner_pid = -1
    if owner_pid != os.getpid() and not args.force:
        print(f"refusing to release lock owned by pid={owner_pid}", file=sys.stderr)
        return 1
    lock.unlink()
    print("released")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    lock = _lock_path(args.number)
    if not lock.exists():
        print("inactive")
        return 0
    content = lock.read_text().strip().splitlines()
    try:
        owner_pid = int(content[0])
        since = content[1] if len(content) > 1 else "?"
    except (ValueError, IndexError):
        print("corrupt")
        return 0
    alive = _is_alive(owner_pid)
    state = "active" if alive else "stale"
    print(f"{state} pid={owner_pid} since={since}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn, helps in (
        ("acquire", cmd_acquire, "claim the task lock (refuse if live owner)"),
        ("release", cmd_release, "release a lock owned by this PID"),
        ("status", cmd_status, "show current owner / alive / stale / inactive"),
    ):
        p = sub.add_parser(name, help=helps)
        p.add_argument("number", type=int)
        if name == "release":
            p.add_argument(
                "--force",
                action="store_true",
                help="release even if a different PID owns the lock",
            )
        p.set_defaults(func=fn)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
