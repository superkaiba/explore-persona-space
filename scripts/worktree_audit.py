#!/usr/bin/env python3
"""Stale-worktree sweep — safety net for the `/issue` Step 10d worktree
removal that does not always fire (e.g. when the merge gate is skipped, an
agent/Workflow worktree is abandoned, or a per-issue session is closed
without merging). Mirrors ``scripts/pod_audit.py`` (the stale-pod audit).

Without this, auto-generated worktrees under ``.claude/worktrees/`` pile up
unbounded — 102 worktrees / 161 GB had accumulated by 2026-05-28.

Scope: ONLY the auto-generated worktree name patterns are ever touched —
``issue-<N>`` (canonical /issue worktree), ``agent-<hex>`` (Agent
``isolation=worktree``), and ``wf_<id>`` (Workflow). Human-named worktrees
(``exp-*``, ``dashboard-*``, ``experiment-*``, ``sagan-*``, ``task-workflow``,
``issue-<N>-<suffix>`` variants, ...) are NEVER auto-removed — manual cleanup
only.

A targeted worktree is removed only when it is provably idle. It is KEPT
(skipped) if ANY of these hold:
  1. a live process has it as cwd, or references its path in argv;
  2. ``issue-<N>`` whose task status is non-terminal
     (planning / plan_pending / approved / running / verifying /
     interpreting / reviewing / blocked);
  3. it was modified within the grace window (default 6h);
  4. it has uncommitted TRACKED changes (real unmerged source — untracked
     generated files like ``eval_results/`` or scratch scripts do NOT block).

Default is dry-run. Pass ``--apply`` to actually remove (the cron wrapper
does). Removal uses ``git worktree remove --force`` (after ``git worktree
unlock`` for locked agent worktrees); a worktree git refuses to remove is
logged and skipped, never ``rm -rf``'d, so an unattended run can never lose
data it cannot account for.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

from explore_persona_space.task_workflow import repo_root, tasks_dir

# Statuses whose issue worktree is still in active use -> never auto-remove.
KEEP_STATUSES = frozenset(
    {
        "planning",
        "plan_pending",
        "approved",
        "running",
        "verifying",
        "interpreting",
        "reviewing",
        "blocked",
    }
)

# Worktree names the sweep is allowed to consider. Everything else is
# human-named and left for manual cleanup.
_TARGET_NAME_RE = re.compile(r"^(issue-\d+|agent-[0-9a-fA-F]+|wf_.+)$")
_ISSUE_NAME_RE = re.compile(r"^issue-(\d+)$")

DEFAULT_GRACE_HOURS = 6.0


@dataclass
class Decision:
    name: str
    remove: bool
    reason: str


@dataclass
class AuditResult:
    removed: list[str] = field(default_factory=list)
    kept: list[Decision] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def should_remove(
    name: str,
    *,
    status: str | None,
    is_live: bool,
    age_hours: float,
    has_tracked_changes: bool,
    grace_hours: float = DEFAULT_GRACE_HOURS,
) -> Decision:
    """Pure decision logic (unit-tested). ``status`` is the issue's task
    status for ``issue-<N>`` worktrees, else ``None``. Returns a Decision
    whose ``reason`` explains the keep/remove call."""
    if not _TARGET_NAME_RE.match(name):
        return Decision(name, False, "human-named worktree (out of sweep scope)")
    if is_live:
        return Decision(name, False, "held by a live process")
    if status is not None and status in KEEP_STATUSES:
        return Decision(name, False, f"issue status is non-terminal ({status})")
    if age_hours < grace_hours:
        return Decision(name, False, f"modified {age_hours:.1f}h ago (< {grace_hours}h grace)")
    if has_tracked_changes:
        return Decision(name, False, "has uncommitted tracked changes")
    detail = f"status={status}" if status is not None else "ephemeral agent/workflow worktree"
    return Decision(name, True, f"idle and reapable ({detail})")


def _issue_statuses() -> dict[int, str]:
    """Map issue number -> status by scanning the ``tasks/<status>/<id>/``
    filesystem tree, which is the AUTHORITATIVE source (the parent folder
    name IS the status). REGISTRY.json is a denormalized cache that lags the
    filesystem — task #407 was at ``tasks/running/407`` but absent from the
    registry on 2026-05-28, so trusting the registry alone would have flagged
    a running experiment's worktree for removal."""
    out: dict[int, str] = {}
    for status_dir in tasks_dir().iterdir():
        if not status_dir.is_dir():
            continue
        status = status_dir.name
        for task_dir in status_dir.iterdir():
            if not task_dir.is_dir():
                continue
            try:
                out[int(task_dir.name)] = status
            except ValueError:
                continue
    return out


def _live_worktree_names(wt_root: str) -> set[str]:
    """Worktree names currently referenced by any process: as a cwd
    (``/proc/<pid>/cwd``) or anywhere in argv (``/proc/<pid>/cmdline``)."""
    names: set[str] = set()
    marker = ".claude/worktrees/"

    def harvest(text: str) -> None:
        idx = text.find(marker)
        while idx != -1:
            rest = text[idx + len(marker) :]
            # name is up to the next path sep or NUL/space
            m = re.match(r"[A-Za-z0-9_.\-]+", rest)
            if m:
                names.add(m.group(0))
            idx = text.find(marker, idx + 1)

    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        # /proc entries are volatile (the process can exit between listdir
        # and read); skipping a vanished pid is expected, not a swallowed bug.
        with contextlib.suppress(OSError):
            harvest(os.readlink(f"/proc/{pid}/cwd"))
        with contextlib.suppress(OSError), open(f"/proc/{pid}/cmdline", "rb") as fh:
            harvest(fh.read().replace(b"\x00", b" ").decode("utf-8", "replace"))
    return names


def _has_tracked_changes(wt_path: str) -> bool:
    """True if the worktree has uncommitted TRACKED changes. Untracked
    files (``??`` porcelain lines) are generated output and do NOT count."""
    try:
        out = subprocess.run(
            ["git", "-C", wt_path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        # Cannot determine -> be conservative, treat as having changes.
        return True
    return any(line and not line.startswith("??") for line in out.splitlines())


def _git_remove(wt_path: str) -> bool:
    """Unlock (if locked) then ``git worktree remove --force``. Returns
    True on success. Never falls back to rm -rf in the unattended path."""
    subprocess.run(
        ["git", "worktree", "unlock", wt_path],
        capture_output=True,
        text=True,
    )
    rc = subprocess.run(
        ["git", "worktree", "remove", "--force", wt_path],
        capture_output=True,
        text=True,
    )
    return rc.returncode == 0


def audit(apply: bool, grace_hours: float, now: float | None = None) -> AuditResult:
    now = time.time() if now is None else now
    root = repo_root()
    wt_root_rel = ".claude/worktrees/"
    wt_dir = root / ".claude" / "worktrees"
    res = AuditResult()
    if not wt_dir.is_dir():
        return res

    statuses = _issue_statuses()
    live = _live_worktree_names(wt_root_rel)

    # Clear any admin entries for worktree dirs that were already deleted.
    subprocess.run(["git", "worktree", "prune"], cwd=str(root), capture_output=True)

    for child in sorted(wt_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        status = None
        m = _ISSUE_NAME_RE.match(name)
        if m:
            status = statuses.get(int(m.group(1)))
        age_hours = (now - child.stat().st_mtime) / 3600.0
        # Only pay for the git status call on otherwise-removable worktrees.
        provisional = should_remove(
            name,
            status=status,
            is_live=name in live,
            age_hours=age_hours,
            has_tracked_changes=False,
            grace_hours=grace_hours,
        )
        if provisional.remove and _has_tracked_changes(str(child)):
            provisional = Decision(name, False, "has uncommitted tracked changes")

        if not provisional.remove:
            res.kept.append(provisional)
            continue
        if not apply:
            res.removed.append(name)  # would-remove (dry-run)
            continue
        if _git_remove(str(child)):
            res.removed.append(name)
        else:
            res.failed.append(name)

    if apply:
        subprocess.run(["git", "worktree", "prune"], cwd=str(root), capture_output=True)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stale-worktree sweep (safety net).")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove (default: dry-run, report would-remove).",
    )
    ap.add_argument(
        "--grace-hours",
        type=float,
        default=DEFAULT_GRACE_HOURS,
        help=f"Skip worktrees modified within this many hours (default {DEFAULT_GRACE_HOURS}).",
    )
    ap.add_argument("--json", action="store_true", help="Emit a JSON summary.")
    args = ap.parse_args(argv)

    res = audit(apply=args.apply, grace_hours=args.grace_hours)
    verb = "removed" if args.apply else "would remove"

    if args.json:
        print(
            json.dumps(
                {
                    "apply": args.apply,
                    "grace_hours": args.grace_hours,
                    "removed": res.removed,
                    "failed": res.failed,
                    "kept": [{"name": d.name, "reason": d.reason} for d in res.kept],
                }
            )
        )
    else:
        print(
            f"worktree_audit: {verb} {len(res.removed)} | "
            f"kept {len(res.kept)} | failed {len(res.failed)}"
        )
        for name in res.removed:
            print(f"  - {verb}: {name}")
        for name in res.failed:
            print(f"  ! FAILED to remove: {name}")
        # Keep reasons only matter for debugging; show targeted-but-kept ones.
        for d in res.kept:
            if _TARGET_NAME_RE.match(d.name):
                print(f"  . kept: {d.name} ({d.reason})")

    # Exit 2 when something was (or would be) removed, mirroring pod_audit;
    # the cron wrapper swallows it so cron does not email on every sweep.
    return 2 if (res.removed or res.failed) else 0


if __name__ == "__main__":
    sys.exit(main())
