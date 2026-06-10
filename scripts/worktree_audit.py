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
  3. it was modified within the grace window (default 6h; tightened to 1h
     under disk pressure — see below);
  4. it has uncommitted TRACKED changes (real unmerged source — untracked
     generated files like ``eval_results/`` or scratch scripts do NOT block).

Disk-pressure mode (2026-06-10, #543): the VM root disk hit 100% mid-pipeline
with ``.claude/worktrees/`` holding 264 GB, intermittently killing git /
task.py across all concurrent sessions. The audit now always reports the
usage of the filesystem holding the worktrees plus a per-worktree ``du``;
when usage is at/above a threshold (default 90%, override via
``EPM_WORKTREE_DISK_PRESSURE_PCT``) the grace window in guard 3 tightens to
``PRESSURE_GRACE_HOURS`` (1h). Pressure changes ONLY the grace window —
guards 1, 2 and 4 and the human-named exclusion are unaffected.

Triage reporting (2026-06-10, #543 follow-up): tightening grace cannot reclaim
worktrees held by guards 1 and 4, which under real pressure ARE the backlog
(observed: ~10 worktrees of long-completed issues, ~13G each, kept only by
uncommitted tracked changes). So the report now (a) surfaces the
manual-triage backlog — worktrees that passed every guard EXCEPT tracked
changes — as a count + du total (text line under pressure; JSON always), and
(b) names the holding pid + trimmed cmdline for every live-process keep, so
zombie sessions pinning terminal-status worktrees are identifiable. Reporting
only — no reaping behavior change.

Default is dry-run. Pass ``--apply`` to actually remove (the cron wrapper
does). Removal uses ``git worktree remove --force`` (after ``git worktree
unlock`` for locked agent worktrees); a worktree git refuses to remove is
logged and skipped, never ``rm -rf``'d, so an unattended run can never lose
data it cannot account for (this also rules out deleting gitignored caches
inside KEPT worktrees under pressure — a held worktree may have a live
process using those files).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

from explore_persona_space.task_workflow import repo_root, tasks_dir

# Issue statuses whose worktree is DONE + merged -> eligible for reaping.
# This is an ALLOWLIST (fail-closed): an issue-<N> worktree is reaped only
# when its status is explicitly one of these. ANY other non-None status —
# an in-flight state (running/planning/...), `blocked`, OR an unrecognized /
# corrupt folder name — keeps the worktree. (agent-/wf- worktrees carry no
# status and are reaped on the idle guards alone.) `awaiting_promotion` is
# intentionally reapable: the clean-result is already merged to main and the
# park-and-wait promotion uses the main repo's tasks/, not the worktree.
REAPABLE_ISSUE_STATUSES = frozenset({"completed", "archived", "awaiting_promotion"})

# Worktree names the sweep is allowed to consider. Everything else is
# human-named and left for manual cleanup. The wf_ branch is restricted to
# the same char class harvest() extracts, so liveness detection can never
# false-negative on a name with chars it would not match.
_TARGET_NAME_RE = re.compile(r"^(issue-\d+|agent-[0-9a-fA-F]+|wf_[A-Za-z0-9_.\-]+)$")
_ISSUE_NAME_RE = re.compile(r"^issue-(\d+)$")

DEFAULT_GRACE_HOURS = 6.0

# Disk-pressure mode: at/above this filesystem usage the grace window
# tightens to PRESSURE_GRACE_HOURS. Threshold overridable via env.
DEFAULT_PRESSURE_THRESHOLD_PCT = 90.0
PRESSURE_GRACE_HOURS = 1.0

# Single source for the tracked-changes keep reason: emitted by should_remove
# / _classify and matched by tracked_changes_backlog, so the backlog counter
# can never drift out of sync with the decisions it summarizes.
_TRACKED_CHANGES_REASON = "has uncommitted tracked changes"


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
    # Reporting (always on): per-worktree disk usage in bytes (None when du
    # failed), usage pct of the filesystem holding the worktrees, and the
    # pressure state actually applied to this run.
    sizes_bytes: dict[str, int | None] = field(default_factory=dict)
    disk_pct: float | None = None
    pressure_threshold_pct: float = DEFAULT_PRESSURE_THRESHOLD_PCT
    pressure: bool = False
    grace_hours_effective: float = DEFAULT_GRACE_HOURS
    # Reporting only: worktree name -> ["pid <pid>: <trimmed cmdline>", ...]
    # for every process referencing it (initial liveness snapshot), so a
    # live-process keep names its holders (zombie-session triage).
    live_holders: dict[str, list[str]] = field(default_factory=dict)


def effective_grace_hours(grace_hours: float, disk_pct: float, threshold_pct: float) -> float:
    """Pure pressure rule (unit-tested): at/above ``threshold_pct`` usage the
    grace window tightens to ``PRESSURE_GRACE_HOURS``; an explicitly tighter
    ``grace_hours`` is never loosened. Below the threshold, unchanged."""
    if disk_pct >= threshold_pct:
        return min(grace_hours, PRESSURE_GRACE_HOURS)
    return grace_hours


def _pressure_threshold_pct() -> float:
    """Pressure threshold (% filesystem usage), env-overridable."""
    return float(
        os.environ.get("EPM_WORKTREE_DISK_PRESSURE_PCT", str(DEFAULT_PRESSURE_THRESHOLD_PCT))
    )


def _disk_usage_pct(path: str) -> float:
    """Percent used of the filesystem holding ``path``."""
    usage = shutil.disk_usage(path)
    return 100.0 * usage.used / usage.total


def _worktree_size_bytes(path: str) -> int | None:
    """Disk usage of one worktree via ``du -sx`` (REPORTING ONLY — a du
    failure or timeout degrades to None and never blocks the sweep).

    Caveat: content hardlinked across worktrees (uv-managed ``.venv``\\s) is
    counted once PER worktree, so the per-worktree sum overstates unique
    disk usage (observed 2026-06-10: du-sum 1146G vs ~264G actual)."""
    try:
        out = subprocess.run(
            ["du", "-sx", "--block-size=1", path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return int(out.stdout.split()[0])
    except (ValueError, IndexError):
        return None


def _fmt_size(n: int | None) -> str:
    """Human-readable GB string for report lines ('?' when du failed)."""
    return f"{n / 1e9:.1f}G" if n is not None else "?"


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
    # issue-<N>: reap ONLY when the status is an explicitly reapable terminal
    # state. Any other non-None status — in-flight, blocked, or an
    # unrecognized/corrupt folder name — fails closed and keeps the worktree.
    # status is None for an orphan issue (folder gone) and for agent-/wf-
    # worktrees; those fall through to the idle guards.
    if _ISSUE_NAME_RE.match(name) and status is not None and status not in REAPABLE_ISSUE_STATUSES:
        return Decision(name, False, f"issue status not reapable ({status})")
    if age_hours < grace_hours:
        return Decision(name, False, f"modified {age_hours:.1f}h ago (< {grace_hours}h grace)")
    if has_tracked_changes:
        return Decision(name, False, _TRACKED_CHANGES_REASON)
    detail = f"status={status}" if status is not None else "ephemeral agent/workflow worktree"
    return Decision(name, True, f"idle and reapable ({detail})")


def tracked_changes_backlog(
    kept: list[Decision], sizes_bytes: dict[str, int | None]
) -> tuple[int, int]:
    """Pure backlog summary (unit-tested): count + total du bytes of kept
    worktrees held ONLY by uncommitted tracked changes — i.e. they passed
    every other guard (in-scope, idle, reapable status, past grace) and would
    have been reaped otherwise. This is the reclaimable-pending-manual-triage
    set the daily cron log surfaces under disk pressure. Substring match also
    catches the ``became unsafe mid-audit: ...`` variant; a None du value
    counts as 0 bytes (and the du sum is hardlink-overcounted, like every
    size this report prints)."""
    matching = [d for d in kept if _TRACKED_CHANGES_REASON in d.reason]
    total = sum(sizes_bytes.get(d.name) or 0 for d in matching)
    return len(matching), total


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


def _live_worktree_holders(wt_root: str) -> dict[str, list[str]]:
    """Worktree names currently referenced by any process: as a cwd
    (``/proc/<pid>/cwd``) or anywhere in argv (``/proc/<pid>/cmdline``).

    Returns name -> ["pid <pid>: <trimmed cmdline>", ...] so the report can
    say WHICH process pins a kept worktree (zombie-session triage). The
    liveness test itself is unchanged — ``name in holders`` is exactly the
    old set membership; the values are reporting-only."""
    holders: dict[str, list[str]] = {}
    marker = ".claude/worktrees/"

    def harvest(text: str, found: set[str]) -> None:
        idx = text.find(marker)
        while idx != -1:
            rest = text[idx + len(marker) :]
            # name is up to the next path sep or NUL/space
            m = re.match(r"[A-Za-z0-9_.\-]+", rest)
            if m:
                found.add(m.group(0))
            idx = text.find(marker, idx + 1)

    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        found: set[str] = set()
        cmdline = ""
        # /proc entries are volatile (the process can exit between listdir
        # and read); skipping a vanished pid is expected, not a swallowed bug.
        with contextlib.suppress(OSError):
            harvest(os.readlink(f"/proc/{pid}/cwd"), found)
        with contextlib.suppress(OSError), open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmdline = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
            harvest(cmdline, found)
        for name in found:
            holders.setdefault(name, []).append(f"pid {pid}: {cmdline[:120] or '?'}")
    return holders


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


def _classify(
    child, statuses: dict[int, str], live: dict[str, list[str]], grace_hours: float, now: float
) -> Decision:
    """Full keep/remove decision for one worktree dir, including the
    (fresh) tracked-changes git call. ``statuses`` and ``live`` are the
    snapshots to decide against (liveness uses ``live``'s keys only)."""
    name = child.name
    status = None
    m = _ISSUE_NAME_RE.match(name)
    if m:
        status = statuses.get(int(m.group(1)))
    age_hours = (now - child.stat().st_mtime) / 3600.0
    decision = should_remove(
        name,
        status=status,
        is_live=name in live,
        age_hours=age_hours,
        has_tracked_changes=False,
        grace_hours=grace_hours,
    )
    # Only pay for the git status call on otherwise-removable worktrees.
    if decision.remove and _has_tracked_changes(str(child)):
        return Decision(name, False, _TRACKED_CHANGES_REASON)
    return decision


def audit(apply: bool, grace_hours: float, now: float | None = None) -> AuditResult:
    now = time.time() if now is None else now
    root = repo_root()
    wt_root_rel = ".claude/worktrees/"
    wt_dir = root / ".claude" / "worktrees"
    res = AuditResult(grace_hours_effective=grace_hours)
    if not wt_dir.is_dir():
        return res

    # Disk-pressure check: at/above the threshold the grace window tightens.
    # ONLY the grace window changes — the live-process, issue-status,
    # tracked-changes and human-named guards are pressure-independent.
    res.disk_pct = _disk_usage_pct(str(wt_dir))
    res.pressure_threshold_pct = _pressure_threshold_pct()
    res.pressure = res.disk_pct >= res.pressure_threshold_pct
    res.grace_hours_effective = effective_grace_hours(
        grace_hours, res.disk_pct, res.pressure_threshold_pct
    )
    grace_hours = res.grace_hours_effective

    statuses = _issue_statuses()
    live = _live_worktree_holders(wt_root_rel)
    res.live_holders = live

    # Clear any admin entries for worktree dirs that were already deleted.
    subprocess.run(["git", "worktree", "prune"], cwd=str(root), capture_output=True)

    for child in sorted(wt_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        res.sizes_bytes[name] = _worktree_size_bytes(str(child))
        decision = _classify(child, statuses, live, grace_hours, now)
        if not decision.remove:
            res.kept.append(decision)
            continue
        if not apply:
            res.removed.append(name)  # would-remove (dry-run)
            continue
        # Re-derive status + liveness FRESH immediately before the
        # destructive call, to close the snapshot->remove race: a session
        # that cd'd in, or a `task.py set-status` that flipped the issue to
        # a non-reapable state, after the initial snapshot must still be
        # honored (M1/M2).
        fresh = _classify(
            child, _issue_statuses(), _live_worktree_holders(wt_root_rel), grace_hours, now
        )
        if not fresh.remove:
            res.kept.append(Decision(name, False, f"became unsafe mid-audit: {fresh.reason}"))
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
        help=(
            f"Skip worktrees modified within this many hours (default {DEFAULT_GRACE_HOURS}; "
            f"tightened to {PRESSURE_GRACE_HOURS} under disk pressure)."
        ),
    )
    ap.add_argument("--json", action="store_true", help="Emit a JSON summary.")
    args = ap.parse_args(argv)

    res = audit(apply=args.apply, grace_hours=args.grace_hours)
    verb = "removed" if args.apply else "would remove"
    backlog_count, backlog_bytes = tracked_changes_backlog(res.kept, res.sizes_bytes)

    if args.json:
        print(
            json.dumps(
                {
                    "apply": args.apply,
                    "grace_hours": args.grace_hours,
                    "grace_hours_effective": res.grace_hours_effective,
                    "disk_pct": res.disk_pct,
                    "pressure_threshold_pct": res.pressure_threshold_pct,
                    "disk_pressure": res.pressure,
                    "removed": res.removed,
                    "failed": res.failed,
                    "kept": [
                        {
                            "name": d.name,
                            "reason": d.reason,
                            "holders": res.live_holders.get(d.name, []),
                        }
                        for d in res.kept
                    ],
                    "sizes_bytes": res.sizes_bytes,
                    # Manual-triage backlog: kept ONLY by uncommitted tracked
                    # changes (would have been reaped otherwise).
                    "tracked_changes_only": {
                        "count": backlog_count,
                        "bytes": backlog_bytes,
                    },
                }
            )
        )
    else:
        if res.disk_pct is not None:
            total = sum(n for n in res.sizes_bytes.values() if n is not None)
            print(
                f"worktree_audit: disk {res.disk_pct:.1f}% used "
                f"(pressure threshold {res.pressure_threshold_pct:.0f}%) | "
                f"worktrees du-sum {_fmt_size(total)} across {len(res.sizes_bytes)} "
                f"(hardlinks counted per worktree)"
            )
            if res.pressure:
                print(
                    f"  !! DISK PRESSURE: grace window tightened "
                    f"{args.grace_hours:g}h -> {res.grace_hours_effective:g}h"
                )
                # Grace tightening cannot reclaim these — surface the
                # manual-triage backlog so the cron log makes it actionable.
                print(
                    f"  !! pressure: {backlog_count} worktrees held only by "
                    f"uncommitted tracked changes, {_fmt_size(backlog_bytes)} total "
                    f"(manual triage)"
                )
        print(
            f"worktree_audit: {verb} {len(res.removed)} | "
            f"kept {len(res.kept)} | failed {len(res.failed)}"
        )
        for name in res.removed:
            print(f"  - {verb}: {name} [{_fmt_size(res.sizes_bytes.get(name))}]")
        for name in res.failed:
            print(f"  ! FAILED to remove: {name} [{_fmt_size(res.sizes_bytes.get(name))}]")
        # Keep reasons only matter for debugging; show targeted-but-kept ones.
        for d in res.kept:
            if _TARGET_NAME_RE.match(d.name):
                print(f"  . kept: {d.name} [{_fmt_size(res.sizes_bytes.get(d.name))}] ({d.reason})")
                # Name the pinning process(es) so a zombie session holding a
                # terminal-status worktree is identifiable from the log alone.
                for holder in res.live_holders.get(d.name, []):
                    print(f"      # held by {holder}")

    # Exit 2 when something was (or would be) removed, mirroring pod_audit;
    # the cron wrapper swallows it so cron does not email on every sweep.
    return 2 if (res.removed or res.failed) else 0


if __name__ == "__main__":
    sys.exit(main())
