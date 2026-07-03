#!/usr/bin/env python3
"""Stale-worktree sweep — safety net for the `/issue` Step 10d worktree
removal that does not always fire (e.g. when the merge gate is skipped, an
agent/Workflow worktree is abandoned, or a per-issue session is closed
without merging). Mirrors ``scripts/pod_audit.py`` (the stale-pod audit).

Without this, auto-generated worktrees under ``.claude/worktrees/`` pile up
unbounded — 102 worktrees / 161 GB had accumulated by 2026-05-28.

Scope: ONLY the auto-generated worktree name patterns are ever touched —
``issue-<N>`` (canonical /issue worktree), ``issue-<N>-<suffix>``
(session-created same-issue follow-up rounds — mapped to issue N for the
status lookup; in scope as of 2026-06-12, after 10+ such worktrees were
misclassified human-named and became immortal during the 201 GB disk-bloat
incident), ``agent-<hex>`` (Agent ``isolation=worktree``), and ``wf_<id>``
(Workflow). Genuinely human-named worktrees (``exp-*``, ``dashboard-*``,
``experiment-*``, ``sagan-*``, ``task-workflow``, ``compute-router``, ...)
are NEVER auto-removed — manual cleanup only.

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

Active remediation (2026-06-10, disk-full incident): guards 1 and 4 had two
recurring false-keep classes that pinned ~35 worktrees of long-completed
tasks while the root disk filled to 100%: (a) ORPHAN-PINNED — ``codex
app-server`` / openai-codex plugin processes spawned with an issue-worktree
cwd never exit after their companion task completes, so guard 1 held 11
worktrees of weeks-completed tasks indefinitely; (b) JUNK-DIRTY —
uncommitted changes confined to runtime-noise files (agent memories,
pods.conf, pods_ephemeral.json) tripped guard 4 and held 13G worktrees
hostage for 2KB diffs. For issue worktrees whose task status is DONE AND
MERGED (``REMEDIATION_ISSUE_STATUSES`` = completed/archived/
awaiting_promotion — awaiting_promotion added 2026-06-12: the worktree
auto-merges to main at the Step 9b awaiting_promotion transition and the
watcher's session-reconcile pass auto-stops parked sessions after ~2h
idle, so the original "may be live-parked" rationale is obsolete; a
genuinely live session is still protected by the real-holder guard), the
audit now classifies both cases loudly
in every report, and under ``--apply`` (NEVER in dry-run) remediates:
(a) kills the exact orphan pids — cmdline re-verified against
``ORPHAN_HOLDER_PATTERNS`` immediately before every signal (PID-reuse
guard); SIGTERM, brief wait, SIGKILL; any survivor or any non-matching
holder keeps the worktree — then re-derives every guard against fresh
state; (b) rescue-copies the allowlisted dirty files plus a full ``git diff
HEAD`` to ``.claude/cache/worktree-rescue-<date>/<name>/`` BEFORE treating
the tree as clean for removal (rescue strictly precedes removal). Dirt
outside ``RESCUE_DIRTY_ALLOWLIST``, any real (non-orphan) holder, and
non-terminal statuses keep today's behavior unchanged.

Venv-reap arm (2026-07-03, #912): a KEPT worktree's ``.venv`` is a pure
build artifact ``uv run``/``uv sync`` regenerates in minutes from the shared
uv cache (``pyproject.toml`` + ``uv.lock`` are checked in at every worktree
root), yet every keep reason (dirty / active issue / human-named) retained it
indefinitely — 34 of 35 worktrees carried one on the 95%-full boot disk. On
every sweep the plain-KEEP branch now additionally deletes JUST the
``.venv`` (rename-aside to ``.venv.reap-tmp-<pid>`` first, post-rename
holder re-check, then rmtree — never the worktree itself) when the worktree
has been idle >= ``DEFAULT_VENV_IDLE_DAYS`` (7d; 2d under the same
disk-pressure predicate as the grace window; env
``EPM_WORKTREE_VENV_IDLE_DAYS``) AND no process holds it — cwd/argv harvest
PLUS a dedicated ``/proc/<pid>/exe`` probe catching interpreters exec'd from
the venv itself. Underscore-prefixed managed worktrees (``_task-main-pin``),
symlinked roots, symlinked ``.venv``\\s, and any worktree whose idleness
cannot be established are never touched; deletion never widens beyond
``<wt>/.venv`` + ``<wt>/.venv.reap-tmp-*``. Dry-run only classifies
(would-reap candidates). Kill switch: ``EPM_WORKTREE_VENV_REAP=0``.

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
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

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
# human-named and left for manual cleanup. ``issue-<N>-<suffix>`` variants
# (session-created for same-issue follow-up rounds) are in scope as of
# 2026-06-12 and map to issue N for the status lookup — previously they
# fell out of this regex, were misclassified human-named, and became
# immortal (10+ of the 53 worktrees in the 201 GB disk-bloat incident).
# The issue-suffix and wf_ branches are restricted to the same char class
# harvest() extracts, so liveness detection can never false-negative on a
# name with chars it would not match.
_TARGET_NAME_RE = re.compile(
    r"^(issue-\d+(?:-[A-Za-z0-9_.\-]+)?|agent-[0-9a-fA-F]+|wf_[A-Za-z0-9_.\-]+)$"
)
_ISSUE_NAME_RE = re.compile(r"^issue-(\d+)(?:-[A-Za-z0-9_.\-]+)?$")

DEFAULT_GRACE_HOURS = 6.0

# Disk-pressure mode: at/above this filesystem usage the grace window
# tightens to PRESSURE_GRACE_HOURS. Threshold overridable via env.
DEFAULT_PRESSURE_THRESHOLD_PCT = 90.0
PRESSURE_GRACE_HOURS = 1.0

# Venv-reap arm (#912): a KEPT worktree's .venv is a pure build artifact,
# regenerable by `uv run`/`uv sync` from the shared uv cache (pyproject.toml
# + uv.lock are checked in at every worktree root). Reap it when the
# worktree has been idle past this window and NOTHING is running in it.
# Pressure tightens the window via effective_venv_idle_days (same predicate
# as PRESSURE_GRACE_HOURS). Kill switch: EPM_WORKTREE_VENV_REAP=0.
DEFAULT_VENV_IDLE_DAYS = 7.0  # env: EPM_WORKTREE_VENV_IDLE_DAYS
PRESSURE_VENV_IDLE_DAYS = 2.0  # tightened under the SAME disk-pressure predicate
_VENV_DIRNAME = ".venv"
_VENV_REAP_TMP_PREFIX = ".venv.reap-tmp-"

# Worktree-name component followed by a `.venv`-PREFIXED path component —
# deliberately also matches `.venv.reap-tmp-*` leftovers so a process
# exec'd from a renamed-aside venv still protects it (post-rename gate).
# The char class is the same one harvest() extracts.
_VENV_EXE_RE = re.compile(r"([A-Za-z0-9_.\-]+)/\.venv")

# Single source for the tracked-changes keep reason: emitted by should_remove
# / _classify and matched by tracked_changes_backlog, so the backlog counter
# can never drift out of sync with the decisions it summarizes.
_TRACKED_CHANGES_REASON = "has uncommitted tracked changes"

# Single source for the live-process keep reason: emitted by should_remove
# and matched by the remediation triage (_remediation_kind), same
# anti-drift coupling as _TRACKED_CHANGES_REASON above.
_LIVE_PROCESS_REASON = "held by a live process"

# Issue statuses eligible for ACTIVE remediation (orphan-holder kill /
# junk-dirty rescue) under --apply. `awaiting_promotion` included as of
# 2026-06-12 (disk-bloat incident: 10-17 GB awaiting_promotion worktrees
# pinned by orphaned codex holders or junk dirt sat unreclaimable for the
# whole multi-week promotion backlog): the worktree has already
# auto-merged to main at the Step 9b awaiting_promotion transition (its
# purpose is complete), and the watcher's session-reconcile pass
# auto-stops parked awaiting_promotion sessions after ~2h idle, so the
# original "may be live-parked awaiting the user's promotion call"
# rationale is obsolete. A genuinely live-parked session stays protected:
# any real (non-orphan) holder blocks remediation, as do dirt outside the
# rescue allowlist and the grace window. Non-issue worktrees (agent-/wf-,
# status None) are never remediated.
REMEDIATION_ISSUE_STATUSES = frozenset({"completed", "archived", "awaiting_promotion"})

# Conservative orphaned-codex holder patterns, matched against a holder's
# cmdline. `codex app-server` workers and openai-codex plugin node
# processes routinely outlive their companion task and keep their spawn cwd
# (an issue worktree) pinned forever — 2026-06-10 disk-full incident: 11
# worktrees of weeks-completed tasks (issues 331/344/365/377/389/390/396/
# 398/405/406/448, ~10-15G each) were held ONLY by such processes. A holder
# matching NONE of these is a real holder (live happy/claude session,
# shell) and blocks any kill. Kills are exact-pid with a cmdline re-verify
# immediately before each signal — never pkill-by-name.
ORPHAN_HOLDER_PATTERNS = (
    re.compile(r"codex app-server"),
    re.compile(r"plugins/cache/openai-codex/"),
)

# Junk-dirty rescue allowlist: runtime-noise files whose uncommitted
# changes must not pin a 13G worktree forever. Entries ending in "/" are
# prefix matches; others are EXACT relative paths. Deliberately exactly
# these three families (the observed 2026-06-10 incident set: issue-507 /
# issue-477 = agent-memory dirt, issue-470 = pods_ephemeral.json,
# issue-489 = all three) — do NOT add figures/ or eval_results/ (a dirty
# eval JSON can be unique unmerged work; figure dirt is harder to prove
# safe). Anything outside this list keeps today's keep behavior.
RESCUE_DIRTY_ALLOWLIST = (
    ".claude/agent-memory/",
    "scripts/pods_ephemeral.json",
    "scripts/pods.conf",
)


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
    # #681 boot-time bind check: True when the data-disk bind at
    # .claude/worktrees is EXPECTED (EPS_WORKTREE_REQUIRE_BIND=1) but MISSING —
    # a half-mounted boot state where a worktree would silently land on `/`.
    # Escalate-only (loud warning), never a reap trigger.
    bind_missing: bool = False
    # Venv-reap arm (#912) — additive reporting fields, defaults keep every
    # existing consumer (cron log, main() stub tests) unchanged.
    venv_enabled: bool = True
    venv_idle_days_effective: float = DEFAULT_VENV_IDLE_DAYS
    venv_reaped: list[str] = field(default_factory=list)
    venv_candidates: list[str] = field(default_factory=list)  # dry-run would-reap
    venv_failed: list[str] = field(default_factory=list)
    # has-venv but kept, with reason (the silent no-venv majority excluded).
    venv_skipped: list[Decision] = field(default_factory=list)
    # du of .venv (apparent bytes; hardlink-overcounted like every size this
    # report prints). Measured pre-verify; dropped on a became-unsafe skip.
    # Leftover-sweep entries ("<name>/<tmp>") carry None (unmeasured).
    venv_bytes: dict[str, int | None] = field(default_factory=dict)


def _data_disk_bind_missing(wt_dir: Path) -> bool:
    """True when the data-disk bind at ``wt_dir`` is EXPECTED but MISSING (#681).

    Opt-in via ``EPS_WORKTREE_REQUIRE_BIND=1`` (default OFF — a no-op before the
    cutover lands the bind). The probe is ``findmnt --mountpoint <wt_dir>`` (NOT
    ``--target``: ``--target`` walks UP to the containing filesystem and returns
    rc=0 for ANY ordinary directory on a mounted fs, so a MISSING bind would go
    undetected — #681 round-2 Critical; ``--mountpoint`` succeeds ONLY when the
    path is itself a mount, the bind we are asserting). A test/pre-cutover seam
    ``EPS_WORKTREE_BIND_PROBE`` overrides it (``true`` force-pass / ``false``
    force-fail). A missing bind means a NEW worktree would silently land on the
    boot disk `/`, so the audit escalates it (loud warning) — it never deletes
    on this signal. Returns False when the check is not opted in or the bind is
    live."""
    if os.environ.get("EPS_WORKTREE_REQUIRE_BIND", "0") != "1":
        return False
    probe = os.environ.get("EPS_WORKTREE_BIND_PROBE", "").strip()
    if probe:
        rc = subprocess.run(["bash", "-c", probe], capture_output=True, check=False)
        return rc.returncode != 0
    rc = subprocess.run(
        ["findmnt", "--noheadings", "--mountpoint", str(wt_dir)],
        capture_output=True,
        check=False,
    )
    return rc.returncode != 0


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


def effective_venv_idle_days(idle_days: float, disk_pct: float, threshold_pct: float) -> float:
    """Pure pressure rule mirroring ``effective_grace_hours``: at/above the
    threshold the venv idle window tightens to ``PRESSURE_VENV_IDLE_DAYS``;
    an explicitly tighter ``idle_days`` is never loosened."""
    if disk_pct >= threshold_pct:
        return min(idle_days, PRESSURE_VENV_IDLE_DAYS)
    return idle_days


def _venv_idle_days() -> float:
    """Venv idle window in days, env-overridable (mirrors _pressure_threshold_pct)."""
    return float(os.environ.get("EPM_WORKTREE_VENV_IDLE_DAYS", str(DEFAULT_VENV_IDLE_DAYS)))


def _venv_reap_enabled() -> bool:
    """Kill switch for the venv arm (default ON; EPM_WORKTREE_VENV_REAP=0 disables)."""
    return os.environ.get("EPM_WORKTREE_VENV_REAP", "1") != "0"


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
        return Decision(name, False, _LIVE_PROCESS_REASON)
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


def should_reap_venv(
    name: str,
    *,
    worktree_kept: bool,  # the worktree itself survived should_remove
    has_venv: bool,  # (wt/.venv).is_dir() and not is_symlink()
    venv_is_symlink: bool,  # (wt/.venv).is_symlink()
    is_live: bool,  # name in the cwd/argv holder snapshot
    exe_in_venv: bool,  # name in the /proc/<pid>/exe venv snapshot
    newest_activity_ts: float | None,  # None = could not be established
    now: float,
    idle_days_required: float,
) -> Decision:
    """Pure venv-reap decision (unit-tested; #912). ``Decision.remove=True``
    means 'reap this worktree's .venv' — the worktree itself is NEVER removed
    by this arm. Gate order is strictly fail-toward-keep. Deliberately
    keep-reason-INDEPENDENT: every plain-keep class (human-named,
    status-not-reapable, tracked-changes) is eligible; the live-process class
    is blocked by ``is_live``/``exe_in_venv``; grace-window keeps are
    auto-excluded because a fresh root mtime makes ``newest_activity_ts``
    fresh (plan §11 D4)."""
    if not worktree_kept:
        return Decision(name, False, "venv: worktree itself is being removed")
    if not has_venv and not venv_is_symlink:
        return Decision(name, False, "venv: none present")
    if name.startswith("_"):
        # _task-main-pin (task_workflow._MANAGED_MAIN_WORKTREE_NAME) and any
        # future managed worktree — the leading underscore is its contract
        # (task_workflow.py, _MANAGED_MAIN_WORKTREE_NAME comment). An
        # anti-drift test pins the constant + its underscore prefix.
        return Decision(name, False, "venv: managed worktree (never touched)")
    if venv_is_symlink:
        return Decision(name, False, "venv: .venv is a symlink (never followed)")
    if is_live:
        return Decision(name, False, f"venv: {_LIVE_PROCESS_REASON}")
    if exe_in_venv:
        return Decision(name, False, "venv: a running process executes from this .venv")
    if newest_activity_ts is None:
        return Decision(name, False, "venv: idleness could not be established")
    idle_days = (now - newest_activity_ts) / 86400.0
    if idle_days < idle_days_required:
        return Decision(
            name,
            False,
            f"venv: active {idle_days:.1f}d ago (< {idle_days_required:g}d idle)",
        )
    return Decision(
        name,
        True,
        f"venv: idle {idle_days:.1f}d (>= {idle_days_required:g}d), reapable",
    )


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


def _is_orphan_cmdline(cmdline: str) -> bool:
    """True if a holder cmdline matches a known orphaned-codex pattern."""
    return any(p.search(cmdline) for p in ORPHAN_HOLDER_PATTERNS)


def classify_holders(holders: list[tuple[int, str]]) -> tuple[list[int], bool]:
    """Pure holder classification (unit-tested). ``holders`` is the
    ``[(pid, cmdline), ...]`` list pinning one worktree. Returns
    ``(orphan_pids, all_orphan)``: ``orphan_pids`` are the pids whose
    cmdline matches an ORPHAN_HOLDER_PATTERNS entry; ``all_orphan`` is True
    only when there is at least one holder AND every holder matches — a
    single non-matching holder (live happy/claude session, shell, editor)
    makes the worktree non-remediable. An empty holder list is NOT
    all-orphan (nothing to kill; the plain idle guards apply)."""
    orphan_pids = [pid for pid, cmd in holders if _is_orphan_cmdline(cmd)]
    return orphan_pids, bool(holders) and len(orphan_pids) == len(holders)


def _path_in_allowlist(path: str) -> bool:
    """True if a porcelain-reported relative path is rescue-allowlisted.
    Entries ending in "/" are prefix matches, others exact. Quoted paths
    (porcelain wraps exotic names in double quotes) fail closed."""
    if path.startswith('"'):
        return False
    for entry in RESCUE_DIRTY_ALLOWLIST:
        if entry.endswith("/"):
            if path.startswith(entry):
                return True
        elif path == entry:
            return True
    return False


def dirty_paths_within_allowlist(porcelain: str) -> tuple[list[str], bool]:
    """Pure dirty-set classification (unit-tested). Parses ``git status
    --porcelain`` output; untracked (``??``) lines are ignored, consistent
    with ``_has_tracked_changes``. Returns ``(tracked_dirty_paths,
    all_within)`` where ``all_within`` is True only when EVERY tracked
    dirty path sits inside RESCUE_DIRTY_ALLOWLIST. Fail-closed: a rename /
    copy line requires BOTH sides allowlisted, and an unparseable or
    quoted path counts as outside."""
    paths: list[str] = []
    all_within = True
    for line in porcelain.splitlines():
        if not line or line.startswith("??"):
            continue
        body = line[3:]
        if not body:
            all_within = False
            continue
        # Rename/copy porcelain lines carry "orig -> dest"; both must pass.
        for part in body.split(" -> "):
            paths.append(part)
            if not _path_in_allowlist(part):
                all_within = False
    return paths, all_within


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


def _live_worktree_holders(wt_root: str) -> dict[str, list[tuple[int, str]]]:
    """Worktree names currently referenced by any process: as a cwd
    (``/proc/<pid>/cwd``) or anywhere in argv (``/proc/<pid>/cmdline``).

    Returns name -> [(pid, cmdline), ...] so the report can say WHICH
    process pins a kept worktree (zombie-session triage) and the
    orphan-holder remediation can classify + signal exact pids. The
    liveness test itself is unchanged — ``name in holders`` is exactly the
    old set membership."""
    holders: dict[str, list[tuple[int, str]]] = {}
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
            holders.setdefault(name, []).append((int(pid), cmdline))
    return holders


def _format_holders(live: dict[str, list[tuple[int, str]]]) -> dict[str, list[str]]:
    """Display form of the structured holder map ("pid <pid>: <cmdline>"),
    used by the JSON summary + the kept-reason print loop."""
    return {
        name: [f"pid {pid}: {cmd[:120] or '?'}" for pid, cmd in entries]
        for name, entries in live.items()
    }


def _venv_exe_holders() -> set[str]:
    """Worktree names whose ``.venv*`` contains a RUNNING executable:
    ``/proc/<pid>/exe`` resolves under ``.claude/worktrees/<name>/.venv...``.
    Complements ``_live_worktree_holders`` (cwd/argv): a process exec'd FROM
    the venv interpreter whose cwd and argv reference neither path (e.g. a
    daemonized ``python`` with cwd ``/``) is invisible to the harvest but
    would crash if its ``sys.path``/stdlib under ``.venv`` vanished.
    Unreadable /proc entries (vanished pid, other-uid EPERM) are skipped —
    same posture as the existing harvest. ``_VENV_EXE_RE`` matches any
    ``.venv``-prefixed component so ``.venv.reap-tmp-*`` leftovers are also
    protected."""
    found: set[str] = set()
    marker = ".claude/worktrees/"
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        with contextlib.suppress(OSError):
            exe = os.readlink(f"/proc/{pid}/exe")
            idx = exe.find(marker)
            if idx != -1:
                m = _VENV_EXE_RE.match(exe[idx + len(marker) :])
                if m:
                    found.add(m.group(1))
    return found


def _venv_newest_activity_ts(child: Path) -> float | None:
    """Newest activity timestamp for the venv-reap idle test (#912): max of
    (a) worktree root dir mtime (the existing age signal, ``_classify``),
    (b) ``.venv`` dir mtime via lstat (venv creation / uv sync churn; a
        symlinked or missing ``.venv`` is skipped — the pure gate handles
        those cases on its own flags),
    (c) the worktree's git-admin activity — mtime of ``HEAD`` and ``index``
        under the gitdir the worktree's ``.git`` FILE points at
        (``gitdir: <path>``), which captures commits / checkouts /
        status-refreshes that touch no top-level entry (root dir mtime only
        moves on direct-child churn). Missing individual admin files are
        skipped, not an error.
    Returns ``None`` when the ``.git`` file is absent / unreadable /
    unparseable or any required signal cannot be read — the pure gate then
    fails toward keep."""
    try:
        signals = [child.stat().st_mtime]
        venv = child / _VENV_DIRNAME
        # A missing .venv is skipped (the pure gate short-circuits on has_venv).
        with contextlib.suppress(FileNotFoundError):
            signals.append(venv.lstat().st_mtime)
        gitfile = child / ".git"
        text = gitfile.read_text()
        if not text.startswith("gitdir:"):
            return None
        gitdir = Path(text.split(":", 1)[1].strip())
        if not gitdir.is_absolute():
            gitdir = child / gitdir
        for admin in ("HEAD", "index"):
            try:
                signals.append((gitdir / admin).stat().st_mtime)
            except FileNotFoundError:
                continue
        return max(signals)
    except OSError:
        return None


def _read_cmdline(pid: int) -> str | None:
    """NUL-joined cmdline of ``pid`` as one string, or None if the process
    is gone. A zombie reads as the empty string (kernel reports no argv)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return None


def _pid_running(pid: int) -> bool:
    """True if ``pid`` exists and is not a zombie (empty cmdline)."""
    return bool(_read_cmdline(pid))


def _kill_orphan_pids(pids: list[int]) -> tuple[list[int], list[int]]:
    """SIGTERM -> brief wait -> SIGKILL the given orphan-holder pids.

    PID-reuse guard: each pid's cmdline is re-read and re-verified against
    ORPHAN_HOLDER_PATTERNS IMMEDIATELY before every signal — a pid recycled
    by an unrelated process between the liveness harvest and the kill is
    skipped (and reported as a survivor, which keeps the worktree). Returns
    ``(gone, leftover)``: only ``gone`` pids are confirmed dead; any
    ``leftover`` (survivor or reuse-skip) blocks removal."""
    pending: list[int] = []
    skipped: list[int] = []
    for pid in pids:
        cmd = _read_cmdline(pid)
        if not cmd:
            continue  # already gone (or zombie) — nothing to signal
        if not _is_orphan_cmdline(cmd):
            skipped.append(pid)  # PID reused by a non-orphan — never signal
            continue
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
        pending.append(pid)
    deadline = time.time() + 5.0
    while pending and time.time() < deadline:
        time.sleep(0.2)
        pending = [pid for pid in pending if _pid_running(pid)]
    for pid in pending:
        cmd = _read_cmdline(pid)  # re-verify before escalating to SIGKILL
        if cmd and _is_orphan_cmdline(cmd):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    survivors = [pid for pid in pending if _pid_running(pid)]
    leftover = sorted(set(survivors) | set(skipped))
    gone = [pid for pid in pids if pid not in leftover]
    return gone, leftover


def _git_porcelain(wt_path: str) -> str | None:
    """``git status --porcelain`` output for one worktree, or None when it
    cannot be determined (callers treat None conservatively)."""
    try:
        out = subprocess.run(
            ["git", "-C", wt_path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _has_tracked_changes(wt_path: str) -> bool:
    """True if the worktree has uncommitted TRACKED changes. Untracked
    files (``??`` porcelain lines) are generated output and do NOT count."""
    out = _git_porcelain(wt_path)
    if out is None:
        # Cannot determine -> be conservative, treat as having changes.
        return True
    return any(line and not line.startswith("??") for line in out.splitlines())


def _rescue_dirty(wt_path: str, rescue_dir: Path, dirty_paths: list[str]) -> str | None:
    """Rescue-copy a junk-dirty worktree's tracked dirt before removal.

    Layout (the manual 2026-06-10 cleanup convention):
    ``<rescue_dir>/<original-relative-path>`` per dirty file plus a full
    ``git diff HEAD`` at ``<rescue_dir>/UNCOMMITTED.diff``. A deleted file
    has nothing to copy — the diff captures it. Returns None on success,
    else an error string; the caller then KEEPS the worktree (rescue
    strictly precedes removal, a failed rescue never loses data)."""
    try:
        rescue_dir.mkdir(parents=True, exist_ok=True)
        diff = subprocess.run(
            ["git", "-C", wt_path, "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if diff.returncode != 0:
            return f"git diff HEAD failed: {diff.stderr[:200]}"
        (rescue_dir / "UNCOMMITTED.diff").write_text(diff.stdout)
        for rel in dirty_paths:
            src = Path(wt_path) / rel
            if not src.is_file():
                continue  # deleted file — captured by the diff
            dst = rescue_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


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


def _issue_status_of(name: str, statuses: dict[int, str]) -> str | None:
    """Task status for an ``issue-<N>`` worktree name, else None."""
    m = _ISSUE_NAME_RE.match(name)
    return statuses.get(int(m.group(1))) if m else None


def _classify(
    child,
    statuses: dict[int, str],
    live: dict[str, list[tuple[int, str]]],
    grace_hours: float,
    now: float,
) -> Decision:
    """Full keep/remove decision for one worktree dir, including the
    (fresh) tracked-changes git call. ``statuses`` and ``live`` are the
    snapshots to decide against (liveness uses ``live``'s keys only)."""
    name = child.name
    status = _issue_status_of(name, statuses)
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


def _remediation_kind(
    name: str,
    decision: Decision,
    status: str | None,
    holders: list[tuple[int, str]],
    wt_path: str,
) -> tuple[str, str] | None:
    """Triage a KEPT worktree as actively remediable, or None.

    Returns ``(kind, detail)`` with kind in {"orphan-pinned", "junk-dirty"}:
      - "orphan-pinned": issue worktree of a remediation-eligible task
        whose ONLY holders are orphaned codex processes (--apply kills
        those exact pids, then re-derives every guard fresh).
      - "junk-dirty": issue worktree of a remediation-eligible task whose
        entire tracked dirty set sits inside RESCUE_DIRTY_ALLOWLIST
        (--apply rescue-copies it, then removes).
    Anything else — non-terminal / unknown status, a real (non-orphan)
    holder, dirt outside the allowlist, agent-/wf-/human-named worktrees —
    returns None: today's keep behavior, untouched.
    """
    if status not in REMEDIATION_ISSUE_STATUSES:
        return None
    if decision.reason == _LIVE_PROCESS_REASON:
        orphan_pids, all_orphan = classify_holders(holders)
        if all_orphan:
            return "orphan-pinned", f"held only by orphaned codex pid(s) {orphan_pids}"
        return None
    if _TRACKED_CHANGES_REASON in decision.reason:
        porcelain = _git_porcelain(wt_path)
        if porcelain is None:
            return None
        dirty, all_within = dirty_paths_within_allowlist(porcelain)
        if dirty and all_within:
            return "junk-dirty", f"{len(dirty)} allowlisted dirty file(s)"
        return None
    return None


def _execute_remediation(
    child, wt_root_rel: str, grace_hours: float, now: float, rescue_root: Path
) -> Decision:
    """APPLY-MODE ONLY: kill orphan holders and/or rescue allowlisted dirt
    for one remediation-eligible issue worktree, re-verifying EVERY guard against
    fresh state at each step. Returns the final decision; remove=True only
    when the worktree is provably idle post-remediation. This function's
    internal fresh checks REPLACE the loop's snapshot->remove re-check
    (re-running ``_classify`` after a rescue would re-keep the rescued but
    still-dirty tree). Never signals a pid that was not re-verified as an
    orphan-pattern match moments before; never proceeds to removal when a
    rescue failed."""
    name = child.name
    # FRESH snapshots — the loop's initial snapshot may be minutes old.
    status = _issue_status_of(name, _issue_statuses())
    if status not in REMEDIATION_ISSUE_STATUSES:
        return Decision(name, False, f"became unsafe mid-audit: status now {status}")
    live = _live_worktree_holders(wt_root_rel)
    killed_note = ""
    holders = live.get(name, [])
    if holders:
        orphan_pids, all_orphan = classify_holders(holders)
        if not all_orphan:
            return Decision(
                name,
                False,
                f"{_LIVE_PROCESS_REASON} (non-orphan holder appeared mid-audit)",
            )
        gone, leftover = _kill_orphan_pids(orphan_pids)
        if leftover:
            return Decision(
                name,
                False,
                f"orphan-pinned: kill incomplete (pid(s) {leftover} survived/skipped)",
            )
        killed_note = f"killed orphaned codex pid(s) {gone}; "
        # Re-verify liveness from a fresh harvest after the kill.
        if name in _live_worktree_holders(wt_root_rel):
            return Decision(
                name,
                False,
                f"became unsafe mid-audit: {_LIVE_PROCESS_REASON} after orphan kill",
            )
    # Re-derive the non-tracked guards fresh (a `task.py set-status` or a
    # recent write since the snapshot must still be honored).
    base = should_remove(
        name,
        status=status,
        is_live=False,
        age_hours=(now - child.stat().st_mtime) / 3600.0,
        has_tracked_changes=False,
        grace_hours=grace_hours,
    )
    if not base.remove:
        return Decision(name, False, f"became unsafe mid-audit: {base.reason}")
    # Junk-dirty rescue (fresh dirty read; rescue strictly precedes removal).
    porcelain = _git_porcelain(str(child))
    if porcelain is None:
        return Decision(name, False, f"{_TRACKED_CHANGES_REASON} (unreadable mid-remediation)")
    dirty, all_within = dirty_paths_within_allowlist(porcelain)
    if dirty:
        if not all_within:
            return Decision(name, False, f"{_TRACKED_CHANGES_REASON} (outside rescue allowlist)")
        err = _rescue_dirty(str(child), rescue_root / name, dirty)
        if err is not None:
            return Decision(name, False, f"junk-dirty rescue FAILED ({err})")
        return Decision(
            name,
            True,
            f"{killed_note}junk-dirty: rescued {len(dirty)} file(s) to "
            f"{rescue_root / name}; reapable (status={status})",
        )
    return Decision(name, True, f"{killed_note}idle and reapable (status={status})")


def _reap_venv(
    child: Path,
    wt_root_rel: str,
    *,
    exe_holders=_venv_exe_holders,
    live_holders=_live_worktree_holders,
) -> str | None:
    """Rename ``.venv`` aside, POST-RENAME holder re-check, then rmtree the
    renamed dir; restore the worktree root dir's ``(atime, mtime)`` afterwards.
    Returns None on success, else an error string.

    Rename-first makes the rename the SERIALIZATION POINT: a process can
    start from ``.venv/bin/python`` in the sub-second window AFTER the fresh
    pre-rename probes but BEFORE the rename; after the rename its
    ``/proc/<pid>/exe`` resolves under ``.venv.reap-tmp-*``, which
    ``_VENV_EXE_RE`` deliberately matches — so the post-rename
    ``exe_holders()`` + ``live_holders()`` re-check catches it, and on ANY
    holder we do NOT rmtree: the renamed tmp dir is left as a PROTECTED
    leftover (the next apply sweep's leftover pass re-attempts it, under the
    same holder gates). A partial delete can never leave a half-broken dir
    NAMED ``.venv`` — after the rename the worktree simply has no venv, and
    the next ``uv run`` recreates it from the shared cache. The mtime restore
    keeps this arm from resetting the MAIN removal arm's root-mtime idleness
    signal (deleting a direct child bumps the parent dir mtime). The holder
    probes are injectable keyword-default callables for fault-injection
    tests."""
    venv = child / _VENV_DIRNAME
    tmp = child / f"{_VENV_REAP_TMP_PREFIX}{os.getpid()}"
    name = child.name
    try:
        st = child.stat()
        os.rename(venv, tmp)
    except OSError as exc:
        return f"rename failed: {exc}"
    # POST-RENAME, PRE-RMTREE holder re-check — the race-closing gate.
    if name in exe_holders() or name in live_holders(wt_root_rel):
        with contextlib.suppress(OSError):
            os.utime(child, (st.st_atime, st.st_mtime))
        return f"holder appeared post-rename; left protected leftover {tmp.name}"
    try:
        shutil.rmtree(tmp)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.utime(child, (st.st_atime, st.st_mtime))
        return f"rmtree failed (leftover at {tmp.name}): {exc}"
    with contextlib.suppress(OSError):
        os.utime(child, (st.st_atime, st.st_mtime))
    return None


def _sweep_venv_leftovers(
    child: Path,
    res: AuditResult,
    wt_root_rel: str,
    *,
    exe_holders=_venv_exe_holders,
    live_holders=_live_worktree_holders,
) -> None:
    """APPLY-MODE ONLY: rmtree pre-existing ``.venv.reap-tmp-*`` leftovers
    (condemned by a prior apply run that crashed / was vetoed post-rename)
    under one KEPT worktree — but only when FRESH probes show no holder (a
    leftover whose interpreter is STILL running stays protected, matching
    the post-rename gate in ``_reap_venv``). Restores the worktree root
    ``(atime, mtime)`` after each delete, exactly as ``_reap_venv`` does.
    Never called in dry-run."""
    name = child.name
    leftovers = sorted(
        p
        for p in child.iterdir()
        if p.name.startswith(_VENV_REAP_TMP_PREFIX) and p.is_dir() and not p.is_symlink()
    )
    if not leftovers:
        return
    if name in exe_holders() or name in live_holders(wt_root_rel):
        res.venv_skipped.append(
            Decision(
                name,
                False,
                f"venv: {len(leftovers)} protected leftover(s) kept ({_LIVE_PROCESS_REASON})",
            )
        )
        return
    for tmp in leftovers:
        entry = f"{name}/{tmp.name}"
        try:
            st = child.stat()
        except OSError as exc:
            res.venv_skipped.append(
                Decision(name, False, f"venv: leftover reap FAILED ({tmp.name}): {exc}")
            )
            continue
        try:
            shutil.rmtree(tmp)
        except OSError as exc:
            with contextlib.suppress(OSError):
                os.utime(child, (st.st_atime, st.st_mtime))
            res.venv_skipped.append(
                Decision(name, False, f"venv: leftover reap FAILED ({tmp.name}): {exc}")
            )
            continue
        with contextlib.suppress(OSError):
            os.utime(child, (st.st_atime, st.st_mtime))
        res.venv_reaped.append(entry)
        res.venv_bytes[entry] = None  # unmeasured — condemned by a prior run


def _venv_arm(
    child: Path,
    live: dict[str, list[tuple[int, str]]],
    venv_exe: set[str],
    res: AuditResult,
    apply: bool,
    idle_days_required: float,
    wt_root_rel: str,
    now: float,
    *,
    exe_holders=_venv_exe_holders,
    live_holders=_live_worktree_holders,
) -> None:
    """Venv-reap arm for one KEPT worktree (#912). Dry-run: classify + record
    only — NEVER deletes. Apply: leftover sweep, then classification against
    the audit-start snapshots, then a fresh re-verify (liveness + exe +
    idleness re-derived against fresh state, mirroring the pre-removal fresh
    ``_classify``), then rename-aside + rmtree via ``_reap_venv``. Never
    touches the worktree itself. Fires only from the plain-KEEP branch of
    ``audit()`` — removal / remediation paths never call it."""
    if not res.venv_enabled:
        return
    name = child.name
    if child.is_symlink():
        # Root-symlink containment: a symlinked entry under .claude/worktrees/
        # (e.g. `manual-link -> ~/other-project`) passes the loop's is_dir()
        # (which follows symlinks); the venv arm must NEVER classify or delete
        # THROUGH it — the target's .venv is outside the physical worktree
        # area and not established as our regenerable artifact. Fail-closed
        # BEFORE any venv classification or leftover sweep.
        res.venv_skipped.append(
            Decision(name, False, "venv: worktree root is a symlink (never followed)")
        )
        return
    if apply:
        _sweep_venv_leftovers(
            child, res, wt_root_rel, exe_holders=exe_holders, live_holders=live_holders
        )
    venv = child / _VENV_DIRNAME
    d = should_reap_venv(
        name,
        worktree_kept=True,
        has_venv=venv.is_dir() and not venv.is_symlink(),
        venv_is_symlink=venv.is_symlink(),
        is_live=name in live,
        exe_in_venv=name in venv_exe,
        newest_activity_ts=_venv_newest_activity_ts(child),
        now=now,
        idle_days_required=idle_days_required,
    )
    if not d.remove:
        if d.reason != "venv: none present":  # silence the no-venv majority
            res.venv_skipped.append(d)
        return
    res.venv_bytes[name] = _worktree_size_bytes(str(venv))  # measure BEFORE
    if not apply:
        res.venv_candidates.append(name)
        return
    # APPLY: fresh re-verify immediately before the destructive call.
    fresh = should_reap_venv(
        name,
        worktree_kept=True,
        has_venv=venv.is_dir() and not venv.is_symlink(),
        venv_is_symlink=venv.is_symlink(),
        is_live=name in live_holders(wt_root_rel),
        exe_in_venv=name in exe_holders(),
        newest_activity_ts=_venv_newest_activity_ts(child),
        now=now,
        idle_days_required=idle_days_required,
    )
    if not fresh.remove:
        res.venv_skipped.append(
            Decision(name, False, f"venv: became unsafe mid-audit: {fresh.reason}")
        )
        res.venv_bytes.pop(name, None)  # drop the stale pre-verify size entry
        return
    err = _reap_venv(child, wt_root_rel, exe_holders=exe_holders, live_holders=live_holders)
    if err is None:
        res.venv_reaped.append(name)
    else:
        res.venv_failed.append(name)
        res.venv_skipped.append(Decision(name, False, f"venv: reap FAILED ({err})"))


def audit(apply: bool, grace_hours: float, now: float | None = None) -> AuditResult:
    now = time.time() if now is None else now
    root = repo_root()
    wt_root_rel = ".claude/worktrees/"
    wt_dir = root / ".claude" / "worktrees"
    res = AuditResult(grace_hours_effective=grace_hours)
    # #681 boot-time bind check: a missing-but-expected data-disk bind means a
    # NEW worktree would silently land on `/`. Escalate-only (set the flag +
    # loud report line in main()); never a reap trigger. Probed even when
    # wt_dir does not yet exist (a missing bind can manifest as a missing dir).
    res.bind_missing = _data_disk_bind_missing(wt_dir)
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
    res.live_holders = _format_holders(live)
    # Venv-reap arm (#912) snapshots: the /proc exe pass is skipped entirely
    # when the kill switch is off; the idle window tightens under the SAME
    # pressure predicate as the grace window (7d -> 2d).
    res.venv_enabled = _venv_reap_enabled()
    res.venv_idle_days_effective = effective_venv_idle_days(
        _venv_idle_days(), res.disk_pct, res.pressure_threshold_pct
    )
    venv_exe = _venv_exe_holders() if res.venv_enabled else set()
    rescue_root = (
        root
        / ".claude"
        / "cache"
        / f"worktree-rescue-{time.strftime('%Y-%m-%d', time.localtime(now))}"
    )

    # Clear any admin entries for worktree dirs that were already deleted.
    subprocess.run(["git", "worktree", "prune"], cwd=str(root), capture_output=True)

    for child in sorted(wt_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        res.sizes_bytes[name] = _worktree_size_bytes(str(child))
        decision = _classify(child, statuses, live, grace_hours, now)
        if not decision.remove:
            # Active-remediation triage for remediation-eligible issue worktrees:
            # orphan-pinned (only orphaned codex holders) or junk-dirty
            # (dirt confined to the rescue allowlist). Classified loudly in
            # EVERY report; remediated (kill / rescue) only under --apply.
            remediation = _remediation_kind(
                name,
                decision,
                _issue_status_of(name, statuses),
                live.get(name, []),
                str(child),
            )
            if remediation is None:
                res.kept.append(decision)
                # Venv-reap arm (#912): plain-KEEP branch ONLY. Removal /
                # remediation paths reap the whole worktree (venv included)
                # or fail toward keep this pass; the arm's own first guards
                # handle the kill switch + root-symlink containment.
                _venv_arm(
                    child,
                    live,
                    venv_exe,
                    res,
                    apply,
                    res.venv_idle_days_effective,
                    wt_root_rel,
                    now,
                )
                continue
            kind, detail = remediation
            if not apply:
                # Dry-run NEVER kills or rescues — report-only.
                res.kept.append(
                    Decision(name, False, f"{kind}: {detail} (--apply remediates + removes)")
                )
                continue
            final = _execute_remediation(child, wt_root_rel, grace_hours, now, rescue_root)
            if not final.remove:
                res.kept.append(final)
                continue
            # _execute_remediation's internal fresh checks ARE the
            # pre-removal re-derivation for this path (a _classify re-run
            # would re-keep a rescued-but-still-dirty tree).
            print(f"  * remediated {name}: {final.reason}", file=sys.stderr)
            if _git_remove(str(child)):
                res.removed.append(name)
            else:
                res.failed.append(name)
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


# Single-instance lock: the daily 09:47 cron audit, the watcher's low-disk
# `--apply` invocation (autonomous_session_watch.py `_vm_remediate_worktrees`),
# and manual runs can overlap; overlap degrades to benign per-item failures,
# but it is a needless race. Mirrors the watcher's `_acquire_lock` pattern.
_LOCK_PATH = Path.home() / ".task-workflow" / "worktree-audit.lock"


def acquire_single_instance_lock(lock_path: Path | None = None) -> object | None:
    """Hold a non-blocking flock for the lifetime of this audit run.

    Returns the held file object (the lock is released when the process
    exits — a context manager would close it and drop the lock early, so
    the bare open is deliberate), or ``None`` when another audit run holds
    it. Callers treat ``None`` as a clean skip (exit 0): both the cron
    wrapper and the watcher's fail-soft subprocess call must never classify
    the skip as a failure."""
    if lock_path is None:
        lock_path = _LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")  # noqa: SIM115 — held for process lifetime
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fd.close()
        return None
    return fd


def _print_venv_report(res: AuditResult, apply: bool) -> None:
    """Text-report block for the venv-reap arm (#912). Bytes are du-APPARENT:
    venv content is hardlink-deduplicated against ~/.cache/uv, so freed <=
    this (shared blocks free only when the uv cache itself is pruned)."""
    if not res.venv_enabled:
        print("worktree_audit venv-reap: DISABLED (EPM_WORKTREE_VENV_REAP=0)")
        return
    acted = res.venv_reaped if apply else res.venv_candidates
    venv_total = sum(res.venv_bytes.get(n) or 0 for n in acted)
    venv_verb = "reaped" if apply else "candidates (would reap)"
    print(
        f"worktree_audit venv-reap: idle window "
        f"{res.venv_idle_days_effective:g}d | {venv_verb} {len(acted)} "
        f"(~{_fmt_size(venv_total)} apparent; hardlinked vs uv cache — "
        f"freed <= this) | skipped {len(res.venv_skipped)} | "
        f"failed {len(res.venv_failed)}"
    )
    for entry in res.venv_reaped:
        label = "venv leftover reaped" if "/" in entry else "venv reaped"
        print(f"  - {label}: {entry} [{_fmt_size(res.venv_bytes.get(entry))} apparent]")
    for name in res.venv_candidates:
        print(f"  - venv would-reap: {name} [{_fmt_size(res.venv_bytes.get(name))} apparent]")
    for name in res.venv_failed:
        print(f"  ! venv reap FAILED: {name}")
    for d in res.venv_skipped:
        print(f"  . venv kept: {d.name} ({d.reason})")


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

    lock = acquire_single_instance_lock()
    if lock is None:
        if args.json:
            print(json.dumps({"skipped": "another worktree_audit run holds the lock"}))
        else:
            print("another worktree_audit run holds the lock; exiting")
        return 0

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
                    "bind_missing": res.bind_missing,
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
                    # Venv-reap arm (#912). bytes_apparent is du-apparent —
                    # hardlink-deduplicated vs ~/.cache/uv, so freed <= this.
                    "venv_reap": {
                        "enabled": res.venv_enabled,
                        "idle_days_effective": res.venv_idle_days_effective,
                        "reaped": res.venv_reaped,
                        "candidates": res.venv_candidates,
                        "failed": res.venv_failed,
                        "skipped": [{"name": d.name, "reason": d.reason} for d in res.venv_skipped],
                        "bytes_apparent": res.venv_bytes,
                    },
                }
            )
        )
    else:
        if res.bind_missing:
            print(
                "  !! BIND MISSING: the data-disk bind at .claude/worktrees is "
                "EXPECTED (EPS_WORKTREE_REQUIRE_BIND=1) but NOT live — a new worktree "
                "would silently land on the boot disk /. Check: findmnt "
                "--mountpoint .claude/worktrees ; sudo mount -a",
                file=sys.stderr,
            )
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
        _print_venv_report(res, args.apply)

    # Exit 2 when something was (or would be) removed, mirroring pod_audit;
    # the cron wrapper swallows it so cron does not email on every sweep.
    return 2 if (res.removed or res.failed) else 0


if __name__ == "__main__":
    sys.exit(main())
