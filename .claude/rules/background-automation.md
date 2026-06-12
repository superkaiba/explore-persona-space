---
description: Background cron automations — stale-pod audit, stale-worktree sweep, autonomous-session watcher (crash-recovery, pod-safety, reconcile + zombie-wrapper + idle-unmapped passes) — full predicates, env-var overrides, and incident history (loads when you touch the audit / watcher scripts)
paths:
  - "scripts/worktree_audit.py"
  - "scripts/cron_worktree_audit.sh"
  - "scripts/autonomous_session_watch.py"
  - "scripts/cron_autonomous_session_watch.sh"
  - "scripts/pod_audit.py"
  - "scripts/cron_pod_audit.sh"
  - "scripts/codex_task.py"
---

# Background cron automations

CLAUDE.md § Pods carries the always-on one-paragraph summary (which crons
exist + their user-visible effects); this file is the full predicate spec.

## Stale-pod audit (09:37 daily + on `pod.py provision`)

Auto-terminate pods EXITED >24h — EXEMPT when the owning task carries the
`keep-running` tag, reported as `kept-exited` instead.

## Stale-worktree sweep (09:47 daily, `worktree_audit.py --apply`)

Reaps idle auto-generated worktrees under `.claude/worktrees/` — removed only
when not held by a live process, not an `issue-<N>` with a non-terminal
status, older than a 6h grace window (tightened to 1h when the filesystem
holding the worktrees is ≥90% full — disk-pressure mode, threshold via
`EPM_WORKTREE_DISK_PRESSURE_PCT`; the audit always reports disk usage +
per-worktree sizes), and with no uncommitted tracked changes. Human-named
worktrees are never touched (`issue-<N>-<suffix>` follow-up worktrees ARE in
sweep scope as of 2026-06-12, mapped to issue N for the status lookup).

For done-and-merged (`completed`/`archived`/`awaiting_promotion` — the latter
added 2026-06-12: the worktree auto-merged to main at the Step 9b transition
and the reconcile pass auto-stops parked sessions; any real non-orphan holder
still blocks) issue worktrees, `--apply` additionally remediates two
false-keep classes (2026-06-10 disk-full incident): kills orphaned codex
`app-server` holder pids (exact-pid, cmdline re-verified immediately before
each signal; never when any real holder is present) and rescue-copies
allowlisted runtime-noise dirt (agent memories, `pods.conf`,
`pods_ephemeral.json`) to `.claude/cache/worktree-rescue-<date>/` BEFORE
removal; dry-run only classifies, never kills or rescues.

`codex_task.py` complements this by pinning every codex-companion dispatch to
the main checkout root (`DISPATCH_ROOT`), so new codex workers never root
themselves in a worktree.

## Autonomous-session watcher (every 10 min, `3-59/10 * * * *`, `autonomous_session_watch.py`)

Passes: crash-recovery respawn, pod-safety reconciliation, stalled-session
detector, orphan-file sweep, and three session reapers — the session-vs-status
reconcile pass, the zombie-wrapper pass, and the idle-unmapped pass.

**Reconcile pass (auto-stop of parked sessions).** An issue-mapped session
whose task is parked/terminal (`awaiting_promotion`/`completed`/`archived`)
is AUTO-STOPPED after ≥2 consecutive checks once ALL hold: no live follow-up
inferred from events.jsonl (latest
`epm:run-launched`/`epm:followup-scope`/`epm:free-analysis-followup-run`
OLDER than the latest done-transition
`epm:promoted`/`epm:status-changed`/`epm:pod-terminated`/`epm:step-completed`),
every non-watcher marker + self-report idle > ~2h
(`EPM_SESSION_RECONCILE_IDLE_S`), no RUNNING `pod-<N>`, and no `keep-running`
tag (auto-stop default per user request 2026-06-10 — "stop the happy sessions
once they reach awaiting promotion" — superseding the same-day alert-only
decision; `EPM_SESSION_RECONCILE_AUTOSTOP=0` reverts to alert-only); sessions
of tasks at any other status (ACTIVE, `followups_running`, `blocked`), the PM
session, and unmapped chat sessions are never touched by this pass.

**Zombie-wrapper pass.** A live EPS session whose process tree has carried NO
inner Claude process for ≥2 consecutive checks AND ≥2h
(`EPM_ZOMBIE_WRAPPER_GRACE_S`) is auto-stopped REGARDLESS of issue mapping
(the 2026-06-11 class: 25 unmapped finished-issue sessions showed as
"running" indefinitely); never touched: the PM session (registered via
`spawn_session.py register-pm` / `spawn-pm` / the `/pm` bootstrap),
non-EPS-cwd sessions, and issue-mapped sessions at
active/`blocked`/`plan_pending` statuses; `EPM_ZOMBIE_WRAPPER_REAP=0` reverts
to alert-only.

**Idle-unmapped pass.** A third session reaper — auto-stops UNMAPPED EPS-cwd
sessions (no registry entry, no `issue-<N>` worktree cwd) whose resolved
Claude transcript has been idle ≥12h (`EPM_UNMAPPED_IDLE_REAP_S`) on ≥2
consecutive checks — the class BOTH other session reapers structurally
exclude (live-but-idle inner Claude + unmapped; the 2026-06-12 VM-lag
incident: 25 such sessions idle 19-43h held ~23 GB RSS); never touched: the
PM session, non-EPS cwds, issue-mapped sessions, wrappers holding a
controlling TTY (a live user terminal), and sessions whose transcript cannot
be resolved (a missing idleness signal FAILS TOWARD KEEP);
`EPM_UNMAPPED_IDLE_REAP=0` reverts to alert-only; records land in
`~/.eps-autonomous/idle-unmapped-events.jsonl` (an unmapped session has no
task to carry a marker).
