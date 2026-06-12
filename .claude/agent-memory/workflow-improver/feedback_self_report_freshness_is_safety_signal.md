---
name: self-report-freshness-is-safety-signal
description: Never add an automation that periodically rewrites ~/.eps-autonomous/issue-progress/<N>.json or posts periodic markers — their AGE is the staleness signal the watcher's stalled-detector + session-reconcile passes act on
metadata:
  type: feedback
---

Any new automation that writes the per-issue self-report file
(`~/.eps-autonomous/issue-progress/<N>.json`, via `session_progress_report.py`)
or posts periodic `epm:*` markers on a schedule will SILENTLY DISABLE watcher
safety passes: `decide_session_stalled` requires the self-report AND the
newest non-watcher marker to BOTH be stale, and `_session_idle_signals`
(reconcile auto-stop) takes `min(ages)` over the same two signals. A
periodic writer keeps them permanently fresh.

**Why:** hit twice while implementing the 2026-06-12 anti-stall redesign:
(1) a per-pass watcher title refresh would have disabled stalled detection
entirely; (2) even a transition-keyed refresh churned against the
terminal-status GC for completed tasks (GC reaps the transition state file
each tick → pass re-detects a "transition" → rewrites the self-report every
pass → reconcile idle never reached). Fix shape: rewrite ONLY on a real
status transition (which already posts `epm:status-changed`, so no
staleness signal is masked), only for EXISTING self-reports (creating one
flips the stalled-detector's deliberate None-skip eligibility), and skip
completed/archived candidates entirely.

**How to apply:** before adding ANY writer of issue-progress files, markers,
or other `~/.eps-autonomous` state, grep the watcher for consumers of that
file's mtime/ts (`_self_report_age_seconds`, `_latest_nonwatcher_event_ts`,
`_session_idle_signals`, `decide_session_stalled`) and check the GC sweep
set (`_GC_TARGETS`) for reap-recreate loops. Related:
[[watcher-two-test-files]], [[ff-worktree-to-main-before-edit]].
