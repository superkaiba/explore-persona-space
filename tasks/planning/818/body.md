---
title: 'workflow-fix: watcher misses zombie wrappers on orphaned tmux TTYs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b32dfc2dde55
created_at: '2026-07-01T22:19:04Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation during interactive tmux/Happy cleanup 2026-07-01:
  31 zombie Happy wrappers (claude_local_launcher.cjs child, 0 CPU, 1-7 days, ~4.8GB
  RSS) on an orphaned tmux server (socket deleted) were never reaped by autonomous_session_watch.py''s
  zombie-wrapper pass because their pts looked like live user TTYs.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation during an interactive tmux/Happy session cleanup (2026-07-01, chat-mode — no parent task).

## Goal

Make the autonomous-session watcher's zombie-wrapper pass reap Happy wrappers whose only descendant is a dead-weight `claude_local_launcher.cjs` (no `claude` process below it, ~0 accumulated CPU), including when the wrapper holds a pts TTY belonging to an orphaned tmux server whose socket no longer exists.

## Workflow gap

- **Bug observed:** 31 zombie Happy wrapper sessions (no inner `claude` process; child was a `claude_local_launcher.cjs` node process with 00:00:00 accumulated CPU over 1–7 days; no issue mapping) accumulated ~4.8 GB RSS and were never reaped by the 10-minute watcher. Their pts TTYs were panes of a 99-day-old orphaned tmux server (pid 2842652, session `sweep-bash_claude_n100_v2`) whose socket file had been deleted from `/tmp/tmux-1001/` — no client could ever attach again, so the "live user TTY" the watcher protects was unreachable by construction.
- **Why it is a workflow gap:** `scripts/autonomous_session_watch.py`'s zombie-wrapper pass ("auto-stops wrapper sessions with no inner Claude process ≥2h") fails toward keep when the wrapper holds a TTY, but does not distinguish a genuinely live user TTY from a pts held by an orphaned tmux server with no socket. It also appears not to classify a wrapper as zombie when a launcher child exists but no `claude` descendant does. Manual cleanup had to do what the watcher exists to do.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/autonomous_session_watch.py (zombie-wrapper pass):
+ def _tty_is_live_user_tty(pts):
+     # a pts whose controlling tmux server has no socket under
+     # /tmp/tmux-<uid>/ (or whose server pid's cmdline is tmux but
+     # socket path is gone) is NOT a live user TTY
+ zombie predicate: wrapper has no descendant with comm == "claude"
+   (a claude_local_launcher.cjs child with ~0 cputime does NOT count
+    as an inner Claude), sustained >= 2h
+ TTY guard: only spare the wrapper when _tty_is_live_user_tty(...)
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the zombie-wrapper predicate before editing
  (`grep -rln 'zombie' scripts/ .claude/ CLAUDE.md`) and update every hit;
  list them in the plan (CLAUDE.md § background-automation summary +
  `.claude/rules/background-automation.md` likely describe the pass).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- Fail toward keep on any ambiguous TTY/liveness signal — the new predicate must
  only widen reaping for the provably-orphaned case (no socket for the owning
  tmux server; no `claude` descendant; ~0 launcher CPU).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: b32dfc2dde55

Orchestrator observation (verbatim summary): during a manual cleanup, 31 Happy
wrappers with `claude_local_launcher.cjs` children (0 CPU, 1–7 days old, no issue
mapping, ~4.8 GB RSS total) were found parked on panes of an orphaned tmux server
(pid 2842652, socket deleted); the watcher's zombie-wrapper pass never reaped them
because the pts TTYs looked like live user TTYs. A 32nd process on the same server
was a stale duplicate My Goat Telegram-channel claude, and a separate 54-day-old
init-parented launcher (`--resume 83ac2dd5`, unknown to the Happy daemon) had also
escaped all passes.
