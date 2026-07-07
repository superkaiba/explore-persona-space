---
title: 'workflow-fix: reap/retry half-spawned session on daemon webhook timeout'
kind: infra
tags:
- wf-fix
- wf-fix-fp:547d7b5f5d26
created_at: '2026-07-04T03:54:10Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed candidate (chat 2026-07-03): spawn-issue --issue
  952 --auto failed twice with ''Session webhook timeout for PID <n>'' while the daemon
  had already forked live sessions, leaking two empty unmapped sessions; proposed:
  on webhook-timeout 500, stop or complete-registration of the half-spawned session
  and retry once with backoff.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised during chat-mode orchestration (spawning the #952 session; emitting agent: orchestrator).

## Goal

On a webhook-timeout HTTP 500 from the daemon /spawn-session, spawn_session.py stops (or retries the handshake for) the already-forked half-spawned session instead of leaving an idle unmapped session running.

## Workflow gap

- **Bug observed:** `spawn-issue --issue 952 --auto` returned HTTP 500 `Session webhook timeout for PID <n>` twice (PIDs 3698, 4687) while the Happy daemon had ALREADY forked live claude sessions for both attempts — leaking two live-but-empty unmapped sessions (no initial prompt delivered, no issue mapping, no crash-recovery registration) that required manual `spawn_session.py stop`. A third identical invocation succeeded, so the timeout is load-transient, but every timed-out attempt leaks a session.
- **Why it is a workflow gap:** the spawn helper reports failure while leaving daemon-side state half-created; nothing cleans it up (the watcher's idle-unmapped pass only reaps after ≥12h transcript idle), and the caller has no way to know a session leaked. The kickoff path is also missing any retry/backoff for this known-transient daemon error.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
In scripts/spawn_session.py spawn-issue/spawn-pm/spawn-campaign error path:
+ on HTTP 500 whose error matches 'Session webhook timeout for PID (\d+)':
+   1. re-query the daemon session list for a session whose pid == captured PID
+   2. if found: either (a) complete the registration (deliver initial prompt +
+      issue mapping + crash-recovery registration) if the daemon API allows, or
+      (b) stop the half-spawned session before raising, so a failed spawn
+      leaves no orphan
+   3. retry the spawn once with backoff (the error is load-transient; a
+      third manual attempt succeeded on 2026-07-03)
+ log the orphan-reap/retry decision to stderr so callers see it
```

## Scope / surfaces

- Primary target: `scripts/spawn_session.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'webhook timeout\|spawn-session' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan. Check whether `autonomous_session_watch.py`'s idle-unmapped pass should also learn a fast-path for zero-transcript sessions younger than 12h whose spawn was never registered.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/spawn_session.py
- fingerprint: 547d7b5f5d26

Raised from chat-mode orchestration 2026-07-03 while spawning the #952 autonomous session: two consecutive `spawn-issue --issue 952 --auto` calls failed with `Happy daemon /spawn-session returned HTTP 500: {'success': False, 'error': 'Session webhook timeout for PID 3698'}` (then PID 4687); `spawn_session.py list` showed both PIDs as live running sessions with no issue mapping; both were stopped manually; the third attempt succeeded (session cmr5tu41o1uagwc0uchgi7huy).
