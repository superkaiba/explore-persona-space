---
title: 'workflow-fix: watcher honors paused-takeover sentinel; stop resolves unregistered
  sessions'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4d4dab6de0e6
created_at: '2026-07-02T23:49:30Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #866: watcher orphan-respawn spawned
  a duplicate orchestrator during a deliberate master-session takeover (issue-866.json.paused-takeover-20260702
  rename made the takeover invisible to the registry-keyed respawn predicate); spawn_session.py
  stop --session-id failed on the daemon-untracked takeover session ({''success'':
  False}), making the prescribed stop-the-stale-session replacement protocol unexecutable.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #866 (emitting agent: /issue orchestrator).

## Goal

Make the autonomous-session watcher's orphan-respawn pass honor a deliberate session takeover (the `issue-<N>.json.paused-takeover-*` registration rename), and make `spawn_session.py stop --session-id` able to resolve/stop a live-but-unregistered session, so a takeover never collides with a watcher respawn.

## Workflow gap

- **Bug observed:** On #866 (2026-07-02), a master-plan session executed a deliberate takeover of a stalled autonomous session: it renamed the registration to `issue-866.json.paused-takeover-20260702` and dispatched its own composing subagent. The watcher's orphan-respawn pass, blind to the sentinel, saw "no live registered session + no fresh markers" and spawned a DUPLICATE `/issue 866 --auto` orchestrator 21 minutes into the takeover's active work — a #778-class double-writer averted only by manual process forensics (index.lock collision + a foreign commit were the first symptoms). `spawn_session.py stop --session-id <takeover-session>` also failed (`{'success': False}`) because the session was daemon-untracked, so the prescribed replacement protocol (stop the stale session first) was unexecutable.
- **Why it is a workflow gap:** the takeover rename convention and the watcher's orphan-respawn predicate are mutually blind: the rename REMOVES the registration signal the watcher keys on, converting a deliberate handoff into an orphan-respawn trigger. No workflow surface documents the takeover sentinel or teaches the watcher/stop tooling to respect it.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ autonomous_session_watch.py (orphan-respawn pass):
+   before respawning issue <N>, glob ~/.eps-autonomous/issue-<N>.json.paused-takeover-*
+   if a sentinel exists and is FRESH (mtime < EPS_TAKEOVER_TTL, default ~6h):
+     SKIP the respawn; log "issue <N> under deliberate takeover (sentinel <path>)"
+   a STALE sentinel (>TTL) is ignored (takeover presumed dead -> respawn as today)
+ spawn_session.py stop:
+   fall back to resolving a daemon-unknown session id via the claude-projects
+   transcript map / process ancestry, or return a structured "unknown session"
+   error naming the fallback (kill-by-pid recipe) instead of bare success:False
+ document the takeover sentinel convention in the watcher rule file
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`, `scripts/spawn_session.py`, `.claude/rules/background-automation.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'paused-takeover' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The sentinel-skip must FAIL OPEN: a missing/stale sentinel preserves today's respawn behavior (crash recovery must never be weakened).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, scripts/spawn_session.py, .claude/rules/background-automation.md
- fingerprint: 4d4dab6de0e6

Origin: task #866 collision, 2026-07-02 22:13-22:40Z (takeover sentinel issue-866.json.paused-takeover-20260702; watcher orphan-respawn at 22:34:15Z; coordination marker events v23 on #866).
