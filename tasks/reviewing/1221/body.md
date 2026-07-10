---
title: forward rc==0 child stderr in remaining subprocess callers
kind: infra
tags:
- wf-fix
- wf-fix-fp:4fa97def4110
- daily-auto-filed
created_at: '2026-07-10T06:53:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): task.py/spawn_session.py
  children exiting rc==0 with stderr warnings have that stderr silently discarded
  in callers outside the two #1150 target files; file_infra_task spawn hop forwards
  stderr only on rc!=0'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1150.

## Goal
Adopt the #1130 _forward_marker_child_stderr helper (or equivalent) in the remaining mutating-subprocess callers: poll_pipeline.py, tick_triage.py, post_step_completed.py task.py children, and file_infra_task.py's rc==0 spawn_session.py hop.

## Workflow gap
- **Bug observed:** task.py / spawn_session.py child subprocesses that exit rc==0 with a stderr warning (e.g. the post-marker deferred-commit ERROR) have that stderr silently discarded in callers outside the two #1150 target files; file_infra_task.py forwards the spawn child's stderr only on rc!=0 (verified on main 2026-07-09: rc==0 branch at scripts/file_infra_task.py ~373-381 prints stdout first line only).
- **Why it is a workflow gap:** The #1130/#1150 fix closed the swallow at two call sites; the same rc==0-with-stderr contract is produced by every task.py child, so every other subprocess caller re-opens the class.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
One-shot `grep -rln` for mutating task.py/spawn_session.py subprocess callers (poll_pipeline.py, tick_triage.py, post_step_completed.py, file_infra_task.py spawn hop); thread `_forward_marker_child_stderr(child, label)` after each rc==0 return. Helper currently adopted only in autonomous_session_watch.py + file_infra_task.py (task.py-new leg).

## Scope / surfaces
- Primary target: `scripts/poll_pipeline.py, scripts/tick_triage.py, scripts/post_step_completed.py, scripts/file_infra_task.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/poll_pipeline.py, scripts/tick_triage.py, scripts/post_step_completed.py, scripts/file_infra_task.py
- fingerprint: d59a387b0261

Merged from TWO sibling parked candidates on #1150 (2026-07-09T08:31:19Z prose-followup, Alternatives critic: sibling rc==0-stderr-forwarding audit of other task.py subprocess callers; 2026-07-09T08:47:53Z prose-followup, code-reviewer round 1: file_infra_task.py:323-area spawn_session.py child's rc==0 stderr still discarded — second-hop swallow, distinct from #1150's task.py-child scope).
