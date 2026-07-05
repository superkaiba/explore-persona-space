---
title: 'workflow-fix: watcher alert for on_hold task with RUNNING pod'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:9596c6f95357
created_at: '2026-07-04T07:11:59Z'
has_clean_result: false
origin_prompt: 'parked — running under workflow_fix_target recursion guard (see .claude/rules/workflow-fix-on-bug.md
  § Recursion guard). Two reviewer-surfaced candidates for the next human/orchestrator
  pass, NOT auto-routed: (1) target_file: scripts/autonomous_session_watch.py — watcher-side
  defense-in-depth for prose holds: before an orphan-respawn, grep the task''s recent
  markers for a ''USER PAUSE''-shaped note and escalate instead of respawning (source:
  Alternatives critic, mechanizable: yes; the #919 doc routing defines the canonical
  note format a parser would key on). (2) target_file: scripts/autonomous_session_watch.py,
  tests/test_autonomous_session_watch.py — pod-safety visibility for the on_hold+RUNNING-pod
  state: add on_hold to an escaped-pod alert/stop class (or a test pinning that _status_class(''on_hold'')
  is deliberately ''other''), so a crash inside the pause window is loud at the watcher
  level too (source: Codex methodology critic Must-Fix, mitigated in #919 by the teardown-first
  ORDER; the code-level alert remains defense-in-depth). Both are watcher CODE changes
  — outside #919''s doc-only must-ask fence, hence parked rather than folded in.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

add on_hold to an escaped-pod alert/stop class (or a test pinning that _status_class('on_hold') is deliberately 'other') so a crash inside the pause window is loud at the watcher level

## Workflow gap

- **Bug observed:** the on_hold+RUNNING-pod state is invisible to the watcher's escaped-pod alert classes; a crash inside the #919 pause window would bill silently
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

add on_hold to an escaped-pod alert/stop class (or a test pinning that _status_class('on_hold') is deliberately 'other') so a crash inside the pause window is loud at the watcher level

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py, tests/test_autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, tests/test_autonomous_session_watch.py
- fingerprint: 9596c6f95357

parked — running under workflow_fix_target recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard). Two reviewer-surfaced candidates for the next human/orchestrator pass, NOT auto-routed: (1) target_file: scripts/autonomous_session_watch.py — watcher-side defense-in-depth for prose holds: before an orphan-respawn, grep the task's recent markers for a 'USER PAUSE'-shaped note and escalate instead of respawning (source: Alternatives critic, mechanizable: yes; the #919 doc routing defines the canonical note format a parser would key on). (2) target_file: scripts/autonomous_session_watch.py, tests/test_autonomous_session_watch.py — pod-safety visibility for the on_hold+RUNNING-pod state: add on_hold to an escaped-pod alert/stop class (or a test pinning that _status_class('on_hold') is deliberately 'other'), so a crash inside the pause window is loud at the watcher level too (source: Codex methodology critic Must-Fix, mitigated in #919 by the teardown-first ORDER; the code-level alert remains defense-in-depth). Both are watcher CODE changes — outside #919's doc-only must-ask fence, hence parked rather than folded in.
