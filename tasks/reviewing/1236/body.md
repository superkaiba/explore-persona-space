---
title: sync batch-datagen memory L13 to pid-file launch contract
kind: infra
tags:
- wf-fix
- wf-fix-fp:ccdaeefb90ed
- daily-auto-filed
created_at: '2026-07-10T06:54:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): L13 (''How to apply'' step
  1) prescribes a nohup launch recipe lacking setsid and with an unspecified pid-capture
  method — after #1200 it is the last experimenter memory prescribing a non-trio launch
  (v'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1200.

## Goal
Add setsid to the L13 launch recipe and point the pid capture at the launcher-internal `echo $$ >` / atomic tmp+mv contract forms.

## Workflow gap
- **Bug observed:** L13 ('How to apply' step 1) prescribes a nohup launch recipe lacking setsid and with an unspecified pid-capture method — after #1200 it is the last experimenter memory prescribing a non-trio launch (verified still present on main). #1200's acceptance grep did not catch it (no `echo $! >` literal); pre-existing and outside #1200's fenced two-file scope.
- **Why it is a workflow gap:** Agent memories are always-loaded workflow surface steering the experimenter; a memory prescribing a non-setsid launch recipe will be copied verbatim, re-introducing the SIGHUP-reap class the pid-file launch contract (pod-side-reporting.md, #1070) closed.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
- 1. Launch with full `nohup ... < /dev/null > log 2>&1 &`, capture the PID, ...
  + 1. Launch with full `setsid nohup ... < /dev/null > log 2>&1 &`, capture the PID per the pid-file launch contract (launcher-internal `echo $$ >` pre-exec, or atomic tmp+mv), ...

## Scope / surfaces
- Primary target: `.claude/agent-memory/experimenter/feedback_datagen_anthropic_batch_long_running.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/agent-memory/experimenter/feedback_datagen_anthropic_batch_long_running.md
- fingerprint: 35e9ec429bb9

Workflow-fix candidate received from Alternatives critic (Phase 2, plan v1 review) — PARKED, NOT ROUTED: this session runs under the workflow-fix recursion guard (task #1200 carries a workflow_fix_target: Provenance line; see .claude/rules/workflow-fix-on-bug.md § Recursion guard). routed: parked: EPM_WORKFLOW_FIX_SESSION. Next human/orchestrator pass may file it. source: prose-followup

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agent-memory/experimenter/feedback_datagen_anthropic_batch_long_running.md
bug_observed: L13 PRESCRIBES a launch recipe ("Launch with full `nohup ... < /dev/null > log 2>&1 &`, capture the PID") lacking `setsid` and with an unspecified pid-capture method — after #1200 lands it is the last experimenter memory prescribing a non-trio launch. No `echo $! >` literal, so #1200's acceptance grep does not catch it; pre-existing and outside #1200's fenced two-file scope.
why_workflow_gap: Agent memories are always-loaded workflow surface steering the experimenter; a memory prescribing a non-setsid launch recipe will be copied verbatim, re-introducing the SIGHUP-reap class the pid-file launch contract (pod-side-reporting.md, #1070) closed.
proposed_change: Add `setsid` to the L13 launch recipe and point the pid capture at the launcher-internal `echo $$ >` / atomic tmp+mv contract forms.
diff_sketch: |
  - 1. Launch with full `nohup ... < /dev/null > log 2>&1 &`, capture the PID, ...
  + 1. Launch with full `setsid nohup ... < /dev/null > log 2>&1 &`, capture the PID per the pid-file launch contract (launcher-internal `echo $$ >` pre-exec, or atomic tmp+mv), ...
confidence: medium
related_task: #1200
<!-- /workflow-fix-candidate -->

