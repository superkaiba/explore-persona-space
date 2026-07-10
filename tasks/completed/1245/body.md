---
title: background + rc-file pattern for Step 10d lint gate
kind: infra
tags:
- wf-fix
- wf-fix-fp:f50a26ba2769
- daily-auto-filed
created_at: '2026-07-10T06:55:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): The Step 10d pre-push workflow-lint
  gate blocks run 2x no-flags lint legs (~4.5-6 min each) + parity + TG pytest legs
  (~9-12+ min total) in ONE foreground fenced Bash invocation, exceeding the 600s
  fo'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1211.

## Goal
Port the Step 9c background + rc-file pattern to the Step 10d gate blocks (or split the legs across invocations), keeping the SHA-bound verdict-file contract byte-unchanged.

## Workflow gap
- **Bug observed:** The Step 10d pre-push workflow-lint gate blocks run 2x no-flags lint legs (~4.5-6 min each) + parity + TG pytest legs (~9-12+ min total) in ONE foreground fenced Bash invocation, exceeding the 600s foreground Bash tool cap — a tool-timeout SIGKILL of the whole gate shell is a live kill family choom does not address. Verified on main: no run_in_background in the gate blocks.
- **Why it is a workflow gap:** The gate blocks carry no run_in_background + rc/verdict-file polling pattern, unlike the Step 9c 1b/1c/1d gates which were explicitly converted to background + rc-file for exactly this cap (#991/#996/#1129 class).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
run the shared gate block via Bash(run_in_background=true), writing /tmp/issue-<N>-lint-verdict.txt as today; the binding sites already consume the FILE, so only the invocation mode + a completion-read step change. Same for form (iii)'s baseline/gated leg pairs.

## Scope / surfaces
- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: ba907fdc95c8

parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see workflow-fix-on-bug § Recursion guard. source: prose-followup (Alternatives critic, #1211 Phase-2 round 1).

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: The Step 10d pre-push workflow-lint gate blocks run 2× no-flags lint legs (~4.5-6 min each) + parity + TG pytest legs (~9-12+ min total) in ONE foreground fenced Bash invocation, exceeding the 600s foreground Bash tool cap — a tool-timeout SIGKILL of the whole gate shell is a live kill family choom does not address.
why_workflow_gap: The gate blocks carry no run_in_background + rc/verdict-file polling pattern, unlike the Step 9c 1b/1c/1d gates which were explicitly converted to background + rc-file for exactly this cap (#991/#996/#1129 class).
proposed_change: Port the Step 9c background + rc-file pattern to the Step 10d gate blocks (or split the legs across invocations), keeping the SHA-bound verdict-file contract byte-unchanged.
diff_sketch: |
  + run the shared gate block via Bash(run_in_background=true), writing
  + /tmp/issue-<N>-lint-verdict.txt as today; the binding sites already
  + consume the FILE, so only the invocation mode + a completion-read
  + step change. Same for form (iii)'s baseline/gated leg pairs.
confidence: medium
related_task: #1211
<!-- /workflow-fix-candidate -->
