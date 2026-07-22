---
title: 'daily-fix: clarify autonomous-block reconcile re-park scope'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6bc6beb0dcc3
- daily-auto-filed
created_at: '2026-07-22T06:44:37Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): autonomous-block step-1
  reconcile scope ambiguous between re-evaluating the step-3 same partition and parking
  after step R'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1575 (emitting agent: Methodology critic, #1575 plan review round 1).

## Goal

Resolve and document ONE reading of the `/issue` SKILL.md autonomous-block step-1 reconcile scope: whether a re-park with `epm:follow-ups-autospawned v1` present re-evaluates the step-3 same partition, or parks after step R.

## Workflow gap

- **Bug observed:** the autonomous-block step-1 reconcile scope is ambiguous between two readings — (a) a re-park with `epm:follow-ups-autospawned v1` present re-evaluates the step-3 same partition (making the expensive 2-round cap + a final-slot moment reachable), or (b) it parks after step R (making the loop step-5 expensive cap-of-2 and the Step-10b-area summary lines ~:822-831 dead text).
- **Why it is a workflow gap:** concrete tension between SKILL.md :7843 (step-1 idempotency), :7969-7974 (RECONCILE 'only verifies filing'), :8288-8307 (step-5 caps), :822-831 (summary). #1575's landed fix is correct under EITHER reading (primary step-3 moment + defensive-parity step-4 clause), but the contract itself remains undecided — future editors will re-derive it.
- **Confidence (emitter):** medium; mechanizable: no (semantic contract choice).
- verified-at-filing: `sed -n '7843p;7969p;8288p;822,831p' .claude/skills/issue/SKILL.md` → all cited spans exist and carry follow-up-loop reconcile/cap/summary content (2026-07-22; presence claim, per-target hits confirmed); `task.py view 1575` events after 2026-07-21T07:36:43Z show the #1575 fix landing (epm:done 09:01Z) with no retraction of this candidate.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up) Pick one reading of the reconcile contract (the planner decides with the file open), then make the four cited spans consistent with it; if reading (b), delete or rescope the dead summary text at ~:822-831.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface (`grep -rln 'follow-ups-autospawned' .claude/ CLAUDE.md scripts/`) and reconcile every consumer of the chosen reading; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 6bc6beb0dcc3

- workflow_fix_target: .claude/skills/issue/SKILL.md

Verbatim parked candidate (task #1575 events, 2026-07-21T07:36:43Z): "parked — running under workflow_fix_target Provenance (recursion guard, workflow-fix-on-bug.md § Recursion guard); source: prose-followup (Methodology critic, #1575 plan review round 1). target_file: .claude/skills/issue/SKILL.md. proposed_change: clarify autonomous-block step 1 reconcile scope — whether a re-park with epm:follow-ups-autospawned v1 present re-evaluates the step-3 same partition (making the expensive 2-round cap + a final-slot moment reachable) or parks after step R (making the loop step-5 expensive cap-of-2 and the Step-10b-area summary lines ~:822-831 dead text). Concrete tension between SKILL.md :7843 (step-1 idempotency), :7969-7974 (RECONCILE 'only verifies filing'), :8288-8307 (step-5 caps), :822-831 (summary). #1575's fix is correct under EITHER reading (primary step-3 moment + defensive-parity step-4 clause). confidence: medium. related_task: #1575. mechanizable: no (semantic contract choice)."
