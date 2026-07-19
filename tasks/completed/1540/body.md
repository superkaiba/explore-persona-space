---
title: 'workflow-fix: reconcile planned cells per lane completion'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6e7313c90a81
- daily-auto-filed
created_at: '2026-07-19T07:07:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): #1481''s sycophancy bare
  arm was never re-swept and surfaced only when the user asked hours later; planned-vs-actual
  coverage reconciliation runs only at clean-result time, not per lane completion
  (c3-P18).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-18 /daily
transcript problem sweep (c3-P18). Route-2 filing.

## Goal

Run planned-vs-realized cell/condition reconciliation (After-Every-Experiment
item 8) at EACH lane completion DURING the run, not only at clean-result
time, so a planned arm that never got swept surfaces before the user has to
ask.

## Workflow gap

- **Bug observed:** #1481's sycophancy bare arm was never re-swept; only
  Thomas's 07:37 question surfaced it ("bare never banded and was never
  re-swept"), after which he had to push "run it in parallel now."
- **Why it is a workflow gap:** planned-vs-actual coverage reconciliation is
  specified only at clean-result time (After-Every-Experiment item 8 /
  `verify_task_body.py` check 11b). During a multi-lane run, a lane that
  completes without covering a planned cell is invisible until the terminal
  analysis — so a whole planned arm can silently go unrun mid-flight.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'lane completion\|per-lane\|each lane' .claude/skills/issue/SKILL.md` → 1 hit (line ~3498, the cluster-poll `per-lane attempts` ladder — NOT a planned-vs-realized reconciliation); the only planned-vs-actual coverage reference (line ~6731) is at the clean-result/analysis gate. No per-lane-completion reconciliation duty exists (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
# SKILL.md Step 7/8 (run monitoring / lane completion): add a per-lane-
# completion reconciliation duty —
+ On each lane/phase completion, reconcile the lane's REALIZED cells against
+ the plan's declared cells for that lane; a planned cell the completed lane
+ did not cover posts an epm:progress note naming the missing cell (so it is
+ visible before the terminal analysis), and the orchestrator decides
+ re-sweep vs documented-drop per the autonomous/interactive rules.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Locate the lane-completion / phase-completion handling in Step 7/8; add the
  reconciliation duty there, cross-referencing After-Every-Experiment item 8.

## Constraints / invariants

- Workflow-surface only. Do not double-run the terminal clean-result
  reconciliation; this is an EARLIER, per-lane surfacing of the same check.
- `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 48b8281e2c40

Surfaced problem (c3-P18): #1481's bare sycophancy arm never banded / never
re-swept; surfaced only by the user's question hours after the other lanes
completed.
