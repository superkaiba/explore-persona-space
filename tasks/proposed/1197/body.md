---
title: 'workflow-fix: background Step 1d pristine compare (600s cap)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0e0bdeeda488
- daily-auto-filed
created_at: '2026-07-09T07:00:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Step 1d compare (incl.
  pristine single-file oracle runs) is prescribed as a short/bounded FOREGROUND call,
  but a healthy pristine run of a SLOW_TESTS file legitimately takes ~640-1950s —
  past the 600s foreground Bash cap — shifting an in-process exit 2 into a tool-layer
  kill with COMPARE_OUT lost.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08, slice 5) from a
candidate parked on task #1129 by a recursion-guarded workflow-fix session.

## Goal

Background the Step 9c Step-1d compare invocation (run_in_background + rc-file dataflow, mirroring the Step-1b gate-background precedent) whenever --run-pristine is passed — or at minimum when the pristine bucket may contain a SLOW_TESTS file — so a legitimate ~640-1950s pristine oracle run cannot be killed by the 600s foreground Bash tool cap.

## Workflow gap

- **Bug observed:** The Step 1d invocation-envelope prescription ('separate, short/bounded foreground call') contradicts the compare's own legitimate runtime (post-#1129 derived bound ~640-1950s per SLOW_TESTS pristine file; 5 pristine files x 600s could already bust it; the #1098 1200s recovery already needed >600s).
- **Why it is a workflow gap:** A tool-layer kill loses COMPARE_OUT entirely, converting a classifiable in-process exit-2 into an unexplained gate failure.
- **Confidence (emitter):** medium (consistency-checker prose finding on #1129)
- **Triage evidence (2026-07-08):** NOT fixed on main: SKILL.md ~L7621 still prescribes 'Step 1d compare — including its pristine single-file oracle runs — executes in a separate, short/bounded foreground call and is accepted unprotected'; no run_in_background/rc-file pattern exists for the compare (the 1b gate has the background precedent). Candidate parked earlier today; completed #1053/#1052/etc target SKILL.md for different bugs — not dupes. No retraction on #1129.

## Proposed change (candidate diff sketch — refine in planning)

```
+ Step 1d (with --run-pristine): run scripts/step9c_baseline.py compare as a
+ background Bash call; persist COMPARE_OUT + COMPARE_RC to
+ /tmp/step9c-compare-issue-<N>.{json,rc}; consume the FILES (1b rc-file
+ pattern); keep the foreground form for the no-pristine fast path.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: parked candidate on task #1129 at 2026-07-08T08:33:03Z

Verbatim parked note:

source: consistency-checker prose finding on #1129. target_file: .claude/skills/issue/SKILL.md. bug_observed: Step 1d compare (incl. pristine single-file oracle runs) is prescribed as a short/bounded FOREGROUND call, but a healthy pristine run of a SLOW_TESTS file legitimately takes ~640-1950s (post-#1129 derived bound), past the 600s foreground Bash tool cap — the incident case can shift from in-process exit 2 to a tool-layer kill with COMPARE_OUT lost. why_workflow_gap: the invocation-envelope prescription contradicts the compare's own legitimate runtime; pre-existing (5 pristine files x 600s = 3000s could already bust it; the #1098 1200s recovery already needed >600s). proposed_change: background the Step 1d compare (run_in_background + rc-file pattern, mirroring the 1b gate-background precedent) when --run-pristine is passed, or at least when the pristine bucket may contain a SLOW_TESTS file. confidence: medium. related_task: #1129. routed: parked — this session's task is itself a workflow-fix (wf-fix tag; recursion-guard spirit: log + notify, never auto-route). Pickup: /daily route-2 sweep or PM triage.
