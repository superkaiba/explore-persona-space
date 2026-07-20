---
title: 'daily-fix: check (g) parity probe per adapter recipe-class'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ed0462b01f10
- daily-auto-filed
created_at: '2026-07-20T06:46:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): check (g) singular parity
  probe ambiguous under multi-recipe-class reuse'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-19 parked-candidate sweep (Step C) from a workflow-fix candidate parked on task #1543 (emitting agent: Alternatives critic finding 5, plan review round 1; parked under the recursion guard).

## Goal

Amend `.claude/rules/artifact-reuse.md` check (g) so the rsLoRA apply-and-read parity probe runs once per reused adapter RECIPE-CLASS, not one global probe.

## Workflow gap

- **Bug observed:** check (g)'s "1-adapter apply-and-read parity probe" (singular) has the same one-global-probe reading as the pre-#1543 (h)(iv) text when a plan reuses adapters of MULTIPLE recipe classes.
- **Why it is a workflow gap:** #1543 closed exactly this ambiguity for the (h)(iv) staging probe; check (g) carries the sibling ambiguity — a single global parity probe can miss a recipe-class whose scaling application diverges (#601 class).
- **Confidence (emitter):** low (filed per the standing any-confidence directive; the spawned planner may deflect with a reasoned no-change report)
- verified-at-filing: `grep -n "1-adapter apply-and-read parity probe" .claude/rules/artifact-reuse.md` → 1 hit (line 154), phrase still singular; context read: no per-recipe-class qualifier present. The #1543 merge (041383903d, PR #1301) amended (h)(iv) + 4 sibling phrase sites only — check (g) untouched (2026-07-19).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up: "probe once per reused adapter recipe-class in check (g)")

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'apply-and-read parity' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: 8bb703d4e936

Verbatim parked candidate (epm:workflow-fix-candidate on #1543, 2026-07-19T07:39:55Z): "source: prose-followup (Alternatives critic finding 5, plan review round 1). target_file: .claude/rules/artifact-reuse.md. bug_observed: check (g)'s '1-adapter apply-and-read parity probe' (singular) has the same one-global-probe reading as the pre-#1543 (h)(iv) text when a plan reuses adapters of MULTIPLE recipe classes. proposed_change: probe once per reused adapter recipe-class in check (g). confidence: low. related_task: #1543."
