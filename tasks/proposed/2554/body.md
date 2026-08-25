---
title: 'workflow-fix: verify_plan binding-addendum coverage check (plan vs BINDING
  body scope decisions)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T20:45:16Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2546 Phase-2 critic ensemble: plan v3 inverted a
  recorded BINDING user scope addendum and passed verify_plan 0/0'
workflow: v1
---
## Goal
Add a binding-addendum coverage check to the plan-verification surface so a plan cannot pass mechanical verification while contradicting a recorded BINDING user scope decision on its own task body.

## Incident (#2546, 2026-08-24)
The user posted `## SCOPE ADDENDUM — user decision 2026-08-24, BINDING` to the task body (+ a verbatim `epm:progress` marker, 19:34:06Z) one minute AFTER the v1 planner spawn. Plans v1–v3 never absorbed it: the plan deferred two of three user-required model arms, captured a 3-point t-grid instead of the required 9-point grid, and misattributed the deferral to "the body's round-1 recommendation" (text that no longer existed). verify_plan PASSed 0 FAIL / 0 WARN and the fact-checker confirmed 41 items — nothing compares the plan against the body's binding sections. Caught only by the critic ensemble (all three lenses REVISE + consistency BLOCK).

## Proposed check (from the alternatives-lens critic, mechanizable sketch)
In `scripts/verify_plan.py` (--issue mode; new check id): when the task body contains an H2 matching `SCOPE ADDENDUM.*BINDING` (or a configurable binding-directive pattern), extract its table's model ids / named deliverables / named grids and FAIL any id absent from the plan's §9 compute table, and FAIL any id appearing only inside a sentence carrying `defer|follow-up|must ask`. WARN when a binding H2 exists but nothing is extractable (surface for the consistency-checker). Consider a consistency-checker spec bullet as the semantic backstop (it caught this one, but only because the addendum was fresh in events.jsonl).

## Acceptance criteria
1. A fixture reproducing the #2546 shape (binding addendum tables 3 model ids; plan §9 carries 1; deferral sentence present) FAILs the new check.
2. A plan carrying all addendum-named ids in §9 PASSes.
3. A body with no binding H2 SKIPs.
4. Check bundled into the no-flags default run with a pin test, or explicitly documented as --issue-mode-only.
