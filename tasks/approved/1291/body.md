---
title: 'daily-fix: kind-aware c8/c15 WARN gating in verify_plan'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6b16705fd624
- daily-auto-filed
created_at: '2026-07-13T06:44:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): c8_success_kill_criteria
  and c15_failloud_test_coverage WARNed on 4 kind:infra plans in one night (2026-07-12),
  each hand-waived in prose — recurring always-waived WARNs on infra plans are alarm-fatigue
  noise.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 problem sweep (transcript-mined; sessions 004c5304 (#1279), aaf2482e (#1276), 06281fbd (#1275)).

## Goal

Stop `verify_plan.py` c8/c15 from over-firing WARNs on `kind: infra` plans, so orchestrators stop hand-writing the same waiver notes every night.

## Workflow gap

- **Bug observed:** on 2026-07-12, 4 hand-written waiver notes for the same two WARN shapes landed in one night: `c8_success_kill_criteria` WARNed on #1279 and #1276, `c15_failloud_test_coverage` WARNed twice on #1275 — all `kind: infra` workflow-fix plans where the checks' experiment-oriented framing (success/kill criteria; fail-loud eval coverage) does not map cleanly, and each orchestrator dispositioned the WARN in prose.
- **Why it is a workflow gap:** the global `EXEMPT_KINDS` gate (scripts/verify_plan.py:554) exempts some checks for infra kinds, but c8/c15 still fire; a recurring WARN that is ~always waived on a kind is noise that trains orchestrators to waive reflexively (the alarm-fatigue failure mode).
- **Confidence (emitter):** low-medium — the right fix may be kind-aware gating, a softened infra-specific rubric, or keeping the WARNs and adding a documented standard disposition; the spawned planner decides with the file open (a reasoned no-change report is an acceptable outcome).
- verified-at-filing: `grep -n "kind in EXEMPT_KINDS" scripts/verify_plan.py` → single global gate at :554; c8/c15 not individually kind-gated (2026-07-13). Waiver-note evidence in the three session transcripts.

## Proposed change (candidate diff sketch — refine in planning)

Downgrade or skip c8_success_kill_criteria and c15_failloud_test_coverage for `kind: infra|batch` plans (or emit an infra-tailored variant), keeping them binding for `kind: experiment`.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py` (+ `tests/test_verify_plan.py` pins).

## Constraints / invariants

- Workflow-surface only. Must not weaken any `kind: experiment` check. Corpus replay for retro-WARN changes per the established verify_plan migration pattern (#1262/#1264). Lint + ruff pass. Recursion guard applies.

## Provenance

- fingerprint: 6b16705fd624

- workflow_fix_target: scripts/verify_plan.py

Origin: /daily 2026-07-12 transcript sweep (4 waiver notes across 004c5304/aaf2482e/06281fbd).
