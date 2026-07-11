---
title: 'daily-fix: recent_clean_results v4 skim reads Takeaways bloc'
kind: infra
tags:
- wf-fix
- wf-fix-fp:070e1083a120
- daily-auto-filed
created_at: '2026-07-11T06:51:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): v4 bodies route through
  the v2/legacy ''## TL;DR'' skim path (is_v3_body at :88 matches the v3 sentinel
  only; :214 routes non-v3 to RE_MD_TLDR which a v4 body never matches), falling back
  to whole-body truncation instead of the ''## Takeaways'' block'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

add v4-sentinel awareness so the skim extracts ## Takeaways for v4 bodies

## Workflow gap

- **Bug observed:** v4 bodies route through the v2/legacy '## TL;DR' skim path (is_v3_body at :88 matches the v3 sentinel only; :214 routes non-v3 to RE_MD_TLDR which a v4 body never matches), falling back to whole-body truncation instead of the '## Takeaways' block
- **Provenance / evidence:** code-reviewer r1 prose follow-up, #1226 (parked 2026-07-10T08:26:33Z). #1226 fixed is_v2 in audit_clean_results_body_discipline.py - a DIFFERENT file; verified live: zero v4 mentions in recent_clean_results.py.

## Scope / surfaces

- Primary target: `scripts/recent_clean_results.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 070e1083a120

- workflow_fix_target: scripts/recent_clean_results.py
