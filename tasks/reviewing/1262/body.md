---
title: 'daily-fix: route c7 escape through the standalone helper'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9f7461b8bc83
- daily-auto-filed
created_at: '2026-07-11T06:51:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): c7''s N/A escape is a bare
  doc-global re.search(''not a replication'') over strip_fences (verify_plan.py:874)
  - any sentence containing the phrase self-escapes it; the sibling NA_RE escapes
  were migrated by #1237, which missed c7 (no NA_RE prefix, so #1237''s grep set missed
  it)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

give c7 the NA_RE-prefixed phrase and route it through _standalone_na_declared with red-green fixtures mirroring the #1237 migration

## Workflow gap

- **Bug observed:** c7's N/A escape is a bare doc-global re.search('not a replication') over strip_fences (verify_plan.py:874) - any sentence containing the phrase self-escapes it; the sibling NA_RE escapes were migrated by #1237, which missed c7 (no NA_RE prefix, so #1237's grep set missed it)
- **Provenance / evidence:** Narrowed residual of the #1238 formal candidate fp 3755904ce2b5 (parked 2026-07-10T11:35:56Z; fact-checker + methodology critic, #1238 plan v2/v3 review). Bulk of the original candidate already fixed by #1237 (zero re.search(NA_RE sites remain); c7 is the surviving site.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 9f7461b8bc83

- workflow_fix_target: scripts/verify_plan.py
