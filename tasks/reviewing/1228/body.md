---
title: gate daily driver wf-fix tags per SKILL route-2 variant
kind: infra
tags:
- wf-fix
- wf-fix-fp:b661ddc44982
- daily-auto-filed
created_at: '2026-07-10T06:53:55Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): daily SKILL.md says non-workflow-surface
  route-2 items DROP wf-fix tags, but _filer_cmd applies wf-fix + wf-fix-fp to EVERY
  route-2 item unconditionally — the experiment-code variant is unreachable via the
  batch driver'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1173.

## Goal
Gate the wf-fix tag set in daily_drive_filings.py per the daily SKILL.md route-2 variants: thread a manifest flag (e.g. wf_fix: false) through _filer_cmd + ensure_wf_fix_provenance so non-workflow-surface route-2 items keep daily-auto-filed but DROP wf-fix/wf-fix-fp tags — or document that experiment-code items must file directly.

## Workflow gap
- **Bug observed:** daily SKILL.md (~:380) says non-workflow-surface route-2 bugs should DROP the wf-fix tags on single-item filings, but scripts/daily_drive_filings.py _filer_cmd (~:350) applies wf-fix + wf-fix-fp + daily-auto-filed to EVERY route-2 item unconditionally (route-gated only), and ensure_wf_fix_provenance injects the Provenance block regardless — the experiment-code variant is unreachable via the batch driver (verified on main 2026-07-09).
- **Why it is a workflow gap:** Two authoritative surfaces (the skill and its driver) prescribe contradictory tagging for the same route, so every non-workflow-surface route-2 filing mis-tags as workflow-surface and pollutes the wf-fix dedup key space.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
+ manifest item key wf_fix (default true); _filer_cmd: route-2 tags = ['daily-auto-filed'] + (['wf-fix', 'wf-fix-fp:<fp>'] if item.get('wf_fix', True) else []); skip ensure_wf_fix_provenance when wf_fix is false; sync SKILL.md wording.

## Scope / surfaces
- Primary target: `scripts/daily_drive_filings.py, .claude/skills/daily/SKILL.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/daily_drive_filings.py, .claude/skills/daily/SKILL.md
- fingerprint: 2e689dde6ccc

Merged from TWO sibling parked candidates: #1173 planner prose observation, 2026-07-09T14:16:38Z (SKILL.md:380 vs daily_drive_filings.py:302-303 discrepancy; distinct bug, out of #1173 scope) and #1180 alternatives-critic prose-followup, 2026-07-09T16:28:16Z (_filer_cmd/ensure_wf_fix_provenance route-gated only; experiment-code route-2 variant unreachable; pre-existing, not worsened by #1180).
