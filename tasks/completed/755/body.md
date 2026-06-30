---
title: 'daily-fix: reconciler must persist EVERY upheld finding'
kind: infra
tags:
- wf-fix
- wf-fix-fp:06bf8a47b132
- daily-auto-filed
created_at: '2026-06-30T06:44:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-06-29 auto-filed route-2: The reconciler persisted only
  a subset of the findings it upheld (2 of 4 in #715), forcing the orchestrator to
  manually re-raise the missing BLOCKER+CONCERN; it also overrode a structure-lens
  FAIL without quoting the SPEC clause.'
---
## Overview / Motivation
Auto-filed by /daily 2026-06-29 problem sweep. Recurring reconciler gaps across #664 and #715.

## Goal
A reconciler verdict never silently drops an upheld finding, and a structure-lens override is grounded in the exact spec text.

## Workflow gap
- **Bug observed:** reconciler persisted only 2 of 4 upheld findings (#715); orchestrator had to manually re-raise the missing BLOCKER + CONCERN for the next brief. Separately, a reconciler underapplied SPEC text when overriding a clean-result-critic structure FAIL.
- **Why it is a workflow gap:** the reconciler agent spec is workflow surface.
- **Confidence (emitter):** medium; two sessions + an existing agent-memory note on the spec-underapply.

## Proposed change
- reconciler.md: require the verdict to enumerate and persist EVERY upheld finding into the brief; add an explicit completeness check ("N upheld -> N persisted").
- When overriding a structure-lens FAIL, the reconciler must quote the exact SPEC.md clause it relies on.

## Scope / surfaces
- `.claude/agents/reconciler.md`.

## Constraints / invariants
- Workflow surface only. Recursion guard: EPM_WORKFLOW_FIX_SESSION=1.

## Provenance
- workflow_fix_target: .claude/agents/reconciler.md
- fingerprint: 06bf8a47b132

Sessions: ca9d2d46 (issue-715), and reconciler spec-underapply memory.
