---
title: 'workflow-fix: LESSONS.md byte-cap headroom restructure'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:565a75f95569
created_at: '2026-07-04T07:12:58Z'
has_clean_result: false
origin_prompt: 'parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target — recursion
  guard (.claude/rules/workflow-fix-on-bug.md § Recursion guard); logged for the next
  human/orchestrator pass, NOT auto-routed. Three concrete prose follow-ups surfaced
  by the #955 implementer + code reviewers: (1) target_file: src/explore_persona_space/llm/api_dispatch.py
  — runtime system-role guard (raise-or-lift) at the dispatch_calls seam so a system-bearing
  messages list can never reach the wire un-lifted; SEQUENCE AFTER #906''s branch
  merges (its datagen fix is in flight); confidence: medium. (2) target_file: .claude/rules/LESSONS.md
  — byte-cap headroom fix: file now at 7992/8000 bytes after information-preserving
  trims; the next index addition hits the cap again (raise the cap or restructure
  the index); confidence: high. (3) target_file: scripts/archive/generate_leakage_data_sync.py
  — archived call_api seam is forward-fragile (would forward a system-bearing list
  verbatim if revived); annotate-or-delete on next archive sweep; confidence: low.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

raise the LESSONS.md byte cap or restructure the index to restore headroom

## Workflow gap

- **Bug observed:** the file is at 7992/8000 bytes after information-preserving trims; the next index addition hits the cap again
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** high

## Proposed change (candidate sketch — refine in planning)

raise the LESSONS.md byte cap or restructure the index to restore headroom

## Scope / surfaces

- Primary target: `.claude/rules/LESSONS.md`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: .claude/rules/LESSONS.md
- fingerprint: 565a75f95569

parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target — recursion guard (.claude/rules/workflow-fix-on-bug.md § Recursion guard); logged for the next human/orchestrator pass, NOT auto-routed. Three concrete prose follow-ups surfaced by the #955 implementer + code reviewers: (1) target_file: src/explore_persona_space/llm/api_dispatch.py — runtime system-role guard (raise-or-lift) at the dispatch_calls seam so a system-bearing messages list can never reach the wire un-lifted; SEQUENCE AFTER #906's branch merges (its datagen fix is in flight); confidence: medium. (2) target_file: .claude/rules/LESSONS.md — byte-cap headroom fix: file now at 7992/8000 bytes after information-preserving trims; the next index addition hits the cap again (raise the cap or restructure the index); confidence: high. (3) target_file: scripts/archive/generate_leakage_data_sync.py — archived call_api seam is forward-fragile (would forward a system-bearing list verbatim if revived); annotate-or-delete on next archive sweep; confidence: low.
