---
title: 'daily-fix: system-role guard at api_dispatch dispatch_calls'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-04T07:12:53Z'
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

add a runtime system-role guard (raise-or-lift) at the dispatch_calls seam so a system-bearing messages list can never reach the wire un-lifted; SEQUENCE AFTER #906's branch merges

## Workflow gap

- **Bug observed:** a system-bearing messages list can reach the wire un-lifted through the dispatch_calls seam (#955/#906 family); the datagen fix in #906 is in flight
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

add a runtime system-role guard (raise-or-lift) at the dispatch_calls seam so a system-bearing messages list can never reach the wire un-lifted; SEQUENCE AFTER #906's branch merges

## Scope / surfaces

- Primary target: `src/explore_persona_space/llm/api_dispatch.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: src/explore_persona_space/llm/api_dispatch.py
- fingerprint: b037d930e545

parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target — recursion guard (.claude/rules/workflow-fix-on-bug.md § Recursion guard); logged for the next human/orchestrator pass, NOT auto-routed. Three concrete prose follow-ups surfaced by the #955 implementer + code reviewers: (1) target_file: src/explore_persona_space/llm/api_dispatch.py — runtime system-role guard (raise-or-lift) at the dispatch_calls seam so a system-bearing messages list can never reach the wire un-lifted; SEQUENCE AFTER #906's branch merges (its datagen fix is in flight); confidence: medium. (2) target_file: .claude/rules/LESSONS.md — byte-cap headroom fix: file now at 7992/8000 bytes after information-preserving trims; the next index addition hits the cap again (raise the cap or restructure the index); confidence: high. (3) target_file: scripts/archive/generate_leakage_data_sync.py — archived call_api seam is forward-fragile (would forward a system-bearing list verbatim if revived); annotate-or-delete on next archive sweep; confidence: low.
