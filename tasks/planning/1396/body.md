---
title: 'daily-fix: upload extraction store before long fits'
kind: infra
tags:
- wf-fix
- wf-fix-fp:287c7eb4864a
- daily-auto-filed
created_at: '2026-07-16T07:21:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): #825 Track-S run 2 hung
  in a serial CPU MLP fit BEFORE the upload phase, stranding the turnstore off HF;
  recovery cost a full fresh GPU re-extraction'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Sequence the extraction-store / expensive-input upload BEFORE (or concurrent with) any long fit/analysis phase, so a fit hang never strands the store; planner §9 orders the phases accordingly.

## Workflow gap

- **Bug observed:** #825 Track-S run 2 hung in a serial CPU MLP fit phase BEFORE the upload phase, stranding the turnstore off HF — recovering it cost a full fresh GPU re-extraction (09f28ede 04:20Z "The turnstore is not on HF (fit hung before [phase=upload])").
- **Why it is a workflow gap:** the upload policy sequences uploads relative to POD RELEASE (#664: GPU released before the final bulk upload) but nothing orders the expensive-input upload relative to LONG FIT phases — the default phase order (extract → fit → upload) puts hours of hang risk between an expensive artifact's creation and its persistence.
- **Severity:** high
- verified-at-filing: `grep -n 'before.*fit\|fit.*before\|sequenc' .claude/rules/upload-policy.md` → 1 hit (L585, "#664 sequencing unchanged. The GPU pod is released before the FINAL bulk upload") — pod-release sequencing exists, upload-before-long-fit sequencing absent; planner.md §9 has no phase-ordering clause for expensive-input persistence relative to fit phases (grep 'upload' in planner.md §9 region shows no such ordering) (2026-07-16 UTC).

## Proposed change (refine in planning)

Add to `.claude/rules/upload-policy.md` a phase-ordering rule: an expensive extraction store / regeneration-costly input consumed by a downstream fit/analysis phase is uploaded (or its upload launched concurrently) BEFORE any long (> ~15-30 min) fit/analysis phase begins — a fit hang or crash must never strand the store. Mirror the ordering requirement into `.claude/agents/planner.md` §9 (the plan names the upload point in the phase sequence). This is the intra-run sibling of the existing #664 pod-release sequencing (L585) and the "persist by default" rule.

## Scope / surfaces

- Primary target: `.claude/rules/upload-policy.md` (anchor: L585 #664 sequencing bullet)
- Secondary: `.claude/agents/planner.md` §9 (phase sequencing)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 287c7eb4864a

- workflow_fix_target: .claude/rules/upload-policy.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 09f28ede (#825) 04:20Z "The turnstore is not on HF (fit hung before [phase=upload])" (batch 08 P8 leg b).
