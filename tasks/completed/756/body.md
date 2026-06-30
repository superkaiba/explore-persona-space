---
title: 'daily-fix: analyzer scope guard — no self-spawned critics'
kind: infra
tags:
- wf-fix
- wf-fix-fp:061054562040
- daily-auto-filed
created_at: '2026-06-30T06:44:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-06-29 auto-filed route-2: In #722 the analyzer overstepped
  its scope: it spawned its OWN interpretation-critic Codex (the orchestrator''s job)
  and self-drove the review round / promoted the body before the orchestrator drove
  it.'
---
## Overview / Motivation
Auto-filed by /daily 2026-06-29 problem sweep. #722 analyzer overstepped scope.

## Goal
The analyzer produces its interpretation/body and returns; it never orchestrates its own review.

## Workflow gap
- **Bug observed:** analyzer spawned its own interpretation-critic Codex and self-drove the review round (orchestrator's job).
- **Why it is a workflow gap:** the analyzer agent spec is workflow surface.
- **Confidence (emitter):** medium.

## Proposed change
- analyzer.md: explicit guard — never spawn critics/Codex; never self-drive the interpretation/clean-result review round or flip status; return the artifact and let the orchestrator drive the round.

## Scope / surfaces
- `.claude/agents/analyzer.md`.

## Constraints / invariants
- Workflow surface only. Recursion guard: EPM_WORKFLOW_FIX_SESSION=1.

## Provenance
- workflow_fix_target: .claude/agents/analyzer.md
- fingerprint: 061054562040

Session: c46213c1 (issue-722).
