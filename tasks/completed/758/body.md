---
title: 'daily-fix: codex-critic must source numbers from reviewed plan'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d4a409d8c5a3
- daily-auto-filed
created_at: '2026-06-30T06:44:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-06-29 auto-filed route-2: A codex-critic prompt for #722
  inlined predicted-positive numbers (+0.74-0.80, MLP -2.17/-6.12) that did NOT appear
  in the reviewed plan v3 or #658 artifacts — the prompt-composer conflated a scratch
  artifact. Codex critics also issued REVISE off stale events.jsonl reads.'
---
## Overview / Motivation
Auto-filed by /daily 2026-06-29 problem sweep. #722/#664 codex-critic prompt hygiene.

## Goal
A codex-critic never reviews fabricated numbers and is aware its marker snapshot may be stale.

## Workflow gap
- **Bug observed:** codex-critic prompt inlined PP numbers not present in the reviewed plan (scratch-artifact conflation); codex critics also REVISE'd off stale events.jsonl twice.
- **Why it is a workflow gap:** the codex-critic prompt-composer spec is workflow surface.
- **Confidence (emitter):** medium.

## Proposed change
- codex-critic.md: the composer sources predicted/numeric values only from the plan/artifacts under review; assert every inlined number traces to the inlined plan body.
- Pin the marker/state snapshot at spawn and note it may be behind; orchestrator surfaces the latest markers in-prompt.

## Scope / surfaces
- `.claude/agents/codex-critic.md`.

## Constraints / invariants
- Workflow surface only. Recursion guard: EPM_WORKFLOW_FIX_SESSION=1.

## Provenance
- workflow_fix_target: .claude/agents/codex-critic.md
- fingerprint: d4a409d8c5a3

Sessions: d59f26d1 (issue-722), 6b23a4ae (issue-664).
