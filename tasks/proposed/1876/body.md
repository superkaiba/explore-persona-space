---
title: 'daily-fix: compliant git-commit forms at snippet sites + blo'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e0bbcbffb0d9
- daily-auto-filed
- trigger-dense
created_at: '2026-07-30T07:13:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): Piped git commit/push compositions
  were guard-blocked >=9 times across >=6 sessions today (F-P3 x4+1, A-P4 x2, D-P6,
  E-P4, J-P3) — the guards work but compose-first keeps failing; the block message
  does not lead with the exact compliant one-liner'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miners F-P3, A-P4/P5, D-P6, E-P4, J-P3 — 9+ blocked firings in one day (each counted as the hook's own is_error tool_result)).

## Goal

Reduce the recurring compose-time failure by making the compliant form copy-paste available at the exact moments it is needed (snippet sites + the block message itself).

## Workflow gap

- **Bug observed:** All firings were caught by the guards (no damage); cost is ~1 wasted turn each, ~9+/day fleet-wide, stable across days (4 on 07-28).
- **Why it is a workflow gap:** the rules exist; the residual is composition — copy sources beat rules for this class (the CLAUDE.md 'compose this form FIRST' note has not moved the rate).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'tmp/push.out' .claude/hooks/guard_piped_git_push.sh` -> 0 (block message lacks the compliant one-liner) (2026-07-30, this run).

## Proposed change (refine in planning)

Two small edits: snippet-site addition in SKILL.md; block-message lead in the hook (message text only — no matcher change).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, .claude/hooks/guard_piped_git_push.sh
- fingerprint: e0bbcbffb0d9
