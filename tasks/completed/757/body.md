---
title: 'daily-fix: same-issue loop re-reads latest followup-scope marker'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7287d98a4f84
- daily-auto-filed
created_at: '2026-06-30T06:44:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-06-29 auto-filed route-2: In #658 the same-issue follow-up
  loop entered keyed on a stale `epm:followup-scope` version (the narrow pre-correction
  scope); later superseding scope markers (judge-filter corrections) landed after
  the orchestrator had already snapshotted, so it would have replanned against the
  stale scope.'
---
## Overview / Motivation
Auto-filed by /daily 2026-06-29 problem sweep. #658 same-issue follow-up loop keyed on a stale scope marker; recovery was a manual stop+respawn.

## Goal
The follow-up loop always plans against the newest followup-scope, never a superseded entry-time snapshot.

## Workflow gap
- **Bug observed:** loop entered keyed on `epm:followup-scope v3` (narrow); v5/v6 corrections landed after the orchestrator snapshotted -> would have replanned against stale scope.
- **Why it is a workflow gap:** the same-issue follow-up loop recipe lives in SKILL.md.
- **Confidence (emitter):** medium; one session, workaround-only.

## Proposed change
- SKILL.md same-issue follow-up loop: re-read the latest (highest-version / newest) `epm:followup-scope` marker immediately before the planner snapshots; do not cache the entry-time version.

## Scope / surfaces
- `.claude/skills/issue/SKILL.md` (same-issue follow-up loop).

## Constraints / invariants
- Workflow surface only. `--check-references`/`--check-asks` green. Recursion guard: EPM_WORKFLOW_FIX_SESSION=1.

## Provenance
- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 7287d98a4f84

Session: b68b9316 (issue-658).
