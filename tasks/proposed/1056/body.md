---
title: 'daily-fix: critics verify figures against pinned blob only'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6341947a2e2e
- daily-auto-filed
created_at: '2026-07-05T07:03:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): #922 burned a full reconciler
  round on 2026-07-04 (03:00-03:05 UTC) because one clean-result critic read STALE
  UNTRACKED figure copies at the repo root instead of the SHA-pinned blob, producing
  a spurious FAIL on a correct body.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Instruct both clean-result critics to verify figures against the SHA-pinned blob (or the worktree copy) and never an untracked repo-root duplicate; optionally have the figure-commit step clean repo-root strays.

## Workflow gap

- **Bug observed:** #922 burned a full reconciler round on 2026-07-04 (03:00-03:05 UTC) because one clean-result critic read STALE UNTRACKED figure copies at the repo root instead of the SHA-pinned blob, producing a spurious FAIL on a correct body.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/agents/clean-result-critic.md, .claude/agents/codex-clean-result-critic.md`
- Session: a0064fda (#922).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/agents/clean-result-critic.md, .claude/agents/codex-clean-result-critic.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
