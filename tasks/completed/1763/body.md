---
title: 'daily-fix: Step 10d merge-SHA check fetches before compare'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7a0dd1cb09b1
- daily-auto-filed
created_at: '2026-07-28T07:05:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): #1735''s Step 10d merged-marker
  recipe ran the merge-SHA subject check before fetching origin/main -> false ''fatal:
  bad object'' MISMATCH; the epm:merged post was aborted one round'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session f56bbb5d (#1735), 2026-07-27T20:21Z (miner H P6).

## Goal

The merge-SHA verification must fetch before it compares.

## Workflow gap

- **Bug observed:** the squash-merge SHA returned by gh existed only on the remote; the local subject check ran pre-fetch and read 'fatal: bad object', producing a false MISMATCH that delayed the epm:merged post one round.
- **Why it is a workflow gap:** the recipe orders the subject check before any fetch of the just-created merge commit.
- **Confidence (emitter):** medium
- verified-at-filing: unverified hypothesis — verify at plan time: exact current recipe ordering in SKILL.md Step 10d merged-marker block (it churned this week); the incident evidence is the session's own 'fatal: bad object' output (miner-quoted).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/issue/SKILL.md` (Step 10d merged-marker recipe): insert `git fetch origin main` before the merge-SHA subject check, or switch the check to `gh pr view <PR> --json mergeCommit` (no local object needed).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 7a0dd1cb09b1

- workflow_fix_target: .claude/skills/issue/SKILL.md
