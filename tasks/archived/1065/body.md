---
title: 'daily-held: Step 10d task-state merge-conflict contract'
kind: infra
tags:
- daily-held
created_at: '2026-07-05T07:04:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 3): Every long-lived issue
  branch now conflicts at the Step 10d rebase-merge because tasks/<status>/<N>/ paths
  git-mv under it on main (3 sessions on 2026-07-04: #953, #967, #906 at 6,738 commits
  behind); each ran the recovery merge successfully, but the recovery path is narrated
  as exceptional while being the de-facto normal.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 3: judgment call -> needs-human), from the nightly transcript problem sweep.

## Goal

Decide the contract: either exclude tasks/** events.jsonl churn from issue-branch commits (a task-state commit-contract change, near the public-contract carve-out) or bless recovery-merge as the normal Step 10d path in SKILL.md. Held: public-contract-adjacent / genuinely ambiguous.

## Workflow gap

- **Bug observed:** Every long-lived issue branch now conflicts at the Step 10d rebase-merge because tasks/<status>/<N>/ paths git-mv under it on main (3 sessions on 2026-07-04: #953, #967, #906 at 6,738 commits behind); each ran the recovery merge successfully, but the recovery path is narrated as exceptional while being the de-facto normal.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Sessions: e4081178 (#953), e4951a90 (#967), 4ea4c2b6 (#906).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
