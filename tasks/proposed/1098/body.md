---
title: 'daily-fix: repo-root guard false-positive on pod-side git in'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4ee669e33d08
- daily-auto-filed
created_at: '2026-07-07T06:49:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-06 problem sweep (route 2): The hook blocked `ssh pod-779
  ''git reset --hard origin/main''` — a destructive git op targeting the POD''s own
  /workspace clone, not the shared repo root; the #779 session (5664c4f8, 2026-07-06)
  had to move the reset into a pod-side script to proceed. The guard''s flat-stream
  scan cannot see that the gated verb lives inside an ssh remote-command argument.
  It also fires on gated-verb substrings inside'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-06 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Exempt gated git verbs that appear only inside an `ssh <host> '...'` remote-command argument (and consider grep-pattern arguments), OR add a documented override env (like the piped-push guard's) for pod-side remote git ops — keeping the repo-root protection semantics intact; add positive+negative test fixtures.

## Workflow gap

- **Bug observed:** The hook blocked `ssh pod-779 'git reset --hard origin/main'` — a destructive git op targeting the POD's own /workspace clone, not the shared repo root; the #779 session (5664c4f8, 2026-07-06) had to move the reset into a pod-side script to proceed. The guard's flat-stream scan cannot see that the gated verb lives inside an ssh remote-command argument. It also fires on gated-verb substrings inside read-only grep pattern arguments.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- source: /daily 2026-07-06 problem sweep (transcript-mined)
