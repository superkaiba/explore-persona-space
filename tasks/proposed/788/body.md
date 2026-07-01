---
title: 'daily-fix: reviewer sandbox read-only tempdir silently skips pytest'
kind: infra
tags:
- wf-fix
- wf-fix-fp:56826a25891c
- daily-auto-filed
created_at: '2026-07-01T06:53:30Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: code-reviewer/test-verdict runs
  in a read-only sandbox where `uv run pytest` cannot create temp/cache files (~/.cache/uv
  read-only, tempfile.gettempdir() not writable via torch->dill), so the Step 9c'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Document + set a writable-tempdir fallback for reviewers/test-verdict in the sandbox (TMPDIR/UV_CACHE_DIR/XDG_CACHE_HOME -> a worktree-local writable dir before `uv run pytest`), and require the reviewer to flag 'tests NOT actually run' loudly rather than treating a code-only review as a PASS.

## Workflow gap

- **Bug observed:** code-reviewer/test-verdict runs in a read-only sandbox where `uv run pytest` cannot create temp/cache files (~/.cache/uv read-only, tempfile.gettempdir() not writable via torch->dill), so the Step 9c test-verdict and workflow_lint checks silently cannot run and get skipped with 'reviewed the code path directly instead' — a code-only review reported as PASS.
- **Evidence:** issues 761/753/739 on 2026-06-30. Sources: /daily miner batch 10.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md
- fingerprint: 56826a25891c
- source: /daily route-2 (2026-06-30)
