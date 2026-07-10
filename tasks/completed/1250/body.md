---
title: guard_repo_root_pull false-positives on quoted 'git pull' in
kind: infra
tags:
- wf-fix
- wf-fix-fp:fa3c0da47b8c
- daily-auto-filed
created_at: '2026-07-10T06:55:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): hook blocked task.py post-marker
  because the note string contained the words git pull (#1201 session, 23:06Z)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-09 from the nightly transcript problem sweep (miner 02).

## Goal

Fix the `guard_repo_root_pull.sh` PreToolUse matcher so it no longer blocks non-git commands whose quoted ARGUMENT TEXT merely contains the words "git pull" (false positive on `task.py post-marker --note '... git pull ...'`).

## Workflow gap

- **Bug observed:** On #1201 (2026-07-09T23:06Z) the hook blocked `uv run python scripts/task.py post-marker ... --note '...git pull...'` because the note string contained the literal "git pull"; the session worked around it by switching to `--file`. The hook shipped the same day (#1201) and false-positived within hours.
- **Why it is a workflow gap:** The matcher fires on the whole command text rather than on an actual `git ... pull` invocation shape, so any marker note, commit message, or echo that MENTIONS a pull is blocked — marker notes about git recoveries are common (this is exactly the #1098 trigger-dense-text class, but for a hook matcher instead of the content filter).

## Proposed change (refine in planning)

Tighten the matcher to fire only when a `git` invocation with a `pull` subcommand appears as a command unit (mirroring `guard_piped_git_push.sh`'s unit-parsing approach and `guard_repo_root_branch.sh`'s verb detection), not on quoted-string content; add self-test cases: `task.py post-marker --note 'recovered via git pull --rebase'` must PASS, a real repo-root non-ff `git pull` must still BLOCK.

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_pull.sh` (shipped by #1201; confirm actual path/name in `.claude/settings.json` hooks)
- Secondary: its self-test / `tests/` twin

## Constraints / invariants

- Workflow-surface only. `bash -n` + self-test PASS; `scripts/workflow_lint.py --check-asks` passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: fa3c0da47b8c

- workflow_fix_target: scripts/guard_repo_root_pull.sh
