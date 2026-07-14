---
title: 'daily-fix: block repo-root branch merges in the guard hook'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2ebfeff2cb59
- daily-auto-filed
created_at: '2026-07-08T06:58:40Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): Session c86ff35c (#1090)
  07:00:50-07:02:17Z: a branch merge run at the SHARED repo root conflicted on 2 files,
  leaving conflict markers in the shared tree ~70s until the merge was aborted — a
  concurrent session staging files in that window could sweep markered content.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

extend guard_repo_root_branch.sh to block branch merges executed in the shared repo-root tree (pointing at the worktree / scratch-worktree recovery), the same way it fences the other destructive verbs

## Workflow gap

- **Bug observed:** Session c86ff35c (#1090) 07:00:50-07:02:17Z: a branch merge run at the SHARED repo root conflicted on 2 files, leaving conflict markers in the shared tree ~70s until the merge was aborted — a concurrent session staging files in that window could sweep markered content.
- **Why it is a workflow gap:** The shared repo root is the canonical commit target for every concurrent session; the guard fences the other tree-mutating verbs but not merges, which have the same blast radius.

## Proposed change

Add the merge verb to the blocked set for repo-root cwd, with a worktree-path (-C worktree) carve-out; block message suggests the scratch-worktree recipe; tests pin both directions.

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: c86ff35c (#1090) 07:00-07:02Z.
