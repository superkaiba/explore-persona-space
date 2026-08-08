---
title: 'daily-fix: root-commit guard honors excluding pathspec'
kind: infra
tags:
- wf-fix
- wf-fix-fp:99c407a64d00
- daily-auto-filed
created_at: '2026-08-03T07:05:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): guard_root_code_commit.sh
  blocked a commit whose explicit pathspec EXCLUDED the uncertified staged script:
  the first block (uncertified code payload staged by a subagent) was correct, but
  the retry `git commit -- <non-code paths>` was blocked too because the guard keys
  on the STAGED INDEX, not the commit pathspec; resolved only by unstaging the foreign
  file -- which the own-files-only contract dis'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner3, session 55419495, task #1739).

## Goal

A pathspec-limited commit that cannot sweep an uncertified staged file is not blocked by that file's presence in a concurrent session's staged index.

## Workflow gap

- **Bug observed:** guard_root_code_commit.sh blocked a commit whose explicit pathspec EXCLUDED the uncertified staged script: the first block (uncertified code payload staged by a subagent) was correct, but the retry `git commit -- <non-code paths>` was blocked too because the guard keys on the STAGED INDEX, not the commit pathspec; resolved only by unstaging the foreign file -- which the own-files-only contract discourages (session 55419495, 20:20:18/20:20:29Z, 2 block firing events, task #1739). unverified hypothesis -- verify at plan time: the guard reads only `git diff --cached --name-only` with no pathspec parse (miner-inferred; hook source not read).
- **Why it is a workflow gap:** The shared root routinely holds sibling sessions' staged files; the own-files-only contract prescribes pathspec-limited commits precisely so foreign staged entries are inert -- the guard's index-grain check re-couples them.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'cached\|pathspec' .claude/hooks/guard_root_code_commit.sh` to be re-run by the planner; compose-time read of the hook's block text confirms no pathspec-aware branch is mentioned in its header (the CWD-blind bare-commit case is documented; the pathspec case is not).

## Proposed change (refine in planning)

either honor an explicit `git commit -- <pathspec>` whose named paths exclude every uncertified staged file (git-commit --only semantics make the pathspec authoritative over the index), or emit unstage-first as the canonical recovery in the block message.

## Scope / surfaces

- Primary target: `.claude/hooks/guard_root_code_commit.sh`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 99c407a64d00

- workflow_fix_target: .claude/hooks/guard_root_code_commit.sh

