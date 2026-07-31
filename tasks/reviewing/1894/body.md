---
title: 'workflow-fix: concurrent-committers rule mandates pathspec-limited COMMITS
  (bare git commit sweeps foreign staged files)'
kind: infra
tags:
- wf-fix
created_at: '2026-07-30T19:56:17Z'
has_clean_result: false
origin_prompt: 'uslot-impl stand-down report 2026-07-30: git commit -F msg without
  a pathspec swept 4 concurrent-session task files into 5e4124acc9; rule says stage-by-path
  but not commit-by-pathspec'
workflow: v1
---
## Overview / Motivation

Auto-filed from a surfaced prose follow-up (user-slot-recapture implementer stand-down report, 2026-07-30). The CLAUDE.md "Concurrent repo-root committers" rule mandates STAGING by explicit path ("never git add -A") but says nothing about the COMMIT command itself. Incident: commit 5e4124acc9 staged its own two files by explicit path, but ran `git commit -F msg` WITHOUT a pathspec — a parallel session had populated the shared index in the gap, so four foreign task files (tasks/planning/1841/events.jsonl, tasks/proposed/1892/*) were swept into the commit. Content correct, nothing lost, but mis-attributed to an unrelated commit message, and the same shape can sweep half-staged foreign WORK in the general case. The repo-root commit guard (guard_root_code_commit.sh) already recognizes and prefers pathspec-limited commits ("a pathspec-limited commit is never blocked by foreign staged files") — the missing piece is the RULE stating the commit side, and optionally a guard nudge.

- verified-at-filing: `git show --stat 5e4124acc9 | grep -c "tasks/"` shows the swept task files in the commit; CLAUDE.md § Concurrent repo-root committers carries "Stage by explicit path only" with no commit-pathspec clause (grep 'Stage by explicit path' CLAUDE.md -> 1 hit, no 'commit .* -- ' guidance in that paragraph) (2026-07-30).

## Goal

Repo-root commits are always pathspec-limited so a concurrently-populated shared index can never be swept into an unrelated commit.

## Proposed change

1. CLAUDE.md § "Concurrent repo-root committers": extend "Stage by explicit path only" to "Stage by explicit path only, and COMMIT with an explicit pathspec (`git commit -m/-F <msg> -- <paths>`) — a bare `git commit` at the repo root commits whatever the shared index holds, including concurrent sessions' staged files".
2. Optional mechanical nudge: `guard_root_code_commit.sh` (or a sibling PreToolUse guard) WARNs (not blocks) on a repo-root `git commit` with no `--` pathspec when the staged set contains paths outside the command's obvious scope. Keep it warn-only — task.py's own commits are legitimately pathspec-less within its flock.
3. Check `.claude/skills/issue/SKILL.md` Step 9a-ter inline-commit recipes for the same clause.

## Constraints

- Workflow-surface only; `task.py`'s internal commit flow (flock-serialized) is exempt by design — scope any guard to interactive/orchestrator commits.
- est_gpu_hours: 0
