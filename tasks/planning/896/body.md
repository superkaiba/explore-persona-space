---
title: Reconcile diverged shared repo-root main (~65 local / ~176 remote) + harden
  root sync against concurrent-rebase corruption
kind: infra
tags:
- root-divergence
created_at: '2026-07-02T22:17:02Z'
has_clean_result: false
origin_prompt: 'issue-810 session incident recovery 2026-07-02: pull --rebase on the
  shared root collided with concurrent committers, corrupted the rebase state, and
  stranded the autostash; root main pushes are blocked fleet-wide by the divergence'
---
## Overview / Motivation

Filed by the issue-810 session (2026-07-02) after a live incident: the SHARED
repo-root `main` has diverged from `origin/main` (~65 local commits ahead /
~176 behind at filing time and growing — local commits are mostly `task.py`
marker/state commits from many concurrent sessions; origin/main advances via
server-side PR merges). Consequences observed live:

- Every `git push origin main` from the root is rejected (non-fast-forward),
  so "commit + push immediately" flows (agent-memory lessons, methodology docs,
  workflow-surface fixes) silently strand their commits locally.
- The documented recovery (`git pull --rebase=merges --autostash`) is UNSAFE at
  current fleet concurrency: the 2026-07-02 attempt hit conflicts on
  agent-memory files, then a concurrent committer corrupted the in-progress
  rebase state (`.git/rebase-merge` reduced to just `autostash`), stranding the
  autostash (14 files of concurrent sessions' uncommitted work). Recovered by
  hand in the issue-810 session (see #810 events, `epm:progress` v59; restore
  commit `2f9627204a`).

## Goal

Safely reconcile the diverged shared repo-root `main` with `origin/main`
(no lost commits, no lost dirty state, no repo-root `reset --hard`), and land a
guard so a shared-root pull/rebase can no longer be corrupted by concurrent
committers.

## Numbered asks

1. Reconcile the divergence: land the local-only commits onto `origin/main`
   (scratch worktree detached at `origin/main` + cherry-pick/merge with
   `merge=union` semantics for `events.jsonl`/`comments.jsonl`, union for
   agent-memory `.md` conflicts), then fast-forward the root `main`.
   NEVER `git reset --hard` on the shared root (CLAUDE.md hard rule).
2. Drain the two rescue stash entries ("autostash rescue from broken
   pull-rebase 2026-07-02" + git's own duplicate): confirm the
   `scripts/pods.conf` / `scripts/pods_ephemeral.json` stale copies are
   genuinely superseded, then drop the entries.
3. Recurrence guard (workflow surface): (a) add
   `.claude/agent-memory/**/*.md merge=union` (or equivalent) to
   `.gitattributes` so memory-file append-append conflicts self-resolve;
   (b) serialize root pull/rebase against concurrent `task.py` committers —
   e.g. take the same `~/.task-workflow/lock` flock around any
   root-`main` pull/rebase, or document scratch-worktree-push as the ONLY
   sanctioned root sync path during fleet activity (CLAUDE.md § Concurrent
   repo-root committers already hints at this; make it binding).

## Constraints / invariants

- Content-preserving throughout; every conflict resolution must be
  union/both-sides for append-only files.
- No pause of the live fleet is assumed; the reconciliation must tolerate
  concurrent commits (re-run/converge, not one-shot).
- `scripts/workflow_lint.py` passes after any workflow-surface edit.

## Provenance

Origin: issue-810 session incident recovery, 2026-07-02 (~21:55-22:15Z).
Evidence: #810 `epm:progress` v59; restore commit `2f9627204a`; stash entries
`autostash rescue from broken pull-rebase 2026-07-02`.
