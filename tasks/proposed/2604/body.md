---
title: 'git stash stack is SHARED across worktrees — a pop from an issue worktree
  can steal another session''s autostash (near-miss on #2546 r15)'
kind: infra
tags: []
created_at: '2026-08-26T14:23:01Z'
has_clean_result: false
origin_prompt: 'Near-miss on task #2546 review round v15: the orchestrator''s brief
  told a reviewer to verify pre-fix failure by in-place stash round-trip; on a clean
  post-commit tree the push no-opped and the pop applied a FOREIGN autostash. Reviewer
  detected, reverted, preserved the entry, and recorded base-blob checkout as the
  correct method. Shared stash stack confirmed live: worktree and repo-root ''git
  stash list'' are byte-identical, 10+ entries deep, including four named rescues
  from prior incidents (2026-08-07, 2026-08-11).'
workflow: v1
---
---
kind: infra
---

# `git stash` operates on a stack SHARED by every worktree — a pop can steal another session's autostash

## Goal

Add a rule-surface warning (and, if review agrees, a guard) for a cross-session data-loss hazard that is reachable from an ordinary-looking instruction and has already bitten this repo at least twice.

`git stash` stores entries as `refs/stash` in the **common git dir**, not per-worktree. With ~15 concurrent sessions each working in its own `.claude/worktrees/issue-<N>/`, every one of them shares a single stash stack. Consequences:

- `git stash pop` / `git stash apply` from ANY worktree consumes and applies whatever is at `stash@{0}` — which may belong to a different session, or be a `--autostash` entry created by a concurrent rebase (`sync_repo_root.py` sets `rebase.autoStash=true`).
- On a CLEAN tree, `git stash push` is a NO-OP (nothing to stash) — so a naive `stash push` / do-work / `stash pop` round-trip does not restore your own state; it applies a FOREIGN entry.

Observed live on 2026-08-26 in `.claude/worktrees/issue-2546`: `git stash list` from the worktree and from the repo root return byte-identical lists, 10+ entries deep.

## The near-miss that surfaced it (task #2546, review round v15)

The orchestrator's review brief instructed the reviewer to verify a pre-fix test failure by "in-place stash round-trip, not a symlinked shadow tree". That instruction was CORRECT for the implementer of the previous round, which had uncommitted changes. It was INVALID for a reviewer working post-commit on a clean tree: the push no-opped and the pop applied a foreign autostash.

The reviewer detected it, fully reverted the application, restored the worktree byte-clean, preserved the foreign stash entry, and recorded base-blob checkout (`git show <base-sha>:<path>`) as the correct method for verifying pre-fix behavior on a COMMITTED round. No data was lost. The orchestrator independently verified afterward: worktree clean, all stash entries intact.

It nearly went the other way, and the instruction that caused it looked entirely reasonable.

## Evidence this is recurring, not hypothetical

The shared stash stack currently holds four entries whose names are themselves incident records:

```
stash@{2}: rescued autostash #3 from repeated rebase husk (issue-2232 session, 2026-08-11)
stash@{3}: rescued autostash #2 from repeated rebase husk (issue-2232 session, 2026-08-11)
stash@{4}: rescued autostash from stale rebase-merge husk (issue-2232 session sync, 2026-08-11)
stash@{5}: rescued autostash from corrupt rebase state (userchat transport round, 2026-08-07)
```

Someone has already had to rescue orphaned autostashes twice, on two separate dates. The #1806 `stash_rescue_audit_pass` (watcher pass 34) exists as a standing recovery channel for that residue. What is missing is the PREVENTIVE half: nothing warns an agent that popping is a cross-session operation.

## Proposed change

1. **Rule text.** Add to `.claude/rules/repo-root-uncommitted-state.md` (which already owns the pre-commit stash-race mechanics) or a sibling: the stash stack is SHARED across all worktrees via the common git dir; `git stash pop`/`apply` is therefore a cross-session mutation and is never a safe way to restore your own state. On a clean tree, `stash push` is a no-op, so the round-trip silently becomes "apply a stranger's work".
2. **The correct recipe, stated positively.** To verify pre-fix behavior on a COMMITTED round, read the base blob — `git show <base-sha>:<path>` — or check out the base into a scratch path. Never stash. (For an UNCOMMITTED round, the implementer's own in-place stash of its own dirty tree is fine, because the push is not a no-op there — the distinction is committed-vs-uncommitted, and that is exactly what the #2546 brief got wrong.)
3. **Consider a guard.** A `PreToolUse` check on `git stash pop` / `git stash apply` / `git stash drop` from inside `.claude/worktrees/*` would make the hazard unreachable rather than merely documented, in the family of `guard_repo_root_branch.sh` and `guard_repo_root_pull.sh`. Whether to block outright or require an explicit override is a design call for the implementing round; `git stash push` should stay open.
4. Brief-writing guidance for agents that compose review briefs: the pre-fix verification recipe depends on whether the round is committed. Recommend base-blob reads by default.

## Scope

Rule-surface change plus an optional hook. Not an experiment; no GPU. Do NOT alter `sync_repo_root.py`'s `rebase.autoStash=true` — that setting is load-bearing for the #2182 rejected-push recovery and is not the defect here.

## Files of record

Task #2546 review round v15 (the near-miss; `epm:code-review v15`, and the reviewer's own memory commit `d5676c0358` recording the correct method). #1806 (stash-rescue audit pass — the recovery half). #2182 (`sync_repo_root.py` autostash). #2015 / `.claude/rules/repo-root-uncommitted-state.md` (the pre-commit stash race — a DIFFERENT mechanism sharing the word "stash"; do not dedupe onto it).
