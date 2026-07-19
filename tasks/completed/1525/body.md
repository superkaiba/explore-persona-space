---
title: 'workflow-fix: sync_repo_root conflict-abort needs the full pointer-convergence
  defusal recipe'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7c447f2c0b16
created_at: '2026-07-19T00:16:01Z'
has_clean_result: false
origin_prompt: 'analyzer workflow-fix-candidate block on #1489 promotion pass (see
  body Provenance)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a formal `<!-- workflow-fix-candidate v1 -->` block raised on task #1489 (emitting agent: analyzer, clean-result promotion pass, 2026-07-18).

## Goal

Extend `scripts/sync_repo_root.py`'s conflict-abort guidance (or add a `--converge` mode) with the full defusal recipe for a genuine content conflict, so the stranded local commit stops re-mining every future sync and no session improvises working-tree materialization over shared task state.

## Workflow gap

- **Bug observed:** On a genuine content conflict sync_repo_root aborts and prints a manual next step (scratch worktree, resolve, push HEAD:main) that lands the CONTENT but leaves the conflicting commit in local main's history, where its replay re-conflicts on every future sync — and gives no guidance for converging the local pointer, inviting improvised (and risky) working-tree materialization over shared task state. Live incident (#1489 promotion pass, 2026-07-18): an eval_results/INDEX.md append conflict → scratch cherry-pick landed the content, then the improvised pointer convergence briefly reverted live task-state dirs for #1417/#1523 (restored within minutes).
- **Why it is a workflow gap:** the documented recovery is incomplete for the recurring append-conflict class (eval_results/INDEX.md rows) on the always-busy shared root; the missing pointer-convergence step is exactly where sessions improvise destructively.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c "scratch" scripts/sync_repo_root.py` → 2 hits (the abort message names the scratch-worktree step but no pointer-convergence recipe — presence-of-site binds); `git log --oneline --since='7 days ago' -- scripts/sync_repo_root.py` → 0 commits (no just-landed fix) (2026-07-18)

## Proposed change (candidate diff sketch — refine in planning)

```
+ On conflict, print (or execute under --converge):
+   git worktree add /tmp/sync-<pid> tmp-sync-branch   # branch -f at local main tip
+   git -C /tmp/sync-<pid> rebase origin/main          # resolve YOUR commit by taking origin's side (drops to empty)
+   git -C /tmp/sync-<pid> push origin HEAD:main
+   flock ~/.task-workflow/lock ... git reset --soft origin/main && <pathspec index sync>
+   # NEVER materialize origin-side tasks/ paths — local task state is often ahead of origin
```

## Scope / surfaces

- Primary target: `scripts/sync_repo_root.py`
- Note: a root `git reset --soft` interacts with the repo-root destructive-git guard rules — the planner must design the convergence step to stay inside the sanctioned envelope (soft reset only, flock-held, no working-tree overwrite of `tasks/`), and update `CLAUDE.md` § Concurrent repo-root committers if the documented recovery text changes.

## Constraints / invariants

- Workflow-surface only; `workflow_lint.py --check-asks` + ruff pass; the existing single-flight/rescue semantics preserved; never a repo-root `reset --hard` (CLAUDE.md hard rule).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/sync_repo_root.py
- fingerprint: 7c447f2c0b16

<!-- workflow-fix-candidate v1 -->
target_file: scripts/sync_repo_root.py
bug_observed: On a genuine content conflict sync_repo_root aborts and prints a manual next step (scratch worktree, resolve, push HEAD:main) that lands the CONTENT but leaves the conflicting commit in local main's history, where its replay re-conflicts on every future sync — and gives no guidance for converging the local pointer, inviting improvised (and risky) working-tree materialization over shared task state.
why_workflow_gap: The documented recovery is incomplete for the recurring append-conflict class (eval_results/INDEX.md rows): the stranded local commit permanently mines the shared sync path until the local main pointer is moved off it.
proposed_change: Extend the conflict-abort message (or add a --converge mode) with the full defusal recipe: rebase the local stack onto origin/main in a scratch worktree resolving the conflicting commit to origin's side (it rebases to empty and drops), push, then at the root under the ~/.task-workflow/lock flock run `git reset --soft origin/main` + pathspec-scoped, symlink-aware index/file sync from HEAD — explicitly warning never to write origin-side tasks/ content into the shared tree.
confidence: medium
related_task: #1489
<!-- /workflow-fix-candidate -->
