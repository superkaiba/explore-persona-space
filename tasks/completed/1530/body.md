---
title: 'workflow-fix: assert HEAD==main at task.py commit site'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4e2fbb227843
- daily-auto-filed
created_at: '2026-07-19T07:06:03Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): During a new_worktree.sh
  branch-creation window (2026-07-18 ~14:08-14:09Z), four task.py marker/status commits
  from three sessions landed on branch issue-1509 instead of main (9fb6221300, 76cbf6e3ce,
  07581c38ef, 8d9484a323 - all verified NOT ancestors of origin/main).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1509 (emitting agent: orchestrator-observation; parked under
the recursion guard, routed by the 2026-07-18 /daily Step C parked-candidate
sweep).

## Goal

Investigate the `new_worktree.sh` branch-creation window during which task.py
marker/status commits landed on branch `issue-1509` instead of `main`; add a
hard HEAD==main assert at the task_workflow COMMIT site (not just at path
resolution) that retries/fails loud instead of committing to a non-main HEAD,
and make `new_worktree.sh`'s branch creation never touch the shared commit
target's HEAD.

## Workflow gap

- **Bug observed:** During/just after
  `new_worktree.sh .claude/worktrees/issue-1509 issue-1509` (2026-07-18
  ~14:08-14:09Z; its output included an anomalous "Already on 'issue-1509'"
  line), FOUR task.py marker/status commits from THREE different sessions
  (this session's stage-dispatch breadcrumb 14:09:16Z + approved→running flip
  14:09:18Z; #1426's epm:merged 14:09:58Z; #1508's epm:code-review 14:10:49Z)
  landed on branch issue-1509 instead of main (commits 9fb6221300,
  76cbf6e3ce, 07581c38ef, 8d9484a323). Content was NOT lost (append-only
  files re-carried the rows in later main commits), but the branch's
  Step-10d own-diff carried foreign tasks/ state and had to be rebuilt
  (reset + cherry-pick) before merge.
- **Why it is a workflow gap:** task.py is documented as branch-guarded to
  main, yet its commits demonstrably landed on a feature branch for a ~95s
  window coinciding with worktree/branch creation — either new_worktree.sh
  transiently repoints the shared commit target's HEAD, or the task_workflow
  commit path (possibly via the `_task-main-pin` managed worktree) resolves
  the branch at a racy moment. Cross-session blast radius (3 sessions
  affected).
- **Confidence (emitter):** medium
- verified-at-filing: `git rev-parse --verify --quiet '<sha>^{commit}'` on all four cited commits → all resolve (9fb6221300234f..., 76cbf6e3ce6094..., 07581c38ef15b9..., 8d9484a323c52c...); `git merge-base --is-ancestor 9fb6221300 origin/main` → NOT an ancestor of origin/main (stranded-on-branch consistent with the claim); context read of the commit site (`src/explore_persona_space/task_workflow.py` ~L6260-6300) → the `git commit --only` call has NO immediate HEAD==main assert (branch safety rests on resolver-time routing + flock + the routed-path CAS `_advance_main_ref`), so the proposed commit-site assert is not landed (absence bind); `git log --oneline --since='7 days ago' -- scripts/new_worktree.sh` → 0 commits (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
+ # task_workflow commit path, immediately before `git commit`:
+ assert current_branch(repo_root) == "main", "refusing to commit task state on non-main HEAD"
+ # new_worktree.sh: create branch via `git worktree add -b <branch> <path> origin/main`
+ # (never a checkout that can transit the shared root/managed pin through the branch)
```

## Scope / surfaces

- Primary target: `scripts/new_worktree.sh, src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln '_task-main-pin\|Already on' .claude/ CLAUDE.md scripts/ src/explore_persona_space/task_workflow.py`)
  and reconstruct the 14:08-14:09Z race from the reflog before choosing the
  fix point; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only (`task_workflow.py` is an explicitly in-scope src/
  exception); never experiment code, `configs/`, or `tasks/` content edits.
- The fix must fail loud/retry, never silently drop a marker commit; existing
  flock + CAS semantics and `tests/test_task_workflow*.py` invariants hold.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: scripts/new_worktree.sh, src/explore_persona_space/task_workflow.py
- fingerprint: d92505bf78aa

<!-- workflow-fix-candidate v1 -->
target_file: scripts/new_worktree.sh, src/explore_persona_space/task_workflow.py
bug_observed: During/just after `new_worktree.sh .claude/worktrees/issue-1509 issue-1509` (2026-07-18 ~14:08-14:09Z; its output included an anomalous "Already on 'issue-1509'" line), FOUR task.py marker/status commits from THREE different sessions (this session's stage-dispatch breadcrumb 14:09:16Z + approved→running flip 14:09:18Z; #1426's epm:merged 14:09:58Z; #1508's epm:code-review 14:10:49Z) landed on branch issue-1509 instead of main (commits 9fb6221300, 76cbf6e3ce, 07581c38ef, 8d9484a323). Content was NOT lost (append-only files re-carried the rows in later main commits), but the branch's Step-10d own-diff carried foreign tasks/ state and had to be rebuilt (reset + cherry-pick) before merge.
why_workflow_gap: task.py is documented as branch-guarded to main, yet its commits demonstrably landed on a feature branch for a ~95s window coinciding with worktree/branch creation — either new_worktree.sh transiently repoints the shared commit target's HEAD, or the task_workflow commit path (possibly via the _task-main-pin managed worktree) resolves the branch at a racy moment. Cross-session blast radius (3 sessions affected).
proposed_change: Investigate the window (new_worktree.sh branch-creation sequence + task_workflow commit-target resolution); add a hard HEAD==main assert at the task_workflow COMMIT site (not just at path resolution) that retries/fails loud instead of committing to a non-main HEAD, and make new_worktree.sh's branch creation never touch the shared commit target's HEAD.
diff_sketch: |
  + # task_workflow commit path, immediately before `git commit`:
  + assert current_branch(repo_root) == "main", "refusing to commit task state on non-main HEAD"
  + # new_worktree.sh: create branch via `git worktree add -b <branch> <path> origin/main`
  + # (never a checkout that can transit the shared root/managed pin through the branch)
confidence: medium
related_task: #1509
<!-- /workflow-fix-candidate -->
