---
title: 'workflow-fix: dispatch repo-branch default should use issue worktree branch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1d54c293b213
created_at: '2026-07-01T23:49:01Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #812: dispatch_issue.py repo-branch
  auto-default inert when invoked from repo root (main); GCE cloned main, scripts
  absent, rc=2 crash att-20260701-232740'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #812 (emitting agent: /issue orchestrator, after the
gcp_clone_wrong_branch failure att-20260701-232740).

## Goal

When `--repo-branch` is absent and the invoking checkout is on `main`, `scripts/dispatch_issue.py` should default `repo_branch` to the issue worktree's checked-out branch (resolved via the already-computed `_issue_worktree_git_root(args.issue)`), not silently ship `main`.

## Workflow gap

- **Bug observed:** A dispatch invoked from the repo root (on `main`) for task #812 — whose approved code lives on the `issue-812` worktree branch — silently cloned `main` on the GCE instance. `scripts/issue812_*.py` were absent, the workload crashed rc=2 within 3 minutes, and the instance powered off (crash diagnostics at HF `issue812_partial/att-20260701-232740/`).
- **Why it is a workflow gap:** `dispatch_issue.py`'s repo-branch auto-default (the #535 r6 / fix19 mirror) reads `_current_git_branch()` — the branch of the INVOKING checkout — and only fires when that is non-`main`. But the orchestrator/experimenter conventionally runs from the shared repo root, which is pinned to `main` by project rule, so the auto-default is inert in exactly the common case. The function ALREADY resolves `_issue_worktree_git_root(args.issue)` a few lines below (for the #685 git-check root); the branch of that worktree is one `git -C <root> branch --show-current` away.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
# scripts/dispatch_issue.py, in the repo-branch defaulting elif:
  branch = _current_git_branch()
+ if (not branch or branch == "main"):
+     wt_root = _issue_worktree_git_root(args.issue)
+     if wt_root is not None:
+         wt_branch = _git_branch_of(wt_root)   # git -C <root> branch --show-current
+         if wt_branch and wt_branch != "main":
+             branch = wt_branch
  if branch and branch != "main":
      logging...info("repo-branch defaulted to %r ...", branch)
      extra["repo_branch"] = branch
```

Also verify the pushed-branch guard still applies (the GCE clone pulls from
origin; keep/extend the "ensure it is pushed" INFO, or harden to a
`git ls-remote` check that FAILS loud when the worktree branch is not pushed).

## Scope / surfaces

- Primary target: `scripts/dispatch_issue.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln '_current_git_branch' scripts/ src/explore_persona_space/backends/`) and update every hit;
  list them in the plan. Consider whether `backends/issue_dispatch.py` has the same defaulting logic.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Explicit `--repo-branch` always wins; the new worktree-branch default fires ONLY when the flag is absent AND the invoking checkout is on `main` AND an issue worktree exists on a non-main branch.
- SLURM's `_assert_repo_branch_synced` refusal behavior must be preserved (a non-main repo_branch still makes SLURM refuse and the auto chain advance).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/dispatch_issue.py
- fingerprint: 1d54c293b213

<!-- workflow-fix-candidate v1 -->
target_file: scripts/dispatch_issue.py
bug_observed: Dispatch from repo root (on main) for issue 812 whose code lives on the issue-812 worktree branch silently cloned main on the GCE instance; scripts absent, workload crashed rc=2
why_workflow_gap: dispatch_issue.py's repo-branch auto-default reads _current_git_branch() (the invoking checkout) and is inert when invoked from the repo root pinned to main — even though _issue_worktree_git_root(args.issue) is already resolved a few lines below and carries the correct branch
proposed_change: When --repo-branch is absent and the invoking checkout is on main, default repo_branch to the issue worktree checked-out branch (via _issue_worktree_git_root) instead of only _current_git_branch()
diff_sketch: |
  + if (not branch or branch == "main"):
  +     wt_root = _issue_worktree_git_root(args.issue)
  +     if wt_root is not None:
  +         wt_branch = _git_branch_of(wt_root)
  +         if wt_branch and wt_branch != "main":
  +             branch = wt_branch
confidence: high
related_task: #812
<!-- /workflow-fix-candidate -->
