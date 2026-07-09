---
title: 'workflow-fix: root rebase/cherry-pick fence in branch guard'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4e821145fafd
- daily-auto-filed
created_at: '2026-07-09T07:00:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Root `git rebase <branch>`
  (and `git cherry-pick`) is a pre-existing ungated tree/history-mutating verb on
  the same shared-repo-root threat model the #1128 merge fence closes — the guard''s
  own comments defer it as a separate candidate.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1128 (recursion-guarded workflow-fix session).

## Goal

Add a root-rebase/cherry-pick fence sibling to the #1128 merge detector in guard_repo_root_branch.sh (allow `--abort` forms; same worktree-path carve-outs) plus test coverage mirroring the merge-fence tests.

## Workflow gap

- **Bug observed:** Root `git rebase <branch>` (and `git cherry-pick`) is a pre-existing ungated tree/history-mutating verb on the same shared-repo-root threat model the #1128 merge fence closes — the guard's own comments defer it as a separate candidate.
- **Why it is a workflow gap:** A conflicting root rebase strands conflict markers in the shared tree exactly like the #1090 root merge did, and the guard already owns this threat model.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

In guard_repo_root_branch.sh verb detection (~l.517 family):
+ also match `git rebase <ref>` / `git cherry-pick <ref>` at the repo root
+ allow `git rebase --abort` / `git cherry-pick --abort|--quit`
+ mirror the #1128 merge-fence tests for the new verbs

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- origin: parked candidate on task #1128 at 2026-07-08T07:47:07Z

Verbatim parked note:

source: prose-followup (methodology critic round 1, task #1128). target_file: scripts/guard_repo_root_branch.sh. bug_observed: root 'git rebase <branch>' (and cherry-pick) is a pre-existing ungated tree/history-mutating verb on the same shared-root threat model the #1128 merge fence closes. proposed_change: a root-rebase fence sibling to the #1128 merge detector. routed: parked — this session is itself a workflow-fix session (wf-fix tag on #1128; recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard) and never auto-files more workflow-fix tasks; surfaced for the next orchestrator/human pass. Distinct fingerprint from #1128 (different verb family).
