---
title: 'daily-fix: block worktree-local ''git -C <wt> merge main'''
kind: infra
tags:
- wf-fix
- wf-fix-fp:a74db93ab02c
- daily-auto-filed
created_at: '2026-07-20T06:46:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): worktree-local merges of
  LOCAL main not hook-blocked; #1530 deferred D6'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-19 parked-candidate sweep (Step C) from a workflow-fix candidate parked on task #1530 (emitting agent: planner Phase 1, prose follow-up; parked under the recursion guard).

## Goal

Extend `scripts/guard_repo_root_branch.sh` (or a sibling PreToolUse hook) to block `git -C <path-under-.claude/worktrees> merge main` / `merge --ff-only main` (LOCAL-main form only; `origin/main` forms stay open), with an env escape hatch.

## Workflow gap

- **Bug observed:** worktree-local merges of the LOCAL main branch are not blocked by any hook; #1530 root-caused the stale-local-main FF corruption class and deferred the mechanical block as plan D6.
- **Why it is a workflow gap:** a worktree merging the shared root's possibly-stale LOCAL `main` re-introduces the exact corruption class #1530 fixed at the commit site; the hook is the mechanical enforcement the rule text now prescribes only in prose.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "merge" scripts/guard_repo_root_branch.sh` → merge handling targets the repo ROOT only; `git -C <worktree>` is the documented deliberate-override (hook lines 38-49), so no clause blocks a worktree-scoped local-main merge (absence-of-guard claim; semantic probe: #1530's `concern-addressed` marker `d6-worktree-merge-main-hook-block-deferred` records the deliberate deferral). Landed-fix history: `git log --oneline --since='2026-07-17' -- scripts/guard_repo_root_branch.sh` → 28ffbede95 (#1538 pipe-chain waiver), bb4d72e7fb (#1530 commit-site guard), 8a556b1a7e (#1528 deny-event sidecar) — none adds a worktree local-main merge block (2026-07-19).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; see #1530 plan v1 §11 D6 for the recorded blast-radius concerns: blocking worktree merges risks wedging legitimate improvised Step 10d conflict recoveries — the plan should weigh an allowlist/escape hatch.)

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the workflow surface for the pattern before editing (`grep -rln 'merge main' .claude/ scripts/guard_repo_root*.sh`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- fingerprint: 5ccd189626d2

Verbatim parked candidate (epm:workflow-fix-candidate on #1530, 2026-07-19T09:07:17Z): "source: prose-followup (planner Phase 1). Candidate: extend guard_repo_root_branch.sh (or a sibling PreToolUse hook) to block 'git -C <path-under-.claude/worktrees> merge main' / 'merge --ff-only main' (LOCAL-main form only; origin/main forms stay open), with an env escape hatch. Rationale + blast-radius concerns: plan v1 §11 D6 (blocking worktree merges risks wedging legitimate improvised 10d conflict recoveries)."
