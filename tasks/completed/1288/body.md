---
title: 'daily-fix: Step 10d merge contention in same-batch fleets'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d1856d16bb7a
- daily-auto-filed
created_at: '2026-07-13T06:44:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): 4/4 same-batch wf-fix sessions
  on 2026-07-12 failed their first gh pr merge --rebase under fleet churn (0/4 rebase
  success, one real merge-conflict recovery, ~30 min + ~7 redundant lint-gate re-runs
  aggregate); the watcher cap-fills dispatches so sibling sessions race Step 10d by
  construction.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 problem sweep (transcript-mined across 8 same-batch workflow-fix sessions: #1275/#1276/#1278/#1279/#1280/#1281/#1282/#1283).

## Goal

Stop same-batch autonomous infra fleets from racing each other through Step 10d — cut the recurring ~30 min/night of merge-conflict recovery and redundant lint-gate re-runs.

## Workflow gap

- **Bug observed:** on 2026-07-12, all 4 sessions of the 07:5x batch (#1275/#1276/#1278/#1279) failed their first `gh pr merge --rebase` ("Base branch was modified" / "can't be rebased" / real merge conflicts on `tasks/REGISTRY.json` + rename/rename on sibling task folders); 0/4 rebase-merges succeeded, all fell to `--squash`; #1278 needed a full scratch-worktree conflict recovery (~16 min). Same class the 07-11 daily counted in 10/24 sessions. Aggregate cost today ≈30 min wall + ~7 redundant lint-gate certification re-runs; a side effect was a transient duplicate `tasks/planning/1280` folder (the #644/#1253 class, auto-remediated by the post-merge guard).
- **Why it is a workflow gap:** the watcher's `proposed_infra_sweep` dispatches up to the 5-session cap in one pass, so sibling infra sessions complete within minutes of each other; each one's status-move commits (git mv + REGISTRY rewrite) land on main between a sibling's lint certification and its merge attempt. Step 10d has no cross-session coordination, so the contention is guaranteed by construction, not incidental.
- **Confidence (emitter):** medium (the mechanism is confirmed; the best remedy is a design choice for the planner — merge-lock vs dispatch stagger vs squash-first default).
- verified-at-filing: transcript evidence 2026-07-12 sessions 004c5304/6fc58be1/aaf2482e/06281fbd (0/4 first-attempt merges; conflict quotes in #1278's transcript at 07:59–08:15Z); behavioral gap — not grep-verifiable as a single pattern.

## Proposed change (candidate diff sketch — refine in planning)

Options for the planner to weigh (any one suffices):
1. A shared merge flock (e.g. `~/.task-workflow/step10d-merge.lock`) held around the Step 10d certify→merge window, so sessions serialize the final stretch;
2. Stagger `proposed_infra_sweep` dispatches (e.g. one per watcher pass instead of cap-filling one pass);
3. Default Step 10d to `--squash` for autonomous infra sessions (rebase is 0/N under fleet churn anyway), keeping the certification SHA-bound retry ladder as-is.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 10d), `scripts/autonomous_session_watch.py` (proposed_infra_sweep dispatch pacing).
- Related shipped work: #1280 (Guard-1 three-dot scoping) removes the false-positive strip pressure but not the merge contention itself.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes.
- Must not reintroduce repo-root branch work (the guard-blocked class) or force-push.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: d1856d16bb7a

- workflow_fix_target: .claude/skills/issue/SKILL.md

Origin: /daily 2026-07-12 transcript sweep (miner reports, sessions 004c5304/6fc58be1/aaf2482e/06281fbd + 3077c244/d78028e0/b49bb04a); 07-11 daily recorded the same class at 10/24 sessions.
