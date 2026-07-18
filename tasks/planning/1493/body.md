---
title: 'daily-fix: extend squash-first merge routing to experiments'
kind: infra
tags:
- wf-fix
- wf-fix-fp:03d919b1251d
- daily-auto-filed
created_at: '2026-07-18T06:45:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): kind:experiment branches
  keep --rebase at the Step 9b/10d merge and predictably burn attempt 1 under fleet
  marker churn — 10/10 conflicted experiment merges since 2026-07-12 had the first
  --rebase refused and landed --squash at attempt 2; the SKILL.md merge-form rationale
  still says ''the empirical record does not cover them'' (SKILL.md ~line 10077),
  which is now stale.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-17 parked-candidate sweep (Step C) from a formal workflow-fix candidate parked on task #1065 (emitting agent: planner prose-followup, plan v1 residual risk 1; fact-checked 9/9 CONFIRMED in that session). #1065 itself was archived as superseded-by-intervening-fixes, and its archival note explicitly hands THIS residual to the nightly sweep.

## Goal

Extend the #1288 squash-first MERGE_FORM routing to `kind: experiment` Step 9b merges — or, minimally, update the stale "empirical record does not cover them" rationale clause with the 07-12→07-17 evidence and a revisit criterion.

## Workflow gap

- **Bug observed:** kind:experiment branches keep `--rebase` at the Step 9b/10d merge and predictably burn attempt 1 under fleet marker churn — 10/10 conflicted experiment merges since 2026-07-12 (#1005, #1090, #1092, #1310, #1315, #1332, #1335, #1345, #1415, #1434) had the first `--rebase` refused, ran the in-worktree recovery, and landed `--squash` at attempt 2; the SKILL.md merge-form rationale still says "the empirical record does not cover them", which is now stale.
- **Why it is a workflow gap:** the #1288 squash-first routing deliberately excluded experiments on a no-evidence rationale that the post-07-12 record has since filled in; the exclusion now costs a predictably-doomed rebase attempt + recovery machinery per conflicted experiment merge with no realized per-commit-revert benefit (every conflicted case lands squashed anyway).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "empirical record does not cover" .claude/skills/issue/SKILL.md` → 1 hit at line 10082 (the stale rationale clause is present in-target) (2026-07-18 UTC). Empirical premise re-derived by the #1065 session's fact-checker 2026-07-17 (7 CONFIRMED, 3/3 sampled); #1065's archival note (2026-07-17T23:45:07Z) cites zero `epm:merge-failed` since 2026-07-05 and 210 `epm:merged` since 2026-07-12 (160 attempt-1 / 20 attempt-2 / 3 attempt-3).

## Proposed change (candidate diff sketch — refine in planning)

```
case "$TASK_KIND" in infra|batch) MERGE_FORM=--squash ;; esac
+ # experiments: 10/10 conflicted --rebase merges since 2026-07-12 landed
+ # --squash at attempt 2 (see #1065 plan v1 §iii-b); either extend squash
+ # routing to experiment|analysis or keep --rebase for the clean-path
+ # per-commit revert value and update the rationale clause.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the merge-form routing sites before editing (`grep -rn 'MERGE_FORM' .claude/ scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- NOTE: the planner may legitimately deflect toward the minimal option (update the stale clause + revisit criterion, keep `--rebase` for the clean path) — 160/210 merges since 07-12 landed at attempt 1, so the rebase path is only a tax on the CONFLICTED subset.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 683132017c58

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: kind:experiment branches keep --rebase at the Step 9b/10d merge and predictably burn attempt 1 under fleet marker churn — 10/10 conflicted experiment merges since 2026-07-12 (#1005, #1090, #1092, #1310, #1315, #1332, #1335, #1345, #1415, #1434) had the first --rebase refused, ran the in-worktree recovery, and landed --squash at attempt 2; the SKILL.md merge-form rationale still says 'the empirical record does not cover them' (SKILL.md ~line 10077), which is now stale.
why_workflow_gap: the #1288 squash-first routing deliberately excluded experiments on a no-evidence rationale that the post-07-12 record has since filled in; the exclusion now costs a predictably-doomed rebase attempt + recovery machinery per conflicted experiment merge with no realized per-commit-revert benefit (every conflicted case lands squashed anyway).
proposed_change: extend #1288's squash-first MERGE_FORM routing to kind:experiment Step 9b merges (or, minimally, update the stale 'empirical record does not cover them' clause with the 07-12→07-17 evidence and a revisit criterion).
confidence: medium
related_task: #1065
<!-- /workflow-fix-candidate -->
