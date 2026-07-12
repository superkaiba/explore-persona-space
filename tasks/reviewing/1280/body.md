---
title: 'workflow-fix: Step 10d Guard 1 three-dot foreign diff'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3f8b0738867d
- daily-auto-filed
created_at: '2026-07-12T06:52:24Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): Step 10d Guard 1 computes
  FOREIGN from the two-endpoint diff git diff MAIN_SHA HEAD -- tasks/, which on ANY
  behind branch under fleet marker churn lists main-side advancement (33 false-positive
  paths on #1271, whose replayed commit set touched ZERO tasks/ paths); following
  the prescribed checkout+strip-commit there would stage main-advancement content
  into a new branch commit whose server-side rep'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-11 parked-candidate routing pass from a FORMAL workflow-fix candidate block parked on task #1271 (the Step C sweep missed it — see the sibling `sweep-park-predicate` filing; the /daily transcript miner surfaced it).

## Goal

Scope Step 10d Guard 1's FOREIGN set to the branch's OWN commits — three-dot `git diff --name-only origin/main...HEAD -- tasks/` — so the strip fires only when a replayed commit actually touches a foreign tasks/ path.

## Workflow gap

- **Bug observed:** Step 10d Guard 1 computes FOREIGN from the two-endpoint diff `git diff MAIN_SHA HEAD -- tasks/`, which on ANY behind branch under fleet marker churn lists main-side advancement (33 false-positive paths on #1271, whose replayed commit set touched ZERO tasks/ paths); following the prescribed checkout+strip-commit there would stage main-advancement content into a new branch commit whose server-side replay CONFLICTS (the #1128 shape) — the strip would create the very conflict it exists to prevent.
- **Why it is a workflow gap:** the guard's diff form conflates "branch carries foreign tasks/ changes in its replayed commits" (the real hazard) with "main advanced since the merge-base" (benign; rebase keeps main's version of untouched files).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'diff --name-only "$MAIN_SHA"' .claude/skills/issue/SKILL.md` → 3 hits (lines 8673 prose, 8731 + 9840 code snippets), all two-endpoint form (2026-07-12).

## Proposed change (candidate diff sketch — refine in planning)

```diff
- if ! git -C "$WT" -c core.quotePath=false diff --name-only "$MAIN_SHA" HEAD -- 'tasks/' \
+ if ! git -C "$WT" -c core.quotePath=false diff --name-only "$MAIN_SHA"...HEAD -- 'tasks/' \
    > /tmp/issue-<N>-guard1-tasks-diff.txt; then
  (three-dot: merge-base..HEAD = the replayed set; two-endpoint form retired)
```

Related context for the planner (prose follow-up from today's fleet, 5-of-5 first-merge failures in one miner batch): consider also reducing foreign-`tasks/` accumulation at the SOURCE — re-pin the branch's `tasks/` tree to origin/main at PR-open time — so the merge-time strip/certification pass is a no-op under churn. Same Step 10d surface; the plan review decides whether to bundle or defer.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 10d Guard 1 — update BOTH code snippets at ~8731 and ~9840 plus the prose at ~8673)
- Grep the workflow surface for the two-endpoint pattern before editing (`grep -rn 'MAIN_SHA" HEAD' .claude/`) and update every hit.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + default run pass.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 3f8b0738867d

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: Step 10d Guard 1 computes FOREIGN from the two-endpoint diff `git diff MAIN_SHA HEAD -- tasks/`, which on ANY behind branch under fleet marker churn lists main-side advancement (33 false-positive paths on #1271, whose replayed commit set touched ZERO tasks/ paths); following the prescribed checkout+strip-commit there would stage main-advancement content into a new branch commit whose server-side replay CONFLICTS (the #1128 shape) — the strip would create the very conflict it exists to prevent.
why_workflow_gap: the guard's diff form conflates "branch carries foreign tasks/ changes in its replayed commits" (the real hazard) with "main advanced since the merge-base" (benign; rebase keeps main's version of untouched files).
proposed_change: scope Guard 1's FOREIGN set to the branch's OWN commits — three-dot `git diff --name-only origin/main...HEAD -- tasks/` (or per-commit `git show` over `origin/main..HEAD`) — so the strip fires only when a replayed commit actually touches a foreign tasks/ path; keep the reset-to-MAIN_SHA recipe for that genuine case.
diff_sketch: |
  - if ! git -C "$WT" diff --name-only "$MAIN_SHA" HEAD -- 'tasks/' \
  + if ! git -C "$WT" diff --name-only "$MAIN_SHA"...HEAD -- 'tasks/' \
      > /tmp/issue-<N>-guard1-tasks-diff.txt; then
  (three-dot: merge-base..HEAD = the replayed set; two-endpoint form retired)
confidence: high
related_task: #1271
<!-- /workflow-fix-candidate -->
