---
title: 'daily-fix: Step 10d/9c — re-sync before SHA-bound gates'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ef4fffdfe417
- daily-auto-filed
created_at: '2026-07-29T07:14:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): three same-day orderings
  forced full gate re-runs: #1750 and #1766 hit the post-gate spec-freshness re-sync
  moving the tip AFTER the SHA-bound gate bound it (merge refused, ~19-min gate re-run
  guaranteed whenever a freshness commit is needed); #1742''s Step 9c gate round 1
  failed on a stale worktree spec because the Step 5a sync fires hours before the
  gate; plus the Step 10d verdict-read diagnostic'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Sources: group-G P4, group-I P4 + P5, group-C P11 (3 miners, 3 issues).

## Goal

Reorder the Step 10d/9c spec-freshness re-syncs relative to the SHA-bound gates so a needed freshness commit no longer guarantees a full gate re-run.

## Workflow gap

- **Bug observed:** (a) #1750 + #1766 (independent sessions, same day): the Step 10d post-gate re-sync committed spec-freshness files AFTER the lint gate bound the tip SHA → squash refused ('Base branch was modified' / 'Head branch is out of date') → full ~19-min gate re-run. The documented shape-3 handler covers same-tip lag only — this sequence GUARANTEES a moved tip whenever the re-sync syncs >0 files. (b) #1742: Step 9c gate round 1 failed on a stale worktree copy of planner.md whose fix (#1740) landed on main hours after the Step 5a sync — a 55-min second gate run. (c) The Step 10d verdict-read diagnostic compound exits nonzero on a PASS verdict (empty-file grep legs), reading as a tool error.
- **Why it is a workflow gap:** the re-sync is by design inside the merge step, after the gate (SKILL.md:11572/11755); no pre-9c re-sync step exists; the verdict-read compound's tail legs return nonzero on PASS.
- **Confidence (emitter):** medium-high ((a)/(b) probed by miners against SKILL.md line anchors; (c) inferred)
- verified-at-filing: `grep -n 'post-gate re-sync' .claude/skills/issue/SKILL.md` → 11572 + 11755 (inside the merge step, after gate binding); miner-I probe: Step 5a sync documented at ~1200/2359 with no pre-9c re-sync step (label: re-verify the exact insertion point at plan time) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Move/reorder the re-sync ahead of gate tip-binding at Step 10d (keeping fail-closed semantics), add a pre-9c freshness re-sync line, and `; true`-terminate the verdict-read compound. Keep the #1714 lockstep pin tests green.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 10d merge block; Step 9c launch block)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: ef4fffdfe417

- workflow_fix_target: .claude/skills/issue/SKILL.md

