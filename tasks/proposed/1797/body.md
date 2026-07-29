---
title: 'daily-fix: refusal ladder — goal channel + content-trigger r'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cb0051488fb6
- daily-auto-filed
- trigger-dense
created_at: '2026-07-29T07:10:14Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): Refusal storms cost ~3h
  across 4 sessions on 2026-07-28 (#1769: 5 dead planner-phase spawns + orchestrator
  wedge; #1774: 3 implementer kills + 1 thrash + wedge; #1776: 2 kills at the report
  turn; #1092: 1 first-pass kill): the trigger-dense goal: frontmatter rides verbatim
  into every brief; a unit whose CONTENT was demonstrated as the trigger was re-spawned
  on the default model anyway; steering-sc'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Sources: group-E P1/P2, group-G P2, group-J P1 (4 independent miners).

## Goal

Extend the refusal-prevention surface with three measured 2026-07-28 gaps: Goal-frontmatter neutralization, a demonstrated-content-trigger fast path, and steering-surface read discipline.

## Workflow gap

- **Bug observed:** Refusal storms cost ~3h of wall across 4 sessions on 2026-07-28. #1769: 5 dead planner-phase spawns (3 usage-policy, 1 thrash, 1 degenerate) then an orchestrator refusal-wedge at 22:51Z — the task's `goal:` frontmatter itself carries intervention-control phrasing and rode verbatim into every brief. #1774: 3 implementer refusal kills + 1 sonnet thrash; round A ran clean at 100 calls while round B (the steering-module content) was killed at 14 calls — the content trigger was demonstrated, yet the next spawn went to the default model anyway. #1776: 2 kills at the final report turn. #1769's owner session was also killed while grepping steering-script hyperparameters (paging intervention-control vocabulary). (Kill counts are transcript-mined tallies — unverified hypothesis — verify at plan time against the sessions' events/markers; tasks #1769/#1774 both confirmed recovered by successor sessions.)
- **Why it is a workflow gap:** the existing ladder (CLAUDE.md § Spurious usage-policy refusals) covers brief VOCABULARY (rung e) and post-kill recovery (b/b2), but not the Goal-frontmatter channel, not a skip-ahead once content (not model pathway) is the demonstrated trigger, and the trigger-dense-review recognition heuristic omits steering surfaces.
- **Confidence (emitter):** medium (mechanisms inferred from kill patterns, not A/B-probed)
- verified-at-filing: `grep -c 'FIRST-PASS briefs carry BOTH disciplines' CLAUDE.md` → 1; rung (b2) present (grep 2 hits); `grep -n 'intervention-control phrasing' CLAUDE.md` → clause (e) names brief phrasing only, no goal-channel clause; trigger-dense-review.md recognition heuristic has no steering-surface bullet (read 2026-07-29 UTC). NOTE: the first-pass real-corpus brief discipline half of this incident class ALREADY LANDED as #1748 (§ Real-corpus datagen briefs) — deliberately excluded from this filing.

## Proposed change (candidate diff sketch — refine in planning)

See the three amendments in the title fields; each is a short additive clause to an existing section — no ladder restructuring.

## Scope / surfaces

- Primary targets: `CLAUDE.md` (§ Spurious usage-policy refusals, clauses (e)/(b2)), `.claude/rules/trigger-dense-review.md` (recognition heuristic + § Orchestrator ordinary turns)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: cb0051488fb6

- workflow_fix_target: CLAUDE.md

