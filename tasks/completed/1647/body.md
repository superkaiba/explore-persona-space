---
title: 'daily-fix: 9a-ter inline lint gate single-flight hook'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7cfa97f80c30
- daily-auto-filed
created_at: '2026-07-24T06:46:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): the Step 9a-ter inline
  payload lint gate composes background lint runs with no single-flight statement,
  same #1606 pattern distinct site'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-23 parked-candidate routing pass (Step C). Raised as a recursion-guard-parked prose follow-up on task #1627 (planner, plan v1 footer) — deliberately excluded from #1627's minimal edit, conditional on #1627 landing (it landed; #1627 is `completed`).

## Goal

Add a single-flight statement to the `/issue` SKILL.md Step 9a-ter inline payload lint-gate recipe, pointing at the Step 9c 1b single-flight statement, so concurrent inline lint runs cannot race (the #1606 pattern at a distinct site).

## Workflow gap

- **Bug observed:** the Step 9a-ter inline payload lint gate composes its own background lint runs with no single-flight statement — same #1606 hazard class the Step 9c gate got fixed for (a concurrent relaunch clobbers the live run's verdict), distinct site.
- **Why it is a workflow gap:** the 9a-ter recipe is the inline-round lint contract; without the probe, two inline rounds (or an inline round racing a Step 9c gate re-run) can double-launch the lint.
- **Confidence (emitter):** medium
- verified-at-filing: `awk '/9a-ter/,0' .claude/skills/issue/SKILL.md | head -400 | grep -n "single-flight\|pgrep"` → 0 hits inside the 9a-ter section (absence claim, in-target 0-hit is the evidence); the Step 9c gate DOES carry its single-flight probe (#1606 block present near SKILL.md line ~10064) — confirming the asymmetry (2026-07-24 UTC). #1627 (the named blocker) is `completed`.

## Proposed change (candidate diff sketch — refine in planning)

Add a short single-flight hook to the 9a-ter inline lint-gate recipe: before (re)launching, probe for a live gate process (the bracketed exact-issue-scoped pgrep form) and defer/wait per the Step 9c 1b statement.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 9a-ter § Inline payload lint gate)

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 7cfa97f80c30

- workflow_fix_target: .claude/skills/issue/SKILL.md

Origin: parked candidate on #1627 (2026-07-23T13:36:36Z): "the Step 9a-ter inline payload lint gate composes its own background lint runs with no single-flight statement — same #1606 pattern, distinct site."
