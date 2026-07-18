---
title: 'daily-fix: mechanical inline payload lint gate pre-push'
kind: infra
tags:
- wf-fix
- wf-fix-fp:72453e10696d
- daily-auto-filed
created_at: '2026-07-18T06:46:48Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): second occurrence of the
  #1388 class: inline rounds landed dotenv-violating issue1092 scripts on main, Step
  9c red fleet-wide most of 2026-07-17, then 4 NEW violations the same evening — the
  inline payload lint gate exists only as SKILL.md prose with no mechanical enforcement.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-17 (route 2) from a transcript-mined recurrence of the #1388 incident class: inline user-chat analysis rounds landing lint-red scripts on `main`, breaking the Step 9c gate fleet-wide.

## Goal

Add mechanical enforcement of the Step 9a-ter "inline payload lint gate" so an inline round cannot push non-artifact payload (scripts/src files) to `main` without the mapped lint/test scan passing — instead of relying on the orchestrator remembering the SKILL.md recipe.

## Workflow gap

- **Bug observed:** on 2026-07-17 the fleet spent most of the day with Step 9c red because inline-landed `scripts/issue1092_*.py` figure scripts violated `tests/test_shared_vm_thread_caps.py` (the #847 dotenv-before-heavy-imports pin): up to 11 scripts red hitting the #1457/#1422/#1447 gates; #1428 fixed that set, and the same evening the #1092 fair-deep-dive inline leg landed 4 NEW violations (resolved ~01:00Z 07-18). This is the second occurrence of the #1388 class ("two inline-landed lint-red scripts broke the Step 9c gate fleet-wide").
- **Why it is a workflow gap:** the inline payload lint gate exists only as SKILL.md prose (Step 9a-ter § Inline payload lint gate); nothing mechanical blocks the push, so each inline round re-relies on orchestrator discipline and the failure recurs fleet-wide.
- **Confidence:** medium
- verified-at-filing: `grep -rln "inline payload lint" .claude/hooks/ .claude/settings.json` → 0 hits (no hook-level enforcement exists; absence-of-guard claim — the 0-hit result is the evidence); the recipe exists in prose (`grep -l "Inline payload lint gate" .claude/skills/issue/SKILL.md` → present). Both scripts from the evening recurrence read fixed on main at compose time (load_dotenv before numpy/matplotlib), i.e. the INSTANCES are resolved — this filing is for the mechanical-enforcement gap, not tonight's instances. Run 2026-07-18 UTC.

## Proposed change (candidate diff sketch — refine in planning)

Options for the planner: (a) a PreToolUse Bash hook that intercepts `git push` from the repo root when the pending commit set touches `scripts/*.py`/`src/**` outside a worktree-branch flow and runs the no-flags `workflow_lint.py` + the `select_step9c_tests.py --map-files` mapped scan first; (b) a lighter variant scoped to inline rounds via a sentinel; (c) a server-side/CI check. The planner picks the minimal reliable shape.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 9a-ter), `.claude/settings.json` hooks, possibly a new `scripts/` guard helper.
- Cross-check the #1388 task record and the existing guard-hook family for composition.

## Constraints / invariants

- Workflow-surface only. Hook changes are behavior changes — full pipeline review is the point of this route-2 filing.
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 72453e10696d

- workflow_fix_target: .claude/skills/issue/SKILL.md

source: /daily 2026-07-17 transcript sweep (chunk-3 miner) — fleet-wide Step-9c red from inline-landed dotenv-violating scripts, #1388 class recurrence, plus 4 new same-evening violations from a second inline leg.
