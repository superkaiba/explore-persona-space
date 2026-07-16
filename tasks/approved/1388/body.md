---
title: 'daily-fix: fix bare list_repo_tree lint offenders'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-16T07:20:00Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): two bare list_repo_tree
  offenders fail step9c lint fleet-wide'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 parked-candidate routing pass (Step C) from recursion-guard-parked prose follow-ups on tasks #1368 (+ its 01:09Z addendum) and #1369. NOT a workflow-surface fix — experiment scripts; filed as an ordinary infra fix task.

## Goal

Fix the two pre-existing bare-`.list_repo_tree(` hub-verify lint offenders on main — `scripts/issue1073_greedy_cloud_distribution.py:102` (landed in 1b52b9f29e) and `scripts/issue1092_inline_operator_stage.py:92` — via the hub-helper reroute or a `HUB_VERIFY_RETRY_EXEMPT` waiver, so `tests/test_workflow_lint.py` stops failing 3x at every Step 9c gate fleet-wide.

## Workflow gap

- **Bug observed:** two experiment scripts call `api.list_repo_tree(` bare (no hub-helper retry wrapper), failing the hub-verify lint pinned by tests/test_workflow_lint.py on pristine main — every session's Step 9c gate carries the pre-existing failure.
- **Why it matters:** red-on-main lint pins force per-session pre-existing-failure discrimination and can mask genuine regressions.
- **Confidence (emitter):** high (mechanical lint evidence)
- verified-at-filing: `grep -n '\.list_repo_tree(' scripts/issue1073_greedy_cloud_distribution.py` → 1 hit at :102; `sed -n '88,95p' scripts/issue1092_inline_operator_stage.py | grep list_repo_tree` → hit at :92; broader `grep -rn 'list_repo_tree(' scripts/` shows additional callers (issue779, issue922_common, issue1332, issue833 ×2, issue1092_dispatch.sh) — the planner scopes which are lint-flagged vs exempt/allowlisted (2026-07-16 UTC)

## Proposed change (candidate diff sketch — refine in planning)

Reroute the two named offenders through the project hub helper (retry-wrapped list_repo_tree) OR add HUB_VERIFY_RETRY_EXEMPT waivers with an inline reason; run the lint to confirm the Step 9c baseline goes green; audit the other grep hits against the lint's actual flag set.

## Scope / surfaces

- Primary target: `scripts/issue1073_greedy_cloud_distribution.py, scripts/issue1092_inline_operator_stage.py`
- NOTE: two sibling module-top-import offenders named in #1369/#1371 (issue1073_fig_linear_nonlinear.py, issue1092_inline_operator_figure.py) were verified ALREADY FIXED on current main at filing time (test_no_new_torch_before_dotenv_vm_entrypoints passes) — out of this task's scope.

## Constraints / invariants

- Experiment-script fix only; no workflow-surface edits.
- Never weaken the lint; fix the offenders or add a reasoned waiver.

## Provenance

parked prose follow-ups (verbatim excerpts): #1368 2026-07-16T00:23:52Z: "target_file: scripts/issue1073_greedy_cloud_distribution.py (out-of-scope experiment script) — pre-existing hub-verify lint offender at L102 (bare .list_repo_tree(, landed on main in 1b52b9f29e) fails tests/test_workflow_lint.py 3x at every Step 9c gate fleet-wide; fix = hub-helper reroute or HUB_VERIFY_RETRY_EXEMPT waiver." — #1368 addendum 2026-07-16T01:09:58Z: "the lint-gate baseline surfaced a SECOND pre-existing bare-list_repo_tree offender on main — scripts/issue1092_inline_operator_stage.py:92 — alongside issue1073:102; the /daily parked-candidate routing pass should scope the fix to BOTH."
