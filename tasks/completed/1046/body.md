---
title: 'daily-fix: Step 9c timeout sizing + slow-file batch + rc hyg'
kind: infra
tags:
- wf-fix
- wf-fix-fp:eeb56cea365b
- daily-auto-filed
created_at: '2026-07-05T07:02:24Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): On 2026-07-04 Step 9c gate
  runs were timeout-killed (exit 143) in three sessions: #991 twice at 480s (test_workflow_lint.py
  too slow for a shared batch), #996 at its 540s bound (41-file subset), #906 narrowed
  scope after kills; separately #969''s gate reported a false Exit 1 from a trailing
  informational grep with no match, and #928''s Step 9a-quater gist snippet reports
  Exit 1 on success (trailing'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Size the Step 9c gate timeout by selected-file count (or raise the default), give test_workflow_lint.py its own longer-timeout batch, and fix recipe exit-code hygiene: || true (or if-form) after informational trailing greps and after the 9a-quater gist [ -z "$GIST_URL" ] test.

## Workflow gap

- **Bug observed:** On 2026-07-04 Step 9c gate runs were timeout-killed (exit 143) in three sessions: #991 twice at 480s (test_workflow_lint.py too slow for a shared batch), #996 at its 540s bound (41-file subset), #906 narrowed scope after kills; separately #969's gate reported a false Exit 1 from a trailing informational grep with no match, and #928's Step 9a-quater gist snippet reports Exit 1 on success (trailing [ -z ... ] && printf test).
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md, scripts/select_step9c_tests.py`
- Sessions: a873fc2d (#991, 18:53+19:03 UTC), b1484eb1 (#996, 21:33 UTC), 5498934e (#969, 08:32 UTC), 20a0c53b (#928, 09:54 UTC). Related: #1022 (known-red-on-main baseline ledger) covers the pre-existing-failure probes, not timeouts.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, scripts/select_step9c_tests.py
- source: /daily 2026-07-04 problem sweep (transcript-mined)
