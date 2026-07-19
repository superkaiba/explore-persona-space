---
title: 'clear pre-existing #1481 lint reds (hub-verify-retry + phase-done waivers)'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-19T18:29:48Z'
has_clean_result: false
origin_prompt: 'code-reviewer round-1 prose follow-up on #1547: pre-existing issue1481_cjk_audit.py:88
  (hub-verify-retry) + issue1481_phasec.sh:43/49 (phase-done-reserved) keep the fleet-wide
  no-flags workflow_lint run red; 1-2-line waiver/wrap clears it.'
workflow: v1
---
## Overview / Motivation

Filed by the /issue 1547 orchestrator from a code-reviewer prose follow-up (round 1, 2026-07-19): two pre-existing #1481-family offenders keep the fleet-wide no-flags `workflow_lint.py` default run — and `tests/test_workflow_lint.py` on pristine main — red, forcing every session's payload-attribution machinery (Step 9c compare, Step 10d lint gate) to re-classify the same known red.

## Goal

Clear the two pre-existing #1481 lint reds so the no-flags `workflow_lint.py` run and `tests/test_workflow_lint.py` are green on pristine main: waive or fix `scripts/issue1481_cjk_audit.py:88` (`--check-hub-verify-retry`) and `scripts/issue1481_phasec.sh:43+49` (`--check-phase-done-reserved`).

## Problem

- **Bug observed:** `uv run python scripts/workflow_lint.py --check-hub-verify-retry` FAILs (1 error): `scripts/issue1481_cjk_audit.py:88` bare `.list_repo_tree(` (the #920 false-failure class). `--check-phase-done-reserved` FAILs (2 errors): `scripts/issue1481_phasec.sh:43+49` invoke `issue1481_dispatch.sh`, which emits the RESERVED `[phase=done]` token at its line 59 (false `status=done` to poll_pipeline.py, incidents #545/#920).
- **Why it matters:** these are the only reds keeping the no-flags default run non-green (they surfaced as the recurring "pre-existing 3F" baseline in #1547's gate runs); every downstream session pays a re-classification cost.
- **Confidence:** high (live-verified at filing).
- verified-at-filing: `uv run python scripts/workflow_lint.py --check-hub-verify-retry` → FAIL (1 error) naming scripts/issue1481_cjk_audit.py:88; `uv run python scripts/workflow_lint.py --check-phase-done-reserved` → FAIL (2 errors) naming scripts/issue1481_phasec.sh:43,49 (both run on repo-root main, 2026-07-19). Landed-fix check: `git log --oneline --since='7 days ago' -- scripts/workflow_lint.py scripts/issue1481_cjk_audit.py scripts/issue1481_phasec.sh` shows no waiver/fix commit for these offenders. Open-sibling scan across proposed/planning/plan_pending/approved/running/reviewing/blocked/on_hold titles: no task addresses them.

## Proposed change (refine in planning)

```
# 1. scripts/issue1481_cjk_audit.py:88 — EITHER route the list_repo_tree call
#    through hub.list_hf_files_under_path / wrap in hub.retry_transient, OR add
#    '# HUB_VERIFY_RETRY_EXEMPT: <reason >=10 chars>' on/above the call line.
# 2. scripts/issue1481_dispatch.sh:59 — waive the mode-gated standalone-lane
#    terminal with '# noqa: phase-done-reserved' on the emission line (or
#    reword the child's completion line / redirect child stdout per
#    .claude/rules/pod-side-reporting.md).
# 3. Verify: both single-flag runs exit 0; tests/test_workflow_lint.py green
#    on main; the no-flags default run drops these 3 errors.
```

Note: #1481 is parked at awaiting_promotion — these are frozen-ish per-issue scripts; the minimal waiver route (1-3 comment lines) is preferred over refactoring, preserving the reproducibility record. Whichever route, do NOT change dispatcher semantics (the `[phase=done]` line is load-bearing for poll_pipeline).

## Scope / surfaces

- Primary targets: `scripts/issue1481_cjk_audit.py`, `scripts/issue1481_phasec.sh` (and/or `scripts/issue1481_dispatch.sh:59` for the waiver) — experiment scripts (NOT workflow surface; wf_fix: false).
- Out of scope: `scripts/workflow_lint.py` (the checks are correct; do not add allowlist entries there — the waiver comments are the designed escape).

## Constraints / invariants

- No behavior change to the #1481 scripts' execution (comment-only waivers preferred).
- `tests/test_workflow_lint.py` + the two single-flag runs green afterward; ruff on touched files passes.

## Provenance

- Surfaced by: code-reviewer round 1 on task #1547 ("the pre-existing issue1481_cjk_audit.py:88 red keeps the fleet-wide no-flags workflow_lint default run failing; a 1-2-line #1481-scoped waiver/wrap follow-up would clear it").
