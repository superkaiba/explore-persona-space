---
title: 'daily-fix: step9c gate degrade on collection ImportError'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e7a8febc29ab
- daily-auto-filed
created_at: '2026-07-28T06:59:56Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): a single collection-red
  file on origin/main (ImportError) aborts the whole Step 9c pytest run rc=2; compare
  refuses to classify (MF-1b) and the session must diagnose + deselect + re-run the
  full gate — 3 sessions paid ~38-45 min each on 2026-07-27'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Sessions 60cd2d8b (#1720, ~40 min), 206fe014 (#1728, ~45 min), 116cfd96 (#1731, ~40 min), 2026-07-27.

## Goal

One collection-broken test file on main should not abort the entire Step 9c gate run.

## Workflow gap

- **Bug observed:** `tests/test_workflow_lint_inline_round_duty_mirror.py` (ImportError from the #1698 lost-update) aborted collection -> pytest rc=2 -> `step9c_baseline compare` correctly refused to classify a partial run (MF-1b) -> each session hand-diagnosed, deselected the file, and re-ran the FULL gate (~37 min each). The refusal is correct; the abort-the-world collection behavior is the fixable half.
- **Why it is a workflow gap:** the gate recipe runs pytest without `--continue-on-collection-errors`, and the selector has no arm that deselects files already collection-red in the baseline ledger — so a known main-side breakage taxes every touched-scope session ~40 min.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'continue-on-collection-errors' scripts/select_step9c_tests.py scripts/step9c_baseline.py .claude/skills/issue/SKILL.md` -> 0/0/0, compose time.

## Proposed change (candidate diff sketch — refine in planning)

Either (a) add `--continue-on-collection-errors` to the Step 9c gate pytest invocation (SKILL.md recipe) with compare-side handling of collect-error rows as per-file failures (classifiable pre-existing), or (b) teach `select_step9c_tests.py` to auto-deselect files marked collection-red in the fresh baseline ledger, WARN-loudly. Planner weighs (a) vs (b); MF-1b's refusal-to-classify semantics for genuinely partial runs must be preserved.

## Scope / surfaces

- Primary targets: `scripts/select_step9c_tests.py`, `scripts/step9c_baseline.py`, `.claude/skills/issue/SKILL.md` (Step 9c gate recipe)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: e7a8febc29ab

- workflow_fix_target: scripts/select_step9c_tests.py, scripts/step9c_baseline.py, .claude/skills/issue/SKILL.md
