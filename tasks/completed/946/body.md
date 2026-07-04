---
title: 'workflow-fix: verify_plan check — causal-branch wording must not exceed diagnostics'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4f6f8a6a7655
created_at: '2026-07-03T18:39:38Z'
has_clean_result: false
origin_prompt: 'codex-critic alternatives prose follow-up on #810 plan v13: verifier
  check that registered causal branch wording does not exceed what reported diagnostics
  can distinguish'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the codex-critic (alternatives lens) reviewing #810 plan v13, echoed by the Claude alternatives + methodology critics.

## Goal

Add a verify_plan.py check that a plan's registered causal-branch wording does not exceed what its reported diagnostics can distinguish.

## Workflow gap

- **Bug observed:** #810 plan v13's falsification branch was worded as demonstrated "answer-content integration" ("they really were carrying answer content, the echo story dies") although the design cannot distinguish integration from OOD empty-turn degradation; three reviewers independently required a wording scope-down before approval.
- **Why it is a workflow gap:** no mechanical check catches registered branch text whose causal claim outstrips the diagnostics; the pattern recurs whenever an ablation/intervention has a confounded falsification branch.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ In scripts/verify_plan.py, add a check (WARN-level initially): for each registered hypothesis branch,
+ flag unqualified causal verbs/claims (e.g. `integration =>`, `carries/carried <X> content`, `the <account> dies`,
+ `rewrites the <round> interpretation`) unless the same paragraph names the undistinguished alternative
+ (`OOD`, `endpoint`, `not uniquely diagnostic`, `degradation remains live`, or equivalent).

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Sibling: `tests/test_verify_plan.py` (pin the new check)

## Constraints / invariants

- Workflow-surface only; workflow_lint + ruff pass; WARN-level first (no retro hard-FAILs).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 4f6f8a6a7655

Surfaced prose (codex-critic alternatives, #810 v13, verbatim): "This is a recurring workflow-surface verifier candidate: causal branch wording should not exceed what the reported diagnostics can distinguish."
