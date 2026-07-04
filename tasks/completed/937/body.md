---
title: 'workflow-fix: verify_plan check — re-extracted reference arms never silently
  replace a committed headline'
kind: infra
tags:
- wf-fix
created_at: '2026-07-03T15:18:34Z'
has_clean_result: false
origin_prompt: 'Codex Statistics critic note on #811 plan v3 (prose follow-up; orchestrator-synthesized
  candidate)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a Codex critic note raised on task #811 (emitting agent: codex-critic Statistics lens, surfaced by the orchestrator).

## Goal

Add a verify_plan.py check: a follow-up plan that RE-EXTRACTS prior-headline reference arms AND folds into an existing clean-result must explicitly distinguish "same-pass comparator" values from "prior committed headline" values (a reference flip is replication-stability evidence, never a silent headline replacement).

## Workflow gap

- **Bug observed:** #811 plan v3 §6 said "adjudication then uses THIS round's internally consistent store", which would let a regenerated-R flip in a re-extracted reference silently move the already-adjudicated clean-result headline for reasons unrelated to the round's new arms.
- **Why it is a workflow gap:** follow-up rounds that re-extract reference arms + fold into an existing clean-result are a recurring plan shape (#811 rounds 1-2 both did); no mechanical check enforces the committed-cells-remain-evidence rule.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ In scripts/verify_plan.py: parse plan text for a parity/reference clause; if the plan
+ both re-extracts prior headline references AND folds into an existing clean-result
+ (followup-scope marker present + reference arms named), require an explicit sentence
+ distinguishing same-pass comparator values from prior committed headline values;
+ WARN (not FAIL) when absent.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for related parity-clause conventions before editing; list hits in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: (computed at filing)

> Codex Statistics critic note (task #811 plan v3, 2026-07-03): "This is a recurring workflow-surface verifier candidate for follow-up plans that re-extract reference arms and fold into an existing clean-result. ... Check sketch: parse plan text for a parity/reference clause; if it both re-extracts prior headline references and folds into an existing clean-result, require an explicit sentence distinguishing 'same-pass comparator' values from 'prior committed headline' values."
