---
title: 'workflow-fix: marker-training-recipe.md training-row token-budget rule (#260
  sibling)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bb53f8a53a43
created_at: '2026-07-04T07:47:14Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #906 r13 crash-fix (marker row
  truncation)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha candidate raised on task #906 (emitting agent: experiment-implementer, r13 crash-fix round).

## Goal

Add the TRAINING-ROW token-budget rule to the marker training recipe: every marker mix row must be tokenized under the trainer's exact render at BUILD time and pair-dropped fail-loud when over the recipe max_length — the training-side sibling of the existing #260 eval-side truncation rule.

## Workflow gap

- **Bug observed:** #906's marker-only run crashed mid-train (after provisioning) in MarkerOnlyDataCollator: two extreme-tail WildChat prompts (1718-2194 tokens) + capped greedy response + ' ※<|im_end|>' exceeded max_length=2048; SFTTrainer right-truncation cut the marker+end-of-turn loss slot. 4/200 rows affected.
- **Why it is a workflow gap:** .claude/rules/marker-training-recipe.md covers the eval-side truncation rule (#260) but is silent on the TRAINING-row budget; any future marker implant on a real-user bank (unbounded prompt tail) re-hits this after provisioning instead of at build.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ marker-training-recipe.md: "TRAINING-row token budget (the #260 sibling): at mix BUILD time, tokenize every row under the trainer's exact render (prompt+completion in one apply_chat_template call) and pair-drop over-budget rows from BOTH pos+cn (preserving the 1:1 ratio) with a fail-loud rejection-fraction floor + a [marker-mix-budget] log; ground keep-vs-raise-max_length on the measured length distribution, never chase an unbounded prompt tail. Worked example: _enforce_marker_mix_token_budget (scripts/issue906_phase1_pilot.py, #906 r13) + tests/test_issue906_marker_mix_budget.py."

## Scope / surfaces

- Primary target: `.claude/rules/marker-training-recipe.md`
- Also check docs/marker_training_recipe.md (the evidence index) for a per-task index row.

## Constraints / invariants

- Workflow-surface only; scripts/workflow_lint.py passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 and carries a workflow_fix_target: Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/marker-training-recipe.md
- fingerprint: bb53f8a53a43

(verbatim lesson: see epm:failure-lesson v3 on task #906 — marker TRAINING rows carry the #260 truncation trap; build-time tokenized budget gate with pair-drop + rejection floor.)
