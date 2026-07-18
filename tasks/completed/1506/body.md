---
title: 'workflow-fix: check-28 opaque-code classes for bare H<d> + f16/l16 tokens'
kind: infra
tags:
- wf-fix
- wf-fix-fp:330afec73666
created_at: '2026-07-18T06:52:09Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1072 round-1 prose follow-up: check 28 misses
  bare hypothesis codes (H3) + slot-family codes (f16/l16) in rendered figure text'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up
raised on task #1072 (emitting agent: clean-result-critic).

## Goal

Extend verify_task_body.py check-28 _opaque_code_tokens classes to flag bare hypothesis codes (\bH\d\b) and slot-family codes (\bf16\b / \bl16\b) in rendered figure/sidecar text outside path words.

## Workflow gap

- **Bug observed:** check 28 allows bare short-letter hypothesis codes — "(H3)" in #1072's figure-4 panel title and f16/l16 slot-family codes in rendered figure text are not flagged outside path words.
- **Why it is a workflow gap:** The no-opaque-condition-codes rule is enforced mechanically for slug/@L-pin tokens but not for hypothesis/slot-family codes, so a Lens-3 must-fix had to be caught by the LM critic instead of the verifier.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "_opaque_code_tokens" scripts/verify_task_body.py` → 2 hits (def at scripts/verify_task_body.py:7319; check 28 block at 7293-7426, label "figure text opaque config codes (slug / @L-pin tokens)") — presence confirmed per-target; current token classes cover slug/@L-pin only (2026-07-18). Live repro: #1072 exploratory_component_profiles.png panel title "(H3)" + "f16 slots" passed check 28.

## Proposed change (candidate diff sketch — refine in planning)

in _opaque_code_tokens (scripts/verify_task_body.py:7319):
+ add token classes: r"\bH\d\b" (bare hypothesis codes) and
+ r"\b[fl]16\b" / slot-family codes, excluding path-like words and
+ code-span contexts; keep the existing slug/@L-pin classes.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 330afec73666

Verbatim surfaced prose: "verify_task_body.py `_opaque_code_tokens` (check 28) allows bare short-letter hypothesis codes — `\bH\d\b` (\"(H3)\" in #1072's fig-4 panel title) and slot-family codes (`\bf16\b`/`\bl16\b`) in sidecar rendered text are not flagged outside path words; extending the token classes would mechanize the Lens 3 must-fix above."
