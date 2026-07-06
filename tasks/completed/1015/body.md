---
title: 'workflow-fix: audit interval_inline bound-form pattern'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e56b4e867544
created_at: '2026-07-04T18:23:49Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #952 r1 prose follow-up'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by clean-result-critic on task #952 r1.

## Goal

Add the '(upper|lower) bound (+x)' prose form to the interval_inline pattern set in audit_clean_results_body_discipline.py.

## Workflow gap

- **Bug observed:** A clean-result body carried 'the interval's upper bound (+0.023) excludes the 0.05 margin' in Results prose — a bound-form CI leak the interval_inline pattern does not match (#952 r1, caught only by the LM critic).
- **Why it is a workflow gap:** the mechanical verifier is the authoritative pre-pass for every clean-result round; a pattern/check gap means every future body re-hits the class and only an LM critic pass catches it.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; see Goal)

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`

## Constraints / invariants

- Workflow-surface only; existing tests pass; add a regression test for the new pattern/check.

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: e56b4e867544
