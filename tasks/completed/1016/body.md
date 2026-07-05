---
title: 'workflow-fix: verify_task_body per-file HF membership check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:23000a486a4f
created_at: '2026-07-04T18:23:57Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #952 r1 prose follow-up'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by clean-result-critic on task #952 r1.

## Goal

verify_task_body.py: for backtick-named files claimed adjacent to a pinned HF tree link, add a per-file membership check against the pinned listing.

## Workflow gap

- **Bug observed:** #952's body claimed divergence_bank_queries.json at the pinned HF eval_results@5b62649 path but the file is absent from that listing (it lives in git) — link-liveness validated the tree link without checking the named file's membership.
- **Why it is a workflow gap:** the mechanical verifier is the authoritative pre-pass for every clean-result round; a pattern/check gap means every future body re-hits the class and only an LM critic pass catches it.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; see Goal)

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`

## Constraints / invariants

- Workflow-surface only; existing tests pass; add a regression test for the new pattern/check.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 23000a486a4f
