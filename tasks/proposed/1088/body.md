---
title: 'workflow-fix: verify_task_body HF count matcher — blobs-only per-namespace
  claims'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e04955ea9e78
created_at: '2026-07-06T16:00:59Z'
has_clean_result: false
origin_prompt: 'clean-result-critic v3 on #833, Lens 5 procedural fix 1 (mechanizable:
  yes)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a mechanizable finding raised on task #833 (emitting agent: clean-result-critic, epm:clean-result-critique v3).

## Goal

Extend `verify_task_body.py`'s HF file-count reconciliation to match the "N files listed per namespace" claim form and count blobs only.

## Workflow gap

- **Bug observed:** #833's Repro footer claimed "908 files listed per namespace" — that number counted HF `list_repo_tree(recursive=True)` ENTRIES (891 blobs + 17 directory objects). The verifier's existing HF file-count adjacency matcher reported "no file-count claims" against the body, so the miscount shipped and was caught only by the LM critic's Hub spot-check.
- **Why it is a workflow gap:** the mechanical verifier owns file-count reconciliation; a claim form it cannot parse silently escapes the check.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ In verify_task_body.py's HF count reconciliation, add the claim pattern r"(\d[\d,]*) files (?:listed )?per namespace" (and the sibling "N files at the pinned revision" form).
+ When resolving the pinned namespace, count only entries with a non-None size (blobs), excluding directory objects returned by list_repo_tree(recursive=True).
+ FAIL/WARN parity with the existing count-mismatch handling.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep for the existing count-claim matcher + its tests before editing; update `tests/test_verify_task_body.py`.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` passes; ruff clean on touched files.
- This fix must not newly hard-FAIL grandfathered v3/v2 bodies (forward-only; WARN preferred where ambiguous).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: e04955ea9e78

Origin: clean-result-critic epm:clean-result-critique v3 on #833, Lens 5 procedural fix 1 ("mechanizable: yes — extend verify_task_body.py's HF file-count reconciliation to match the 'N files listed per namespace' claim form and count blobs only (its adjacency matcher reported 'no file-count claims' against this body)").
