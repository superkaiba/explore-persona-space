---
title: 'workflow-fix: widen check-28 figure token pattern to arm slugs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:68dcc4ea7de6
created_at: '2026-08-02T06:22:13Z'
has_clean_result: false
origin_prompt: 'clean-result-critic fold-round review on #1768, 2026-08-02 (candidate/prose
  follow-up; verbatim in body Provenance)'
workflow: v1
---
## Overview / Motivation
Auto-filed from a prose follow-up surfaced by clean-result-critic on task #1768 (fold-round review, 2026-08-02).
## Goal
Widen `verify_task_body.py` check 28 (`check_figure_label_codes`) token pattern to catch hyphen-separated arm slugs (e.g. `cas-pers-con-lr1e5-s137`) rendered on figure tick labels, WARN once per body (not per figure).
## Workflow gap
- **Bug observed:** check 28 caught `M_0` in a rendered figure but missed hyphen-separated arm slugs on tick labels across ~10 figures of #1768's folded body; only 4 captions decode them.
- **Why it is a workflow gap:** the opaque-code lens (no-opaque-condition-codes policy) has a mechanical scanner whose token pattern covers math-ish tokens but not the fleet's slug grammar — a real Lens 3 blind spot.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "opaque\|check_figure" scripts/verify_task_body.py` → check 28 = `check_figure_label_codes` (doc L518-520, "must not carry opaque [codes]", WARN class) exists and is the correct target; slug-shaped tokens absent from its pattern per the critic's live read (2026-08-02).
## Proposed change (candidate diff sketch — refine in planning)
diff_sketch: |
  In check_figure_label_codes: add a slug token alternation r"\b[a-z]{2,4}(?:-[a-z0-9]{1,12}){2,}\b"
  (behavior-context-regime-lr-seed grammar), deduped to ONE WARN per body listing offending figures;
  captions that decode the slugs suppress the WARN for that figure.
## Scope / surfaces
- Primary target: `scripts/verify_task_body.py` (+ tests). Nearest sibling check 29 (figures tracked at HEAD) untouched.
## Constraints / invariants
- WARN-class only (never a new hard FAIL on grandfathered bodies); workflow-surface only.
- Recursion guard applies (EPM_WORKFLOW_FIX_SESSION=1 + workflow_fix_target line).
## Provenance
- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 68dcc4ea7de6
Verbatim surfaced prose: "verify_task_body.py check 28's figure-text scanner caught M_0 but misses hyphen-separated arm slugs (cas-pers-con-lr1e5-s137) on tick labels — widening the token pattern (WARN once per body, not per figure) closes a real Lens 3 blind spot."
