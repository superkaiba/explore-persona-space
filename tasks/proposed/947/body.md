---
title: 'workflow-fix: verify_plan check — paired-contrast rows need per-context sources
  on both arms'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fe9772faa82a
created_at: '2026-07-03T18:39:41Z'
has_clean_result: false
origin_prompt: 'four-reviewer prose follow-up on #810 plan v13: verifier check that
  every registered paired-contrast row has a declared per-context source on both arms'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from prose follow-ups surfaced by BOTH methodology critics + the codex statistics critic + the consistency-checker reviewing #810 plan v13 (four independent catches of one gap).

## Goal

Add a verify_plan.py check that every registered paired-contrast row names a per-context data source on BOTH arms.

## Workflow gap

- **Bug observed:** #810 plan v13 registered a 9-row paired bootstrap whose named full-side pack (uh_summaries.pt) lacked 2 of the 9 rows (im_end/turn_nl); the registered success criterion was unsatisfiable from the named inputs and the falsification leg was uncomputable exactly when those rows win. Caught only by reviewer legwork (pack row-name reads).
- **Why it is a workflow gap:** verify_plan has no statistical-input-existence check for paired contrasts; the reuse-fitness (c) required-cells idea exists for artifacts but not for per-row paired-DV sources.
- **Confidence (emitter):** medium-high (four independent reviewer catches)

## Proposed change (candidate diff sketch — refine in planning)

+ In scripts/verify_plan.py: when a plan registers a paired contrast (grep `paired` + a row list),
+ require a declared per-context source per arm covering every registered row
+ (mechanizable shape per the critics: `set(registered_pair_rows) ⊆ union(keys(full_side_sources))`),
+ FAIL when a registered row has no declared source on either arm.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Sibling: `tests/test_verify_plan.py`

## Constraints / invariants

- Workflow-surface only; workflow_lint + ruff pass.

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: fe9772faa82a

Surfaced prose (codex-critic methodology, #810 v13, verbatim): "This should become a workflow verifier check for paired contrasts: every registered row must have per-context sources on both arms."
