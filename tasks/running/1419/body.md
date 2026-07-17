---
title: 'workflow-fix: audit pre_reg pattern catches bare ''registered <noun>'' forms'
kind: infra
tags:
- wf-fix
- wf-fix-fp:534440028de7
created_at: '2026-07-16T08:43:47Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose follow-up on #1345 r1: 6 bare registered-noun
  mentions escaped the pre_reg alternation'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1345 (emitting agent: clean-result-critic, round 1).

## Goal

Extend audit_clean_results_body_discipline.py's pre_reg pattern to catch bare "registered <noun>" forms in prose sections.

## Workflow gap

- **Bug observed:** Six bare "registered <verdict/margin/read/lattice>" pre-registration mentions in #1345's body escaped the current pre_reg alternation ("pre-registered" / "as registered" / "registered hypothesis") — caught only by the critic's Lens 7 read.
- **Why it is a workflow gap:** the audit script owns the mechanical register pass; a recurring register violation that its regex structurally misses lands on the LM critic every time.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "pre_reg\|pre-registered" scripts/audit_clean_results_body_discipline.py` → pattern block present at the cited :74-77 region with the narrow alternation (presence confirmed; the bare registered-noun alternative absent) (2026-07-16).

## Proposed change (candidate diff sketch — refine in planning)

+ pre_reg alternation gains: \bregistered\s+(verdict|margin|read|lattice|criterion|threshold|band)\b
+ scoped to prose sections via the existing v4 _restrict_pre_reg_to_prose_sections mechanism.

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Also extend its test file; grep for sibling audit patterns before editing.

## Constraints / invariants

- Workflow-surface only; prose-section scoping preserved (Methodology/samples exempt per the existing mechanism).
- Tests extended + passing; workflow_lint PASS. Recursion guard applies.

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 534440028de7
