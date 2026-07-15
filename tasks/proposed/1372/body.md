---
title: 'workflow-fix: opaque snake_case slug detection in body-discipline audit'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b533909c219a
created_at: '2026-07-15T23:44:35Z'
has_clean_result: false
origin_prompt: 'clean-result-critic r1 prose follow-up on #1315: neg_reph_curious
  escaped the opaque-condition-label audit'
workflow: v1
---
## Overview / Motivation
Auto-filed from a clean-result-critic prose follow-up on task #1315 (round 1, mechanizable: yes).
## Goal
audit_clean_results_body_discipline.py: extend the opaque-condition-label pattern to backticked 3+-segment snake_case tokens (`[a-z]+_[a-z]+(_[a-z0-9]+)+`) in reader-facing prose sections, with an allowlist for file/field names (mix_meta.json, span_seam, etc.).
## Workflow gap
- **Bug observed:** #1315's body used the opaque panel-context slug `neg_reph_curious` in reader-facing Methodology and Results prose; the audit missed it in both places (caught by the critic's spec-text opaque-code rule).
- **Why it is a workflow gap:** The audit's condition_labels pattern (audit script line ~142) covers C1/D1-style codes but not snake_case slugs, so the no-opaque-condition-codes rule (user memory + SPEC) has no mechanical backstop for the commonest slug shape.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -niE "opaque|condition.label" scripts/audit_clean_results_body_discipline.py` → condition_labels pattern present at line ~142; no snake_case coverage in it (absence claim, inspected at filing) (2026-07-15)
## Proposed change (candidate diff sketch — refine in planning)
+ Add to the condition_labels detector: backticked tokens matching [a-z]+_[a-z]+(_[a-z0-9]+)+ in Takeaways/Goal/Results prose,
+ ALLOWLIST file/field names (endswith .json/.py/.pt, known field names span_seam, mix_meta, judge_raw, etc.).
## Scope / surfaces
- Primary target: `scripts/audit_clean_results_body_discipline.py`; tests alongside.
## Constraints / invariants
- Allowlist must keep existing green bodies green (dry-run over recent completed tasks in the plan); recursion guard applies.
## Provenance
- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: b533909c219a
