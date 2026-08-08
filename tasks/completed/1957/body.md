---
title: 'workflow-fix: audit inline-CI rule misses bracket-less verbal form (CI <num>
  to <num>)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:87b562e5ceae
created_at: '2026-08-01T04:29:07Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from #1946 clean-result-critique v1: audit_clean_results_body_discipline.py
  inline-CI rule misses the bracket-less verbal ''CI <num> to <num>'' form (fourth
  variant of the #382 class).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1946 (emitting agent: clean-result-critic, round 1).

## Goal

Extend the inline-CI rule in scripts/audit_clean_results_body_discipline.py with the bracket-less verbal form (`CI <num> to <num>`) so the Lens-7 mechanical backstop catches the fourth surface variant of the banned inline-credence-interval construct.

## Workflow gap

- **Bug observed:** body #1946 shipped an inline numeric CI in Results prose — "(CI −0.072 to +0.002)" — and the audit PASSed: its inline-CI rule matches only square-bracketed forms and the named-endpoint ("upper bound = X") form, not the bracket-less verbal "CI <num> to <num>" form.
- **Why it is a workflow gap:** Lens 7 bans inline credence intervals and the audit is its mechanical backstop; each uncovered surface form has recurred until pattern-covered (#382 brackets, #649 U+2212 signs, #952 named endpoints — this is the fourth variant), each burning a critic round.
- **Confidence (emitter):** medium
- verified-at-filing: the incident body text "(CI −0.072 to +0.002)" was live in tasks/interpreting/1946/body.md at review time (clean-result-critique v1, 2026-08-01) with the audit PASSing on it — reproduced by the critic's own audit run; `grep -c inline scripts/audit_clean_results_body_discipline.py` → 47 hits (rule family present; no bracket-less 'CI <num> to <num>' alternative among them, checked 2026-08-01).

## Proposed change (candidate diff sketch — refine in planning)

In the inline-CI rule block (~scripts/audit_clean_results_body_discipline.py:295-353), add an alternative:
+ # (4) bracket-less verbal form: `CI −0.072 to +0.002` (#1946) — no square
+ #     brackets, bounds joined by "to"; same sign class + carve-outs.
+ r"\bCI\s*[:=]?\s*[+\-−]?\d[\d.]*\s+to\s+[+\-−]?\d[\d.]*"

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing; add a pin test to the audit's test file per its convention.

## Constraints / invariants

- Workflow-surface only; ruff clean; existing carve-outs (chart annotations, CI-as-test-definition register) preserved; no new FAILs on grandfathered bodies (run the audit fleet-wide once to check the blast radius).
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics via the Provenance line below (recursion guard).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 87b562e5ceae

Verbatim origin: the workflow-fix-candidate v1 block in epm:clean-result-critique v1 on task #1946 (2026-08-01).
