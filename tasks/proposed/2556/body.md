---
title: 'workflow-fix: lattice label-semantics truth-table + threshold-CI-policy check
  (c20 extension or sibling)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T20:52:25Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2546 Codex statistics critic: TraceCarried label
  co-fires with a contradicting answer-presence conjunct; c20 SKIPs threshold lattices
  by design'
workflow: v1
---
## Goal
Extend the verdict-lattice verification surface with a label-semantics truth-table check: every Boolean combination of a lattice's threshold predicates must map to exactly one semantically compatible label, and the plan must declare its threshold-uncertainty (CI) policy.

## Incident (#2546 plan v3, 2026-08-24)
The registered two-predicate lattice (`TraceCarried ⇔ CA_gap_pp >= 15; AnswerAlreadyPresent ⇔ CA_gap_pp < 15 AND A_r2_pp >= 50; Indeterminate ⇔ otherwise`) emits `TraceCarried` on the (≥15, ≥50) cell — where the answer-presence conjunct contradicts the label's own narrative ("carried by the trace, NOT the answer"). No CI policy for a bootstrap interval spanning the 15-pp threshold was declared. verify_plan c20 deliberately SKIPs threshold-form lattices (out of the v1 cell algebra per #1689/#1700 — "co-fire risk on threshold predicates is Statistics-critic scope"); the Codex statistics critic caught it and proposed making the truth-table check a recurring workflow-surface verifier.

## Proposed check (from the Codex statistics critic, mechanizable sketch)
For a tier-1 threshold lattice (the shape c20 currently SKIPs): enumerate the Boolean combinations of its distinct threshold predicates; WARN when a combination is covered only by a label whose name/narrative contradicts a co-holding predicate (heuristic: negation vocabulary in the label's defining prose vs the predicate it co-fires with), or more tractably: WARN when >1 non-otherwise predicate can co-hold and no label names the conjunction cell. Additionally WARN when a plan registers a bootstrap/CI battery AND a threshold lattice but declares no threshold-CI policy (point-estimate-only vs CI-based-indeterminate). Design question for the planner: whether this lives in c20 (lifting part of the #1689 SKIP) or a new check id; the #1689/#1700 SKIP rationale must be preserved for the pure-exhaustiveness question.

## Acceptance criteria
1. Fixture reproducing the #2546 shape (two predicates, co-holdable, no conjunction label, no CI policy) WARNs.
2. A four-state lattice enumerating all combinations + a declared CI policy PASSes.
3. The existing #1689/#1700 threshold+otherwise SKIP behavior for exhaustiveness is unchanged (pin test).
