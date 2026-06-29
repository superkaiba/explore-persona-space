---
title: 'Phase 3 — A3.6-A3.10 + joint factorization on the trained store (leakage program
  #660)'
kind: experiment
tags: []
created_at: '2026-06-25T07:26:50Z'
has_clean_result: false
parent_id: 660
goal: 'Phase 3 — test A3.6-A3.10 + joint factorization on the Phase-2 trained store:
  activation realized gate, whitened key-query gate (key/metric ablations), base-gate
  validity, drift decomposition; clustered CIs + single-context arm.'
---
## Goal

Phase 3 — test A3.6-A3.10 + joint factorization on the Phase-2 trained store: activation realized gate, whitened key-query gate (key/metric ablations), base-gate validity, drift decomposition; clustered CIs + single-context arm. Includes **A3.6c** — the causal context-vector patch (input-vs-map localization, R12-1).

## Design
Designed by /adversarial-planner at dispatch from docs/theory_assumption_test_plan.md (§3 Phase 3 + §4, incl. the A3.6c row + the round-12 revision log) AND the Phase-2 clean-result. READ docs/leakage_theory_paper.tex first. CPU on the trained store EXCEPT the **A3.6c causal-patch arm**, which needs forward passes with hooks on the fleet adapters + base model (no training) → provision ONE `eval`-intent GPU for it; the rest of Phase 3 stays CPU. Can overlap Phase 4 (§10e). Autocontinue.

**A3.6c (causal context-vector patch).** Cross-model activation patching at read layer L on the Phase-2 store: **P↓** run θ⁺ with the base `c0(C)` overwritten at the layer-L context position; **P↑** run θ0 with the FT `c⁺(C)` overwritten. Read the answer profile in BOTH the activation DV `v` and the on-policy behavioral DV `E`. Verdict via the context-vector-mediated fraction `f_CV`: context vector moved ⇔ P↑→(v⁺,E⁺) ∧ P↓→(v0,E0); map changed ⇔ P↑→(v0,E0) ∧ P↓→(v⁺,E⁺). Controls: self-patch identity null, random/other-context-CV floor, norm-matched variant, patch-scope {last-token, full-span}; L-sweep {7,14,21}. It is the nonparametric causal ground truth for A3.10's key+query drift decomposition, and the causal complement to A3.6a (raw `c⁺−c0` drift) + A3.6b (`M0` predictive transfer). Full spec: docs/theory_assumption_test_plan.md §4 (A3.6c) + round-12 revision log.
