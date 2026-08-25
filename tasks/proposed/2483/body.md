---
title: 'verify_plan: flag ratio-form ordering conjuncts over unbounded-below metrics
  in verdict lattices (c20 threshold-SKIP blind spot)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-22T21:49:37Z'
has_clean_result: false
origin_prompt: 'Codex statistics critic on #2476 plan v3 (SQ-A, upheld by the reconciler
  as Real-blocking): the registered lattice conjunct ''tier-ratio > 1'' (tier-0 median
  held-out R2 divided by tier-2 median) is not an ordering test because held-out R2
  is unbounded below - a positive tier-0 median with a negative tier-2 median yields
  a negative ratio and routes the strongest gradient to Indeterminate. The lattice
  PASSed verify_plan because the threshold-form + otherwise clause takes the c20 SKIP
  path, so no coherence check ever ran on the conjunct semantics.'
workflow: v1
---
## Goal

Add a verify_plan.py check (or extend c20) that flags division/ratio-form ordering conjuncts over unbounded-below metrics (held-out R2, log-likelihood deltas) inside registered verdict lattices: a ratio comparison 'A/B > 1' carries no ordering information once signs mix, and the c20 threshold-SKIP path (threshold atoms + otherwise clause, #1689/#1700) means such a conjunct is never parsed for coherence.

## Provenance

Surfaced on #2476 plan v3: the SQ-A Must-Fix (Codex statistics critic; reconciler upheld it as Real-blocking with a truth-table demonstration - m0=+0.30, m2=-0.05 gives ratio -6 < 1 -> strongest gradient mis-routed to Indeterminate; both-negative medians reverse the ordering; zero denominator undefined). Fixed in #2476 v4 by the sign-stable difference form (m0 - m2 > 0 / < 0). Codex marked the check mechanizable: truth-table synthetic positive/negative/zero median cells against the verdict function; reject division-based comparisons involving an unbounded-below metric.

## Design notes

- Trigger surface: c20's clause harvest already extracts predicates; add a scan for 'divided by' / 'ratio' / 'A/B' quantity definitions adjacent to lattice threshold conjuncts whose underlying metric vocabulary is R2/log-prob/delta (unbounded-below list), WARN with the sign-mixing counterexample and the difference-form remedy.
- Distinct from #2396 (decimal-threshold lattices never FAIL-capable - the SKIP/WARN mechanics); this task is about conjunct SEMANTICS slipping through the SKIP path. Cross-link both.
- Pin tests: fixture plans with a ratio conjunct (flag) and a difference conjunct (clean); the #2476 v3->v4 pair is the real-world fixture pair.
