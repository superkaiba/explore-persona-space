---
name: dose-response-fwer-with-shared-randomness
description: Per-adapter × per-probe hero tables with "≥1 of N fires" criteria inflate FWER to N×M cells; cross-adapter dependence via shared templates/queries needs max-stat aggregation, not Bonferroni-N
metadata:
  type: feedback
---

When a hero figure is a "per-adapter × per-probe dose-response table" with success "≥1 of N adapters fires under probe T" for each of M probe types, the family is N×M tests, not M: at α=0.05 with N=4 × M=4, FWER under independence ≈ 1−0.95¹⁶ ≈ 56%. Fixes: (a) Holm across the N×M cells, or (b) per-cell hierarchical conjunction (installation AND drift AND drift-base conditions on the SAME adapter as one test, with "≥1 of N" a single union over adapter-level conjunctions).

**Critical wrinkle — shared randomness:** when probes share input randomness (same drift-template bank, same held-out query set across adapters), the N adapter-level tests within a probe family are NOT independent: Bonferroni overcorrects within-family, the effective denominator is closer to M, but ACROSS-family inflation still applies. Specify the aggregation: "max statistic across N adapters with permutation null" (correct within-family FWER) or "intersection across N adapters" (very conservative).

**Why:** #375 round 2 (2026-05-21) grew from a 4-cell union test (FWER ≈18.5%) to 16 strict cells across two parallel primaries without addressing FWER, with shared template/query randomness unspecified.

**How to apply:** ask (1) what is m_total = N×M? (2) how is per-cell α controlled? (3) do probe families share randomness? If (3) yes, demand the aggregation be specified. Related: [[feedback_alternatives_lens_round2.md]] — bootstrap percentile-CI degeneracy near 0 recurs for weakly-firing drift cells (3-10%); combine FWER correction with BCa or permutation tests there.
