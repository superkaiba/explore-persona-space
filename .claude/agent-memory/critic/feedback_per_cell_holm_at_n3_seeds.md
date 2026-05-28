---
name: Per-cell Wilcoxon at N=3 seeds + Holm-Bonferroni at α/m is structurally unreachable
description: When per-cell test is "Wilcoxon over 3 seeds" and Holm is applied across m cells, the smallest cutoff α/m < 1/2^N — no cell can ever pass Holm. Decision rules requiring "≥1 cell passes Holm" become unreachable.
metadata:
  type: feedback
---

When a plan spec uses Wilcoxon signed-rank (or sign test, or any exact rank
test) over N=3 paired observations, the smallest one-sided p attainable is
`1/2^N = 1/8 = 0.125`. Holm-Bonferroni or Bonferroni at family-wise α=0.05
across m cells has smallest cutoff `α/m`. If `α/m < 0.125`, **no cell can
ever pass Holm** — it is an algebraic impossibility independent of effect
size.

**Why:** The discrete null distribution of the rank statistic at small N
has only `2^N` sign-pattern outcomes (or `N!` for permutation tests). Min
one-sided p:
- N=2: 0.25
- N=3: 0.125
- N=4: 0.0625
- N=5: 0.03125
- N=6: 0.015625

Bonferroni cutoff α/m vs N (when α=0.05):
- m=3: 0.0167 — N=5 just clears
- m=5: 0.010 — N=6 just clears
- m=9: 0.0056 — need N=7+
- m=14: 0.0036 — need N=8+

**How to apply:** When critiquing any plan with:
1. Per-cell test over a small replicate dimension (typically seeds, ≤5).
2. Multiple comparison correction across cells (Holm / Bonferroni / FDR).
3. Decision rules requiring `≥ k cells pass`.

Compute `α / m`. Compare against `1/2^N`. If the cutoff is smaller, flag
as BLOCKER:
- Either raise per-cell N (more seeds, or pool the per-context paired
  test units instead of the per-seed units — typically N becomes
  N_contexts × N_seeds which is 100×–1000×).
- Or use a less-conservative correction (BH-FDR doesn't help here either
  if the underlying test can't even reach raw α=0.05 alone — N=3 min p
  is 0.125 > 0.05).
- Or pool across cells (aggregate test only, no per-cell Holm) and
  abandon the per-cell decision rule.

**Anchor (critic round on #399, 2026-05-26):** Plan had per-cell Wilcoxon
at N=3 seeds + Holm across 9 cells (α/9 = 0.0056). Min per-cell p = 0.125.
Scenarios B and C both required "cells passing Holm" — Scenario C
unreachable, Scenario B ambiguous. Companion to [[feedback_n2_sigma_and_perm_cap]]
(N<5 stdev defect) and [[feedback_spearman_threshold_n12]] (small-N rank
correlation defect).
