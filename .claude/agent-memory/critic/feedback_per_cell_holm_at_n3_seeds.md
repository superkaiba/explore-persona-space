---
name: Per-cell exact rank tests at small N + Holm are structurally unreachable
description: Wilcoxon over N=3 seeds has min one-sided p = 1/2³ = 0.125; Holm/Bonferroni cutoff α/m below that makes "≥1 cell passes Holm" decision rules algebraically impossible (#399)
metadata:
  type: feedback
---

When a plan uses Wilcoxon signed-rank (or sign / any exact rank test) over N paired observations per cell, the smallest one-sided p is `1/2^N` (N=3: 0.125; N=4: 0.0625; N=5: 0.03125; N=6: 0.0156). Holm/Bonferroni at FWER α=0.05 across m cells has smallest cutoff α/m (m=3: 0.0167 — N=5 just clears; m=9: 0.0056 — need N≥7). If α/m < 1/2^N, **no cell can ever pass** — an algebraic impossibility independent of effect size. Note N=3 min p = 0.125 > 0.05, so such cells can't even reach raw significance; BH-FDR doesn't rescue it.

**Why (anchor, #399 critic round 2026-05-26):** per-cell Wilcoxon at N=3 seeds + Holm across 9 cells (α/9 = 0.0056) with scenarios requiring "cells passing Holm" — one scenario unreachable, one ambiguous.

**How to apply:** for any plan with (1) a per-cell test over a small replicate dimension (≤5 seeds), (2) multiple-comparison correction across cells, (3) decision rules requiring "≥k cells pass": compute α/m vs 1/2^N. If the cutoff is smaller, BLOCKER — raise per-cell N (pool per-context paired units instead of per-seed units: N becomes N_contexts × N_seeds), or pool across cells (aggregate test only) and abandon the per-cell rule. Companions: [[feedback_n2_sigma_and_perm_cap]], [[feedback_spearman_threshold_n12]].
