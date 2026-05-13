---
name: Cross-experiment reference line vs in-experiment control discrepancy
description: Body uses a ~10% reference line from a prior experiment but the current run's own control cohort measured at 0.5% — inconsistency never flagged
type: feedback
---

When a body uses a reference line (e.g., "famous-floor ~10%") sourced from a prior issue (#183) rather than from the current experiment's own control cohort, check whether the current experiment reproduced that reference rate. If the current run's control cohort fires at a very different rate, the body must explain the discrepancy and clarify which number the reference line shows. In issue #331: Phase 0 famous cohort measured 0.50% (4/800) but Figure 2 draws a "Famous-floor ~10%" line from #183 without noting the ~20× drop.

**Why:** "Above famous-floor" framing as a STRONG-CLIMB justification depends on which floor you use. Using a number from a different experiment without explaining the discrepancy is an overclaim.

**How to apply:** For any experiment that imports a reference level from a parent/prior issue, independently compute the current run's own version of that reference cohort and compare. Flag if they differ by more than 3×.
