---
name: Comparator-range sibling-cell provenance
description: Quoted parent reference ranges (e.g. clamp gaps "0.75-1.21") can come from sibling cell families, not the actual A/B comparator arm — recompute from the comparator's own JSONs
type: feedback
---

When a plan quotes a parent's registered range as the flag-off / control comparator
("flag-off committed 0.75-1.21"), check WHICH cells produced that range in the parent's
classification JSON. In #613 the 0.75-1.21 clamp range came from #601's six COUNT cells
(`c472_anchor`/`negex_100`/`negex_400` in `eval_results/issue_601/analysis/classification.json`),
while the actual A/B comparator (`dense_200p800n` seeds 42/137) has terminal gaps
1.03 / 1.25 (computed from the committed `dense_trajectory.json`) — seed 42 sits closer
to the 1.5-nat bar than the quoted range suggests.

**Why:** the decision rule may be unchanged (bar still 1.5), but the analyzer narrating
"flag-on vs flag-off 0.75-1.21" against the wrong cells misstates the contrast size.
Sibling of feedback_subset_mismatched_threshold_calibration (different question slice);
this variant is different CELL FAMILY.

**How to apply:** for any reused-comparator read, recompute the quoted reference
statistic from the comparator arm's own committed JSONs (persona-mean vs pooled
coincide only when the design is balanced — verify). Concern-not-REVISE when the
analyzer re-loads exact centers from JSONs at analysis time.
