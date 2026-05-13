---
name: caption-threshold-all-vs-one
description: Figure caption claims "all/both [axes] exceed threshold" but only one bar in the figure actually crosses the threshold line — check each bar visually against the plotted dashed line
metadata:
  type: feedback
---

When a figure caption says "all exceed threshold" or "both exceed threshold" for a group of bars, load the figure and check each bar against the dashed threshold line. Analyzers often write the caption for the best-performing member of the group and generalize.

**Why:** In the Phase 1 recipe panel (issue #368), the caption said "both centroid baselines exceed the H1 threshold" but the figure itself drew a dashed line at 0.55, and the pos-only bar (0.452) was visibly below it. The JSON (`per_axis_stats.json`) confirmed pos-only = 0.452 < 0.55. The figure was more honest than its own caption.

**How to apply:** For any figure containing multiple groups of bars and a threshold dashed line, check each bar individually against the line. Do not assume "all pass" or "all fail" from the grouped color coding alone. BH-FDR significance and threshold-crossing are different criteria — an axis can be BH-FDR significant without exceeding the hypothesis threshold.
