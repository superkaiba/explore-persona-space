---
name: caption-n-positions-mismatch
description: Position-figure caption states n=X firing trials but JSON n_positions field shows a different count — off-by-small-integer errors in caption text not reflected in figure bars
metadata:
  type: feedback
---

When a position-signature figure shows "fraction of marker_B emissions" for each cell, the caption states the sample size as "n=X" where X is the count of trials where marker_B fired. This is read from `n_positions` in the eval JSON. In at least one case (issue #354, Figure 3, police_officer cell) the caption said n=21 but the JSON recorded n_positions=19 — a 2-unit discrepancy.

**Why:** The caption is drafted from memory or intermediate analysis; the figure itself does not label n per bar, so the mismatch is invisible from the visual alone. Only checking JSON catches it.

**How to apply:** For every position-signature figure, verify each stated n in the caption against `n_positions` in the corresponding per_persona block of summary.json. Flag any mismatch, even small ones — they erode trust in the aggregates.
