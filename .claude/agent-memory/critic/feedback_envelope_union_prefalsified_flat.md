---
name: Envelope-union success criteria can be pre-falsified on the "flat" branch
description: For mid-dial / trend designs, compute the anchor-vs-anchor span in seed-SD units under the EXACT statistic H1 names before accepting a "no gradient" framing; sub-metric pooling can mask disjoint clusters (#550)
type: feedback
---

When a plan adds a middle dial point between two anchors and declares success = "all cells inside the envelope [union of the anchors' observed ranges] → no gradient", check two things against the anchors' per-cell JSONs FIRST (#550, 2026-06-10):

1. **Compute the anchor-to-anchor span in pooled seed-SD units under H1's exact statistic.** #550's flatness yardstick was "span ≤ 2× pooled seed SD" — but the anchors ALREADY violated it (worse-of-A/B GD3 clusters disjoint at ≈4.4× pooled SD, monotone; GD1 top-share ≈2.2× SD monotone). The "flat" branch was pre-falsified by data in git; the mid point would likely land between → a small clean monotone gradient, contradicting the planned "no gradient" headline while the envelope criterion still PASSes.
2. **Watch sub-metric pooling that masks separation.** The plan quoted ranges pooled over singletons A and B (apparently overlapping), but the hypothesis statistic was worse-of-A/B, under which the clusters don't overlap at all.

**Why:** envelope-union criteria are ~an order of magnitude wider than seed noise — envelope containment is nearly unfalsifiable as a gradient test; the real test is the seed-SD trend read. A mechanical "success fired → no gradient" write-up flips a true small positive into a false flat.

**How to apply:** usually NOT a REVISE if per-cell values + seeds + realized dial positions are first-class deliverables and a descriptive trend read is pre-specified — flag as the top analyzer concern with the computed numbers. REVISE only if deliverables omit the per-cell values needed for the trend read.
