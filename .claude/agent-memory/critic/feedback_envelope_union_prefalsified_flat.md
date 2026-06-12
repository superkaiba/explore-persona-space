---
name: Envelope-union success criteria can be pre-falsified on the "flat" branch
description: For 3rd-dial-point / trend designs, compute the anchor-vs-anchor span in seed-SD units under the EXACT statistic H1 names before accepting a "no gradient" headline; pooling sub-metrics (a/b mixing) can mask disjoint clusters
type: feedback
---

When a plan adds a middle dial point between two anchors and declares success = "all cells inside the envelope [union of the two anchors' observed ranges] → no gradient," check two things against the anchors' per-cell JSONs BEFORE accepting the framing (#550, 2026-06-10):

1. **Compute the anchor-to-anchor span in pooled seed-SD units under the exact statistic the hypothesis names.** In #550, H1's flatness yardstick was "across-dial span ≤ 2× pooled across-seed SD" — but the two anchors ALREADY violated it: worse-of-A/B GD3 clusters were disjoint (1.359–1.384 vs 1.326–1.341, ≈4.4× pooled SD, monotone down) and GD1 top-1 SV share ≈2.2× SD monotone up. The "flat" branch of H1 was pre-falsified by data sitting in git; only "non-monotone" could save it, and the mid point would likely land between → small clean monotone gradient, contradicting the planned "no gradient at any reachable dial" headline while the envelope criterion still PASSes.
2. **Watch for sub-metric pooling that masks separation.** The plan quoted GD3 ranges pooled over singletons A and B (1.24–1.38 vs 1.22–1.34, apparently overlapping), but the hypothesis statistic was worse-of-A/B, under which the clusters don't overlap at all.

**Why:** envelope-union criteria are ~an order of magnitude wider than seed noise, so envelope-containment is nearly unfalsifiable as a gradient test; the real gradient test is the seed-SD trend read. A mechanical "success fired → no gradient" write-up would flip a true small positive into a false flat.

**How to apply:** usually NOT a REVISE if the per-cell values + seeds + realized dial positions are first-class deliverables and a descriptive trend read is pre-specified (analyzer can recover) — flag as the top analyzer concern with the computed numbers. REVISE only if the plan's deliverables omit the per-cell values needed to run the trend read.
