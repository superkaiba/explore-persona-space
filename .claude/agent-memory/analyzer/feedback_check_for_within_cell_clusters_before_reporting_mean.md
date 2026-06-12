---
name: check-for-within-cell-clusters-before-reporting-mean
description: A "partial saturation" mean can hide multimodal cluster structure; sort by the metric and eyeball the distribution before promoting an average to the TL;DR
metadata:
  type: feedback
---

When an aggregate mean lands mid-range (e.g. "selectivity 0.17" on a 0-1 scale), do not report it as the headline without sorting the underlying cells and checking the distribution shape — a mid-range mean across qualitatively distinct behavior modes misleads the mentor.

**Why:** task #397 round 1 — I reported E1 as "partial: selectivity 0.17". The critic sorted the 24 cells: trimodal — 3 clean (~0.78), 16 lockstep failures (<0.30), 3 dead (~0). The real story was "collapses for 16/24, works for 3/24 (which share substrate B=1 + D=1 — a free lead), fails for 3/24"; the mean suppressed all of it.

**How to apply:** sort per-cell metric within the stratum, print, and bin into behavior modes (clean / lockstep / dead or analogous). If ≥3 cells fall in any non-headline cluster, state the cluster structure explicitly and report cluster-conditional means alongside the overall mean.

Related: `[[position_distribution_when_marker_eval]]` (aggregate rates hiding qualitative structure).
