---
name: reproduction-vs-new-computation
description: When an experiment reuses a prior axis as a reproduction check AND computes a new version of the same axis, the Summary can silently carry the old reproduction value instead of the new one
metadata:
  type: feedback
---

When a body has both a reproduction-sanity check (e.g., "Method-A centroid from #142 reproduces at 0.567 ± 0.03") and a new computation of the same axis type (e.g., "pcentroid_methodA_L20 = 0.788 on Phase 2"), check that the Summary and TL;DR bullets cite the NEW value, not the reproduction-check value.

**Why:** The reproduction check is a sanity gate (confirms the existing implementation works); the new computation is the actual result being reported. Analyzers sometimes copy the reproduction number into the "centroid baseline beat Chen-style" claim because 0.567 appears prominently in `reproduction_sanity.json`, while 0.788 is in `per_axis_stats.json`.

**How to apply:** For any centroid/axis that appears in BOTH a `reproduction_sanity.json` and a `per_axis_stats.json`, cross-check which value the Summary citation uses. If it matches the reproduction value but the per_axis_stats value is different (and higher), flag it as a factual error.
