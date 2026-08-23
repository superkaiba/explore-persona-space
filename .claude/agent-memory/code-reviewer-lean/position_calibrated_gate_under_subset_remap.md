---
name: position-calibrated-gate-under-subset-remap
description: A threshold calibrated to POSITIONS (e.g. "first 4 layers" at 0.999) silently re-binds to the wrong units when an index-subset kwarg remaps positions — check whether bars key on index VALUE or subset POSITION (#1901 R1 g1)
metadata:
  type: feedback
---

When a module gains an index-subset kwarg (`layers=(2,14,19,26)` threaded through
a capture/persist path), re-check every threshold in that module calibrated to
POSITIONS of the full index range. #1901 R1 g1: `equivalence_check`'s two-bar
gate (#779 calibration: per-layer cosine ≥ 0.999 on the FIRST 4 layers, flat
≥ 0.995 absorbing deep-layer bf16 jitter) computed `n_early = min(4, len(layers))`
— under the 4-layer production subset the strict "early" bar re-bound to deep
layers L14/L19/L26 that the calibration deliberately exempted. Disclosed in the
docstring, fail-loud direction, and unexercised by the round's driver → Minor
concern, not blocker.

**Why:** the calibration's unit is "shallow layers" (an index-VALUE property);
the code's unit is "first entries of the list" (a POSITION property). Subsetting
makes the two diverge silently — a strictly-tighter gate can false-HALT a healthy
production run mid-pod.

**How to apply:** whenever a diff threads an index subset (layers, folds, shards,
heads) through code holding position-indexed thresholds/slices (`x[:k]`,
`min(k, len(...))`), ask which semantic the calibration meant. Severity forks on
(a) fail-loud vs silent direction and (b) whether any in-round caller arms the
path — grep the round's driver before rating. Related: [[new-dial-missing-from-resume-regime]]
(same commit shape: new dial's OTHER seams — this one checks its gate seams).
