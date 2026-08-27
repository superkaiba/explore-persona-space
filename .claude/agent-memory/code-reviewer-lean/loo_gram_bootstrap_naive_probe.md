---
name: loo-gram-bootstrap-naive-probe
description: Verify a batched LOO-within-resample Gram-matrix bootstrap by probing it against a naive per-draw loop; check set_counts cache ordering and filtered-vs-unfiltered axis inputs
metadata:
  type: feedback
---

When a diff implements a batched bootstrap of an LOO axis statistic via Gram
algebra (S*_b = counts·obs; member cos vs S*_b − obs_i; denom via
‖S‖²−2·obs·S+‖obs‖²), do NOT review the algebra by eye: write a /tmp probe
that importlib-loads the module and compares every loading surface
(member/non-member × pred/obs sides, plus ss and denom) against a naive
per-draw loop at ~1e-10, plus the weighted-median helper against a naive
weighted median with NaN + zero-weight rows (#2617 r1 g2: probe passed and
converted a FAIL-risk review into targeted CONCERNS in minutes).

**Why:** the algebra has many sign/off-by-one traps (own-obs removal, LOO
denominators, counts-cache ordering) that read plausibly either way; a probe
settles all of them at once.

**How to apply:** also check (a) a `set_counts()`-style mutable cache is
re-set before each draw-set's loadings (per-pair vs family-clustered blocks
ordered correctly); (b) the point estimate and the bootstrap are fed the SAME
rows — a finite-filter applied to one but not the other misaligns member
LOO subtraction (fail-loud shape crash if rowwise_cos requires equal rows,
silent NaN-poisoned CIs otherwise); (c) leave-one-INSTANCE-out (subtract obs_i
once per resampled instance) is the correct resample analog of LOO. Related:
[[shared_null_draw_set_review_recipe]], [[selection_symmetric_band_review_recipe]].
