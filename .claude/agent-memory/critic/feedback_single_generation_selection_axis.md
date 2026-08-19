---
name: single-generation-selection-axis
description: "Capture-generation-mixed max-selection axes: union of near-duplicate cells = conservative-but-power-losing double-counted stringency; single-generation axis + parked robustness arm is the right disposition (#2061 v7)"
metadata:
  type: feedback
---

When a global max/argmax selection axis could union cells from TWO capture
generations with near-duplicate corpora (subset-vs-full row pools, e.g.
gsm8k_train5k vs gsm8k_train_full), the union is the WRONG axis even though
"wider = more stringent" sounds safer.

**Why:** (1) A within-cell permutation null shuffles labels independently per
cell (a shared seed on different-length row sets does not align permutations),
so null cross-cell structure is near-independent while true statistics at
row-overlapping cells are positively correlated ⇒ null max over the union is
stochastically LARGER than the true joint max under H0 ⇒ conservative type-I
but strictly power-losing — the extra cells' stringency is double-counted,
not independent. (2) With ZERO shared stems there is no cross-generation
replication read, so a max attained at the other generation's cell has no
anchor against capture drift — an interpretability cost at the SELECTION
level even when each per-cell delta is within-generation-clean. (3) The union
can re-import degenerate-regime cells (n<d, low-n rare-feature noise) that
inflate the shared band for every well-posed cell.

**How to apply:** approve a single-generation registered axis with the other
generation parked as a documented robustness arm (own approval round). Check
the narrowing is re-registered EVERYWHERE (metric, success/kill, hypothesis,
per-cell spurious-count arithmetic, persistence contract, fail-loud cell-count
guards) and that the grid choice was structure-informed (dir/inventory probe),
never outcome-informed. The MC-SE on a global-max quantile depends on draw
count only, not cell count — axis narrowing does not touch it. (#2061 v7-v9,
2026-08-05; probe marker 2026-08-05T21:48:40Z.)
