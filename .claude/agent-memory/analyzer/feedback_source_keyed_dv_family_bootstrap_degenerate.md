# Source-keyed DV ⇒ target-family bootstrap degenerates (recompute source-clustered)

Context: #722/#833 map-fit line (M: c_C → v; 16 source-keyed inputs × 30 targets).

The function-change DV `|(M(c) − M0(c)) · r̂_B|` is CONSTANT across each source's 30
targets (the map input varies only with source), so the harness's target-family
clustered bootstrap returns DEGENERATE CIs (point = lo = hi) on every
median/mean of it — every family has the same per-source composition, so every
resample yields the identical statistic. Structural, not a bug; #722 shipped it
as a "cross-behavior pattern carries the inference" caveat.

Fix that costs minutes: the cells JSONs persist `per_cell.{source_cids, proj_*}`
— collapse to the 16 per-source values (assert within-source std < 1e-12 first),
then bootstrap over SOURCES (resample 16 with replacement, 2000 draws, seed 42).
In #833 this turned the degenerate reads into real CIs and exposed that the raw
paired on-minus-off delta excludes zero POSITIVE while the floor-unit delta at
L14 excludes zero NEGATIVE (normalization-sensitive — a finding, not noise).

Chain-ρ CIs are NOT affected (the chain correlates against E, which varies per
(source, target) cell).

Sibling note: recomputing the ridge LOCO chain on CPU float64 from the joined
cache reproduces r7e's GPU (FIT_DEVICE=cuda) persisted ρ only to ~0.002 (PRESS-λ
near-tie) — assert with tolerance 1e-2, title figures with the PERSISTED value.
