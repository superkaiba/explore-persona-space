---
name: mixed-regime-selection-axis
description: "Runtime estimator branches / mixed expected-degenerate+fit cells inside a global-max selection axis: benign iff algebraic-identity branch + null recomputes through the identical engine path; the REAL asymmetry to police is lambda-selection/criterion mismatch between observed-fit path and null engine (#2061 v12)"
metadata:
  type: feedback
---

A global-max selection axis whose cells may differ in ESTIMATOR REGIME
(runtime primal-vs-Gram/dual ridge branch keyed on realized n vs d) or in
EXPECTED DEGENERACY (near-constant-input cells beside real fit cells) does
NOT by itself break the selection-symmetric read.

**Why:** (1) primal↔dual ridge is an algebraic identity (nonzero eigenvalues
of XᵀX and XXᵀ coincide; GCV dof identical), so a runtime branch changes cost
+ disclosure, never the statistic — #2061's parent measured 2.2e-16
primal-vs-Gram agreement on the same data; a max over 262k features cares
about gaps between top competitors, orders above float noise. (2)
Expected-degenerate cells (near-constant X ⇒ ΔR²_j ≈ 0) contribute ~nothing
to observed OR null max — worst case they inflate the shared band via
rare-feature noise (power loss, never type-I), diagnosable from persisted
per-cell per-draw max arrays (prefix-band-contribution fraction + a
subset-only re-reduction with matched null, zero recompute). (3) The
asymmetry that ACTUALLY matters for a max statistic is
λ-selection/criterion mismatch between the observed-fit path (full GCV
criterion) and the null engine (subsampled criterion) — demand a persisted
per-cell identity delta (recompute the no-flip statistic through the SAME
engine path) + a registered analyzer duty to use the engine-path observed
max for an exactly-symmetric headline read when the gap is non-trivial.

**How to apply:** on a regime-flip disposition, RETAIN + matched-null-by-
construction + per-cell disclosure beats exclusion (data-dependent axis
surgery — the selection-asymmetry class the null exists to prevent).
Check: (a) the null engine fail-louds rather than silently fitting an
undeclared regime; (b) per-draw per-cell max arrays persisted so subset
re-reductions are post-hoc recoverable; (c) MC-SE of the global-max
quantile depends on DRAW COUNT only, never axis width (each draw yields one
realization of the selection statistic). (#2061 v11-v12, 2026-08-06.)
