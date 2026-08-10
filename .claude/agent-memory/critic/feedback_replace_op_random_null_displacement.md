---
name: replace-op-random-null-displacement
description: "Random-direction nulls for component-REPLACE ops (vs cap ops): norm-matching does not impact-match — realized displacement |proj_def − ⟨h,v̂⟩| is anisotropy-confounded (axis≈PC1); demand |Δproj| telemetry on BOTH contrast sides; single-seed + per-layer-independent draws are scope caveats not REVISEs (#2203 v8)"
metadata:
  type: feedback
---

Alternatives-lens disposition for norm-matched random-direction controls on
component-REPLACE interventions (first seen #2203 plan v8, axis-component-
replace random null).

Rule: for a CAP op, re-computing τ as a percentile of the random direction's
OWN projections impact-matches the null by construction (see
[[capping-steering-plan-stats-traps]] item 1). For a REPLACE op
(`h + v̂·(proj_def − ⟨h,v̂⟩)`) there is NO such mechanism: the realized edit
magnitude is the projection gap along the direction, and when the real axis
is high-variance (cos(axis,PC1)=0.80 in #2203), an isotropic norm-matched
random direction displaces the state ~|d|/√dim (≈1/60 at d=3584) of what
the real axis does. A "Confirmed axis-specificity" read is then consistent
with "any large-displacement component replace works."

**Why:** this survived an otherwise excellent #2203 v8 amendment that got
footprint-matching, seed parity, Indeterminate-first lattice, and all-rows
retention right. It is recoverable (APPROVE + Concern), NOT a REVISE,
WHEN the plan persists realized projection before/after for the new arms —
but check whether the REUSED comparator side has displacement telemetry:
#2203's parent committed only counts (`total_positions_edited`,
`mean_fired_frac`), so the A-side |Δproj| was unrecoverable without a rerun.

**How to apply:** any plan contrasting a real-direction REPLACE/patch arm
against a random-direction null. (1) Classify the displacement mismatch as
an analyzer Concern with a named residual control (variance-matched /
top-PC-subspace random direction) — REVISE only if neither side's |Δproj|
is reported anywhere (then a Confirmed is unweighable). (2) Near-zero
R-side |Δproj| ⇒ the control is a near-no-op; narration leans Indeterminate.
(3) Single seeded draw + per-layer-independent random directions vs a
cross-layer-coherent axis are scope caveats (convention-matched to parent
null designs), never blockers. (4) CI-overlap⇒Falsified lattices: ask the
analyzer to narrate partial genericity from point estimates, not the binary
label.
