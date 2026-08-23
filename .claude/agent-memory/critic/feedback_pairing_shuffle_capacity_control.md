---
name: pairing-shuffle-capacity-control
description: A refit-on-shuffled-pairings map control likely COLLAPSES (GCV to grid edge, near-zero output); pair it with the spectrum/capacity-matched weight-permutation control of the TRUE map or the modal branch is unweighable (#1739 claim4)
metadata:
  type: feedback
---

When a plan's load-bearing control for a "probe-on-mapped-features beats
probe-on-raw-features" claim is a map REFIT on pairing-destroyed data
(shuffled context–answer pairs, same recipe), the modal outcome is control
COLLAPSE: a pairing-free ridge/GCV fit shrinks to the grid edge and emits
near-constant output, so the probe on it is trivially weak and the margin
(Δ_true − Δ_shuf) > 0 is minted for capacity reasons, not mechanism. In
that branch the design cannot distinguish "learned pairing is load-bearing"
from "any variance/spectrum-preserving linear transform buys the probe gain
(ridge-geometry artifact)" — λ diagnostics + kNN manipulation checks let
the analyzer NAME the collapse but not RESOLVE the alternative.

**Fix (Must-Fix shape):** have the complementary capacity-matched control
ride the same pass — a weight-row-permutation of the TRUE fitted map
(#1739's `arm20_shuffled_map_ridge` / `shuffled_map_weights`,
rank/Frobenius-preserving; output = W·P·x, an invertible linear transform
of x, so it stays probe-feedable by construction). Near-free (consumes the
already-fitted map, closed-form). The two controls have complementary
blind spots: pairing-shuffle matches fit PROCEDURE but risks capacity
collapse; weight-perm matches CAPACITY/spectrum but not procedure. With
both, every branch is interpretable.

**Why:** #1739 claim4-controls (2026-08-19) planned only the
pairing-shuffle under P-B and itself flagged collapse as "PARTLY EXPECTED"
(§8 risk row); the historical precedent control that MATCHED the old
protocol's gains was the weight-permutation (body.md L107: control matched
+0.008…+0.017; fair_v2 arm20 reads 0.14–0.18 — non-degenerate), so a
skeptic's first question about a strong-form verdict is "did the old
control family run at the new protocol?".

**How to apply:** any shuffled/null-refit control design where the null
family can degenerate under the fit procedure — check whether a
matched-capacity consumption-side control (permute the FITTED artifact,
not the training pairs) exists in the codebase and is cheap to ride along.
Related: [[null-calibration-fixed-design-matrix]] (#555 discriminators for
null families), [[rank1-mechanism-test-confounds]] (anisotropy nulls for
cosines).
