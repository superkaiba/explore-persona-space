---
name: trap-f-space-gates-shrinking-denominators
description: Fraction-of-swap F traps — pooled gates must respect the anchor-separation exclusion (degenerate pairs have unbounded F noise), and depth-shrinking denominators turn CI-straddle verdict branches into failure-to-reject mislabels
metadata:
  type: feedback
---

Two recurring statistics traps in the #2094/#2162 fraction-of-swap F line
(F = (Δ_patched − Δ_floor)/(Δ_ceiling − Δ_floor), pre-registered exclusion
|ceiling − floor| ≥ 0.5):

1. **Any gate or pooled statistic over PAIRS must pool SURVIVING pairs only.**
   A non-surviving pair's F has a near-zero denominator, so its per-pair noise
   is unbounded (observed: a parent pair at sep 0.095 with F = −1.105). A halt
   gate stating "all N pairs" while grounding its σ on a surviving-pair figure
   has an unbounded false-HALT channel from a handful of degenerate pairs.
   Check: recompute the gate's stated n against the survivors in the committed
   f_cells.jsonl (`abs(separation) >= 0.5`).

2. **A verdict-lattice branch keyed on "95% CI straddles 0" is a
   failure-to-reject read, not a finding** — especially where the anchor gap
   SHRINKS with the design axis (in #2162 the gap fell 1.56 → 0.57 from depth
   1 → 5 while surviving n fell 36 → 18, so F-space noise inflates ~3× exactly
   where the verdict is read). Require the straddle branch to ALSO exclude the
   pre-registered effect size (CI upper bound below the operationalized
   alternative) before it earns a substantive label; otherwise route to
   No-verdict/underpowered. This is the lattice form of the user's standing
   null-framing correction ("indistinguishable from null given the variance",
   never "confirms the null").

**Why:** both fired in #2162 plan v7 (turn-boundary-multipatch review,
2026-08-14); the denominators/exclusion are inherited by every follow-up on
this bank, so the traps recur per round.
**How to apply:** whenever a plan in this line registers a pooled pair-level
gate or a CI-based verdict lattice, run the two checks above against the
committed f_metrics artifacts before passing the gate section. Related:
[[trap-minimal-pair-span-locus-degeneracy]].
