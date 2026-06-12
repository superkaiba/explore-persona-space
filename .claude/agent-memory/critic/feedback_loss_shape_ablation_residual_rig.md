---
name: Loss-shape ablation residual-rig scoping + deliberate-saturation geometry regime
description: "#599-class single-variable loss-mask ablations: FALSIFY never exhausts the rig (lr/schedule/steps/corpus stay unmatched vs the EM band) — claim-scoping concern, not REVISE; saturated endpoints are the MATCHED regime for shift-spectrum geometry DVs when all reference arms are saturated"
type: feedback
---

Two judgments from the #599 plan review (whole-response-loss marker re-train, parent #561):

1. **Residual-rig scoping on loss-shape ablations.** A single-variable
   loss-mask swap (marker-only → EM-style whole-response CE) tests exactly
   the Goal, but the EM reference arm also differs in lr (2e-5 vs 2e-6),
   schedule (linear vs cosine), steps (200 vs 600), and corpus. A FALSIFY
   read scoped as "rig-component space exhausted" overstates — only the
   loss-shape component is ruled out. **Why:** the Goal is loss-shape-
   specific, so the design answers its own question; the overclaim lives
   in hypothesis prose the analyzer/interpretation-critic can re-scope.
   **How to apply:** concern-for-analyzer, not REVISE, when the plan's own
   prior-work section names the residual mismatch and reads concentration
   against the EM *band* rather than as a recipe-matched comparison.

2. **Deliberate saturation is the matched regime for geometry DVs.** When
   the primary DV is shift-spectrum concentration and ALL reference arms
   (persisted tensors) were read at saturated/fixed-step endpoints, keeping
   the saturated endpoint (band-stop analog off) is correct — a band-stopped
   checkpoint would be the regime MISMATCH. The marker-training-recipe
   band-stop default targets *leakage-resolution* DVs, not geometry DVs
   with demonstrated dynamic range on saturated adapters (#551/#561:
   marker 0.31–0.35 vs EM 0.52–0.60 at saturation). Don't fire Methodology
   item 11 on this pattern when lr is inside the clean window (≤5e-6) and
   the manipulation check still gates regime match.

3. Minor recurring pattern: after a fact-checker corrects a plan (e.g.
   "B0 skips callbacks"), check for STALE parentheticals elsewhere claiming
   the pre-correction behavior (e.g. §4 still saying a trajectory point is
   "visible at B0+50 in the smoke logs"). Textual residue = concern with a
   pointer, not REVISE, when the corrected section names a valid later
   verification point (first production callback fire).
