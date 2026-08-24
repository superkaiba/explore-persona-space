---
name: nested-floor-sweep-review
description: Reviewing threshold/floor-sensitivity sweeps over banked instruments — compositional cross-floor deltas, any-over-grid verdict-mobility claims, defined-only conditioning
metadata:
  type: feedback
---

Nested alive-set threshold sweeps (an eligibility floor swept over a FIXED banked instrument, alive sets nested supersets) have three recurring measurement shapes (#2476 floor-sensitivity round, APPROVE with concerns):

1. **Cross-floor deltas are purely compositional.** Per-unit statistics (per-feature R², retrieval rank) are floor-invariant — the floor only changes which units enter the aggregate. So a cross-floor median change is entirely carried by the newly-admitted band. Demand: per-unit arrays + per-floor masks persisted (analyzer can then decompose into marginal bands [floor_i, floor_{i-1})); recommend the marginal-band read explicitly. Nesting disclosed + per-floor counts in every table ⇒ not a confound, it IS the question — Concern, not REVISE.
2. **"Verdict changes at ≥1 swept point" is an any-over-grid existential** (H-B shape). With nested sets + identical stat seeds across points, the per-point tests are strongly positively dependent, so effective multiplicity ≪ grid size; when all points are reported symmetrically and the decision rule stays registered at the original point only, this is a Concern for the analyzer's narration, not a REVISE under lens item 11's N/A escape.
3. **Defined-only aggregates condition on the sweep variable.** Undefined-DV drops (near-constant features → R² undefined) are activity-selected — exactly the units the loosened floor admits — biasing the defined-only median upward at loose floors. Plan-side remedy: per-floor per-tier dropped counts reported + a census-only demotion when > half a cell drops (adequate); analyzer reads medians and drop counts together.

**Why:** all three passed cleanly in #2476 v6 because the plan persisted per-feature unions + masks and kept the registered cell at the original floor; absent those, shapes 1-2 become REVISEs.
**How to apply:** any plan sweeping an eligibility threshold / alive floor / inclusion cutoff over banked per-unit stats — check persistence of per-unit arrays + masks first, then the registration status of the original cell.
