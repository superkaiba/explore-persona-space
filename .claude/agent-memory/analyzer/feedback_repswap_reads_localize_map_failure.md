---
name: repswap reads localize which representation kills a cross-regime map
description: "In the #825/#1345 reparam battery, ceilings.repswap_* + comp_repmap_* separate a context-side failure from an answer-side/operator failure; delta_terms = recovered - max(within-0.05, null)"
type: feedback
---

When a paired reparam leg (`_layer_battery`, `issue825_map_alignment.py`) gives an
asymmetric verdict (one direction recovers, the other fails), read the battery
internals before narrating "different operator":

- `ceilings.repswap_i2b` = DIRECT ridge from regime-i CONTEXT to regime-b ANSWER
  activations (paired rows). If this is high while regime-b's own within-R² is
  negative, the b-side ANSWER representation is intact and the failure is
  localized to the b-side CONTEXT slot (errors-in-variables attenuation), not the
  operator. #1345 paired stories: chat-ctx→story-answer 0.56 vs story within
  −0.31; story→chat samefn recovery 0.61 (above the matched chat ceiling 0.24)
  while chat→story recovery −0.17 fell below its capacity null −0.06.
- `delta_terms[d] = recov[d] − max(within[d] − 0.05, null_recovery_r2[d])` — the
  published per-direction delta is NOT recovered−within; when within is deeply
  negative the binding term is the matched-capacity null.
- Composition recovery can EXCEED the target's own matched-n ceiling (3 ridge
  stages out-generalize 1 at small n); matched-capacity nulls are what license
  the claim, not the ceiling comparison.

**Why:** #1345 r3 (2026-07-17) — the framing-effect verdict would have read as
"story operator is different" without the repswap split; instead the honest
claim is "one operator, story-side context coordinates unstable."

**How to apply:** any asymmetric reparam verdict on this line → quote repswap
both ways + the null the delta actually bound against.
